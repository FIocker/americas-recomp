"""Check ds_swizzle_b32 quad exchanges and s_wqm_b32 expansion in a pixel shader.

The first fragment program exchanges each pixel with its horizontal and vertical
neighbour inside the 2x2 quad, subtracts, and exports the values and differences.
The second narrows EXEC to one lane per quad, expands it with s_wqm_b32, and
exports a constant; a second run activates two quads only, so the other quads
must stay clear. Both run with XBOX_NO_FRAGMENT_BARY=1, the host perspective
path; the quad operations are independent of the barycentric extension.

The two-quad mask assumes the 32-lane pixel wave runs as one host subgroup.
exec builds pixel shaders for the device's subgroupSize without requesting a
size, and the driver may run the fragment stage narrower: Intel ANV runs it 16
wide (and cannot pin it, FRAGMENT is not in its requiredSubgroupSizeStages), so
EXEC applies per 16 lanes and twice the quads light. A third program exports
each lane's v_mbcnt_lo_u32_b32 index; when the observed lane span (highest index
plus one) is below 32, the two-quad case is skipped rather than failed. This is
a heuristic: it reads the lanes that wrote pixels in the probe shader, which is
compiled separately from the s_wqm shader. A probe image that does not look
like lane indices counts as a failure, and both s_wqm cases still run.
"""
import argparse
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile


WIDTH = HEIGHT = 32
TOLERANCE = 1e-5
QUAD_VERTEX = """#version 450
layout(location=0) out vec4 p;
void main() {
    vec2 v[3] = vec2[](vec2(-1,-1), vec2(3,-1), vec2(-1,3));
    gl_Position = vec4(v[gl_VertexIndex], 0, 1);
    p = vec4(v[gl_VertexIndex], 0, 1);
}
"""
WQM_VERTEX = """#version 450
void main() {
    vec2 v[3] = vec2[](vec2(-1,-1), vec2(3,-1), vec2(-1,3));
    gl_Position = vec4(v[gl_VertexIndex], 0, 1);
}
"""


def interp_pair(destination, channel):
    first = 0xC8000000 | (destination << 18) | (channel << 8)
    second = 0xC8000000 | (destination << 18) | (1 << 16) | (channel << 8) | 1
    return first, second


def swizzle(destination, source, offset):
    return 0xD8000000 | (53 << 18) | (offset & 0xffff), (destination << 24) | (source & 0xff)


def subtract(destination, first, second):
    return (4 << 25) | (destination << 17) | (second << 9) | (256 + first)


def export(target, sources):
    return 0xF800080F | (target << 4), sum((sources[i] & 0xff) << (8 * i) for i in range(4))


def move(destination, constant):
    return 0x7E000000 | (destination << 17) | (1 << 9) | constant


def quad_program():
    words = []
    words += interp_pair(2, 0)
    words += interp_pair(3, 1)
    words += swizzle(4, 2, 0x80B1)
    words.append(subtract(5, 4, 2))
    words += swizzle(6, 3, 0x804E)
    words.append(subtract(7, 6, 3))
    words += export(0, [2, 3, 5, 7])
    words.append(0xBF810000)
    return struct.pack(f"<{len(words)}I", *words)


def wqm_program(active_mask):
    words = [move(2, 242), 0xBEFE03FF, active_mask, 0xBEFE097E, *export(0, [2, 2, 2, 2]), 0xBF810000]
    return struct.pack(f"<{len(words)}I", *words)


def lane_program():
    # v_mbcnt_lo_u32_b32 v2, -1, 0; v_cvt_f32_u32 v2, v2: each pixel exports its lane index.
    words = [0xD7650002, 0x020100C1, 0x7E040D02, *export(0, [2, 2, 2, 2]), 0xBF810000]
    return struct.pack(f"<{len(words)}I", *words)


def lane_span(image):
    lanes = set()
    for value in image[0::4]:
        if not 0.0 <= value < 64.0 or value != int(value):
            raise SystemExit(f"exported {value!r}, not a lane index")
        lanes.add(int(value))
    span = max(lanes) + 1
    if lanes != set(range(span)) or span < 4 or span & (span - 1):
        raise SystemExit(f"lanes {sorted(lanes)[:8]} (of {len(lanes)}) are not a contiguous power-of-two span")
    return span


def compile_vertex(compiler, work, name, source):
    (work / f"{name}.vert").write_text(source, encoding="ascii")
    subprocess.run([compiler, "-V", f"{name}.vert", "-o", f"{name}.spv"],
                   cwd=work, check=True, timeout=60)
    return work / f"{name}.spv"


def run(runner, work, vertex, program, interpolants):
    (work / "fragment.bin").write_bytes(program)
    (work / "scenario.txt").write_text(
        f"stage ps\nwave 32\ntarget {WIDTH} {HEIGHT}\ninputaddr 0x2\ninterp {interpolants} 0\n"
        f"vs {vertex.name}\ncode fragment.bin\nexport 0 9 0xe4\nout out\n", encoding="ascii")
    result = subprocess.run([runner, "scenario.txt"], cwd=work, capture_output=True, text=True,
                            timeout=90, env=dict(os.environ, XBOX_NO_FRAGMENT_BARY="1"))
    if result.returncode:
        raise SystemExit((result.stdout + result.stderr).strip()[-800:])
    text = result.stdout.strip()
    if "RAN" not in text:
        raise SystemExit(text[-800:] or result.stderr[-400:])
    data = (work / "out" / "target0.bin").read_bytes()
    if len(data) != WIDTH * HEIGHT * 16:
        raise SystemExit(f"unexpected RGBA32F byte count: {len(data)}")
    return struct.unpack(f"<{WIDTH * HEIGHT * 4}f", data)


def verify_quad(image):
    maximum = 0.0
    bad = []
    for py in range(HEIGHT):
        for px in range(WIDTH):
            x = 2.0 * (px + 0.5) / WIDTH - 1.0
            y = 2.0 * (py + 0.5) / HEIGHT - 1.0
            nx = 2.0 * ((px ^ 1) + 0.5) / WIDTH - 1.0
            ny = 2.0 * ((py ^ 1) + 0.5) / HEIGHT - 1.0
            want = (x, y, nx - x, ny - y)
            got = image[(py * WIDTH + px) * 4:(py * WIDTH + px) * 4 + 4]
            error = max(abs(a - b) for a, b in zip(got, want))
            maximum = max(maximum, error)
            if error > TOLERANCE:
                bad.append(((px, py), got, want))
    if bad:
        raise SystemExit(f"{len(bad)} pixels differ; first {bad[:3]}")
    return maximum


def verify_wqm(image, expected):
    written = [value for value in image if value == 1.0]
    other = [(i // 4, value) for i, value in enumerate(image) if value != 1.0]
    if len(written) != expected or any(value != 0.0 for _, value in other):
        raise SystemExit(
            f"{len(written)} channels at 1.0, expected {expected}; "
            f"{sum(value != 0.0 for _, value in other)} other channels are not clear")
    return len(written)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exec", type=Path,
                        default=Path(__file__).resolve().parents[2] / "build-xbox" / "exec.exe")
    runner = parser.parse_args().exec.resolve()
    if not runner.is_file() and runner.with_suffix(".exe").is_file():
        runner = runner.with_suffix(".exe")
    compiler = shutil.which("glslangValidator")
    if not runner.is_file() or compiler is None:
        parser.error("provide a built --exec runner and put glslangValidator on PATH")
    failures = 0
    with tempfile.TemporaryDirectory(prefix="quad test ") as temp:
        work = Path(temp)
        quad_vertex = compile_vertex(compiler, work, "quad", QUAD_VERTEX)
        wqm_vertex = compile_vertex(compiler, work, "wqm", WQM_VERTEX)
        try:
            maximum = verify_quad(run(runner, work, quad_vertex, quad_program(), 1))
            print(f"PASS quad swizzle: {WIDTH * HEIGHT} pixels, max error {maximum:.3g}")
        except SystemExit as error:
            failures += 1
            print("FAIL quad swizzle:", error)
        try:
            span = lane_span(run(runner, work, wqm_vertex, lane_program(), 0))
        except SystemExit as error:
            failures += 1
            span = None
            print("FAIL lane probe:", error)
        for mask, expected, label in ((0x11111111, 4096, "all quads"),
                                      (0x00000011, 1024, "two quads")):
            if label == "two quads" and span is not None and span < 32:
                print(f"SKIP s_wqm ({label}): the pixel shader ran with lane indices 0..{span - 1}, "
                      f"so a 32-lane EXEC mask applies per {span} lanes")
                continue
            try:
                count = verify_wqm(run(runner, work, wqm_vertex, wqm_program(mask), 0), expected)
                print(f"PASS s_wqm ({label}): {count} channels at 1.0, rest clear")
            except SystemExit as error:
                failures += 1
                print(f"FAIL s_wqm ({label}):", error)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
