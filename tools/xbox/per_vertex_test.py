"""Check per-vertex VINTRP parameters and six simultaneous colour targets on the Vulkan runner."""
import argparse
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile


WIDTH = HEIGHT = 32
VERTEX_SHADER = """#version 450
layout(location=0) out vec4 color0;
layout(location=1) out vec4 color1;
void main() {
    vec2 vertices[3] = vec2[](vec2(-1,-1), vec2(3,-1), vec2(-1,3));
    vec3 rgb[3] = vec3[](vec3(1,0,0), vec3(0,1,0), vec3(0,0,1));
    vec4 alt[3] = vec4[](vec4(0.25,0.5,0.75,1.0), vec4(0.5,0.75,1.0,1.25), vec4(1.0,1.5,2.0,2.5));
    int i = gl_VertexIndex;
    gl_Position = vec4(vertices[i], 0, 1);
    color0 = vec4(rgb[i], 1);
    color1 = alt[i];
}
"""


def interpolant(destination, mode, attribute, channel):
    return 0xC8000000 | (destination << 18) | (2 << 16) | (attribute << 10) | (channel << 8) | mode


def fragment_program():
    words = []
    for attribute, base in ((0, 2), (1, 14)):
        for channel in range(4):
            words.append(interpolant(base + channel, 2, attribute, channel))
        for channel in range(4):
            words.append(interpolant(base + 4 + channel, 0, attribute, channel))
        for channel in range(4):
            words.append(interpolant(base + 8 + channel, 1, attribute, channel))
    for target in range(6):
        source = 2 + target * 4
        words += [0xF800080F | (target << 4), source | (source + 1) << 8 | (source + 2) << 16 | (source + 3) << 24]
    words.append(0xBF810000)
    return struct.pack(f"<{len(words)}I", *words)


def verify(target, expected):
    if len(target) != WIDTH * HEIGHT * 16:
        raise ValueError(f"unexpected RGBA32F byte count: {len(target)}")
    for i, rgba in enumerate(struct.iter_unpack("<4f", target)):
        if not all(math.isfinite(value) for value in rgba):
            raise ValueError(f"pixel {i}: non-finite value {rgba}")
        if rgba != expected:
            y, x = divmod(i, WIDTH)
            raise ValueError(f"pixel ({x}, {y}): expected {expected}, read {rgba}")


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
    with tempfile.TemporaryDirectory(prefix="per vertex ") as temp:
        work = Path(temp)
        (work / "color.vert").write_text(VERTEX_SHADER, encoding="ascii")
        subprocess.run([compiler, "-V", "color.vert", "-o", "color.spv"],
                       cwd=work, check=True, timeout=60)
        (work / "color.bin").write_bytes(fragment_program())
        (work / "scenario.txt").write_text(
            f"stage ps\nwave 32\ntarget {WIDTH} {HEIGHT}\ninputaddr 0x2\ninterp 2 0\n"
            "vs color.spv\ncode color.bin\n" + "".join(f"export {t} 9 0xe4\n" for t in range(6)) + "out out\n",
            encoding="ascii")
        result = subprocess.run([runner, "scenario.txt"], cwd=work, capture_output=True, text=True, timeout=90)
        if result.returncode:
            if "fragment_shader_barycentric" in result.stdout + result.stderr:
                raise SystemExit("this test needs VK_KHR_fragment_shader_barycentric")
            raise SystemExit((result.stdout + result.stderr).strip()[-800:])
        expected = (
            (1.0, 0.0, 0.0, 1.0),
            (-1.0, 1.0, 0.0, 0.0),
            (-1.0, 0.0, 1.0, 0.0),
            (0.25, 0.5, 0.75, 1.0),
            (0.25, 0.25, 0.25, 0.25),
            (0.75, 1.0, 1.25, 1.5),
        )
        for target, want in enumerate(expected):
            verify((work / "out" / f"target{target}.bin").read_bytes(), want)
    print(f"PASS: P0, P10 and P20 of two attributes matched at all {WIDTH * HEIGHT} pixels in six targets")


if __name__ == "__main__":
    main()
