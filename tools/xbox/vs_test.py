"""Run a synthetic RDNA2 vertex shader through the execution harness.

The guest vertex program reads the vertex index from v5 and builds the triangle
(-1,-1), (4,-1), (-1,4), which covers the whole target. It exports that position
and one parameter holding (x, y, vertex index, 1). A GLSL fragment shader writes
the interpolated parameter to an RGBA32F target.

W is 1 at every vertex, so interpolation is linear in screen space: at each
pixel centre the parameter must equal the pixel's own clip-space x and y, and
its third channel ((x + 1) + 2 (y + 1)) / 5, the weighted vertex index. That
checks the vertex index input, the position export and the parameter export.
"""
import argparse
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rdna_asm

WIDTH, HEIGHT = 13, 7
TOLERANCE = 0.00001
FRAGMENT = """#version 450
layout(location=0) in vec4 parameter;
layout(location=0) out vec4 outColor;
void main() {
    outColor = parameter;
}
"""
ONE, MINUS_ONE = ("r", 242), ("r", 243)


def program():
    p = rdna_asm.Program()
    p.vop2("v_and_b32", 0, 1, "v5")
    p.vop2("v_lshrrev_b32", 1, 1, "v5")
    p.vop1("v_cvt_f32_u32", 0, "v0")
    p.vop1("v_cvt_f32_u32", 1, "v1")
    p.vop1("v_cvt_f32_u32", 4, "v5")
    p.vop1("v_mov_b32", 2, ("lit", struct.unpack("<I", struct.pack("<f", 5.0))[0]))
    p.vop2("v_mul_f32", 0, "v2", "v0")
    p.vop2("v_mul_f32", 1, "v2", "v1")
    p.vop2("v_add_f32", 0, MINUS_ONE, "v0")
    p.vop2("v_add_f32", 1, MINUS_ONE, "v1")
    p.vop1("v_mov_b32", 2, 0)
    p.vop1("v_mov_b32", 3, ONE)
    p.words += [0xF80008CF, 0x03020100]
    p.words += [0xF8000A0F, 0x03040100]
    p.sopp("s_endpgm")
    return p.bytes()


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
    with tempfile.TemporaryDirectory(prefix="vertex ") as temp:
        work = Path(temp)
        (work / "fs.frag").write_text(FRAGMENT, encoding="ascii")
        subprocess.run([compiler, "-V", "fs.frag", "-o", "fs.spv"], cwd=work, check=True, timeout=60)
        (work / "vs.bin").write_bytes(program())
        (work / "scenario.txt").write_text(
            f"stage vs\nwave 64\ntarget {WIDTH} {HEIGHT}\nfs fs.spv\ncode vs.bin\nout out\n",
            encoding="ascii")
        subprocess.run([runner, "scenario.txt"], cwd=work, check=True, timeout=90)
        actual = (work / "out" / "target0.bin").read_bytes()
    if len(actual) != WIDTH * HEIGHT * 16:
        raise SystemExit(f"unexpected RGBA32F readback size: {len(actual)} bytes")
    worst = 0.0
    for i, rgba in enumerate(struct.iter_unpack("<4f", actual)):
        y, x = divmod(i, WIDTH)
        cx, cy = (x + 0.5) / WIDTH * 2 - 1, (y + 0.5) / HEIGHT * 2 - 1
        expected = (cx, cy, ((cx + 1) + 2 * (cy + 1)) / 5, 1.0)
        error = max(abs(a - b) for a, b in zip(rgba, expected))
        if not error <= TOLERANCE:
            raise SystemExit(f"pixel ({x}, {y}): expected {expected}, read {rgba}")
        worst = max(worst, error)
    print(f"PASS: vertex shader positioned and shaded {WIDTH * HEIGHT} pixels; max error {worst:.3g}")


if __name__ == "__main__":
    main()
