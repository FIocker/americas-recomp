"""Render a coordinate image with the RDNA2 fragment recompiler and check every pixel."""
import argparse
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exec", type=Path,
                        default=Path(__file__).resolve().parents[2] / "build-xbox" / "exec.exe")
    runner = parser.parse_args().exec.resolve()
    if not runner.is_file() and runner.with_suffix(".exe").is_file():
        runner = runner.with_suffix(".exe")
    compiler = shutil.which("glslangValidator")
    if not runner.is_file():
        parser.error(f"missing Vulkan execution harness: {runner}")
    if compiler is None:
        parser.error("add the Vulkan SDK's glslangValidator to PATH")

    width, height = 13, 7
    vertex_shader = """#version 450
void main() {
    vec2 vertices[3] = vec2[](vec2(-1,-1), vec2(3,-1), vec2(-1,3));
    gl_Position = vec4(vertices[gl_VertexIndex], 0, 1);
}
"""
    mov_v2_zero = 0x7E040280
    mov_v3_one_float = 0x7E0602F2
    export_rgba_done = (0xF800180F, 0x03020100)
    end_program = 0xBF810000
    with tempfile.TemporaryDirectory(prefix="pixel smoke ") as temp:
        work = Path(temp)
        (work / "triangle.vert").write_text(vertex_shader, encoding="ascii")
        subprocess.run([compiler, "-V", "triangle.vert", "-o", "triangle.spv"],
                       cwd=work, check=True, timeout=60)
        (work / "fragment.bin").write_bytes(struct.pack("<5I", mov_v2_zero, mov_v3_one_float,
                                                       *export_rgba_done, end_program))
        (work / "scenario.txt").write_text(
            f"stage ps\nwave 32\ntarget {width} {height}\ninputaddr 300\nexport 0 9 0xe4\n"
            "vs triangle.spv\ncode fragment.bin\nout out\n", encoding="ascii")
        subprocess.run([runner, "scenario.txt"], cwd=work, check=True, timeout=90)
        actual = (work / "out" / "target0.bin").read_bytes()
    if len(actual) != width * height * 16:
        raise SystemExit(f"unexpected RGBA32F readback size: {len(actual)} bytes")
    for i, rgba in enumerate(struct.iter_unpack("<4f", actual)):
        y, x = divmod(i, width)
        expected = (x + 0.5, y + 0.5, 0.0, 1.0)
        if rgba != expected:
            raise SystemExit(f"pixel ({x}, {y}): expected {expected}, read {rgba}")
    print(f"PASS: all {width * height} pixels ({len(actual)} RGBA32F bytes) matched")


if __name__ == "__main__":
    main()
