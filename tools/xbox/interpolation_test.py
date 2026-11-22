"""Check perspective-correct vertex colors and triangle coverage on the Vulkan runner."""
import argparse
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile


WIDTH = HEIGHT = 32
CLEAR = (-777.0,) * 4
VERTEX_SHADER = """#version 450
layout(location=0) out vec4 color;
void main() {
    vec2 ndc[3] = vec2[](vec2(-0.75, -0.75), vec2(0.75, -0.75), vec2(-0.75, 0.25));
    float clipW[3] = float[](1, 2, 4);
    vec3 rgb[3] = vec3[](vec3(1,0,0), vec3(0,1,0), vec3(0,0,1));
    int i = gl_VertexIndex;
    gl_Position = vec4(ndc[i] * clipW[i], 0, clipW[i]);
    color = vec4(rgb[i], 1);
}
"""


def fragment_program():
    words = []
    for channel in range(4):
        destination = 2 + channel
        interp = 0xC8000000 | (destination << 18) | (channel << 8)
        words.extend((interp, interp | (1 << 16) | 1))
    export_rgba_done = (0xF800180F, 0x05040302)
    words.extend((*export_rgba_done, 0xBF810000))
    return struct.pack(f"<{len(words)}I", *words)


def verify(image):
    if len(image) != WIDTH * HEIGHT * 16:
        raise ValueError(f"unexpected RGBA32F byte count: {len(image)}")
    covered = 0
    max_error = 0.0
    for i, rgba in enumerate(struct.iter_unpack("<4f", image)):
        y, x = divmod(i, WIDTH)
        green = (x + 0.5 - 4) / 24
        blue = (y + 0.5 - 4) / 16
        red = 1 - green - blue
        if min(red, green, blue) < 0:
            if rgba != CLEAR:
                raise ValueError(f"pixel ({x}, {y}): expected clear, read {rgba}")
            continue
        covered += 1
        weights = (red, green / 2, blue / 4)
        denominator = sum(weights)
        expected = tuple(w / denominator for w in weights) + (1.0,)
        errors = [abs(a - b) for a, b in zip(rgba, expected)]
        if not all(math.isfinite(e) and e <= 1e-5 for e in errors):
            raise ValueError(f"pixel ({x}, {y}): expected {expected}, read {rgba}")
        max_error = max(max_error, *errors)
    print(f"PASS: {covered} blended, {WIDTH * HEIGHT - covered} clear; max error {max_error:.3g}")


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
    with tempfile.TemporaryDirectory(prefix="vertex colors ") as temp:
        work = Path(temp)
        (work / "color.vert").write_text(VERTEX_SHADER, encoding="ascii")
        subprocess.run([compiler, "-V", "color.vert", "-o", "color.spv"],
                       cwd=work, check=True, timeout=60)
        (work / "color.bin").write_bytes(fragment_program())
        (work / "scenario.txt").write_text(
            f"stage ps\nwave 32\ntarget {WIDTH} {HEIGHT}\ninputaddr 0x2\ninterp 1 0\n"
            "vs color.spv\ncode color.bin\nexport 0 9 0xe4\nout out\n", encoding="ascii")
        subprocess.run([runner, "scenario.txt"], cwd=work, check=True, timeout=90)
        verify((work / "out" / "target0.bin").read_bytes())


if __name__ == "__main__":
    main()
