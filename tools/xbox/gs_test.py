"""Run synthetic RDNA2 geometry shaders through the execution harness.

The guest geometry programs write vertices to the GSVS LDS slots, emit them
with s_sendmsg, and export a position and one colour parameter. A GLSL vertex
shader supplies a full-screen triangle and a GLSL fragment shader writes the
geometry shader's colour to an 8x8 RGBA32F target.

The first program emits a full-screen triangle, so every pixel must be red. The
second emits a left-half quad, so only the left four columns must be red; the
rest must keep the clear value. The guest program ignores its inputs, so the
same checks run with point, line, triangle, line-adjacency and
triangle-adjacency host input primitives.
"""
import argparse
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile


WIDTH = HEIGHT = 8
CLEAR = -777.0
VERTEX = """#version 450
out gl_PerVertex {
    vec4 gl_Position;
};
void main() {
    vec2 p[3] = vec2[](vec2(-1,-1), vec2(4,-1), vec2(-1,4));
    gl_Position = vec4(p[gl_VertexIndex % 3], 0, 1);
}
"""
FRAGMENT = """#version 450
layout(location=0) in vec4 color;
layout(location=0) out vec4 outColor;
void main() {
    outColor = color;
}
"""
FULL = (
    (243, 243, 128, 242, 242, 128, 128, 242),
    (246, 243, 128, 242, 242, 128, 128, 242),
    (243, 246, 128, 242, 242, 128, 128, 242),
)
HALF = (
    (243, 243, 128, 242, 242, 128, 128, 242),
    (128, 243, 128, 242, 242, 128, 128, 242),
    (243, 242, 128, 242, 242, 128, 128, 242),
    (128, 242, 128, 242, 242, 128, 128, 242),
)


def move(destination, constant):
    return 0x7E000000 | (destination << 17) | (1 << 9) | constant


def store(address, value, offset):
    return 0xD8000000 | (13 << 18) | (offset & 0xffff), (value << 8) | address


def send(immediate):
    return 0xBF900000 | (immediate & 0xffff)


def program(vertices):
    words = [move(1, 128)]
    for vertex, values in enumerate(vertices):
        base = (64 + vertex * 8) * 4
        for slot, constant in enumerate(values):
            words.append(move(2, constant))
            words += store(1, 2, base + slot * 4)
        words.append(send(0x22))
    words.append(send(0x12))
    words += [0xF80008CF, 0x02020202]
    words += [0xF8000A0F, 0x02020202]
    words.append(0xBF810000)
    return struct.pack(f"<{len(words)}I", *words)


def compile_shader(compiler, work, name, source, suffix):
    path = work / f"{name}{suffix}"
    path.write_text(source, encoding="ascii")
    subprocess.run([compiler, "-V", path.name, "-o", f"{name}.spv"],
                   cwd=work, check=True, timeout=60)
    return work / f"{name}.spv"


def read_target(path):
    data = path.read_bytes()
    if len(data) != WIDTH * HEIGHT * 16:
        raise SystemExit(f"unexpected RGBA32F byte count: {len(data)}")
    return struct.unpack(f"<{WIDTH * HEIGHT * 4}f", data)


def verify_full(data):
    red = (1.0, 0.0, 0.0, 1.0)
    for pixel in range(WIDTH * HEIGHT):
        got = data[pixel * 4:(pixel + 1) * 4]
        if got != red:
            raise SystemExit(f"pixel {pixel}: expected {red}, read {got}")


def verify_half(data):
    red = (1.0, 0.0, 0.0, 1.0)
    clear = (CLEAR, CLEAR, CLEAR, CLEAR)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            pixel = y * WIDTH + x
            got = data[pixel * 4:(pixel + 1) * 4]
            expected = red if x < WIDTH // 2 else clear
            if got != expected:
                raise SystemExit(f"pixel ({x}, {y}): expected {expected}, read {got}")


def run_case(runner, compiler, work, name, vertices, verify, input_primitive=3):
    compile_shader(compiler, work, "vs", VERTEX, ".vert")
    compile_shader(compiler, work, "fs", FRAGMENT, ".frag")
    (work / "gs.bin").write_bytes(program(vertices))
    (work / "scenario.txt").write_text(
        f"stage gs\nwave 64\ntarget {WIDTH} {HEIGHT}\nvs vs.spv\nfs fs.spv\n"
        f"geom {input_primitive} 3 {len(vertices)} 8 8 1 1\ncode gs.bin\nexport 0 9 0xe4\nout out\n",
        encoding="ascii")
    subprocess.run([runner, "scenario.txt"], cwd=work, check=True, timeout=90)
    verify(read_target(work / "out" / "target0.bin"))


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
    with tempfile.TemporaryDirectory(prefix="geometry ") as temp:
        for input_primitive in (1, 2, 3, 4, 5):
            run_case(runner, compiler, Path(temp), f"full-{input_primitive}",
                     FULL, verify_full, input_primitive)
            run_case(runner, compiler, Path(temp), f"half-{input_primitive}",
                     HALF, verify_half, input_primitive)
    print(f"PASS: geometry shader filled {WIDTH * HEIGHT} pixels and clipped a half-screen quad "
          f"with point, line, triangle and adjacency inputs")


if __name__ == "__main__":
    main()
