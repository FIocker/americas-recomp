"""Check s_quadmask_b32 and s_quadmask_b64 on a compute shader.

One 32-lane workgroup reduces four 32-bit masks and two 64-bit masks to one bit
per group of four input bits. The cases cover all ones, one bit per quad, zero,
a high bit, and a 64-bit mask whose set bits are only in the high word. Every
lane must store the expected eight words.
"""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile


LANES = 32
REGION_SIZE = LANES * 32
EXPECTED = (0xFF, 0xFF, 0x00, 0x81, 0xFFFF, 0x0000, 0xFF00, 0x0000)


def mov_literal(sdst, value):
    return 0xBE800000 | (sdst << 16) | (3 << 8) | 255, value


def quadmask32(destination, source):
    return 0xBE800000 | (destination << 16) | (44 << 8) | source


def quadmask64(destination, source):
    return 0xBE800000 | (destination << 16) | (45 << 8) | source


def move(destination, source):
    return 0x7E000000 | (destination << 17) | (1 << 9) | source


def store(value, offset):
    return 0xE0701000 | offset, 0x80020009 | (value << 8)


def program():
    words = [0xF4080202, 0xFA000000, 0xBF8C0000, 0x34120085]
    words += mov_literal(0, 0xFFFFFFFF)
    words.append(quadmask32(1, 0))
    words.append(move(1, 1))
    words += mov_literal(0, 0x11111111)
    words.append(quadmask32(1, 0))
    words.append(move(2, 1))
    words += mov_literal(0, 0x00000000)
    words.append(quadmask32(1, 0))
    words.append(move(3, 1))
    words += mov_literal(0, 0x80000001)
    words.append(quadmask32(1, 0))
    words.append(move(4, 1))
    words += mov_literal(0, 0xFFFFFFFF)
    words += mov_literal(1, 0xFFFFFFFF)
    words.append(quadmask64(2, 0))
    words.append(move(5, 2))
    words.append(move(6, 3))
    words += mov_literal(0, 0x00000000)
    words += mov_literal(1, 0xFFFFFFFF)
    words.append(quadmask64(2, 0))
    words.append(move(7, 2))
    words.append(move(8, 3))
    for index in range(8):
        words += store(index + 1, index * 4)
    words.append(0xBF810000)
    return struct.pack(f"<{len(words)}I", *words)


def verify(data):
    if len(data) != REGION_SIZE:
        raise SystemExit(f"unexpected readback size: {len(data)}")
    actual = struct.unpack(f"<{LANES * 8}I", data)
    for lane in range(LANES):
        got = actual[lane * 8:(lane + 1) * 8]
        if got != EXPECTED:
            raise SystemExit(f"lane {lane}: expected {EXPECTED}, read {got}")
    return LANES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exec", type=Path,
                        default=Path(__file__).resolve().parents[2] / "build-xbox" / "exec.exe")
    runner = parser.parse_args().exec.resolve()
    if not runner.is_file() and runner.with_suffix(".exe").is_file():
        runner = runner.with_suffix(".exe")
    if not runner.is_file():
        parser.error(f"missing {runner}; build the harness with --with-exec")
    with tempfile.TemporaryDirectory(prefix="quadmask ") as temp:
        work = Path(temp)
        (work / "code.bin").write_bytes(program())
        (work / "scenario.txt").write_text(
            "wave 32\ngroups 1 1 1\ncode code.bin\nout out\n"
            "region 0 256\nregion 1 1024\nptrlo 0 0 1\nptrhi 0 4 1 0\n"
            "poke 0 8 1024\npoke 0 12 0x30027fac\nuser 4 0\n", encoding="ascii")
        subprocess.run([runner, "scenario.txt"], cwd=work, check=True, timeout=90)
        lanes = verify((work / "out" / "region1.bin").read_bytes())
    print(f"PASS: s_quadmask_b32 and s_quadmask_b64 matched {lanes} lanes")


if __name__ == "__main__":
    main()
