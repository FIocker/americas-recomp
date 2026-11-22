"""Check v_mad_legacy_f32 clamp on a compute shader.

One 32-lane workgroup computes three values with v_mad_legacy_f32 and the VOP3
clamp bit, then stores them per lane: 2.0 * 4.0 + 4.0 -> 1.0,
-2.0 * 4.0 + 4.0 -> 0.0, and 0.5 * 0.5 + 0.0 -> 0.25. Without the clamp bit the
same inputs produce 12.0, -4.0 and 0.25, so the test distinguishes the clamp
from an ordinary multiply-add.
"""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile


LANES = 32
REGION_SIZE = LANES * 16
ZERO = 128
HALF = 240
TWO = 244
NEG_TWO = 245
FOUR = 246


def mad_legacy(destination, first, second, third):
    control = 0xD4000000 | (0x140 << 16) | (1 << 15) | destination
    sources = (third << 18) | (second << 9) | first
    return control, sources


def program():
    words = [0xF4080202, 0xFA000000, 0xBF8C0000, 0x34080084]
    words += mad_legacy(0, TWO, FOUR, FOUR)
    words += mad_legacy(1, NEG_TWO, FOUR, FOUR)
    words += mad_legacy(2, HALF, HALF, ZERO)
    words += [0xE0701000, 0x80020004]
    words += [0xE0701004, 0x80020104]
    words += [0xE0701008, 0x80020204]
    words.append(0xBF810000)
    return struct.pack(f"<{len(words)}I", *words)


def verify(data):
    if len(data) != REGION_SIZE:
        raise SystemExit(f"unexpected readback size: {len(data)}")
    actual = struct.unpack(f"<{LANES * 4}f", data)
    expected = (1.0, 0.0, 0.25, 0.0)
    for lane in range(LANES):
        got = actual[lane * 4:(lane + 1) * 4]
        if got != expected:
            raise SystemExit(f"lane {lane}: expected {expected}, read {got}")
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
    with tempfile.TemporaryDirectory(prefix="mad legacy clamp ") as temp:
        work = Path(temp)
        (work / "code.bin").write_bytes(program())
        (work / "scenario.txt").write_text(
            "wave 32\ngroups 1 1 1\ncode code.bin\nout out\n"
            "region 0 256\nregion 1 1024\nptrlo 0 0 1\nptrhi 0 4 1 0\n"
            "poke 0 8 1024\npoke 0 12 0x30027fac\nuser 4 0\n", encoding="ascii")
        subprocess.run([runner, "scenario.txt"], cwd=work, check=True, timeout=90)
        lanes = verify((work / "out" / "region1.bin").read_bytes()[:REGION_SIZE])
    print(f"PASS: v_mad_legacy_f32 clamp matched {lanes} lanes at 1.0, 0.0 and 0.25")


if __name__ == "__main__":
    main()
