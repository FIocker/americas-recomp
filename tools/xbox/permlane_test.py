"""Check v_permlane16_b32 and v_permlanex16_b32 lane selection.

One 32-lane workgroup uses the same 64-bit lane-select value for both
instructions: each nibble selects 15 - lane within a 16-lane group. The first
pair of results comes from v_permlane16_b32, the second from
v_permlanex16_b32. Each instruction runs once with op_sel[3] set and once
without it; the results must match, which checks the assumption that Xbox
shaders may set that bit on a 32-bit permute.
"""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile


LANES = 32
REGION_SIZE = LANES * 16


def mov_literal(sdst, value):
    return 0xBE800000 | (sdst << 16) | (3 << 8) | 255, value


def permlane(destination, opcode, op_sel):
    control = 0xD4000000 | (opcode << 16) | (op_sel << 11) | destination
    sources = (1 << 18) | 256
    return control, sources


def program():
    words = [0xF4080202, 0xFA000000, 0xBF8C0000, 0x340A0084]
    words += mov_literal(0, 0x89ABCDEF)
    words += mov_literal(1, 0x01234567)
    words += permlane(1, 0x377, 0x8)
    words += permlane(2, 0x377, 0x0)
    words += permlane(3, 0x378, 0x8)
    words += permlane(4, 0x378, 0x0)
    words += [0xE0701000, 0x80020105]
    words += [0xE0701004, 0x80020205]
    words += [0xE0701008, 0x80020305]
    words += [0xE070100C, 0x80020405]
    words.append(0xBF810000)
    return struct.pack(f"<{len(words)}I", *words)


def expected(lane):
    group = lane & 16
    offset = 15 - (lane & 15)
    normal = group + offset
    cross = (group ^ 16) + offset
    return normal, normal, cross, cross


def verify(data):
    if len(data) != REGION_SIZE:
        raise SystemExit(f"unexpected readback size: {len(data)}")
    actual = struct.unpack(f"<{LANES * 4}I", data)
    for lane in range(LANES):
        got = actual[lane * 4:(lane + 1) * 4]
        want = expected(lane)
        if got != want:
            raise SystemExit(f"lane {lane}: expected {want}, read {got}")
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
    with tempfile.TemporaryDirectory(prefix="permlane ") as temp:
        work = Path(temp)
        (work / "code.bin").write_bytes(program())
        (work / "scenario.txt").write_text(
            "wave 32\ngroups 1 1 1\ncode code.bin\nout out\n"
            "region 0 256\nregion 1 1024\nptrlo 0 0 1\nptrhi 0 4 1 0\n"
            "poke 0 8 1024\npoke 0 12 0x30027fac\nuser 4 0\n", encoding="ascii")
        subprocess.run([runner, "scenario.txt"], cwd=work, check=True, timeout=90)
        lanes = verify((work / "out" / "region1.bin").read_bytes()[:REGION_SIZE])
    print(f"PASS: v_permlane16/x16_b32 selected {lanes} lanes with op_sel[3] ignored")


if __name__ == "__main__":
    main()
