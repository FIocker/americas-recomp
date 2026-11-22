"""Check that v_mad_legacy_f32 (VOP3 0x140) and v_mac_legacy_f32 (VOP2 0x06) are fused, as RDNA2 runs them.

boykopovar/AnyPS5#1281 measured VOP3 0x140 on an RX 6700 XT (gfx1031): the result equals v_fma_f32 bit for bit, while the zero rule
and the denormal-source flush stay those of the gfx1010 v_mad_legacy_f32. One 32-lane workgroup computes, per lane:

  0  v_mad_legacy_f32  (1 + 2^-12)^2 - 1      3F800800 * 3F800800 + BF800000   fused 3A000400, product rounded first 3A000000
  1  v_mad_legacy_f32  max * 2 - max          7F7FFFFF * 40000000 + FF7FFFFF   fused 7F7FFFFF, product rounded first 7F800000
  2  v_mad_legacy_f32  zero rule: +0 * +inf + 1                                 3F800000 either way
  3  v_mad_legacy_f32  denormal source: +tiny * 1 + -0                         00000000 either way
  4  v_mac_legacy_f32  row 0 with the addend in the destination                 as row 0
  5  v_mad_legacy_f32  denormal source: -tiny * 1 + -0                         00000000 either way

Rows 0 and 1 are the #1281 discriminator rows and rows 3 and 5 its denormal-flush rows (v_fma_f32 keeps the tiny there). Row 2 is
the zero rule as docs/dev/debt/shader.md describes it for gfx1010. The recompiler is run twice: by default (fused) and with
XBOX_NO_FUSED_LEGACY_MAD=1 (product rounded first, the gfx1010 behaviour). Both must match their column.

The operands are loaded from the output buffer (bytes 32..63 of each lane's record, written by this test) rather than moved in as
literals: with literals the recompiler folds them into an Fma whose three operands are constants, and Intel's Windows driver
(101.6127, see the GPU matrix) folds that as a rounded multiply followed by an add, so the test then measures the driver's constant
folder instead of the GPU. With run-time operands the same driver fuses.
"""
import argparse
import os
from pathlib import Path
import struct
import subprocess
import tempfile


LANES = 32
STRIDE = 64
INPUTS = 32
ONE, NEG_ONE, TWO = 242, 243, 244
ZERO = 128
V = 256
FUSED = (0x3A000400, 0x7F7FFFFF, 0x3F800000, 0x00000000, 0x3A000400, 0x00000000)
ROUNDED = (0x3A000000, 0x7F800000, 0x3F800000, 0x00000000, 0x3A000000, 0x00000000)


def buffer_load_dword(data, offset):
    return [0xE0301000 | offset, 0x80020004 | (data << 8)]


def v_mad_legacy(destination, first, second, third):
    return [0xD4000000 | (0x140 << 16) | destination, (third << 18) | (second << 9) | first]


def v_mac_legacy(destination, first, second):
    return [(0x06 << 25) | (destination << 17) | (second << 9) | first]


def buffer_store_dword(data, offset):
    return [0xE0701000 | offset, 0x80020004 | (data << 8)]


INPUT_VALUES = {10: 0x3F800800, 11: 0x7F7FFFFF, 13: 0xFF7FFFFF, 14: 0x7F800000, 15: 0x00000001, 16: 0xBF800000,
                17: 0x80000000, 18: 0x80000001}


def program():
    words = [0xF4080202, 0xFA000000, 0xBF8C0000, 0x34080086]
    for slot, register in enumerate(INPUT_VALUES):
        words += buffer_load_dword(register, INPUTS + 4 * slot)
    words.append(0xBF8C0000)
    words += v_mad_legacy(20, V + 10, V + 10, NEG_ONE)
    words += v_mad_legacy(21, V + 11, TWO, V + 13)
    words += v_mad_legacy(22, ZERO, V + 14, ONE)
    words += v_mad_legacy(23, V + 15, ONE, V + 17)
    words += v_mac_legacy(16, V + 10, 10)
    words += v_mad_legacy(24, V + 18, ONE, V + 17)
    for row, register in enumerate((20, 21, 22, 23, 16, 24)):
        words += buffer_store_dword(register, 4 * row)
    words.append(0xBF810000)
    return struct.pack(f"<{len(words)}I", *words)


def initial_region():
    record = bytes(INPUTS) + struct.pack(f"<{len(INPUT_VALUES)}I", *INPUT_VALUES.values())
    return (record + bytes(STRIDE - len(record))) * LANES


def run(runner, env):
    with tempfile.TemporaryDirectory(prefix="mad legacy fused ") as temp:
        work = Path(temp)
        (work / "code.bin").write_bytes(program())
        (work / "region1.bin").write_bytes(initial_region())
        (work / "scenario.txt").write_text(
            "wave 32\ngroups 1 1 1\ncode code.bin\nout out\n"
            f"region 0 256\nregion 1 {LANES * STRIDE} init region1.bin\nptrlo 0 0 1\nptrhi 0 4 1 0\n"
            f"poke 0 8 {LANES * STRIDE}\npoke 0 12 0x30027fac\nuser 4 0\n", encoding="ascii")
        subprocess.run([runner, "scenario.txt"], cwd=work, env=env, check=True, timeout=90, stdout=subprocess.DEVNULL)
        return (work / "out" / "region1.bin").read_bytes()


def verify(data, expected, mode):
    for lane in range(LANES):
        got = struct.unpack_from("<6I", data, lane * STRIDE)
        if got != expected:
            raise SystemExit(f"{mode}, lane {lane}: expected {' '.join('%08X' % v for v in expected)}, "
                             f"read {' '.join('%08X' % v for v in got)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exec", type=Path,
                        default=Path(__file__).resolve().parents[2] / "build-xbox" / ("exec.exe" if os.name == "nt" else "exec"))
    runner = parser.parse_args().exec.resolve()
    if not runner.is_file() and runner.with_suffix(".exe").is_file():
        runner = runner.with_suffix(".exe")
    if not runner.is_file():
        parser.error(f"missing {runner}; build the harness first")
    fused_env = {k: v for k, v in os.environ.items() if k != "XBOX_NO_FUSED_LEGACY_MAD"}
    verify(run(runner, fused_env), FUSED, "fused (default)")
    verify(run(runner, {**fused_env, "XBOX_NO_FUSED_LEGACY_MAD": "1"}), ROUNDED, "XBOX_NO_FUSED_LEGACY_MAD=1")
    print(f"PASS: v_mad_legacy_f32 and v_mac_legacy_f32 matched {LANES} lanes fused (default) and with the product rounded first")


if __name__ == "__main__":
    main()
