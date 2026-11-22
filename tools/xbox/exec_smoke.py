"""Dispatch a standalone RDNA2 kernel and verify the complete Vulkan buffer readback."""
import argparse
from pathlib import Path
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
    if not runner.is_file():
        parser.error(f"missing {runner}; build the harness with --with-exec")

    load_descriptor = (0xF4080202, 0xFA000000)
    wait_for_descriptor = 0xBF8C0000
    byte_offset_from_lane = 0x34020082
    add_seventeen_to_lane = 0x4A040091
    store_dword = (0xE0701000, 0x80020201)
    end_program = 0xBF810000
    words = (*load_descriptor, wait_for_descriptor, byte_offset_from_lane,
             add_seventeen_to_lane, *store_dword, end_program)
    with tempfile.TemporaryDirectory(prefix="xbox compute ") as temp:
        work = Path(temp)
        (work / "code.bin").write_bytes(struct.pack(f"<{len(words)}I", *words))
        (work / "scenario.txt").write_text(
            "wave 32\ngroups 1 1 1\ncode code.bin\nout out\n"
            "region 0 256\nregion 1 1024\nptrlo 0 0 1\nptrhi 0 4 1 0\n"
            "poke 0 8 1024\npoke 0 12 0x30027fac\nuser 4 0\n", encoding="ascii")
        subprocess.run([runner, "scenario.txt"], cwd=work, check=True, timeout=90)
        actual = (work / "out" / "region1.bin").read_bytes()
        expected = struct.pack("<32I", *range(17, 49)) + bytes(896)
        if actual != expected:
            offset = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b),
                          min(len(actual), len(expected)))
            raise SystemExit(f"readback mismatch at byte {offset} (received {len(actual)} bytes)")
    print("PASS: lanes 0..31 wrote 17..48; all 1024 output bytes match")


if __name__ == "__main__":
    main()
