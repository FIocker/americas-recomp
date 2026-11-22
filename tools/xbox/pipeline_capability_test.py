"""Check that exec rejects unsupported pipelines and admits supported fragments."""

import argparse
import os
from pathlib import Path
import struct
import subprocess
import tempfile

MISSING_COMPANION = "fragment shader requires a barycentric emulation companion stage that exec does not provide"
MISSING_VERTEX = "a graphics stage needs a `vs <file.spv>` line"
MISSING_NATIVE_BARY = (
    "the device has no VK_KHR_fragment_shader_barycentric; "
    "set XBOX_NO_FRAGMENT_BARY=1 to recompile pixel shaders without it"
)


def expect_result(runner, name, scenario, diagnostic, words=(), *, compiled=False, native_bary=False):
    with tempfile.TemporaryDirectory(prefix="xbox pipeline ") as temp:
        work = Path(temp)
        (work / "scenario.txt").write_text(scenario + "out out\n", encoding="ascii")
        if words:
            (work / "fragment.bin").write_bytes(struct.pack(f"<{len(words)}I", *words))
        env = dict(os.environ, XBOX_EXEC_SPV="shader.spv")
        if native_bary:
            env.pop("XBOX_NO_FRAGMENT_BARY", None)
        else:
            env["XBOX_NO_FRAGMENT_BARY"] = "1"
        result = subprocess.run([runner, "scenario.txt"], cwd=work, env=env,
                                capture_output=True, text=True, timeout=90)
        lines = result.stdout.splitlines()
        skipped = native_bary and result.returncode == 1 and lines == [f"FAIL {MISSING_NATIVE_BARY}"]
        if skipped:
            compiled = False
            diagnostic = MISSING_NATIVE_BARY
        prefix = lines[:-1]
        expected_prefix = len(prefix) == 1 and prefix[0].startswith("compiled: ") if compiled else not prefix
        if result.returncode != 1 or lines[-1:] != [f"FAIL {diagnostic}"] or not expected_prefix:
            raise SystemExit(f"{name}: unexpected exit {result.returncode}\n{result.stdout}{result.stderr}")
        dump = work / "shader.spv"
        if (work / "out").exists() or dump.exists() != compiled or (compiled and dump.stat().st_size == 0):
            raise SystemExit(f"{name}: unexpected shader dump or render output")
        if skipped:
            print(f"SKIP {name}: selected device lacks VK_KHR_fragment_shader_barycentric support")
            return
    print(f"PASS {name}: exit 1, FAIL {diagnostic}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exec", type=Path, required=True, help="path to the execution harness")
    parser.add_argument("--expect-no-geometry", action="store_true",
                        help="also test a device with geometryShader=false")
    args = parser.parse_args()
    runner = args.exec.resolve()
    if not runner.is_file() and runner.with_suffix(".exe").is_file():
        runner = runner.with_suffix(".exe")
    if not runner.is_file():
        parser.error("provide a built --exec runner")

    fragment = "stage ps\nwave 32\ninputaddr 0x2\ncode fragment.bin\nexport 0 9 0xe4\n"
    attribute = fragment + "interp 1 0\n"
    interp_p1 = 0xC8080000
    interp_p2 = 0xC8090001
    interp_mov_p10 = 0xC80A0000
    export_v2 = (0xF800180F, 0x02020202)
    export_perspective_ij = (0xF800180F, 0x01000100)
    end_program = 0xBF810000
    per_vertex = (interp_mov_p10, *export_v2, end_program)

    expect_result(runner, "per-vertex parameters", attribute, MISSING_COMPANION, per_vertex)
    expect_result(runner, "raw barycentric coordinates", fragment, MISSING_COMPANION,
                  (*export_perspective_ij, end_program))
    expect_result(runner, "ordinary P1/P2 interpolation", attribute, MISSING_VERTEX,
                  (interp_p1, interp_p2, *export_v2, end_program), compiled=True)
    expect_result(runner, "native per-vertex compilation", attribute, MISSING_VERTEX,
                  per_vertex, compiled=True, native_bary=True)
    if args.expect_no_geometry:
        expect_result(runner, "unavailable geometry stage", "stage gs\ncode missing.bin\n",
                      "stage gs requires the Vulkan geometryShader feature")


if __name__ == "__main__":
    main()
