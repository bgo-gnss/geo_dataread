"""(Re)capture the golden-master expectations into expected/.

Run from the geo_dataread root:

    uv run python tests/goldenmaster/capture_goldens.py [case_id ...]

With no arguments every case is recaptured; case_id arguments restrict the
capture to just those cases (for ADDING new pins without rewriting the
frozen ones).

ONLY rerun this when a behavior change is intentional and reviewed — the
whole point of the goldens is that they do NOT move during the refactor.
The script runs every case twice and refuses to write if the two passes
differ (guards against nondeterminism sneaking into the baseline).
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cases import CASES, EXPECTED, NEU_FILE_CASES, build_config_dir, run_neu_file_case


def _dicts_equal(a, b):
    if a.keys() != b.keys():
        return False
    for k in a:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.dtype.kind in "fc":
            if not np.array_equal(x, y, equal_nan=True):
                return False
        elif not np.array_equal(x, y):
            return False
    return True


def main(only=()) -> int:
    unknown = sorted(set(only) - set(CASES) - set(NEU_FILE_CASES))
    if unknown:
        raise SystemExit(f"unknown case id(s): {unknown}")
    cases = {c: CASES[c] for c in (only or CASES) if c in CASES}
    neu_cases = [c for c in (only or NEU_FILE_CASES) if c in NEU_FILE_CASES]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        env = {
            "config_dir": build_config_dir(tmp / "gpsconfig"),
            "empty_dir": tmp / "empty",
        }
        env["empty_dir"].mkdir()
        os.environ["GPS_CONFIG_PATH"] = str(env["config_dir"])
        EXPECTED.mkdir(exist_ok=True)

        for case_id, runner in cases.items():
            first = runner(env)
            second = runner(env)
            if not _dicts_equal(first, second):
                print(f"FATAL: {case_id} is nondeterministic — nothing written")
                return 1
            np.savez_compressed(EXPECTED / f"{case_id}.npz", **first)
            print(f"captured {case_id}")

        for d in ("neu1", "neu2"):
            (tmp / d).mkdir(exist_ok=True)
        for case_id in neu_cases:
            out1 = run_neu_file_case(case_id, tmp / "neu1")
            out2 = run_neu_file_case(case_id, tmp / "neu2")
            if out1.read_bytes() != out2.read_bytes():
                print(f"FATAL: {case_id} is nondeterministic — nothing written")
                return 1
            (EXPECTED / f"{case_id}.NEU").write_bytes(out1.read_bytes())
            print(f"captured {case_id}")

    print(f"\n{len(cases) + len(neu_cases)} golden(s) written to {EXPECTED}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
