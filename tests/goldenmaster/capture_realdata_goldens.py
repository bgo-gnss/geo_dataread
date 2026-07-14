"""(Re)capture the REAL-DATA golden expectations into expected/realdata/.

Run from the geo_dataread root:

    uv run python tests/goldenmaster/capture_realdata_goldens.py [case_id ...]

With no arguments every case is recaptured; case_id arguments restrict the
capture to just those cases (for ADDING new pins without rewriting the
frozen ones — e.g. the slice-6 JOIN cases were captured selectively).

Same rules as capture_goldens.py: ONLY rerun when a behavior change is
intentional and reviewed. Every case runs twice; nothing is written if the
two passes differ (nondeterminism guard). This script never touches the
synthetic goldens in expected/ — it only writes expected/realdata/.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from realdata_cases import (  # noqa: E402
    REAL_CASES,
    REAL_EXPECTED,
    REAL_NEU_FILE_CASES,
    build_real_config_dir,
    run_real_neu_file_case,
)


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
    unknown = sorted(set(only) - set(REAL_CASES) - set(REAL_NEU_FILE_CASES))
    if unknown:
        raise SystemExit(f"unknown case id(s): {unknown}")
    cases = {c: REAL_CASES[c] for c in (only or REAL_CASES) if c in REAL_CASES}
    neu_cases = [c for c in (only or REAL_NEU_FILE_CASES) if c in REAL_NEU_FILE_CASES]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        env = {"config_dir": build_real_config_dir(tmp / "gpsconfig")}
        REAL_EXPECTED.mkdir(parents=True, exist_ok=True)

        for case_id, runner in cases.items():
            first = runner(env)
            second = runner(env)
            if not _dicts_equal(first, second):
                print(f"FATAL: {case_id} is nondeterministic — nothing written")
                return 1
            np.savez_compressed(REAL_EXPECTED / f"{case_id}.npz", **first)
            print(f"captured {case_id}")

        for d in ("neu1", "neu2"):
            (tmp / d).mkdir(exist_ok=True)
        for case_id in neu_cases:
            out1 = run_real_neu_file_case(case_id, env, tmp / "neu1")
            out2 = run_real_neu_file_case(case_id, env, tmp / "neu2")
            if out1.read_bytes() != out2.read_bytes():
                print(f"FATAL: {case_id} is nondeterministic — nothing written")
                return 1
            (REAL_EXPECTED / f"{case_id}.NEU").write_bytes(out1.read_bytes())
            print(f"captured {case_id}")

    print(
        f"\n{len(cases) + len(neu_cases)} real-data golden(s) written to {REAL_EXPECTED}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
