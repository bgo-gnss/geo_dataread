"""Golden-master tests: current gps_read behavior must not move (plan §15.6).

Float arrays are compared with a tight tolerance (rtol=1e-9) rather than
bit-exactly, so a BLAS/threading wobble can't produce false alarms; anything
a refactor could plausibly break (units, ordering, filtering, offsets,
detrending) is far outside that band. Everything else (times, strings,
row counts, the .NEU product files) is compared exactly.
"""

import numpy as np
import pytest

from cases import CASES, CRASH_CASES, EXPECTED, NEU_FILE_CASES, run_neu_file_case

RTOL = 1e-9
ATOL = 1e-12


def _assert_dict_matches(actual, case_id):
    golden = np.load(EXPECTED / f"{case_id}.npz")
    assert set(actual.keys()) == set(golden.files), (
        f"{case_id}: output structure changed "
        f"(got {sorted(actual)}, expected {sorted(golden.files)})"
    )
    for key in golden.files:
        got, want = np.asarray(actual[key]), golden[key]
        assert got.shape == want.shape, (
            f"{case_id}/{key}: shape {got.shape} != {want.shape}"
        )
        if want.dtype.kind in "fc":
            np.testing.assert_allclose(
                got,
                want,
                rtol=RTOL,
                atol=ATOL,
                equal_nan=True,
                err_msg=f"{case_id}/{key}",
            )
        else:
            assert np.array_equal(got, want), f"{case_id}/{key}: values changed"


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_golden(case_id, gm_env):
    _assert_dict_matches(CASES[case_id](gm_env), case_id)


@pytest.mark.parametrize("case_id", sorted(CRASH_CASES))
def test_golden_crash(case_id, gm_env):
    runner, exc_type = CRASH_CASES[case_id]
    with pytest.raises(exc_type):
        runner(gm_env)


@pytest.mark.parametrize("case_id", sorted(NEU_FILE_CASES))
def test_golden_neu_file(case_id, gm_env, tmp_path):
    out = run_neu_file_case(case_id, tmp_path)
    got = out.read_text().splitlines()
    want = (EXPECTED / f"{case_id}.NEU").read_text().splitlines()
    assert len(got) == len(want), f"{case_id}: line count {len(got)} != {len(want)}"
    for i, (g, w) in enumerate(zip(got, want)):
        assert g == w, f"{case_id}: line {i + 1} differs\n got: {g!r}\nwant: {w!r}"
