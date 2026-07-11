"""Real-data golden-master tests: gps_read behavior on frozen REAL GLOBK
solutions must not move (data->plot baseline, scope A; safety net for S10.1).

Same comparison policy as test_goldenmaster.py: float arrays at rtol=1e-9
(BLAS wobble can't false-alarm; unit/order/filter regressions are far outside
the band), everything else exact. See realdata_cases.py for fixture
provenance and the per-call GPS_CONFIG_PATH isolation.
"""

import numpy as np
import pytest

from realdata_cases import (
    REAL_CASES,
    REAL_EXPECTED,
    REAL_NEU_FILE_CASES,
    build_real_config_dir,
    run_real_neu_file_case,
)

RTOL = 1e-9
ATOL = 1e-12


@pytest.fixture(scope="session")
def gm_real_env(tmp_path_factory):
    """Rendered realdata gpsconfig dir. Unlike the synthetic gm_env this does
    NOT mutate GPS_CONFIG_PATH session-wide — runners swap it per call."""
    return {"config_dir": build_real_config_dir(tmp_path_factory.mktemp("real-gpsconfig"))}


def _assert_dict_matches(actual, case_id):
    golden = np.load(REAL_EXPECTED / f"{case_id}.npz")
    assert set(actual.keys()) == set(golden.files), (
        f"{case_id}: output structure changed "
        f"(got {sorted(actual)}, expected {sorted(golden.files)})"
    )
    for key in golden.files:
        got, want = np.asarray(actual[key]), golden[key]
        assert got.shape == want.shape, f"{case_id}/{key}: shape {got.shape} != {want.shape}"
        if want.dtype.kind in "fc":
            np.testing.assert_allclose(
                got, want, rtol=RTOL, atol=ATOL, equal_nan=True,
                err_msg=f"{case_id}/{key}",
            )
        else:
            assert np.array_equal(got, want), f"{case_id}/{key}: values changed"


@pytest.mark.parametrize("case_id", sorted(REAL_CASES))
def test_real_golden(case_id, gm_real_env):
    _assert_dict_matches(REAL_CASES[case_id](gm_real_env), case_id)


@pytest.mark.parametrize("case_id", sorted(REAL_NEU_FILE_CASES))
def test_real_golden_neu_file(case_id, gm_real_env, tmp_path):
    out = run_real_neu_file_case(case_id, gm_real_env, tmp_path)
    got = out.read_text().splitlines()
    want = (REAL_EXPECTED / f"{case_id}.NEU").read_text().splitlines()
    assert len(got) == len(want), f"{case_id}: line count {len(got)} != {len(want)}"
    for i, (g, w) in enumerate(zip(got, want)):
        assert g == w, f"{case_id}: line {i + 1} differs\n got: {g!r}\nwant: {w!r}"
