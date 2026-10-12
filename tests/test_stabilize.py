"""stabilize: the loose SINEX day tied to the frame reproduces glorg's org output.

Fixture: the archived 2014-150 day (okada /D/SNX/2014), loose SINEX + glorg org.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geo_dataread import org, sinex, stabilize

FIX = Path(__file__).parent / "fixtures" / "sinex"
SNX = FIX / "IMO0OPSFIN_20141500000_01D_01D_SOL.SNX.gz"
ORG = FIX / "IMO0OPSFIN_20141500000_01D_01D_SOL.org.gz"

# glorg 10.71 printed (iteration 1 and 4): mas, mas, mas, m, m, m
GLORG_IT1 = [-0.07936, -0.12532, -0.00792, -0.01338, 0.00342, 0.00739]
GLORG_IT4 = [-0.04022, -0.11333, -0.01661, -0.01255, 0.00336, 0.00507]
GLORG_IT4_SIG = [0.03048, 0.02998, 0.03606, 0.00089, 0.00090, 0.00100]


@pytest.fixture(scope="module")
def day() -> tuple[sinex.SinexSolution, stabilize.Stabilized]:
    sol = sinex.read_sinex(SNX)
    ref, cfg = org.read_org_setup(ORG)
    return sol, stabilize.stabilize(sol, ref, cfg)


def test_read_org_setup() -> None:
    ref, cfg = org.read_org_setup(ORG)
    assert cfg == stabilize.StabilizeConfig()  # the IMO icel.org.cmd settings
    assert len(ref) == 77 and ref[:3] == ["GOL2", "ALGO", "AMC2"]
    assert "CHUR" in ref and "PIE1" in ref  # '#CHUR PIE1' is a comment line


def test_iterations_match_glorg_print(day) -> None:  # type: ignore[no-untyped-def]
    _, st = day
    it1, it4 = st.iterations[0], st.iterations[-1]
    for got, want in ((it1.theta, GLORG_IT1), (it4.theta, GLORG_IT4)):
        # rotations (mas) can differ in the 4th decimal, translations (m) not
        np.testing.assert_allclose(got[:3], want[:3], atol=2e-3)
        np.testing.assert_allclose(got[3:], want[3:], atol=2e-5)
    np.testing.assert_allclose(it4.theta_sigma, GLORG_IT4_SIG, atol=1e-5)
    assert (round(it1.pre_rms, 4), round(it1.post_rms, 5)) == (0.0087, 0.0055)
    assert (round(it4.pre_rms, 4), round(it4.post_rms, 5)) == (0.0073, 0.004)
    assert [d[0].split(":")[0] for d in it1.deleted] == ["GOL2", "COCO"]
    assert len(it1.sites) == 49 and len(it4.sites) == 47
    assert it1.rel_sigma == (1.0,) * 49
    np.testing.assert_array_equal(st.theta, it4.theta)


def test_every_site_matches_glorg_final_positions(day) -> None:  # type: ignore[no-untyped-def]
    _, st = day
    pos = org.read_org_positions(ORG)
    rows = org.match_sinex_rows(st.solution, pos)
    assert len(rows) == len(pos) == 135
    for name, i in rows.items():
        # org prints 5 decimals (m): agreement is at its resolution
        np.testing.assert_allclose(
            st.solution.estimate[i : i + 3], pos[name].xyz, atol=1.5e-5
        )
        np.testing.assert_allclose(
            st.solution.sigma[i : i + 3], pos[name].sigma, atol=1e-5
        )


def test_condition_holds_and_loose_input_untouched(day) -> None:  # type: ignore[no-untyped-def]
    sol, st = day
    loose = sinex.read_sinex(SNX)
    np.testing.assert_array_equal(sol.estimate, loose.estimate)
    # site sigmas drop from loose (cm–m) to frame-defined (mm)
    i = st.solution.xyz_index("REYK")
    assert loose.sigma[i].min() > 0.02 and st.solution.sigma[i].max() < 0.01
    np.testing.assert_allclose(st.solution.cov, st.solution.cov.T, atol=1e-18)


def test_design_partials() -> None:
    x = np.array([2587384.0, -1043034.0, 5716564.0])
    a = stabilize.design(
        x, ("xrot", "yrot", "zrot", "xtran", "ytran", "ztran", "scale")
    )
    assert a.shape == (3, 7)
    # UT1 is a rotation about Z; pole x/y are −/+ the Helmert ry/rx rotations
    w = 1.0 / stabilize.RAD_TO_MAS
    np.testing.assert_allclose(a[:, 2], np.cross([0, 0, w], x))
    np.testing.assert_allclose(a[:, 0], -np.cross([0, w, 0], x))
    np.testing.assert_allclose(a[:, 1], -np.cross([w, 0, 0], x))
    np.testing.assert_allclose(a[:, 6], x * 1e-9)
    # parameter order is glorg's bit order, whatever order is asked for
    np.testing.assert_array_equal(stabilize.design(x, ("ztran", "xrot")), a[:, [0, 5]])


def test_bad_config_and_too_few_sites() -> None:
    with pytest.raises(ValueError, match="subset"):
        stabilize.StabilizeConfig(params=("xrot", "tilt"))
    sol = sinex.read_sinex(SNX)
    with pytest.raises(ValueError, match="reference sites"):
        stabilize.stabilize(sol, ["ONSA"])  # 3 equations, 6 parameters


def test_numpy_only() -> None:
    for mod in (stabilize, org):
        imports = [
            ln
            for ln in Path(str(mod.__file__)).read_text().splitlines()
            if ln.startswith(("import ", "from "))
        ]
        assert not [
            ln
            for ln in imports
            if "pandas" in ln or "gps_parser" in ln or "gps_read" in ln
        ]
