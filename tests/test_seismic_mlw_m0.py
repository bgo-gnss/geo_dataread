"""Tests for the SIL Mlw → M0 relation (T5b, PLAN-seismicity R1).

``moment_from_mlw`` / ``mlw_from_moment`` / ``MLW_M0_FIT`` and the way
``with_derived_moment`` uses them. The fit itself is
``docs/seismic/fit_mlw_m0.py`` → ``docs/seismic/mlw_m0_relation.md``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geo_dataread.seismic import sil_catalog as sc
from geo_dataread.seismic.filters import (
    DERIVED_MOMENT_FLAG,
    MLW_M0_FIT,
    NO_DERIVED_MOMENT_TYPES,
    FilterError,
    MlwMomentRelation,
    mlw_from_moment,
    moment_from_magnitude,
    moment_from_mlw,
    with_derived_moment,
)
from geo_dataread.seismic.schema import ATTRS_KEY, coerce_catalog, cumulative_moment

FIXTURES = Path(__file__).parent / "fixtures" / "sil"
MLW_GRID = np.linspace(-2.0, 7.0, 9001)


def test_definition_below_first_knee() -> None:
    # SIL's own Mlw = log10(M0) − 10 below the first knee
    m = np.array([-1.0, 0.0, 0.5, 1.0, 1.9])
    np.testing.assert_allclose(np.log10(moment_from_mlw(m)), m + 10.0, atol=1e-12)
    assert MLW_M0_FIT.intercept == 10.0
    assert MLW_M0_FIT.slopes[0] == 1.0


def test_round_trip_exact() -> None:
    np.testing.assert_allclose(mlw_from_moment(moment_from_mlw(MLW_GRID)), MLW_GRID)
    m0 = np.logspace(7.0, 19.0, 500)
    np.testing.assert_allclose(moment_from_mlw(mlw_from_moment(m0)), m0, rtol=1e-12)
    # at and either side of every knee
    for k in MLW_M0_FIT.knots:
        m = np.array([k - 1e-9, k, k + 1e-9])
        np.testing.assert_allclose(mlw_from_moment(moment_from_mlw(m)), m, atol=1e-9)


def test_strictly_monotonic() -> None:
    lg = np.log10(moment_from_mlw(MLW_GRID))
    assert np.all(np.diff(lg) > 0)
    assert np.all(np.diff(mlw_from_moment(np.logspace(7, 19, 1000))) > 0)


def test_continuous_at_knees_with_the_given_slopes() -> None:
    f = MLW_M0_FIT.log10_moment
    eps = 1e-7
    for i, k in enumerate(MLW_M0_FIT.knots):
        left, at, right = (float(f(k - eps)), float(f(k)), float(f(k + eps)))
        assert left == pytest.approx(at, abs=1e-6)
        assert right == pytest.approx(at, abs=1e-6)
        assert (at - left) / eps == pytest.approx(MLW_M0_FIT.slopes[i], rel=1e-5)
        assert (right - at) / eps == pytest.approx(MLW_M0_FIT.slopes[i + 1], rel=1e-5)


def test_fit_record() -> None:
    rec = MLW_M0_FIT.as_dict()
    assert rec["n_events"] > 100_000
    assert rec["fit_start"] < rec["fit_end"]
    assert rec["sigma_by_bin"]
    lo, hi = MLW_M0_FIT.mlw_range
    assert lo < 0.0 < 5.0 < hi
    # SIL's definition: Mlw grows 1, 0.9, 0.8, 0.7, 0.5 per decade of moment;
    # below Hanks–Kanamori's slope 1.5 up to Mlw 5.4, steeper above
    np.testing.assert_allclose(
        1.0 / np.array(MLW_M0_FIT.slopes), [1.0, 0.9, 0.8, 0.7, 0.5]
    )
    assert MLW_M0_FIT.knots == (2.0, 3.0, 4.6, 5.4)
    assert MLW_M0_FIT.sigma < 0.01
    # the Hanks–Kanamori moment of an Mlw 1 event is ~3x too small, of Mlw 4 ~5x
    # too large
    assert float(moment_from_mlw(1.0) / moment_from_magnitude(1.0)) > 2.5
    assert float(moment_from_magnitude(4.0) / moment_from_mlw(4.0)) > 4.5


def _bin_tol(mlw: float) -> float:
    """3σ of the relation's bin plus the 0.005 half-step of Mlw's 0.01 rounding."""
    for lo, hi, _, sigma in MLW_M0_FIT.sigma_by_bin:
        if lo <= mlw < hi:
            return 3.0 * sigma + 0.005 * max(MLW_M0_FIT.slopes)
    raise AssertionError(f"no residual bin for Mlw {mlw}")


@pytest.mark.parametrize(
    "rel", ["1995/jun/15", "2022/dec/12", "2024/aug/22", "2025/may/03"]
)
def test_fixture_moments_within_reported_residual(rel: str) -> None:
    # the catalogue reader's mag_mlw (0.01, from lib.mag/events.lib), not the
    # 0.1-rounded events.fps value
    df, _ = sc.read_sil_day(FIXTURES / rel)
    d = df[(df["moment_nm"] > 0) & df["mag_mlw"].notna()]
    assert len(d) >= 4
    mlw = d["mag_mlw"].to_numpy(dtype=float)
    r = np.log10(d["moment_nm"].to_numpy(dtype=float)) - np.log10(moment_from_mlw(mlw))
    for m, res in zip(mlw, r, strict=True):
        assert abs(res) <= _bin_tol(m), (rel, m, res)


def test_nan_and_nonpositive() -> None:
    assert np.isnan(moment_from_mlw(np.nan))
    out = mlw_from_moment(np.array([np.nan, 0.0, -1.0, 1e10]))
    assert np.isnan(out[:3]).all()
    assert out[3] == pytest.approx(0.0)
    both = moment_from_mlw(np.array([np.nan, 1.0]))
    assert np.isnan(both[0]) and both[1] == pytest.approx(1e11)
    assert np.isnan(MLW_M0_FIT.mlw(np.nan))


def test_scalar_in_zero_d_out() -> None:
    assert np.asarray(moment_from_mlw(1.0)).shape == ()
    assert np.asarray(mlw_from_moment(1e11)).shape == ()
    assert float(mlw_from_moment(1e11)) == pytest.approx(1.0)


def test_relation_validation() -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(MLW_M0_FIT, slopes=(1.0,))
    with pytest.raises(ValueError):
        dataclasses.replace(MLW_M0_FIT, knots=(3.0, 2.0))
    with pytest.raises(ValueError):
        dataclasses.replace(MLW_M0_FIT, slopes=(1.0, 0.0, 1.0))


def test_custom_relation() -> None:
    rel = MlwMomentRelation(
        intercept=9.0,
        knots=(1.0,),
        slopes=(1.0, 1.5),
        fit_start="2000-01-01",
        fit_end="2001-01-01",
        n_events=1,
        mlw_range=(0.0, 2.0),
        sigma=0.0,
        sigma_by_bin=(),
        reference="test",
    )
    assert float(np.log10(moment_from_mlw(2.0, relation=rel))) == pytest.approx(11.5)
    assert float(mlw_from_moment(10**11.5, relation=rel)) == pytest.approx(2.0)


# ------------------------------------------------------- with_derived_moment


def _frame() -> pd.DataFrame:
    t = pd.date_range("2024-08-22", periods=5, freq="h", tz="UTC")
    raw = pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(5)],
            "latitude": 64.0,
            "longitude": -22.0,
            "magnitude": [1.0, 4.0, 2.5, 1.5, np.nan],
            "magnitude_type": ["Mlw", "Mlw", "ML", "aut.mag", "Mlw"],
            "mag_mlw": [1.0, 4.0, np.nan, np.nan, np.nan],
            "moment_nm": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "evaluation_mode": "manual",
        },
        index=pd.DatetimeIndex(t, name="time"),
    )
    return coerce_catalog(raw, source="sil-files")


def test_derived_mlw_uses_sil_relation() -> None:
    out = with_derived_moment(_frame(), magnitude_types=["Mlw"])
    assert out[DERIVED_MOMENT_FLAG].tolist() == [True, True, False, False, False]
    np.testing.assert_allclose(
        out["moment_nm"].to_numpy()[:2], moment_from_mlw(np.array([1.0, 4.0]))
    )
    rec = out.attrs[ATTRS_KEY]["subsets"][-1]
    assert rec["relations"] == {"Mlw": "sil_mlw_m0"}
    assert tuple(rec["mlw_relation"]["knots"]) == MLW_M0_FIT.knots
    assert rec["n_derived"] == 2
    # the cumulative moment uses the derived values unchanged
    cum = cumulative_moment(out)
    assert float(cum.iloc[-1]) == pytest.approx(
        float(moment_from_mlw([1.0, 4.0]).sum())
    )


def test_declared_mw_types_keep_hanks_kanamori() -> None:
    out = with_derived_moment(_frame(), magnitude_types=["Mlw", "ML", "aut.mag"])
    m0 = out["moment_nm"].to_numpy()
    assert m0[2] == pytest.approx(float(moment_from_magnitude(2.5)))
    assert m0[3] == pytest.approx(float(moment_from_magnitude(1.5)))
    assert m0[0] == pytest.approx(float(moment_from_mlw(1.0)))
    rec = out.attrs[ATTRS_KEY]["subsets"][-1]
    assert rec["relations"] == {
        "Mlw": "sil_mlw_m0",
        "ML": "hanks_kanamori_1979",
        "aut.mag": "hanks_kanamori_1979",
    }
    # without Mlw in the list no SIL fit record is attached
    rec2 = with_derived_moment(_frame(), magnitude_types=["ML"]).attrs[ATTRS_KEY][
        "subsets"
    ][-1]
    assert "mlw_relation" not in rec2


def test_mlw_relation_none_is_the_old_hk_path() -> None:
    out = with_derived_moment(_frame(), magnitude_types=["Mlw"], mlw_relation=None)
    np.testing.assert_allclose(
        out["moment_nm"].to_numpy()[:2], moment_from_magnitude(np.array([1.0, 4.0]))
    )
    rec = out.attrs[ATTRS_KEY]["subsets"][-1]
    assert rec["relations"] == {"Mlw": "hanks_kanamori_1979"}
    assert "mlw_relation" not in rec


@pytest.mark.parametrize("mtype", NO_DERIVED_MOMENT_TYPES)
def test_types_without_relation_refused(mtype: str) -> None:
    with pytest.raises(FilterError, match="T18"):
        with_derived_moment(_frame(), magnitude_types=["Mlw", mtype])


def test_nan_magnitude_not_filled_and_existing_kept() -> None:
    df = _frame()
    df.loc[df.index[0], "moment_nm"] = 5e10
    out = with_derived_moment(df, magnitude_types=["Mlw"])
    assert out["moment_nm"].iloc[0] == 5e10
    assert not out[DERIVED_MOMENT_FLAG].iloc[0]
    assert np.isnan(out["moment_nm"].iloc[4])  # NaN magnitude stays NaN
    assert not out[DERIVED_MOMENT_FLAG].iloc[4]
