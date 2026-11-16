"""moment_estimate: relation per magnitude type (PLAN-ts-quakes Q-T1/Q4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geo_dataread.seismic import moment as mm
from geo_dataread.seismic.filters import moment_from_mlw
from geo_dataread.seismic.schema import coerce_catalog


def cat(rows: list[tuple[float, str | None, float]]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=len(rows), freq="h", tz="UTC"),
            "event_id": [f"e{i}" for i in range(len(rows))],
            "latitude": 64.6,
            "longitude": -17.4,
            "magnitude": [r[0] for r in rows],
            "magnitude_type": [r[1] for r in rows],
            "moment_nm": [r[2] for r in rows],
            "evaluation_mode": "manual",
        }
    )
    return coerce_catalog(df.set_index("time"), source="quakes-seiscomp")


def test_relation_chosen_by_type() -> None:
    df = cat(
        [
            (2.0, "Mlw", 3e12),  # measured: kept
            (2.0, "Mlw", np.nan),  # SIL relation
            (2.0, "ML_SIL", np.nan),
            (4.8, "Mpgv_w", np.nan),
            (2.5, "Mpgv_w", np.nan),  # below the Mpgv_w range: not estimated
            (1.0, "aut.mag", np.nan),  # no relation
            (np.nan, "ML_SIL", np.nan),  # no magnitude
        ]
    )
    out = mm.moment_estimate(df)
    m0 = out["moment_nm"].to_numpy()
    assert m0[0] == 3e12
    assert m0[1] == pytest.approx(float(moment_from_mlw(2.0)))
    assert m0[2] == pytest.approx(10 ** (10.41 + 1.045 * 2.0))
    assert m0[3] == pytest.approx(10 ** (8.95 + 1.347 * 4.8))
    assert np.isnan(m0[4]) and np.isnan(m0[5]) and np.isnan(m0[6])
    assert list(out[mm.ESTIMATED_FLAG]) == [
        False,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    rel = [None if pd.isna(v) else v for v in out[mm.RELATION_COLUMN]]
    assert rel == [
        "measured",
        "sil_mlw_m0",
        "seiscomp_ml_sil_m0",
        "seiscomp_mpgv_w_m0",
        None,
        None,
        None,
    ]
    rec = out.attrs["catalog"]["subsets"][-1]
    assert rec["filter"] == "moment_estimate"
    assert rec["unestimated_by_type"] == {"Mpgv_w": 1, "aut.mag": 1}
    assert rec["no_magnitude"] == 1
    assert rec["relations"]["ML_SIL"]["n"] == 2622


def test_input_untouched_and_mlw_switch() -> None:
    df = cat([(2.0, "Mlw", np.nan)])
    out = mm.moment_estimate(df, mlw=False)
    assert np.isnan(out["moment_nm"].iloc[0])
    assert np.isnan(df["moment_nm"].iloc[0])


def test_mismatched_relation_key_raises() -> None:
    with pytest.raises(ValueError, match="is for"):
        mm.moment_estimate(
            cat([(2.0, "ML_SIL", np.nan)]), relations={"Mlw": mm.SEISCOMP_ML_SIL}
        )


def test_missing_column_raises() -> None:
    with pytest.raises(ValueError, match="magnitude_type"):
        mm.moment_estimate(pd.DataFrame({"moment_nm": [], "magnitude": []}))


def test_relations_meet_at_the_mpgv_floor() -> None:
    # Mpgv_w 3 is about ML_SIL 2.6 (Mpgv_w runs ~0.4 above ML_SIL at 2–3):
    # the two relations agree there within the ML_SIL rms (0.29)
    a = mm.SEISCOMP_ML_SIL.log10_moment(2.6)
    b = mm.SEISCOMP_MPGV_W.log10_moment(3.0)
    assert abs(float(a) - float(b)) < 0.29
