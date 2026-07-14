"""Tests for the apply-on-read view toggle (geo_dataread.gps_views).

Internal-delivery slice of DESIGN_live_detrending.md — pinned here:

- raw view bit-identical to the legacy read (regression guard),
- cleaned view: mask-only outlier flags, raw preserved,
- detrended view: PURE apply of a stored record (no re-fit on read),
  exactly matching the leaf evaluation, term selection included,
- graceful degrade: any cleaning/detrend failure warns and serves raw,
- provenance: method tag / frame / record version / fitted_at / borrowed,
- UseSTA borrowing, frame-mismatch hard refusal,
- the revived array paths: getData(ref="detrend"), gamittoNEU(ref="detrend").

Uses the goldenmaster fixtures (frozen GLOBK series + hermetic gpsconfig).
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "goldenmaster"))

from cases import TOT, build_config_dir  # noqa: E402

import geo_dataread.gps_read as gpsr  # noqa: E402
from geo_dataread import gps_views  # noqa: E402
from gps_analysis import (  # noqa: E402
    DETREND_METHOD_ROBUST,
    OutlierParams,
    estimate_detrend,
    evaluate_record,
)

FRAME = "plate:ITRF2008"
WINDOW = (2015.0, 2020.0)  # pre-unrest fit window for the SENG fixture


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def view_env(tmp_path_factory):
    config_dir = build_config_dir(tmp_path_factory.mktemp("gpsconfig-views"))
    old = os.environ.get("GPS_CONFIG_PATH")
    os.environ["GPS_CONFIG_PATH"] = str(config_dir)
    yield {"config_dir": config_dir}
    if old is None:
        os.environ.pop("GPS_CONFIG_PATH", None)
    else:
        os.environ["GPS_CONFIG_PATH"] = old


@pytest.fixture(scope="module")
def seng_plate(view_env):
    """Legacy plate-removed read of the SENG fixture (mm)."""
    yearf, data, Ddata, _ = gpsr.getData("SENG", ref="plate", Dir=TOT)
    return yearf, data, Ddata


@pytest.fixture(scope="module")
def seng_record(seng_plate):
    """Stored detrend record fitted on the pre-unrest SENG window."""
    yearf, data, Ddata = seng_plate
    est = estimate_detrend(
        "lineperiodic", yearf, data, Ddata, window=WINDOW, frame=FRAME
    )
    assert est.detrend_method == DETREND_METHOD_ROBUST
    return est.to_record(fitted_at="2026-07-14T00:00:00Z")


@pytest.fixture()
def params_file(view_env, seng_record, tmp_path):
    """A schema-v1 parameter document holding the SENG record."""
    doc = {
        "schema_version": 1,
        "frame": FRAME,
        "stations": {"SENG": seng_record},
    }
    path = tmp_path / "detrend_params.json"
    path.write_text(json.dumps(doc))
    return path


@pytest.fixture()
def deployed_params(view_env, seng_record):
    """The document at its deployed default location (gpsconfig dir)."""
    doc = {
        "schema_version": 1,
        "frame": FRAME,
        "stations": {"SENG": seng_record},
    }
    path = view_env["config_dir"] / "detrend_params.json"
    path.write_text(json.dumps(doc))
    yield path
    path.unlink()


def _legacy_frame(sta, **kwargs):
    yearf, data, Ddata, _ = gpsr.getData(sta, ref="plate", Dir=TOT, **kwargs)
    return gpsr.convGlobktopandas(yearf, data, Ddata)


def _synthetic(n_spikes=3, seed=7):
    """4-yr daily lineperiodic + noise series with injected spikes."""
    rng = np.random.default_rng(seed)
    t = 2015.0 + np.arange(1461) / 365.25
    y = np.vstack(
        [
            base
            + rate * (t - 2015.0)
            + 3.0 * np.cos(2 * np.pi * t)
            + 1.5 * np.sin(2 * np.pi * t)
            + rng.normal(0.0, 1.0, t.size)
            for base, rate in ((0.0, 20.0), (5.0, -12.0), (-3.0, 2.0))
        ]
    )
    sigma = np.full(y.shape, 1.0)
    spike_idx = np.linspace(200, 1200, n_spikes).astype(int)
    y[:, spike_idx] += 15.0  # 15σ — unambiguous outliers
    return t, y, sigma, spike_idx


# ---------------------------------------------------------------------------
# raw view — the regression guard
# ---------------------------------------------------------------------------


def test_raw_view_bit_identical_to_legacy(view_env):
    df = gps_views.read_gps_view("SENG", Dir=TOT)
    legacy = _legacy_frame("SENG")
    assert list(df.columns) == list(legacy.columns)
    for col in legacy.columns:
        np.testing.assert_array_equal(
            df[col].to_numpy(), legacy[col].to_numpy(), err_msg=col
        )
    assert df.index.equals(legacy.index)
    attrs = df.attrs["gps_view"]
    assert attrs["view"] == "raw"
    assert attrs["clean"] is False
    assert attrs["degraded"] is False


def test_raw_view_window_matches_legacy(view_env):
    df = gps_views.read_gps_view("SENG", start=2020.0, end=2024.5, Dir=TOT)
    legacy = _legacy_frame("SENG", fstart=2020.0, fend=2024.5)
    pd.testing.assert_frame_equal(df, legacy)


def test_invalid_view_and_ref_combinations(view_env):
    with pytest.raises(ValueError, match="view must be one of"):
        gps_views.read_gps_view("SENG", view="detrend", Dir=TOT)
    with pytest.raises(ValueError, match="view='detrended'"):
        gps_views.read_gps_view("SENG", view="raw", ref="detrend", Dir=TOT)
    with pytest.raises(ValueError, match="plate-first"):
        gps_views.read_gps_view("SENG", view="detrended", ref="itrf2008", Dir=TOT)


# ---------------------------------------------------------------------------
# cleaned view — mask only, raw preserved
# ---------------------------------------------------------------------------


def test_detect_view_outliers_flags_spikes():
    t, y, sigma, spike_idx = _synthetic()
    flags, prov = gps_views.detect_view_outliers(t, y, sigma)
    assert flags.shape == y.shape
    assert prov["degraded"] is False and prov["outlier_abort"] is False
    assert flags[:, spike_idx].all(), "injected 15-sigma spikes must be flagged"
    assert prov["n_flagged"] == int(np.count_nonzero(flags))
    # conservative: well under 5% of the series flagged
    assert np.count_nonzero(flags) < 0.05 * y.size


def test_detect_view_outliers_degrades_on_failure():
    t, y, sigma, _ = _synthetic()
    t_bad = t[::-1].copy()  # unsorted epochs -> leaf raises
    with pytest.warns(UserWarning, match="outlier detection failed"):
        flags, prov = gps_views.detect_view_outliers(t_bad, y, sigma)
    assert not flags.any()
    assert prov["degraded"] is True
    assert "serving unflagged" in prov["degrade_reason"]


def test_detect_view_outliers_abort_degrades():
    t, y, sigma, _ = _synthetic(n_spikes=30)
    tight = OutlierParams(max_flag_fraction=0.001)
    with pytest.warns(UserWarning, match="aborted"):
        flags, prov = gps_views.detect_view_outliers(t, y, sigma, outlier_params=tight)
    assert not flags.any()
    assert prov["outlier_abort"] is True and prov["degraded"] is True


def test_cleaned_view_preserves_raw_columns(view_env):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # full SENG spans unrest; abort allowed
        df = gps_views.read_gps_view("SENG", view="cleaned", Dir=TOT)
    legacy = _legacy_frame("SENG")
    for col in legacy.columns:  # raw columns untouched by the mask
        np.testing.assert_array_equal(df[col].to_numpy(), legacy[col].to_numpy())
    for name in ("north", "east", "up"):
        assert df[f"{name}_outlier"].dtype == np.bool_
        flagged = df[f"{name}_outlier"].to_numpy()
        cleaned = df[f"{name}_cleaned"].to_numpy()
        raw = df[name].to_numpy()
        np.testing.assert_array_equal(cleaned[~flagged], raw[~flagged])
        assert np.isnan(cleaned[flagged]).all()
    attrs = df.attrs["gps_view"]
    assert attrs["view"] == "cleaned" and attrs["clean"] is True
    assert attrs["n_flagged"] == int(
        df[[f"{n}_outlier" for n in ("north", "east", "up")]].to_numpy().sum()
    )


# ---------------------------------------------------------------------------
# detrended view — pure apply of the stored record
# ---------------------------------------------------------------------------


def test_detrended_view_is_pure_apply(view_env, params_file, seng_record):
    df = gps_views.read_gps_view(
        "SENG", view="detrended", params=params_file, frame=FRAME, Dir=TOT
    )
    legacy = _legacy_frame("SENG")
    for col in legacy.columns:  # raw stays retrievable next to the view
        np.testing.assert_array_equal(df[col].to_numpy(), legacy[col].to_numpy())
    trend = evaluate_record(seng_record, df["yearf"].to_numpy())
    for c, name in enumerate(("north", "east", "up")):
        np.testing.assert_allclose(
            df[f"{name}_detrended"].to_numpy(),
            df[name].to_numpy() - trend[c],
            rtol=0.0,
            atol=1e-12,
            err_msg=f"{name}: detrended view must equal raw - stored trajectory",
        )
    attrs = df.attrs["gps_view"]
    assert attrs["detrend_method"] == DETREND_METHOD_ROBUST
    assert attrs["frame"] == FRAME
    assert attrs["record_version"] == 1
    assert attrs["fitted_at"] == "2026-07-14T00:00:00Z"
    assert attrs["borrowed"] is None
    assert attrs["params_station"] == "SENG"
    assert attrs["degraded"] is False


def test_detrended_view_term_selection(view_env, params_file, seng_record):
    df = gps_views.read_gps_view(
        "SENG", view="detrended", params=params_file, terms="periodic", Dir=TOT
    )
    trend = evaluate_record(seng_record, df["yearf"].to_numpy(), terms="periodic")
    np.testing.assert_allclose(
        df["north_detrended"].to_numpy(),
        df["north"].to_numpy() - trend[0],
        rtol=0.0,
        atol=1e-12,
    )
    assert df.attrs["gps_view"]["terms"] == "periodic"


def test_detrended_missing_station_degrades_to_raw(view_env, params_file):
    with pytest.warns(UserWarning, match="absent from the detrend parameter"):
        df = gps_views.read_gps_view(
            "ELDC", view="detrended", params=params_file, Dir=TOT
        )
    legacy = _legacy_frame("ELDC")
    for col in legacy.columns:
        np.testing.assert_array_equal(df[col].to_numpy(), legacy[col].to_numpy())
    assert "north_detrended" not in df.columns
    attrs = df.attrs["gps_view"]
    assert attrs["degraded"] is True
    assert "absent" in attrs["degrade_reason"]


def test_detrended_missing_document_degrades_to_raw(view_env, tmp_path):
    with pytest.warns(UserWarning, match="no detrend parameters"):
        df = gps_views.read_gps_view(
            "SENG", view="detrended", params=tmp_path / "nope.json", Dir=TOT
        )
    assert "north_detrended" not in df.columns
    assert df.attrs["gps_view"]["degraded"] is True


def test_detrended_forced_apply_failure_degrades_to_raw(
    view_env, params_file, monkeypatch
):
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic apply failure")

    monkeypatch.setattr(gps_views, "apply_stored_detrend", boom)
    with pytest.warns(UserWarning, match="detrend application failed"):
        df = gps_views.read_gps_view(
            "SENG", view="detrended", params=params_file, Dir=TOT
        )
    legacy = _legacy_frame("SENG")
    np.testing.assert_array_equal(df["north"].to_numpy(), legacy["north"].to_numpy())
    assert "north_detrended" not in df.columns
    attrs = df.attrs["gps_view"]
    assert attrs["degraded"] is True
    assert "synthetic apply failure" in attrs["degrade_reason"]


def test_detrended_corrupt_record_degrades_to_raw(view_env, seng_record, tmp_path):
    broken = dict(seng_record)
    broken["components"] = [
        {**c, "params": c["params"][:3]} for c in seng_record["components"]
    ]
    path = tmp_path / "broken.json"
    path.write_text(json.dumps({"schema_version": 1, "stations": {"SENG": broken}}))
    with pytest.warns(UserWarning, match="detrend application failed"):
        df = gps_views.read_gps_view("SENG", view="detrended", params=path, Dir=TOT)
    assert "north_detrended" not in df.columns
    assert df.attrs["gps_view"]["degraded"] is True


def test_borrowed_use_sta_record(view_env, params_file, seng_record):
    """UseSTA borrowing: ELDC served with SENG's stored parameters."""
    df = gps_views.read_gps_view(
        "ELDC", view="detrended", params=params_file, use_sta="SENG", Dir=TOT
    )
    trend = evaluate_record(seng_record, df["yearf"].to_numpy())
    np.testing.assert_allclose(
        df["north_detrended"].to_numpy(),
        df["north"].to_numpy() - trend[0],
        rtol=0.0,
        atol=1e-12,
    )
    attrs = df.attrs["gps_view"]
    assert attrs["params_station"] == "SENG"
    assert attrs["borrowed"] == {
        "from": "SENG",
        "terms": "all",
        "donor_fitted_at": "2026-07-14T00:00:00Z",
    }
    assert attrs["degraded"] is False


def test_frame_mismatch_is_a_hard_error(view_env, seng_record, tmp_path):
    path = tmp_path / "frames.json"
    path.write_text(
        json.dumps({"schema_version": 1, "stations": {"SENG": seng_record}})
    )
    with pytest.raises(ValueError, match="frame mismatch"):
        gps_views.read_gps_view(
            "SENG", view="detrended", params=path, frame="plate:ITRF2014", Dir=TOT
        )


# ---------------------------------------------------------------------------
# document reader validation
# ---------------------------------------------------------------------------


def test_read_detrend_params_rejects_bad_schema(tmp_path):
    p = tmp_path / "v99.json"
    p.write_text(json.dumps({"schema_version": 99, "stations": {}}))
    with pytest.raises(ValueError, match="schema_version"):
        gps_views.read_detrend_params(p)
    p2 = tmp_path / "nostations.json"
    p2.write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(ValueError, match="stations"):
        gps_views.read_detrend_params(p2)
    with pytest.raises(FileNotFoundError):
        gps_views.read_detrend_params(tmp_path / "missing.json")


def test_default_params_path_falls_back_to_gpsconfig(view_env):
    path = gps_views.default_params_path()
    assert path == view_env["config_dir"] / "detrend_params.json"


# ---------------------------------------------------------------------------
# revived legacy array paths
# ---------------------------------------------------------------------------


def test_getdata_ref_detrend_is_stored_apply(view_env, deployed_params, seng_record):
    yearf_p, data_p, Ddata_p, _ = gpsr.getData("SENG", ref="plate", Dir=TOT)
    yearf_d, data_d, Ddata_d, _ = gpsr.getData("SENG", ref="detrend", Dir=TOT)
    np.testing.assert_array_equal(yearf_d, yearf_p)
    np.testing.assert_array_equal(Ddata_d, Ddata_p)
    trend = evaluate_record(seng_record, yearf_p)
    np.testing.assert_allclose(data_d, data_p - trend, rtol=0.0, atol=1e-9)


def test_getdata_ref_detrend_degrades_to_plate(view_env):
    # no document deployed -> warn + plate-removed series unchanged
    yearf_p, data_p, _, _ = gpsr.getData("SENG", ref="plate", Dir=TOT)
    with pytest.warns(UserWarning, match="no detrend parameters"):
        yearf_d, data_d, _, _ = gpsr.getData("SENG", ref="detrend", Dir=TOT)
    np.testing.assert_array_equal(yearf_d, yearf_p)
    np.testing.assert_array_equal(data_d, data_p)


def test_gamittoneu_ref_detrend_is_stored_apply(view_env, deployed_params, seng_record):
    neu_p = gpsr.gamittoNEU("SENG", mm=True, ref="plate", dstring="yearf", Dir=TOT)
    neu_d = gpsr.gamittoNEU("SENG", mm=True, ref="detrend", dstring="yearf", Dir=TOT)
    yearf = np.asarray(neu_p["yearf"], dtype=np.float64)
    trend = evaluate_record(seng_record, yearf)
    for c in range(3):
        np.testing.assert_allclose(
            np.asarray(neu_d[f"data[{c}]"]),
            np.asarray(neu_p[f"data[{c}]"]) - trend[c],
            rtol=0.0,
            atol=1e-6,  # m-unit apply + mm conversion round-trip
        )
        np.testing.assert_array_equal(neu_d[f"Ddata[{c}]"], neu_p[f"Ddata[{c}]"])
