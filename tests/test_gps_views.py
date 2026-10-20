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
    DETREND_METHOD_RUNNING,
    OutlierParams,
    estimate_detrend,
    evaluate_record,
)

# The REAL convention (detrend_estimate.FRAME; all 33 deployed records carry
# it). This fixture previously invented "plate:ITRF2008", which no record uses
# — it could, because the §2.5 frame guard was dead code: the ref="detrend"
# read paths passed no frame, so nothing ever compared record to series. With
# the guard live the fixture has to tell the truth.
FRAME = "plate_removed"
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
    # robust (an outlier stage ran, no abort); the running window since
    # 2026-10-19, the model detector before
    assert not est.outlier_abort
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


def _synthetic_trailing_spike(offset_from_end=2):
    """Big spike close to the END, with too few epochs after it to judge by.

    The step statistic needs 3 post-samples, so a spike inside the last
    couple of epochs leaves D indeterminate — the shape the provisional
    mask exists for (HOFN 2026-07-24, −58.8 mm, 2 d from the end).
    Returns the spike index too, so a test can assert on that epoch.
    """
    t, y, sigma, _ = _synthetic(n_spikes=0)
    y = y.copy()
    idx = y.shape[1] - 1 - offset_from_end
    y[:, idx] += 60.0
    return t, y, sigma, idx


def test_provisional_marks_recent_indeterminate_candidates():
    t, y, sigma, idx = _synthetic_trailing_spike()
    flags, prov = gps_views.detect_view_outliers(t, y, sigma)
    provisional = np.asarray(prov["provisional"])
    assert provisional.shape == flags.shape
    assert prov["n_provisional"] == int(np.count_nonzero(provisional))
    assert provisional[:, idx].any(), "trailing unrulable spike must be provisional"
    # disjoint by construction: a protected candidate is never flagged
    assert not (flags & provisional).any()


def test_provisional_never_changes_the_flags():
    """The invariant that makes the mask safe to add: it removes nothing."""
    t, y, sigma, _ = _synthetic_trailing_spike()
    with_mask, _ = gps_views.detect_view_outliers(t, y, sigma)
    without, prov = gps_views.detect_view_outliers(t, y, sigma, provisional_days=0.0)
    np.testing.assert_array_equal(with_mask, without)
    assert prov["n_provisional"] == 0


def test_provisional_recency_bound_excludes_older_clusters():
    """Indeterminate but not recent is old news, not pending.

    The spike sits 2 days from the end, so a 1-day window must exclude it
    while the default 14-day window includes it.  Without the bound the
    mask would be dominated by gap-adjacent points deep in the series
    (measured on RHOF: 323 / 3400 / 3400 days from the end) instead of the
    recent ones it is meant to surface.
    """
    t, y, sigma, idx = _synthetic_trailing_spike(offset_from_end=2)
    _f, wide = gps_views.detect_view_outliers(t, y, sigma, provisional_days=14.0)
    _f, narrow = gps_views.detect_view_outliers(t, y, sigma, provisional_days=1.0)
    assert np.asarray(wide["provisional"])[:, idx].any()
    assert narrow["n_provisional"] == 0


def test_provisional_degrades_on_a_reduced_detection(monkeypatch):
    """A detection object without the fields must not raise -- it is a hint."""
    import types

    t, y, sigma, _ = _synthetic_trailing_spike()
    real = gps_views.detect_outliers

    def reduced(*args, **kwargs):
        full = real(*args, **kwargs)
        return types.SimpleNamespace(
            flags=full.flags, excess_flag_abort=full.excess_flag_abort
        )

    monkeypatch.setattr(gps_views, "detect_outliers", reduced)
    flags, prov = gps_views.detect_view_outliers(t, y, sigma)
    assert prov["n_provisional"] == 0
    assert flags.shape == y.shape  # cleaning itself unaffected


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


def test_running_method_needs_no_model_and_no_steps():
    """§15: with an undeclared step and a year of unrest, and nothing
    declared, the running window flags the injected blunders and leaves
    the step alone."""
    t, y, sigma, _ = _synthetic(n_spikes=0)
    y = y.copy()
    n = t.size
    y[:, n // 2 :] += 80.0  # undeclared step
    y[:, n // 4 : n // 2] += np.linspace(0.0, 60.0, n // 2 - n // 4)  # unrest
    blunders = [n // 8, 3 * n // 4]
    y[:, blunders] += 40.0
    running = OutlierParams(method="running")
    flags, prov = gps_views.detect_view_outliers(t, y, sigma, outlier_params=running)
    assert prov["outlier_abort"] is False and prov["degraded"] is False
    assert flags[:, blunders].all()
    assert np.count_nonzero(flags) <= 2 * len(blunders) * y.shape[0]
    assert flags[:, n // 2 - 3 : n // 2 + 3].sum() == 0  # the step is left alone


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
    assert (
        attrs["detrend_method"] == DETREND_METHOD_RUNNING
    )  # the record's own tag, passed through
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


def test_series_frame_tags_each_reference_frame():
    # ITRF2008 (TOT) keeps the original tag, so the deployed records still apply
    assert gps_views.series_frame() == gps_views.PLATE_REMOVED_FRAME
    assert gps_views.series_frame("itrf2008") == "plate_removed"
    assert gps_views.series_frame("igb14") == "plate_removed@itrf14"
    assert gps_views.series_frame("imo.24hr_itrf20") == "plate_removed@itrf20"
    assert gps_views.series_frame("imo.08hr_itrf20") == "plate_removed@itrf20"


def test_params_path_for_frame():
    p = Path("/cfg/detrend_params.json")
    assert gps_views.params_path_for_frame(p) == p
    assert gps_views.params_path_for_frame(p, "itrf08") == p
    assert gps_views.params_path_for_frame(p, "igb14") == Path(
        "/cfg/detrend_params.itrf14.json"
    )
    assert gps_views.params_path_for_frame(p, "itrf20") == Path(
        "/cfg/detrend_params.itrf20.json"
    )


def test_tot_record_refused_on_another_frame(view_env, seng_record, tmp_path):
    # an ITRF2008 (TOT) record must not detrend an itrf14 series; the default
    # frame of a TOT read is ITRF2008, so the same record still applies there
    path = tmp_path / "itrf08.json"
    rec = dict(seng_record, frame=gps_views.PLATE_REMOVED_FRAME)
    path.write_text(json.dumps({"schema_version": 1, "stations": {"SENG": rec}}))
    df = gps_views.read_gps_view("SENG", view="detrended", params=path, Dir=TOT)
    assert df.attrs["gps_view"]["frame"] == "plate_removed"
    with pytest.raises(ValueError, match="frame mismatch"):
        gps_views.read_gps_view(
            "SENG",
            view="detrended",
            params=path,
            frame=gps_views.series_frame("itrf14"),
            Dir=TOT,
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


def test_getdata_ref_detrend_is_plate_removed(view_env, deployed_params, seng_record):
    """ref="detrend" returns plate-removed data (same as ref="plate").
    The record application is now in the plot driver (timesmatplt.plotTime),
    NOT in getData — so ref is just a title label."""
    yearf_p, data_p, Ddata_p, _ = gpsr.getData("SENG", ref="plate", Dir=TOT)
    yearf_d, data_d, Ddata_d, _ = gpsr.getData("SENG", ref="detrend", Dir=TOT)
    np.testing.assert_array_equal(yearf_d, yearf_p)
    np.testing.assert_array_equal(Ddata_d, Ddata_p)
    np.testing.assert_array_equal(data_d, data_p)


def test_getdata_ref_detrend_no_degrading(view_env):
    """No degrade: ref="detrend" just means plate-removed. No warning."""
    yearf_p, data_p, _, _ = gpsr.getData("SENG", ref="plate", Dir=TOT)
    yearf_d, data_d, _, _ = gpsr.getData("SENG", ref="detrend", Dir=TOT)
    np.testing.assert_array_equal(yearf_d, yearf_p)
    np.testing.assert_array_equal(data_d, data_p)


def test_gamittoneu_ref_detrend_is_plate_removed(
    view_env, deployed_params, seng_record
):
    """gamittoNEU(ref="detrend") returns plate-removed data — same as ref="plate"."""
    neu_p = gpsr.gamittoNEU("SENG", mm=True, ref="plate", dstring="yearf", Dir=TOT)
    neu_d = gpsr.gamittoNEU("SENG", mm=True, ref="detrend", dstring="yearf", Dir=TOT)
    for c in range(3):
        np.testing.assert_allclose(
            np.asarray(neu_d[f"data[{c}]"]),
            np.asarray(neu_p[f"data[{c}]"]),
            rtol=0.0,
            atol=1e-6,
        )
        np.testing.assert_array_equal(neu_d[f"Ddata[{c}]"], neu_p[f"Ddata[{c}]"])


# ---------------------------------------------------------------------------
# remove_declared_steps — selective step removal by kind
# ---------------------------------------------------------------------------


def _stepped_record(seng_plate, tmp_path, step_epoch=2017.0):
    """A SENG record fitted with one declared step at ``step_epoch``."""
    yearf, data, Ddata = seng_plate
    # Inject a REAL step so the fitted amplitude is non-trivial: otherwise the
    # estimator finds step_amp ~ 0 (there is no jump in the golden data there).
    data = data.copy()
    data[:, yearf >= step_epoch] += 60.0
    est = estimate_detrend(
        "lineperiodic",
        yearf,
        data,
        Ddata,
        window=WINDOW,
        frame=FRAME,
        step_epochs=[step_epoch],
    )
    return est.to_record(fitted_at="2026-07-14T00:00:00Z")


def _steps_yaml(config_dir, sta="SENG", epoch=2017.0, kind="earthquake"):
    p = config_dir / "steps.yaml"
    p.write_text(
        "schema_version: 1\n"
        "stations:\n"
        f"  {sta}:\n"
        f"    - epoch_yearf: {epoch}\n"
        f"      kind: {kind}\n"
        "      comment: test step\n"
    )
    return p


def test_remove_declared_steps_subtracts_selected_kind(view_env, seng_plate, tmp_path):
    """An earthquake-removal subtracts the recorded step amplitude; the series
    is a new array (input untouched) and the provenance names the removal."""
    record = _stepped_record(seng_plate, tmp_path)
    params = tmp_path / "detrend_params.json"
    params.write_text(
        json.dumps({"schema_version": 1, "frame": FRAME, "stations": {"SENG": record}})
    )
    _steps_yaml(view_env["config_dir"], kind="earthquake")

    yearf, data, _ = seng_plate
    out, prov = gps_views.remove_declared_steps(
        "SENG", yearf, data, kinds=["earthquake"], params=params
    )
    assert prov["applied"] is True and not prov["degraded"]
    assert prov["removed"] == [(2017.0, "earthquake")]
    # input untouched
    assert out is not data
    # the 60 mm injected step was removed: post-step minus pre-step drops by
    # ~60 mm (the trend + seasonal difference REMAINS — this removes steps
    # only, not the background)
    pre = np.nanmean(data[0, yearf < 2017.0])
    post = np.nanmean(data[0, yearf >= 2017.0])
    pre2 = np.nanmean(out[0, yearf < 2017.0])
    post2 = np.nanmean(out[0, yearf >= 2017.0])
    drop = (post - pre) - (post2 - pre2)
    assert drop == pytest.approx(60.0, abs=5.0), f"step removed {drop:.1f} mm"


def test_remove_declared_steps_skips_unselected_kind(view_env, seng_plate, tmp_path):
    """An equipment-removal on a station whose only step is seismic removes
    nothing and degrades with the reason named."""
    record = _stepped_record(seng_plate, tmp_path)
    params = tmp_path / "detrend_params.json"
    params.write_text(
        json.dumps({"schema_version": 1, "frame": FRAME, "stations": {"SENG": record}})
    )
    _steps_yaml(view_env["config_dir"], kind="earthquake")

    yearf, data, _ = seng_plate
    out, prov = gps_views.remove_declared_steps(
        "SENG", yearf, data, kinds=["equipment"], params=params
    )
    assert prov["applied"] is False and prov["degraded"] is True
    assert "not selected" in prov["skipped"][0][1]
    np.testing.assert_array_equal(out, data)


def test_remove_declared_steps_no_kinds_is_a_noop(view_env, seng_plate, tmp_path):
    yearf, data, _ = seng_plate
    out, prov = gps_views.remove_declared_steps("SENG", yearf, data, kinds=[])
    assert prov["removed"] == [] and prov["applied"] is False
    np.testing.assert_array_equal(out, data)


# ---------------------------------------------------------------------------
# window_datum_shift: a windowed read vs the full-span one
# ---------------------------------------------------------------------------


def test_window_datum_shift_is_the_constant_between_reads(view_env, seng_plate):
    """A windowed getData series is the full one minus a per-component
    constant (zeroed and plate-anchored at the window start); the shift is
    that constant, exactly."""
    t_full, d_full, _ = seng_plate
    t_win, d_win, _, _ = gpsr.getData("SENG", fstart=2018.0, ref="plate", Dir=TOT)
    shift = gps_views.window_datum_shift(
        "SENG", fstart=2018.0, fend=None, ref="plate", Dir=TOT
    )
    assert shift.shape == (3,) and np.all(np.abs(shift) > 1.0)
    _, i, j = np.intersect1d(t_full, t_win, return_indices=True)
    np.testing.assert_allclose(d_win[:, j], d_full[:, i] - shift[:, None], atol=1e-9)


def test_window_datum_shift_no_window_is_zero(view_env):
    np.testing.assert_array_equal(
        gps_views.window_datum_shift("SENG", fstart=None, fend=None, Dir=TOT),
        np.zeros(3),
    )


def test_window_datum_shift_aligns_a_full_span_record(view_env, seng_record):
    """The point of the shift: a record fitted on the full read, evaluated
    on a windowed one, lines up once the shift is subtracted."""
    t_win, d_win, _, _ = gpsr.getData(
        "SENG", fstart=2018.5, fend=2019.9, ref="plate", Dir=TOT
    )
    shift = gps_views.window_datum_shift(
        "SENG", fstart=2018.5, fend=2019.9, ref="plate", Dir=TOT
    )
    model = np.asarray(evaluate_record(seng_record, t_win))
    resid = d_win - (model - shift[:, None])
    assert np.all(np.abs(np.nanmedian(resid, axis=1)) < 3.0)
    unaligned = d_win - model
    assert np.max(np.abs(np.nanmedian(unaligned, axis=1))) > 10.0


def test_window_datum_shift_disjoint_window_raises(view_env):
    with pytest.raises(ValueError, match="no data"):
        gps_views.window_datum_shift("SENG", fstart=1990.0, fend=1991.0, Dir=TOT)


def test_icing_days_are_taken_out_then_flagged():
    """An iced block (days of offset) is not an outlier to the running window
    — a run of that length is signal — so only the icing index removes it."""
    t, y, sigma, _ = _synthetic(n_spikes=0)
    y = y.copy()
    block = np.arange(600, 612)
    y[:, block] += 25.0  # twelve iced days
    running = OutlierParams(method="running")
    bare, _ = gps_views.detect_view_outliers(t, y, sigma, outlier_params=running)
    assert not bare[:, block].any()
    flags, prov = gps_views.detect_view_outliers(
        t, y, sigma, outlier_params=running, icing_epochs=t[block], icing_mode="remove"
    )
    assert flags[:, block].all() and prov["n_icing"] == block.size
    assert prov["icing"][:, block].all() and prov["icing"].sum() == 3 * block.size


def test_icing_days_are_flagged_even_when_detection_fails():
    t, y, sigma, _ = _synthetic(n_spikes=0)
    bad = OutlierParams(method="running", max_flag_fraction=1e-9)  # aborts all
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y2 = y.copy()
        y2[:, 300] += 40.0
        flags, prov = gps_views.detect_view_outliers(
            t,
            y2,
            sigma,
            outlier_params=bad,
            icing_epochs=t[[500, 501]],
            icing_mode="remove",
        )
    assert prov["outlier_abort"] is True
    assert flags[:, [500, 501]].all() and flags.sum() == 6


def test_no_harvest_means_no_icing_silently(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert gps_views.resolve_icing_epochs("SENG", tmp_path) == ((), None)
    assert gps_views.resolve_icing_epochs("SENG", False) == ((), None)
    assert gps_views.resolve_icing_epochs("SENG", [2026.5]) == ((2026.5,), "explicit")


def test_icing_mode_keep_keeps_the_days_and_says_which():
    """The default: suspected, not decided -- out of detection, not flagged."""
    t, y, sigma, _ = _synthetic(n_spikes=0)
    y = y.copy()
    block = np.arange(600, 612)
    y[:, block] += 25.0
    y[:, 300] += 40.0  # a real blunder elsewhere
    running = OutlierParams(method="running")
    flags, prov = gps_views.detect_view_outliers(
        t, y, sigma, outlier_params=running, icing_epochs=t[block]
    )
    assert not flags[:, block].any() and flags[:, 300].all()
    assert prov["icing"][:, block].all() and prov["n_icing"] == block.size
    with pytest.raises(ValueError, match="icing_mode"):
        gps_views.detect_view_outliers(
            t, y, sigma, icing_epochs=t[block], icing_mode="hide"
        )


def test_a_one_sided_candidate_before_an_icing_spell_is_decided():
    """GFUM 2026-09: a blunder just before weeks of icing stayed yellow
    (provisional) because "recent" was measured on the detector's input,
    which the iced days are kept out of. Measured from the real end, it is
    old and decided: an outlier. A blunder in the last days stays pending."""
    t, y, sigma, _ = _synthetic(n_spikes=0)
    y = y.copy()
    n = t.size
    iced = np.arange(n - 40, n)  # the last 40 days iced
    y[:, n - 42] += 40.0  # just before the spell: one side only
    running = OutlierParams(method="running")
    flags, prov = gps_views.detect_view_outliers(
        t, y, sigma, outlier_params=running, icing_epochs=t[iced]
    )
    assert (
        flags[:, n - 42].all() and not np.asarray(prov["provisional"])[:, n - 42].any()
    )
    y2 = y.copy()
    y2[:, n - 42] -= 40.0
    y2[:, n - 3] += 40.0  # the newest days, no icing: still pending
    flags2, prov2 = gps_views.detect_view_outliers(t, y2, sigma, outlier_params=running)
    assert (
        not flags2[:, n - 3].any() and np.asarray(prov2["provisional"])[:, n - 3].all()
    )


def test_a_one_sided_candidate_at_the_series_start_stays_protected():
    """The start of a read window is not a blunder: missing BEFORE side."""
    t, y, sigma, _ = _synthetic(n_spikes=0)
    y = y.copy()
    y[:, 2] += 40.0
    flags, _ = gps_views.detect_view_outliers(
        t, y, sigma, outlier_params=OutlierParams(method="running")
    )
    assert not flags[:, 2].any()


def test_station_step_components_masks_by_component(tmp_path):
    """One epoch declared as an N row and an E row ORs to a horizontal-only
    step; an untagged (ALL) step applies everywhere."""
    p = tmp_path / "steps.yaml"
    p.write_text(
        "schema_version: 1\n"
        "stations:\n"
        "  GJOG:\n"
        "    - epoch_yearf: 2020.471311\n"
        "      component: N\n"
        "      kind: earthquake\n"
        "    - epoch_yearf: 2020.471311\n"
        "      component: E\n"
        "      kind: earthquake\n"
        "    - epoch_yearf: 2023.5\n"
        "      kind: equipment\n"
    )
    epochs, mask, source = gps_views.station_step_components("GJOG", steps=p)
    np.testing.assert_array_equal(epochs, [2020.471311, 2023.5])
    np.testing.assert_array_equal(mask, [[True, True, False], [True, True, True]])
    assert source == str(p)


def test_station_step_components_degrades_like_the_epochs(tmp_path):
    epochs, mask, source = gps_views.station_step_components(
        "GJOG", steps=tmp_path / "missing.yaml"
    )
    assert epochs.size == 0 and mask.shape == (0, 3) and source is None
