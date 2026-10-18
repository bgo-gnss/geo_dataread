"""The velocity product: estimator core, record round trip, files, CLI."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from geo_dataread import velocity as vel


def _series(rates=(1.0, -2.0, 3.0), years=0.25, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    t = 2026.0 + np.arange(0.0, years, 1 / 365.25)
    y = np.vstack([r * (t - t[0]) + rng.normal(0, noise, t.size) for r in rates])
    return t, y, np.full_like(y, noise)


def _record(**changes):
    base = vel.VelocityRecord(
        station="TEST", lon=-22.5, lat=63.9, frame="itrf2014", plate="NOAM",
        ref="plate-periodic", ref_requested="plate-periodic",
        series="imo.24hr_itrf14", view="cleaned", uncert=10.0,
        window=(2026.0, 2026.25), span_used=(2026.0, 2026.249),
        n_epochs=(91, 91, 91), largest_gap=0.003, method="wls", model="linear",
        rate=(1.0, -2.0, 3.0), sigma=(2.0, 2.0, 6.0),
        sigma_formal=(0.5, 0.5, 1.5), sigma_method="empirical_k",
        k=(4.0, 4.0, 4.0), k_source={"quiet_span": [2010.0, 2019.0]},
        corr_en=0.0, record=None, steps_in_window=(), outlier_abort=False,
        n_flagged=0, flags=(), refused=None, software={"x": "1"},
        input_fingerprint="abc", snr=(10.0, 2.0), chi2nu=(1.0, 1.1, 0.4),
        f_quadratic=(0.5, 1.0, 2.0),
    )  # fmt: skip
    return dataclasses.replace(base, **changes)


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------


def test_line_rates_are_recovered_per_component():
    t, y, s = _series(noise=0.1)
    wr = vel.velocity_from_arrays(t, y, s, (t[0], t[-1] + 1e-6))
    assert wr.reasons == ()
    np.testing.assert_allclose(wr.rate, [1.0, -2.0, 3.0], atol=0.3)
    assert np.all(wr.sigma_formal > 0) and wr.n_epochs == (t.size,) * 3


def test_nan_epochs_drop_per_component_and_from_the_gates():
    t, y, s = _series()
    y[2, ::2] = np.nan  # outliers in up only
    wr = vel.velocity_from_arrays(t, y, s, (t[0], t[-1] + 1e-6))
    assert wr.n_epochs[0] == t.size and wr.n_epochs[2] < t.size
    assert np.isfinite(wr.rate).all()


def test_count_and_cover_gates_leave_a_short_window_to_the_short_tier():
    t, y, s = _series(years=0.01)  # 4 epochs
    wr = vel.velocity_from_arrays(t, y, s, (t[0], t[0] + 0.25))
    assert any(r.startswith("n_obs") for r in wr.reasons)
    assert any(r.startswith("cover") for r in wr.reasons)
    assert wr.short and np.isfinite(wr.rate).all()
    assert np.all(wr.sigma_short >= wr.sigma_formal - 1e-12)
    # a step inside refuses outright: no short tier
    wr = vel.velocity_from_arrays(t, y, s, (t[0], t[0] + 0.25),
                                  step_epochs=[float(t[1]) + 1e-4])  # fmt: skip
    assert not wr.short and np.isnan(wr.rate).all()


def test_a_step_inside_refuses_unless_allowed():
    t, y, s = _series()
    step = float(t[40])
    wr = vel.velocity_from_arrays(t, y, s, (t[0], t[-1] + 1e-6), step_epochs=[step])
    assert wr.reasons == (f"step_in_window@{step:.4f}",)
    wr = vel.velocity_from_arrays(t, y, s, (t[0], t[-1] + 1e-6), step_epochs=[step],
                                  allow_steps=True)  # fmt: skip
    assert wr.reasons == () and wr.steps_in_window == (step,)


def _table(frame="itrf2014"):
    return {
        "schema": vel.K_TABLE_SCHEMA, "frame": frame, "series": "imo.24hr_itrf14",
        "residuals": {
            "all": {"days": [7, 365], "k": [[1.5, 1.5, 1.5], [5.0, 4.0, 6.0]],
                    "n_stations": [10, 8]},
            "secular": {"days": [7], "k": [[2.0, 2.0, 2.0]], "n_stations": [10]},
        },
    }  # fmt: skip


def test_table_k_interpolates_by_residual():
    t = _table()
    np.testing.assert_allclose(vel.table_k(t, "all", 7 / 365.25), [1.5, 1.5, 1.5])
    np.testing.assert_allclose(vel.table_k(t, "all", 2.0), [5.0, 4.0, 6.0])
    np.testing.assert_allclose(vel.table_k(t, "secular", 1.0), [2.0, 2.0, 2.0])
    assert vel.table_k({"residuals": {}}, "all", 1.0) is None


def test_load_k_table_checks_schema_and_frame(tmp_path):
    path = tmp_path / "k.json"
    path.write_text(json.dumps(_table()))
    table, source = vel.load_k_table(vel.VelocitySettings(k_table=str(path)))
    assert table["frame"] == "itrf2014" and source.startswith(f"{path}#sha256:")
    path.write_text(json.dumps(_table(frame="itrf2020")))
    with pytest.raises(ValueError, match="frame"):
        vel.load_k_table(vel.VelocitySettings(k_table=str(path)))
    assert vel.load_k_table(vel.VelocitySettings(k_table="none")) == (None, None)
    with pytest.raises(FileNotFoundError):
        vel.load_k_table(vel.VelocitySettings(k_table=str(tmp_path / "nope.json")))


def test_k_residual_keeps_the_seasonal_for_plate():
    assert vel.K_RESIDUAL == {
        "plate-periodic": "all", "background": "all", "plate": "secular",
    }  # fmt: skip


def test_window_bounds_take_whole_days():
    t0, t1 = vel.window_bounds(dt.date(2026, 8, 15), dt.date(2026, 8, 16))
    assert t1 - t0 == pytest.approx(2 / 365, rel=1e-6)
    assert vel.window_bounds(2026.1, 2026.2) == (2026.1, 2026.2)


def test_settings_validate():
    with pytest.raises(ValueError, match="ref must be"):
        vel.VelocitySettings(ref="itrf")
    with pytest.raises(ValueError, match="no_record"):
        vel.VelocitySettings(no_record="maybe")


def test_set_name_says_dates_ref_and_frame():
    name = vel.set_name(vel.VelocitySettings(ref="background"),
                        dt.date(2026, 8, 15), dt.date(2026, 10, 15))  # fmt: skip
    assert name == "window-20260815-20261015_background_itrf14"


# ---------------------------------------------------------------------------
# record and files
# ---------------------------------------------------------------------------


def test_record_round_trips_through_json_with_nan_as_null():
    r = _record(rate=(np.nan, np.nan, np.nan), refused="cover=0.5<0.8",
                corr_en=np.nan)  # fmt: skip
    d = json.loads(json.dumps(r.to_dict()))
    assert d["rate"] == [None, None, None] and d["corr_en"] is None
    back = vel.VelocityRecord.from_dict(d)
    assert back.refused == r.refused and np.isnan(back.rate).all()
    assert vel.VelocityRecord.from_dict(_record().to_dict()) == _record()
    with pytest.raises(ValueError, match="schema"):
        vel.VelocityRecord.from_dict({**d, "schema_version": 99})
    # a schema-1 record reads with the standard-tier defaults
    v1 = {k: v for k, v in _record().to_dict().items()
          if k not in ("tier", "snr", "chi2nu", "f_quadratic", "parts_refused",
                       "data_end")}  # fmt: skip
    old = vel.VelocityRecord.from_dict({**v1, "schema_version": 1})
    assert old.tier == "standard" and np.isnan(old.snr).all()
    assert old.schema_version == vel.SCHEMA_VERSION  # rewritten as schema 2


def test_set_files_are_sorted_byte_stable_and_gmt_has_accepted_rows(tmp_path):
    recs = [
        _record(station="ZZZZ"),
        _record(station="AAAA", rate=(np.nan,) * 3, refused="no_record"),
        _record(station="MMMM"),
    ]
    paths = vel.write_velocity_set(recs, tmp_path, "s")
    first = {k: p.read_bytes() for k, p in paths.items()}
    vel.write_velocity_set(list(reversed(recs)), tmp_path, "s")
    assert {k: p.read_bytes() for k, p in paths.items()} == first
    gmt = [ln for ln in paths["gmt"].read_text().splitlines() if not ln.startswith("#")]
    assert [ln.split()[-1] for ln in gmt] == ["MMMM", "ZZZZ"]
    # GMT velo order: lon lat ve vn sve svn corr station
    assert gmt[0].split()[:7] == ["-22.500000", "63.900000", "-2.000", "1.000",
                                  "2.000", "2.000", "0.000"]  # fmt: skip
    csv_rows = paths["csv"].read_text().splitlines()[2:]
    assert [r.split(",")[0] for r in csv_rows] == ["AAAA", "MMMM", "ZZZZ"]
    head = paths["csv"].read_text().splitlines()[1].split(",")
    assert csv_rows[0].split(",")[head.index("refused")] == "no_record"
    assert head[-4:] == ["tier", "snr_h", "snr_u", "parts_refused"]
    back = vel.read_velocity_records(paths["csv"])
    assert [r.station for r in back] == ["AAAA", "MMMM", "ZZZZ"]


def test_a_set_refuses_two_frames(tmp_path):
    with pytest.raises(ValueError, match="one frame"):
        vel.write_velocity_set(
            [_record(), _record(station="B", frame="itrf2020")], tmp_path, "s"
        )
    with pytest.raises(ValueError, match="no velocity records"):
        vel.write_velocity_set([], tmp_path, "s")


# ---------------------------------------------------------------------------
# real data (local only: skipped without the .pos files)
# ---------------------------------------------------------------------------


@pytest.fixture
def _deployed_config_reachable(monkeypatch):
    """The golden-master conftest pins GPS_CONFIG_PATH session-wide; these
    tests need the deployed config (memory: goldenmaster conftest)."""
    monkeypatch.delenv("GPS_CONFIG_PATH", raising=False)


def _have_pos(sta: str) -> bool:
    return (Path.home() / "gps-data" / "pos" / f"{sta}.imo.24hr_itrf14.pos").is_file()


@pytest.mark.skipif(not _have_pos("NYLA"), reason="no local NYLA itrf2014 .pos")
@pytest.mark.usefixtures("_deployed_config_reachable")
def test_nyla_two_month_window_reads_its_own_k_and_refuses_a_step():
    st = vel.VelocitySettings(ref="plate-periodic", k_table="none")
    r = vel.estimate_station_velocity("NYLA", dt.date(2026, 8, 15),
                                      dt.date(2026, 10, 15), st)  # fmt: skip
    assert r.accepted and r.frame == "itrf2014" and r.plate == "NOAM"
    assert r.sigma_method == "empirical_k" and all(k >= 1 for k in r.k)
    assert r.k_source["quiet_span"] == pytest.approx([2007.1594, 2019.3328])
    np.testing.assert_allclose(r.sigma, np.multiply(r.sigma_formal, r.k))
    # NYLA has a declared step at 2025.2479
    r = vel.estimate_station_velocity("NYLA", dt.date(2025, 1, 1),
                                      dt.date(2025, 6, 1), st)  # fmt: skip
    assert r.refused == "step_in_window@2025.2479"


@pytest.mark.skipif(not _have_pos("THOB"), reason="no local THOB itrf2014 .pos")
@pytest.mark.usefixtures("_deployed_config_reachable")
def test_a_station_without_record_falls_back_to_plate_or_is_refused(tmp_path):
    # THOB without its (regional, 2026-10-18) record: the deployed doc less it
    deployed = dict(vel._params_doc(vel.VelocitySettings()) or {"stations": {}})
    deployed["stations"].pop("THOB", None)
    params = tmp_path / "params.json"
    params.write_text(json.dumps(deployed))
    win = (dt.date(2026, 8, 15), dt.date(2026, 10, 15))
    r = vel.estimate_station_velocity(
        "THOB", *win, vel.VelocitySettings(k_table="none", params=str(params))
    )
    assert r.ref == "plate" and r.ref_requested == "plate-periodic"
    assert "fallback:plate" in r.flags and "background:none" in r.flags
    assert r.sigma_method == "default_k" and r.k == (5.0, 5.0, 5.0)
    # with a table: its k on the residual that keeps the seasonal (ref plate)
    path = tmp_path / "k.json"
    path.write_text(json.dumps(_table()))
    r2 = vel.estimate_station_velocity(
        "THOB", *win, vel.VelocitySettings(k_table=str(path), params=str(params))
    )
    assert r2.sigma_method == "empirical_k_table" and r2.k == (2.0, 2.0, 2.0)
    assert r2.k_source["residual"] == "secular"
    assert r2.input_fingerprint != r.input_fingerprint  # k is in the key
    assert "k_table_settings_differ" not in r2.flags
    # a table built on the raw view: used, but the record says so
    path.write_text(json.dumps({**_table(), "view": "raw", "uncert": 10.0}))
    r3 = vel.estimate_station_velocity(
        "THOB", *win, vel.VelocitySettings(k_table=str(path), params=str(params))
    )
    assert "k_table_settings_differ" in r3.flags
    assert r3.k_source["table_settings"] == {"view": "raw"}
    np.testing.assert_allclose(r2.rate, r.rate)
    r = vel.estimate_station_velocity(
        "THOB", *win, vel.VelocitySettings(no_record="refuse", k_table="none", params=str(params))
    )
    assert r.refused == "no_record"


@pytest.mark.skipif(not _have_pos("NYLA"), reason="no local NYLA itrf2014 .pos")
@pytest.mark.usefixtures("_deployed_config_reachable")
def test_a_week_long_window_is_not_refused_by_the_count_floor():
    st = vel.VelocitySettings(k_table="none")
    r = vel.estimate_station_velocity("NYLA", dt.date(2026, 10, 1),
                                      dt.date(2026, 10, 7), st)  # fmt: skip
    assert r.accepted, r.refused
    assert r.sigma_method == "empirical_k"  # its own k at 7 days


def test_records_do_not_depend_on_the_other_stations(monkeypatch):
    """A station's record is the same whichever stations share the run."""
    calls = []

    def fake(sta, start, end, settings, *, doc=None, k_table=None, **_):
        calls.append((sta, k_table))
        return _record(station=sta)

    monkeypatch.setattr(vel, "estimate_station_velocity", fake)
    monkeypatch.setattr(vel, "_params_doc", lambda s: None)
    monkeypatch.setattr(vel, "load_k_table", lambda s: ("T", "src"))
    a = vel.estimate_velocities(["B", "A"], 2026.0, 2026.1)
    b = vel.estimate_velocities(["A"], 2026.0, 2026.1)
    assert [r.station for r in a] == ["A", "B"] and a[0] == b[0]
    assert {c[1] for c in calls} == {("T", "src")}


def test_cli_defaults_are_the_library_defaults():
    """gps-velocities and the map (VelocitySettings()) must give one number."""
    args = vel.build_parser().parse_args(["estimate", "NYLA", "--last", "7"])
    st = vel.VelocitySettings()
    assert (args.min_obs, args.min_cover) == (st.min_obs, st.min_cover)
    assert (args.ref, args.view, args.frame) == (st.ref, st.view, st.frame)
    assert (args.uncert, args.default_k, args.no_record) == (
        st.uncert,
        st.default_k,
        st.no_record,
    )
    assert (args.short_tier, args.short_snr, args.short_snr_pair) == (
        st.short_tier,
        st.short_snr,
        st.short_snr_pair,
    )


def test_last_days_are_whole_days_ending_with_the_end_day():
    assert vel.last_days_start(dt.date(2026, 10, 16), 7) == dt.date(2026, 10, 10)
    t0, t1 = vel.window_bounds(vel.last_days_start(dt.date(2026, 10, 16), 7),
                               dt.date(2026, 10, 16))  # fmt: skip
    assert (t1 - t0) * 365.25 == pytest.approx(7, abs=0.02)
    with pytest.raises(ValueError, match="at least one day"):
        vel.last_days_start(dt.date(2026, 10, 16), 0)


def test_a_window_is_half_open_at_the_midnight_after_its_end():
    t, y, s = _series(years=0.1)
    end = float(t[-1])
    whole = vel.velocity_from_arrays(t, y, s, (float(t[0]), end + 1e-9))
    edge = vel.velocity_from_arrays(t, y, s, (float(t[0]), end))
    assert whole.n_epochs[0] == edge.n_epochs[0] + 1


@pytest.mark.usefixtures("_deployed_config_reachable")
def test_a_station_without_a_pos_file_is_refused_not_raised(tmp_path):
    st = vel.VelocitySettings(pos_dir=str(tmp_path), k_table="none")
    r = vel.estimate_station_velocity("NYLA", dt.date(2026, 10, 1),
                                      dt.date(2026, 10, 7), st)  # fmt: skip
    assert not r.accepted and r.refused.startswith("read:")
    r = vel.estimate_station_velocity("XXXX", dt.date(2026, 10, 1),
                                      dt.date(2026, 10, 7), st)  # fmt: skip
    assert not r.accepted and "no_coordinates" in r.flags


def test_a_given_k_is_not_called_empirical(monkeypatch):
    monkeypatch.setattr(
        vel, "_read", lambda *a, **k: (_ for _ in ()).throw(ValueError("x"))
    )
    r = vel.estimate_station_velocity(
        "NYLA", 2026.0, 2026.1, vel.VelocitySettings(no_record="plate"),
        doc={"stations": {}}, k=(2.0, 2.0, 2.0), coords=(0.0, 0.0),
    )  # fmt: skip
    assert r.sigma_method == "given_k"


def _fake_read(t, y, s):
    return lambda *a, **k: (np.asarray(t, float), np.asarray(y, float),
                            np.asarray(s, float), {})  # fmt: skip


def _short(monkeypatch, t, y, s, **settings):
    monkeypatch.setattr(vel, "_read", _fake_read(t, y, s))
    day = 1 / 365.25
    return vel.estimate_station_velocity(
        "NYLA", float(t[0]) - day / 2, float(t[-1]) + day / 2,
        vel.VelocitySettings(**settings), doc={"stations": {}},
        k=(1.5, 1.5, 1.5), coords=(0.0, 0.0),
    )  # fmt: skip


def test_short_tier_keeps_a_fast_pair_and_drops_the_quiet_up(monkeypatch):
    day = 1 / 365.25
    t = 2015.5 + np.arange(3) * day
    y = np.vstack([20.0 * np.arange(3), -25.0 * np.arange(3), np.zeros(3)])
    r = _short(monkeypatch, t, y, np.full((3, 3), 3.0))
    assert r.accepted and r.tier == "short" and "short:n=3" in r.flags
    assert r.rate[0] == pytest.approx(20.0 * 365.25)
    assert np.isnan(r.rate[2]) and r.parts_refused["up"].startswith("snr=")
    assert r.snr[0] >= vel.SHORT_SNR
    # the fake read has no excluded-epoch catalog: declared fell back to raw
    assert "no_excluded_catalog" in r.flags


def test_nonlinear_flag_needs_enough_epochs(monkeypatch):
    day = 1 / 365.25
    for n, flagged in ((5, False), (8, True)):
        t = 2015.5 + np.arange(n) * day
        x = np.arange(n, dtype=float)
        bend = 3.0 * x * x  # a strong curve, F far above the threshold
        y = np.vstack([bend, bend, np.zeros(n)])
        y[:, 1::2] += 0.5  # not exact: a finite F
        r = _short(monkeypatch, t, y, np.full((3, n), 1.0))
        assert r.accepted
        assert any(f.startswith("nonlinear:") for f in r.flags) is flagged, n


def test_short_tier_refuses_quiet_windows_and_a_blunder(monkeypatch):
    day = 1 / 365.25
    t = 2015.5 + np.arange(3) * day
    quiet = _short(monkeypatch, t, np.zeros((3, 3)), np.full((3, 3), 3.0))
    assert not quiet.accepted and "horizontal snr=" in quiet.refused
    # one bad epoch on 3: the χ² inflates σ, the window stays refused
    bad = np.zeros((3, 3))
    bad[1, 2] = 40.0
    r = _short(monkeypatch, t, bad, np.full((3, 3), 3.0))
    assert not r.accepted
    # two epochs: the stricter pair threshold
    t2, y2 = t[:2], np.vstack([[0.0, 40.0], [0.0, 0.0], [0.0, 0.0]])
    r = _short(monkeypatch, t2, y2, np.full((3, 2), 3.0))
    assert not r.accepted  # snr 40 mm/day at σ 3: below the pair's 8
    loose = _short(monkeypatch, t2, y2, np.full((3, 2), 3.0), short_snr_pair=3.0)
    assert loose.accepted and "short:n=2" in loose.flags
    off = _short(monkeypatch, t2, y2, np.full((3, 2), 3.0), short_tier=False)
    assert not off.accepted and "snr" not in off.refused


def _pre_onset_series():
    day = 1 / 365.25
    t = 2015.0 + np.arange(0, 0.6, day)
    north = 20.0 * (t - 2015.0) + np.where(t > 2015.4, 80.0 * (t - 2015.4), 0.0)
    y = np.vstack([north, np.zeros_like(t), np.zeros_like(t)])
    y += np.random.default_rng(3).normal(0.0, 1.0, y.shape)
    return t, y, np.full(y.shape, 1.0)


def _run(monkeypatch, t, y, s, settings, window=(2015.45, 2015.55)):
    monkeypatch.setattr(vel, "_read", _fake_read(t, y, s))
    return vel.estimate_station_velocity(
        "NYLA", *window, settings, doc={"stations": {}}, k=(1.5, 1.5, 1.5),
        coords=(0.0, 0.0),
    )  # fmt: skip


def test_pre_onset_subtracts_the_station_s_own_pre_window_rate(monkeypatch):
    t, y, s = _pre_onset_series()
    pre = (2015.1, 2015.35)
    r = _run(
        monkeypatch, t, y, s, vel.VelocitySettings(ref="pre-onset", pre_window=pre)
    )
    win = _run(monkeypatch, t, y, s, vel.VelocitySettings())
    before = _run(monkeypatch, t, y, s, vel.VelocitySettings(), window=pre)
    assert r.accepted and r.ref == "pre-onset" and win.ref == "plate"
    assert r.rate[0] == pytest.approx(win.rate[0] - before.rate[0])
    assert r.rate[0] == pytest.approx(80.0, abs=15.0)
    assert r.pre_onset is not None and r.pre_onset["rate"][0] == pytest.approx(
        before.rate[0]
    )
    # the pre-window takes its own k, not the caller's (the window's length)
    assert r.pre_onset["sigma_method"] != "given_k"
    assert r.sigma[0] == pytest.approx(np.hypot(win.sigma[0], r.pre_onset["sigma"][0]))
    # no stored seasonal: the record says the pre-window rate aliases it
    assert {"pre_onset", "seasonal:none", "fallback:plate"} <= set(r.flags)
    assert "pre_overlap" not in r.flags
    back = vel.VelocityRecord.from_dict(json.loads(json.dumps(r.to_dict())))
    assert back.pre_onset == r.pre_onset and back.schema_version == 3


def test_pre_onset_refuses_without_data_before_the_onset(monkeypatch):
    t, y, s = _pre_onset_series()
    st = vel.VelocitySettings(ref="pre-onset", pre_window=(2014.5, 2014.7))
    r = _run(monkeypatch, t, y, s, st)
    assert not r.accepted and r.refused.startswith("pre_onset: ")
    assert np.isnan(r.rate).all()


def test_pre_onset_settings_and_set_name():
    with pytest.raises(ValueError, match="pre_window"):
        vel.VelocitySettings(ref="pre-onset")
    with pytest.raises(ValueError, match="pre_window"):
        vel.VelocitySettings(pre_window=(2023.6, 2023.8))
    st = vel.VelocitySettings(
        ref="pre-onset", frame="itrf2020",
        pre_window=vel.window_bounds(dt.date(2023, 8, 6), dt.date(2023, 10, 24)),
    )  # fmt: skip
    name = vel.set_name(st, dt.date(2023, 11, 1), dt.date(2023, 11, 7))
    assert name == "window-20231101-20231107_pre-onset-20230806-20231024_itrf20"


def test_cli_takes_a_pre_window():
    args = vel.build_parser().parse_args(
        ["estimate", "SENG", "--last", "7", "--ref", "pre-onset",
         "--pre-window", "20230806:20231024"]
    )  # fmt: skip
    assert args.pre_window == (dt.date(2023, 8, 6), dt.date(2023, 10, 24))


def _short_after_pre(noise):
    """A quiet daily pre-window 2015.0–2015.4, then three fast epochs at 2015.5."""
    day = 1 / 365.25
    t = np.concatenate([2015.0 + np.arange(0, 0.4, day), 2015.5 + np.arange(3) * day])
    y = np.random.default_rng(5).normal(0.0, noise, (3, t.size))
    y[:, -3:] = np.vstack([20.0 * np.arange(3), -25.0 * np.arange(3), np.zeros(3)])
    s = np.full(y.shape, float(noise))
    s[:, -3:] = 3.0
    return t, y, s, (2015.5 - day / 2, 2015.5 + 6.5 * day)


def test_pre_onset_refusal_clears_the_window_s_short_tier(monkeypatch):
    monkeypatch.setattr(vel, "_data_end", lambda *a: None)  # no .pos read
    t, y, s, window = _short_after_pre(1.0)
    st = vel.VelocitySettings(ref="pre-onset", pre_window=(2014.5, 2014.7))
    r = _run(monkeypatch, t, y, s, st, window=window)
    assert not r.accepted and r.refused.startswith("pre_onset: ")
    assert r.tier == "standard" and r.parts_refused == {}
    assert np.isnan(r.chi2nu).all() and np.isnan(r.f_quadratic).all()
    assert not any(f.startswith(("short:", "nonlinear")) for f in r.flags)


def test_pre_onset_flags_a_short_part_the_pre_window_sinks(monkeypatch):
    monkeypatch.setattr(vel, "_data_end", lambda *a: None)  # no .pos read
    t, y, s, window = _short_after_pre(1.0)
    pre = (2015.0, 2015.4)
    quiet = _run(monkeypatch, t, y, s, vel.VelocitySettings(ref="pre-onset", pre_window=pre), window=window)  # fmt: skip
    assert quiet.accepted and quiet.tier == "short"
    assert not any(f.startswith("short:pre_snr") for f in quiet.flags)
    t, y, s, window = _short_after_pre(3000.0)
    noisy = _run(monkeypatch, t, y, s, vel.VelocitySettings(ref="pre-onset", pre_window=pre), window=window)  # fmt: skip
    assert noisy.accepted and noisy.tier == "short" and noisy.snr[0] < vel.SHORT_SNR
    assert "short:pre_snr<5:horizontal" in noisy.flags


def test_pre_onset_overlap_and_a_pre_window_after_the_window(monkeypatch):
    t, y, s = _pre_onset_series()
    st = vel.VelocitySettings(ref="pre-onset", pre_window=(2015.2, 2015.47))
    assert "pre_overlap" in _run(monkeypatch, t, y, s, st).flags
    st = vel.VelocitySettings(ref="pre-onset", pre_window=(2015.46, 2015.58))
    with pytest.raises(ValueError, match="not before the window"):
        _run(monkeypatch, t, y, s, st)


def _seasonal_only(monkeypatch, seen):
    rec = {"model": "periodic", "seasonal_only": True, "detrend_method": "regional",
           "window": [2010.0, 2015.0],
           "param_names": ["cos_annual", "sin_annual", "cos_semiannual",
                           "sin_semiannual"]}  # fmt: skip
    monkeypatch.setattr(vel, "_record", lambda doc, sta: rec)
    t = 2015.0 + np.arange(0, 0.25, 1 / 365.25)
    y = np.vstack([10.0 * (t - 2015.0), np.zeros_like(t), np.zeros_like(t)])

    def read(sta, t0, t1, settings, *, terms, doc):
        seen.append(terms)
        return t, y, np.ones_like(y), {}

    monkeypatch.setattr(vel, "_read", read)
    return rec


def test_a_regional_seasonal_is_removed_and_flagged_but_is_no_background(
    monkeypatch,
):
    seen: list = []
    _seasonal_only(monkeypatch, seen)
    table = {"residuals": {"secular": {"days": [30, 61, 91], "k": [[3.0] * 3] * 3}}}
    run = dict(doc={"stations": {}}, coords=(0.0, 0.0), k_table=(table, "t"))
    r = vel.estimate_station_velocity(
        "ELDC", 2015.0, 2015.25, vel.VelocitySettings(), **run
    )
    assert r.accepted and "seasonal:regional" in r.flags and seen == ["periodic"]
    # no regional k measured yet: the seasonal-kept k bounds it
    assert r.sigma_method == "empirical_k_table" and r.k_source["residual"] == "secular"
    b = vel.estimate_station_velocity(
        "ELDC", 2015.0, 2015.25, vel.VelocitySettings(ref="background"), **run
    )
    assert not b.accepted and b.refused.startswith("seasonal_only:")
    got, info = vel.station_rate_scales("ELDC", [0.1], vel.VelocitySettings(), doc={})
    assert got == {} and info["reason"] == "seasonal-only record"


@pytest.mark.skipif(not _have_pos("THOB"), reason="no local THOB itrf2014 .pos")
@pytest.mark.usefixtures("_deployed_config_reachable")
def test_a_regional_station_takes_the_regional_k_or_flags_its_fallback(tmp_path):
    doc = dict(vel._params_doc(vel.VelocitySettings()) or {"stations": {}})
    if not (doc["stations"].get("THOB") or {}).get("seasonal_only"):
        pytest.skip("THOB has no regional record in the deployed doc")
    win = (dt.date(2026, 8, 15), dt.date(2026, 10, 15))
    with_regional = _table()
    with_regional["residuals"][vel.K_RESIDUAL_REGIONAL] = {
        "days": [7], "k": [[3.0, 3.0, 3.0]], "n_stations": [20]}  # fmt: skip
    got = {}
    for name, table in (("regional", with_regional), ("fallback", _table())):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(table))
        got[name] = vel.estimate_station_velocity(
            "THOB", *win, vel.VelocitySettings(k_table=str(path))
        )
    r, f = got["regional"], got["fallback"]
    assert r.ref == "plate-periodic" and "seasonal:regional" in r.flags
    assert r.k == (3.0, 3.0, 3.0) and r.k_source["residual"] == "regional"
    assert "k:regional_fallback" not in r.flags
    assert f.k == (2.0, 2.0, 2.0) and f.k_source["residual"] == "secular"
    assert "k:regional_fallback" in f.flags
