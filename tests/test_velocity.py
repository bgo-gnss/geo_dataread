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
        input_fingerprint="abc",
    )  # fmt: skip
    return dataclasses.replace(base, **changes)


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------


def test_line_rates_are_recovered_per_component():
    t, y, s = _series(noise=0.1)
    wr = vel.velocity_from_arrays(t, y, s, (t[0], t[-1]))
    assert wr.reasons == ()
    np.testing.assert_allclose(wr.rate, [1.0, -2.0, 3.0], atol=0.3)
    assert np.all(wr.sigma_formal > 0) and wr.n_epochs == (t.size,) * 3


def test_nan_epochs_drop_per_component_and_from_the_gates():
    t, y, s = _series()
    y[2, ::2] = np.nan  # outliers in up only
    wr = vel.velocity_from_arrays(t, y, s, (t[0], t[-1]))
    assert wr.n_epochs[0] == t.size and wr.n_epochs[2] < t.size
    assert np.isfinite(wr.rate).all()


def test_gates_refuse_all_three_rates():
    t, y, s = _series(years=0.01)  # 4 epochs
    wr = vel.velocity_from_arrays(t, y, s, (t[0], t[0] + 0.25))
    assert any(r.startswith("n_obs") for r in wr.reasons)
    assert any(r.startswith("cover") for r in wr.reasons)
    assert np.isnan(wr.rate).all()


def test_a_step_inside_refuses_unless_allowed():
    t, y, s = _series()
    step = float(t[40])
    wr = vel.velocity_from_arrays(t, y, s, (t[0], t[-1]), step_epochs=[step])
    assert wr.reasons == (f"step_in_window@{step:.4f}",)
    wr = vel.velocity_from_arrays(t, y, s, (t[0], t[-1]), step_epochs=[step],
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
    assert csv_rows[0].endswith(",no_record")
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
    win = (dt.date(2026, 8, 15), dt.date(2026, 10, 15))
    r = vel.estimate_station_velocity(
        "THOB", *win, vel.VelocitySettings(k_table="none")
    )
    assert r.ref == "plate" and r.ref_requested == "plate-periodic"
    assert "fallback:plate" in r.flags and "background:none" in r.flags
    assert r.sigma_method == "default_k" and r.k == (5.0, 5.0, 5.0)
    # with a table: its k on the residual that keeps the seasonal (ref plate)
    path = tmp_path / "k.json"
    path.write_text(json.dumps(_table()))
    r2 = vel.estimate_station_velocity(
        "THOB", *win, vel.VelocitySettings(k_table=str(path))
    )
    assert r2.sigma_method == "empirical_k_table" and r2.k == (2.0, 2.0, 2.0)
    assert r2.k_source["residual"] == "secular"
    assert r2.input_fingerprint != r.input_fingerprint  # k is in the key
    assert "k_table_settings_differ" not in r2.flags
    # a table built on the raw view: used, but the record says so
    path.write_text(json.dumps({**_table(), "view": "raw", "uncert": 10.0}))
    r3 = vel.estimate_station_velocity(
        "THOB", *win, vel.VelocitySettings(k_table=str(path))
    )
    assert "k_table_settings_differ" in r3.flags
    assert r3.k_source["table_settings"] == {"view": "raw"}
    np.testing.assert_allclose(r2.rate, r.rate)
    r = vel.estimate_station_velocity(
        "THOB", *win, vel.VelocitySettings(no_record="refuse", k_table="none")
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
