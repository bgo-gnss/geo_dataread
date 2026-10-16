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
    t, y, s = _series(years=0.05)  # 19 epochs
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


def test_network_k_is_the_median_of_own_scales_only():
    recs = [
        _record(k=(2.0, 3.0, 4.0)),
        _record(k=(4.0, 5.0, 6.0)),
        _record(k=(9.0, 9.0, 9.0), sigma_method="default_k"),
    ]
    np.testing.assert_allclose(vel.network_k(recs), [3.0, 4.0, 5.0])
    assert vel.network_k([_record(sigma_method="default_k")]) is None


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
    st = vel.VelocitySettings(ref="plate-periodic")
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
def test_a_station_without_record_falls_back_to_plate_or_is_refused():
    win = (dt.date(2026, 8, 15), dt.date(2026, 10, 15))
    r = vel.estimate_station_velocity("THOB", *win, vel.VelocitySettings())
    assert r.ref == "plate" and r.ref_requested == "plate-periodic"
    assert "fallback:plate" in r.flags and "background:none" in r.flags
    assert r.sigma_method == "default_k"
    r = vel.estimate_station_velocity(
        "THOB", *win, vel.VelocitySettings(no_record="refuse")
    )
    assert r.refused == "no_record"
