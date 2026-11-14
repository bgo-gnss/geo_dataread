"""PGV magnitudes from alertMap files (ticket T7b of PLAN-seismicity.md).

Fixtures: ``tests/fixtures/alert`` (README there). No network, no mount.
The golden case is SIL event ``/2026/nov/12/21:50:00/53:43:797`` with
alertMap detection 2026-11-12 #076 and the Mpgv page row for it.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geo_dataread.seismic import sil_alert as sa
from geo_dataread.seismic.schema import MAGNITUDE_TYPES, TIME_DTYPE
from geo_dataread.seismic.sil_files import read_cleanaut_mag

ALERT = Path(__file__).parent / "fixtures" / "alert"
SIL = ALERT / "sil"
GOLDEN_EVENT = ALERT / "2026/nov/12/076.event"
GOLDEN_EVB = SIL / "2026/nov/12/21:50:00/53:43:797.evb"
GOLDEN_ID = "/2026/nov/12/21:50:00/53:43:797"


def page_rows() -> pd.DataFrame:
    """The frozen Mpgv page table (2026-11), as text cells."""
    text = (ALERT / "mpgv/2026-11.html").read_text(encoding="utf-8")
    rows = []
    for row in re.findall(r"<tr>(.*?)</tr>", text, re.S):
        cells = [
            re.sub(r"<.*?>", "", c).strip()
            for c in re.findall(r"<t[dh]>(.*?)</t[dh]>", row, re.S)
        ]
        rows.append(cells[1:12])
    cols = [
        "time",
        "lat",
        "lon",
        "depth",
        "alert_sta",
        "filter_sta",
        "mean",
        "median",
        "mauto",
        "alertid",
        "alertfit",
    ]
    return pd.DataFrame(rows, columns=cols)


def page_row(time: str) -> pd.Series:
    rows = page_rows()
    hit = rows[rows["time"] == time]
    assert len(hit) == 1
    return hit.iloc[0]


def page_origin(row: pd.Series) -> sa.Origin:
    return sa.Origin(pd.Timestamp(row["time"]), float(row["lat"]), float(row["lon"]))


@pytest.fixture(scope="module")
def golden_origin() -> sa.Origin:
    """The origin from the frozen cleanaut.mag line (what the page used)."""
    row = read_cleanaut_mag(SIL / "2026/nov/12/cleanaut.mag").frame.iloc[0]
    assert row["id"] == GOLDEN_ID
    return sa.Origin(
        pd.Timestamp(row["time"]),
        float(row["latitude"]),
        float(row["longitude"]),
        float(row["depth"]),
    )


@pytest.fixture(scope="module")
def golden() -> sa.AlertEvent:
    return sa.read_alert_event(GOLDEN_EVENT)


def r2(x: float) -> float:
    return round(x + 1e-12, 2)


# --------------------------------------------------------------------------
# Model C
# --------------------------------------------------------------------------


def test_hand_values() -> None:
    # Discovery doc §5.2.
    assert float(sa.pgv_magnitude_farfield(20.0, 100.0)) == pytest.approx(
        3.467648, abs=1e-6
    )
    assert float(sa.pgverror(10.0, 5.0)) == pytest.approx(0.140124, abs=1e-6)
    assert float(sa.nearfield_bias(10.0, 5.0)) == pytest.approx(0.2255, abs=1e-4)
    # At r = 0 PGV is 0.2024 m/s at M -> 0 and almost independent of M (g is
    # -b/a rounded: 1.69 * 0.621 = 1.0495, not 1.05).
    assert float(sa.pgv_model_c(0.0, 0.0)) == pytest.approx(0.2024, abs=5e-5)
    at0 = sa.pgv_model_c(np.linspace(0.0, 7.0, 15), 0.0)
    assert np.all(np.abs(at0 / 0.2024 - 1.0) < 0.01)


def test_farfield_matches_legacy_formula() -> None:
    r, pgv = np.array([20.0, 100.0, 230.0]), np.array([500.0, 20.0, 1.3])
    legacy = 1 / 1.05 * (1.69 * np.log10(r) + np.log10(1e-6 * pgv) + 4.96)
    np.testing.assert_allclose(
        sa.pgv_magnitude_farfield(pgv, r), legacy, rtol=0, atol=1e-12
    )


def test_exact_round_trip() -> None:
    m = np.linspace(2.0, 7.0, 26)[:, None]
    r = np.geomspace(1.0, 300.0, 40)[None, :]
    pgv_um_s = sa.pgv_model_c(m, r) * 1e6
    back = sa.pgv_magnitude(pgv_um_s, r)
    assert np.max(np.abs(back - m)) < 1e-9


def test_exact_minus_farfield_is_the_bias() -> None:
    m, r = 5.0, np.array([5.0, 10.0, 30.0, 100.0])
    pgv = sa.pgv_model_c(m, r) * 1e6
    diff = m - sa.pgv_magnitude_farfield(pgv, r)
    np.testing.assert_allclose(diff, sa.nearfield_bias(r, m), atol=1e-9)
    assert np.all(np.diff(diff) < 0)  # the bias grows as r falls


def test_monotone_in_m_and_scalar_equals_vector() -> None:
    ms = np.linspace(1.0, 7.0, 50)
    assert np.all(np.diff(sa.pgv_model_c(ms, 40.0)) > 0)
    vec = sa.pgv_magnitude([5.0, 50.0, 500.0], [30.0, 60.0, 90.0])
    for i, (p, r) in enumerate([(5.0, 30.0), (50.0, 60.0), (500.0, 90.0)]):
        assert float(sa.pgv_magnitude(p, r)) == vec[i]


def test_invalid_inputs_give_nan() -> None:
    for f in (sa.pgv_magnitude, sa.pgv_magnitude_farfield):
        out = f([0.0, -1.0, 10.0, np.nan, 10.0], [10.0, 10.0, 0.0, 10.0, -5.0])
        assert np.isnan(out).all()
    # A PGV above what the relation reaches at that distance has no root.
    assert np.isnan(sa.pgv_magnitude(1e7, 1.0))
    assert np.isnan(sa.nearfield_bias(0.0, 4.0))
    assert np.isnan(sa.pgv_model_c(4.0, -1.0))


def test_nearfield_table() -> None:
    assert sa.nearfield_min_km(3.99) == 10.0
    assert sa.nearfield_min_km(4.0) == 15.0
    assert sa.nearfield_min_km(5.5) == 30.0
    assert sa.nearfield_min_km(6.0) == 50.0
    assert sa.nearfield_min_km(9.0) == 50.0
    assert sa.nearfield_min_km(float("nan")) == 10.0
    with pytest.raises(ValueError):
        sa.nearfield_min_km(4.0, [])


def test_mid_third_mean() -> None:
    assert sa.mid_third_mean([1, 2, 3, 4, 5, 6, 7, 8]) == pytest.approx(
        4.5
    )  # drops 2+2
    assert sa.mid_third_mean([1.0, 2.0, 9.0]) == 2.0
    assert sa.mid_third_mean([5.0, np.nan, 1.0, 3.0]) == 3.0
    assert np.isnan(sa.mid_third_mean([1.0, 2.0]))  # the page leaves n = 2 blank
    assert sa.mid_third_mean([1.0, 2.0], min_values=1) == 1.5
    assert np.isnan(sa.mid_third_mean([]))


# --------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------


def test_golden_event_file(golden: sa.AlertEvent) -> None:
    h, f = golden
    assert h.alert_id == 76 and h.n_stations == 66 and h.index_value == 3307
    assert h.counters == (8023, 11330, 11330)
    assert h.time == pd.Timestamp("2026-11-12 21:53:45", tz="UTC")
    assert len(f) == 66 and f["station"].is_unique
    assert tuple(f.columns) == sa.ALERT_EVENT_COLUMNS
    assert f["arrival_time"].dtype == TIME_DTYPE
    assert f["class_code"].dtype == "Int64"
    first = f.iloc[0]
    assert (
        first["station"] == "ada"
        and first["pgv_um_s"] == 394.46
        and first["class_code"] == 22
    )
    assert first["offset_s"] == 0.13
    # Field 8 = seconds of the minute of the arrival.
    sec = f["arrival_time"].dt.second + f["arrival_time"].dt.microsecond / 1e6
    np.testing.assert_allclose(
        sec.to_numpy() % 60, f["second_of_minute"].to_numpy() % 60, atol=0.006
    )
    assert f.loc[f["station"] == "mko", ["aux11", "aux12"]].isna().all().all()  # --.--


def test_small_event_and_2020_layout() -> None:
    small = sa.read_alert_event(ALERT / "2026/nov/12/001.event")
    assert small.header.n_stations == 2 and list(small.frame["station"]) == [
        "mko",
        "nyl",
    ]
    assert small.header.time == pd.Timestamp("2026-11-12 00:04:42", tz="UTC")
    old = sa.read_alert_event(ALERT / "2020/jun/25/001.event")
    assert old.header.time == pd.Timestamp("2020-06-25 00:44:58", tz="UTC")
    assert len(old.frame) == 10
    assert old.frame["aux10"].isna().sum() == 8
    assert old.frame["aux12"].isna().sum() == 8
    assert (
        not old.frame[["lon", "lat", "offset_s", "pgv_um_s", "fb"]].isna().any().any()
    )


def test_header_only_event() -> None:
    ev = sa.parse_alert_event("001  14  17  28 Thu Nov 12 00:04:42 2026    3  0\n")
    assert ev.frame.empty and tuple(ev.frame.columns) == sa.ALERT_EVENT_COLUMNS
    assert ev.frame["arrival_time"].dtype == TIME_DTYPE


def test_event_errors() -> None:
    head = "001  14  17  28 Thu Nov 12 00:04:42 2026    3  1\n"
    row = "-16.33826  64.97840    3      0.05 mko  17.47 21   -93 42.03  5 41.07  17.47     0.05\n"
    with pytest.raises(sa.AlertFormatError, match="fields"):
        sa.parse_alert_event(head + "-16.3 64.9 3 0.05 mko\n")
    with pytest.raises(sa.AlertFormatError, match="month"):
        sa.parse_alert_event(head.replace("Nov", "Nóv") + row)
    with pytest.raises(sa.AlertFormatError, match="header"):
        sa.parse_alert_event("001 14 17 Thu Nov 12 00:04:42 2026 3 1\n" + row)
    with pytest.raises(sa.AlertFormatError, match="bad field"):
        sa.parse_alert_event(head + row.replace("0.05", "x.05", 1))
    with pytest.raises(sa.AlertFormatError, match="header says"):
        sa.parse_alert_event(head + row + row.replace("mko", "nyl"))
    assert (
        len(
            sa.parse_alert_event(
                head + row + row.replace("mko", "nyl"), strict=False
            ).frame
        )
        == 2
    )
    with pytest.raises(sa.AlertFormatError, match="duplicated"):
        sa.parse_alert_event(head.replace("3  1", "3  2") + row + row)
    with pytest.raises(sa.AlertFormatError, match="empty"):
        sa.parse_alert_event("\n\n")


def test_day_index() -> None:
    idx = sa.read_alert_day_index(ALERT / "2026/nov/12/index.html")
    assert tuple(idx.columns) == sa.DAY_INDEX_COLUMNS
    assert len(idx) == 80 and list(idx["alert_id"]) == list(range(1, 81))
    row = idx.set_index("alert_id").loc[76]
    assert row["time"] == pd.Timestamp("2026-11-12 21:53:45", tz="UTC")
    golden = sa.read_alert_event(GOLDEN_EVENT).header
    assert (row["n_stations"], row["index_value"]) == (
        golden.n_stations,
        golden.index_value,
    )
    assert idx["time"].dtype == TIME_DTYPE
    text = (ALERT / "2026/nov/12/index.html").read_text(encoding="latin-1")
    with pytest.raises(sa.AlertFormatError, match="title says"):
        sa.parse_alert_day_index(text, date(2026, 11, 13))
    untitled = re.sub(r"<title>.*?</title>", "", text)
    with pytest.raises(sa.AlertFormatError, match="no day"):
        sa.parse_alert_day_index(untitled)
    assert len(sa.parse_alert_day_index(untitled, date(2026, 11, 12))) == 80
    assert sa.parse_alert_day_index("<title>Alert Map 2026 Nov 12</title>").empty


def test_day_alert_mag() -> None:
    mag = sa.read_alert_mag(ALERT / "2026/nov/12/alert.mag")
    assert tuple(mag.columns) == sa.ALERT_MAG_COLUMNS and len(mag) == 8
    r = mag.set_index("alert_id").loc[76]
    assert r["time"] == pd.Timestamp("2026-11-12 21:53:49.1", tz="UTC")
    assert (r["latitude"], r["longitude"], r["magnitude"], r["n_stations"]) == (
        64.089,
        -17.052,
        4.29,
        66,
    )
    assert (
        r["class_letter"] == "T" and mag.set_index("alert_id").loc[22, "code"] == "011"
    )
    with pytest.raises(sa.AlertFormatError, match="fields"):
        sa.parse_alert_mag(" 076 20261112 215349.1  64.089\n")
    with pytest.raises(sa.AlertFormatError, match="bad field"):
        sa.parse_alert_mag(
            " 076 2026111 215349.1  64.089  -17.052 4.000  4.29 T 000  0.49  0.56  3.8 66  2.05 988\n"
        )
    assert sa.parse_alert_mag("").empty


def test_golden_evb() -> None:
    evb = sa.read_evb(GOLDEN_EVB)
    o = evb.origin
    assert o.time == pd.Timestamp("2026-11-12 21:53:40.6", tz="UTC")
    assert (o.latitude, o.longitude, o.depth_km) == (64.653, -17.481, 5.3)
    assert (
        evb.time_error_s == 0.08
        and evb.quality == pytest.approx(90.051)
        and evb.model == 5
    )
    p = evb.phases
    assert tuple(p.columns) == sa.EVB_PHASE_COLUMNS and len(p) == 33
    assert p.iloc[0]["station"] == "dyn" and p.iloc[0]["distance_km"] == 16.3
    assert p.iloc[0]["arrival_time"] == pd.Timestamp("2026-11-12 21:53:43.80", tz="UTC")
    assert p["arrival_time"].dtype == TIME_DTYPE
    d = evb.station_distances()
    assert d.index.is_unique and d["mor"] == 209.4


def test_evb_minute_hour_rollover_and_phases() -> None:
    evb = sa.read_evb(SIL / "2024/aug/22/21:55:00/59:59:641.evb")
    p = evb.phases
    assert p.iloc[0]["arrival_time"] == pd.Timestamp("2024-08-22 21:59:59.64", tz="UTC")
    assert p.iloc[1]["arrival_time"] == pd.Timestamp("2024-08-22 22:00:00.02", tz="UTC")
    lag = p[p["station"] == "lag"]
    assert list(lag["phase"]) == ["p", "s"]  # rows, not a duplicated index
    assert len(evb.station_distances()) == p["station"].nunique() < len(p)
    lfe = p[(p["station"] == "lfe") & (p["phase"] == "p")].iloc[0]
    assert (lfe["azimuth_obs_deg"], lfe["azimuth_res_deg"]) == (323.9, -115.1)


def test_evb_2020_and_year() -> None:
    path = SIL / "2020/jun/25/00:05:00/09:33:883.evb"
    evb = sa.read_evb(path)
    assert evb.origin.time == pd.Timestamp("2020-06-25 00:09:28.4", tz="UTC")
    text = path.read_text()
    assert sa.parse_evb(text).origin.time.year == 2020  # pivot
    with pytest.raises(sa.AlertFormatError, match="does not match"):
        sa.parse_evb(text, year=2019)


def test_evb_midnight_synthetic() -> None:
    text = (
        " origin time 26 11 12   23h 59m 58.0s +/- 0.05s\n"
        " latitude  64.000  +/- 0.002 deg.\n"
        " longitude-21.000  +/- 0.006\n"
        " focal depth  5.0  +/-   1.5 km p-def: 51\n"
        "0sta   arr. time   res.  weight dist  azi  az.obs  err\n"
        " aaa p 23 59 59.90  0.11  11.1   9.3  19.6    0.0   0.0   1.0\n"
        " bbb p 00 00  3.10  0.18  11.1  30.4 144.6    0.0   0.0   1.0\n"
        " model:  5   80.000\n"
    )
    p = sa.parse_evb(text).phases
    assert list(p["arrival_time"]) == [
        pd.Timestamp("2026-11-12 23:59:59.90", tz="UTC"),
        pd.Timestamp("2026-11-13 00:00:03.10", tz="UTC"),
    ]


def test_evb_errors() -> None:
    good = GOLDEN_EVB.read_text().splitlines()
    with pytest.raises(sa.AlertFormatError, match="too short"):
        sa.parse_evb("\n".join(good[:3]))
    with pytest.raises(sa.AlertFormatError, match="origin"):
        sa.parse_evb("\n".join(["x", *good[1:]]))
    with pytest.raises(sa.AlertFormatError, match="latitude"):
        sa.parse_evb("\n".join([good[0], "x", *good[2:]]))
    with pytest.raises(sa.AlertFormatError, match="phase header"):
        sa.parse_evb("\n".join([*good[:4], "x", *good[5:]]))
    bad = good[5][:30] + "  x.x" + good[5][35:]
    with pytest.raises(sa.AlertFormatError, match="bad phase row"):
        sa.parse_evb("\n".join([*good[:5], bad, *good[6:]]))


def test_origin_validation() -> None:
    o = sa.Origin(pd.Timestamp("2026-11-12 21:53:40.5"), 64.0, -17.0)
    assert str(o.time.tz) == "UTC"
    assert (
        sa.Origin(pd.Timestamp("2026-11-12 22:53:40.5+01:00"), 64.0, -17.0).time
        == o.time
    )
    with pytest.raises(ValueError):
        sa.Origin(o.time, float("nan"), -17.0)
    with pytest.raises(ValueError):
        sa.Origin(o.time, 95.0, -17.0)


# --------------------------------------------------------------------------
# Travel-time fit (legacy check_event_fit)
# --------------------------------------------------------------------------


def test_golden_fit(golden: sa.AlertEvent, golden_origin: sa.Origin) -> None:
    f = golden.frame
    dist = sa.station_distances_km(
        golden_origin.latitude, golden_origin.longitude, f["lat"], f["lon"]
    )
    fit = sa.travel_time_fit(golden_origin, f["arrival_time"], dist)
    assert (fit.n_fit, fit.n_total, fit.accepted) == (61, 66, True)
    assert fit.label == page_row("2026-11-12 21:53:40.500")["alertfit"] == "61/66 (92%)"
    assert fit.region == "bardarbunga" and fit.post_s == 25.0
    # Without the Bárðarbunga allowance (12 s) stations are lost.
    plain = sa.travel_time_fit(
        golden_origin,
        f["arrival_time"],
        dist,
        settings=sa.FitSettings(special_regions=()),
    )
    assert plain.region is None and plain.post_s == 12.0 and plain.n_fit < fit.n_fit
    # Just outside the box: the default 12 s window.
    out = sa.Origin(golden_origin.time, 64.71, golden_origin.longitude)
    assert sa.travel_time_fit(out, f["arrival_time"], dist).post_s == 12.0


def test_small_detection_rejected(golden_origin: sa.Origin) -> None:
    ev = sa.read_alert_event(ALERT / "2026/nov/12/001.event")
    dist = sa.station_distances_km(64.98, -16.34, ev.frame["lat"], ev.frame["lon"])
    origin = sa.Origin(pd.Timestamp("2026-11-12 00:04:40"), 64.98, -16.34)
    fit = sa.travel_time_fit(origin, ev.frame["arrival_time"], dist)
    assert fit.n_total == 2 and not fit.accepted  # < 10 stations, whatever fits


def test_fit_rules_synthetic() -> None:
    t0 = pd.Timestamp("2026-11-12 23:59:50", tz="UTC")
    origin = sa.Origin(t0, 63.0, -20.0)
    dist = np.full(12, 73.0)  # window (8, 50.42) s after the origin
    offs = [9.0] * 9 + [8.0, 50.5, 30.0]  # bounds are strict
    arr = [t0 + pd.Timedelta(seconds=s) for s in offs]  # cross midnight
    fit = sa.travel_time_fit(origin, arr, dist)
    assert fit.n_fit == 10 and fit.accepted
    assert list(fit.fits) == [True] * 9 + [False, False, True]
    assert fit.label == "10/12 (83%)" and fit.fraction == pytest.approx(10 / 12)
    # Naive arrivals are UTC.
    naive = [a.tz_localize(None) for a in arr]
    assert sa.travel_time_fit(origin, naive, dist).n_fit == 10
    # Fraction rule: 10 of 15 = 67 % < 68 %.
    more = sa.travel_time_fit(origin, arr + [t0] * 3, np.full(15, 73.0))
    assert more.n_fit == 10 and not more.accepted
    # NaN distance never fits but counts.
    d2 = dist.copy()
    d2[0] = np.nan
    assert sa.travel_time_fit(origin, arr, d2).n_fit == 9
    empty = sa.travel_time_fit(origin, [], [])
    assert not empty.accepted and empty.label == "" and np.isnan(empty.fraction)
    with pytest.raises(ValueError, match="arrivals"):
        sa.travel_time_fit(origin, arr, dist[:3])


# --------------------------------------------------------------------------
# Estimate
# --------------------------------------------------------------------------


def test_golden_estimate(golden: sa.AlertEvent, golden_origin: sa.Origin) -> None:
    res = sa.estimate_pgv_magnitude(golden, golden_origin, event_id=GOLDEN_ID)
    page = page_row("2026-11-12 21:53:40.500")
    assert res.magnitude_type == "Mpgv_alert" and res.magnitude_type in MAGNITUDE_TYPES
    assert res.n_alert == int(page["alert_sta"]) == 66
    assert res.n_kept == int(page["filter_sta"]) == 54
    assert res.min_km == 15.0
    assert r2(res.magnitude) == float(page["median"]) == 4.15
    assert res.magnitude == pytest.approx(4.1534, abs=5e-5)
    # The page's 4.17 is the exact inverse's mid-third mean (far field: 4.16).
    assert r2(res.robust_mean) == float(page["mean"]) == 4.17
    assert res.fit is not None and res.fit.label == "61/66 (92%)" and res.fit.accepted
    s = res.summary()
    assert s["n_fit"] == 61 and s["alertfit"] == "61/66 (92%)" and s["alert_id"] == 76

    st = res.stations
    assert tuple(st.columns) == sa.PGV_STATION_COLUMNS and len(st) == 66
    reasons = st["drop_reason"].value_counts().to_dict()
    assert reasons == {
        "beyond_max_km": 8,
        "excluded_class_code": 2,
        "excluded_station": 1,
        "nearfield": 1,
    }
    assert set(st.loc[st["drop_reason"] == "excluded_class_code", "station"]) == {
        "hei",
        "gha",
    }
    assert list(st.loc[st["drop_reason"] == "nearfield", "station"]) == ["von"]
    assert st["kept"].sum() == 54 and st.loc[st["kept"], "drop_reason"].isna().all()
    assert st["fits_window"].sum() == 61
    assert (st["source"] == "alertmap").all() and (st["event_id"] == GOLDEN_ID).all()
    assert (st["alert_day"] == "2026-11-12").all() and (st["alert_id"] == 76).all()
    assert st["arrival_time"].dtype == TIME_DTYPE


def test_golden_farfield(golden: sa.AlertEvent, golden_origin: sa.Origin) -> None:
    res = sa.estimate_pgv_magnitude(
        golden, golden_origin, settings=sa.PgvSettings(inverse="farfield")
    )
    assert res.n_kept == 54 and r2(res.magnitude) == 4.15
    assert r2(res.robust_mean) == 4.16  # the doc's 4.162: not the page's 4.17
    kept = res.stations[res.stations["kept"]]
    assert (kept["mag_exact"] > kept["mag_farfield"]).all()


def test_golden_variants(golden: sa.AlertEvent, golden_origin: sa.Origin) -> None:
    # A fixed 10 km cut keeps von (13.3 km): 55 stations.
    assert (
        sa.estimate_pgv_magnitude(
            golden, golden_origin, settings=sa.PgvSettings(min_km=10.0)
        ).n_kept
        == 55
    )
    # Without the code rule: the discovery doc's 56.
    no_code = sa.PgvSettings(exclude_class_codes=())
    assert (
        sa.estimate_pgv_magnitude(golden, golden_origin, settings=no_code).n_kept == 56
    )
    # The fit can be skipped.
    res = sa.estimate_pgv_magnitude(golden, golden_origin, fit=None)
    assert res.fit is None and res.stations["fits_window"].isna().all()
    assert res.summary()["alertfit"] is None


def test_evb_distances(golden: sa.AlertEvent, golden_origin: sa.Origin) -> None:
    evb = sa.read_evb(GOLDEN_EVB)
    res = sa.estimate_pgv_magnitude(
        golden, golden_origin, evb=evb, settings=sa.PgvSettings(distance="evb")
    )
    st = res.stations
    assert (st["distance_source"] == "evb").all()
    in_evb = st["station"].isin(set(evb.phases["station"]))
    assert (
        st.loc[~in_evb, "drop_reason"]
        .isin(["no_distance", "excluded_class_code", "excluded_station"])
        .all()
    )
    assert st.loc[in_evb, "drop_reason"].ne("no_distance").all()
    assert res.n_kept == 27 and r2(res.magnitude) == 4.14
    # The fit keeps the geodesic rule whatever the distance source.
    assert res.fit is not None and res.fit.label == "61/66 (92%)"
    with pytest.raises(ValueError, match="evb"):
        sa.pgv_stations(golden, golden_origin, settings=sa.PgvSettings(distance="evb"))


def test_empty_and_bad_pgv(golden_origin: sa.Origin) -> None:
    ev = sa.parse_alert_event("001  14  17  28 Thu Nov 12 00:04:42 2026    3  0\n")
    res = sa.estimate_pgv_magnitude(ev, golden_origin)
    assert res.n_alert == 0 and np.isnan(res.magnitude) and np.isnan(res.robust_mean)
    assert res.fit is not None and not res.fit.accepted
    text = (
        "001  14  17  28 Thu Nov 12 21:53:45 2026    3  1\n"
        "-17.26712  64.40655  821      0.00 grf   0.91 14   -10 53.21 96 44.90   3.80    25.22\n"
    )
    st = sa.pgv_stations(sa.parse_alert_event(text), golden_origin)
    assert st["drop_reason"].tolist() == ["bad_pgv"]


# --------------------------------------------------------------------------
# More Mpgv page rows (documented, not hidden: the B4 code-22 rule fails some)
# --------------------------------------------------------------------------

PAGE_CASES = {
    # file, page time: (kept, median, mean) with the default B4 rule
    "2026/nov/05/042.event": ("2026-11-05 17:51:37.600", 15, 3.47, 3.49),
    "2026/nov/01/050.event": ("2026-11-01 20:28:59.400", 2, 3.31, None),
}


@pytest.mark.parametrize("name", sorted(PAGE_CASES))
def test_page_rows_reproduced(name: str) -> None:
    time, kept, med, mean = PAGE_CASES[name]
    row = page_row(time)
    res = sa.estimate_pgv_magnitude(sa.read_alert_event(ALERT / name), page_origin(row))
    assert res.n_kept == kept == int(row["filter_sta"])
    assert r2(res.magnitude) == med == float(row["median"])
    if mean is None:
        assert np.isnan(res.robust_mean) and row["mean"] == ""  # n = 2: blank
    else:
        assert r2(res.robust_mean) == mean == float(row["mean"])


def test_page_row_farfield_differs() -> None:
    # 2026-11-01 #050: the far field rounds to 3.30, the page (exact) to 3.31.
    row = page_row("2026-11-01 20:28:59.400")
    ev = sa.read_alert_event(ALERT / "2026/nov/01/050.event")
    far = sa.estimate_pgv_magnitude(
        ev, page_origin(row), settings=sa.PgvSettings(inverse="farfield")
    )
    assert r2(far.magnitude) == 3.30


def test_code22_rule_deviations() -> None:
    """Two page rows the field-6 ≠ 22 rule (B4) does not reproduce.

    2026-11-03 #009: the page keeps 8 stations, four of them code 22 (gri,
    gil, hla, gra); dropping no station by code gives the page's 3.10/3.01.
    2026-11-10 #050: the page drops two of 56 stations with no code 22 among
    them; dropping hei (code 12 here, 22 on 11-12) and gha (14 here)
    reproduces 54 / 4.39 / 4.40. Field 6 is not a fixed station property.
    """
    row = page_row("2026-11-03 02:28:38.900")
    ev = sa.read_alert_event(ALERT / "2026/nov/03/009.event")
    b4 = sa.estimate_pgv_magnitude(ev, page_origin(row))
    assert (b4.n_kept, r2(b4.magnitude), r2(b4.robust_mean)) == (4, 3.17, 3.17)
    none = sa.estimate_pgv_magnitude(
        ev, page_origin(row), settings=sa.PgvSettings(exclude_class_codes=())
    )
    assert (none.n_kept, r2(none.magnitude), r2(none.robust_mean)) == (8, 3.10, 3.01)
    assert (int(row["filter_sta"]), row["median"], row["mean"]) == (8, "3.1", "3.01")

    row = page_row("2026-11-10 13:37:50.200")
    ev = sa.read_alert_event(ALERT / "2026/nov/10/050.event")
    codes = ev.frame.set_index("station")["class_code"]
    assert (codes["hei"], codes["gha"]) == (12, 14)
    b4 = sa.estimate_pgv_magnitude(ev, page_origin(row))
    assert b4.n_kept == 56 and r2(b4.magnitude) == 4.38
    names = sa.PgvSettings(
        exclude_stations=("ada", "hei", "gha"), exclude_class_codes=()
    )
    alt = sa.estimate_pgv_magnitude(ev, page_origin(row), settings=names)
    assert (alt.n_kept, r2(alt.magnitude), r2(alt.robust_mean)) == (54, 4.39, 4.40)
    assert alt.fit is not None and alt.fit.label == row["alertfit"] == "56/66 (85%)"
