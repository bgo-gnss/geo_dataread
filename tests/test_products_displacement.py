"""gpsprod step 1: the displacement product (ends) on synthetic series."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pytest

from geo_dataread.products import (
    DisplacementRecord,
    DisplacementSettings,
    Source,
    StationView,
    displacement_from_view,
    estimate_displacements,
    resolve_period,
)
from geo_dataread.products import cli, loader
from geo_dataread.products import displacement as pdisp
from geo_dataread.products.files import table, write
from geo_dataread.velocity import window_bounds

DAY = 1.0 / 365.25
PROVENANCE = (
    "frame",
    "series",
    "session",
    "ref",
    "view",
    "method",
    "period",
    "sigma_method",
)


def _view(
    rate=(20.0, -10.0, 5.0),
    *,
    days: int = 400,
    per_day: int = 1,
    noise: float = 1.0,
    seed: int = 0,
    start: float = 2024.0,
) -> StationView:
    rng = np.random.default_rng(seed)
    n = days * per_day
    t = start + (np.arange(n) + 0.5) * DAY / per_day
    y = np.vstack([r * (t - start) for r in rate]) + rng.normal(0, noise, (3, n))
    return StationView("SYNT", t, y, np.full((3, n), noise), "plate-periodic")


def _period(first: dt.date, last: dt.date) -> tuple[float, float]:
    return window_bounds(first, last)


def test_linear_motion_is_rate_times_span() -> None:
    v = _view()
    p = (2024.2, 2024.2 + 200 * DAY)
    vals, flags, refused = displacement_from_view(v, p, DisplacementSettings())
    assert refused is None and flags == []
    span = (
        vals["ends"]["end"]["north"]["t_mean"]
        - vals["ends"]["start"]["north"]["t_mean"]
    )
    for i, r in enumerate((20.0, -10.0, 5.0)):
        assert abs(vals["displacement"][i] - r * span) < 3 * vals["sigma"][i]


def test_window_counts_and_eight_hour_sessions() -> None:
    p = (2024.2, 2024.2 + 100 * DAY)
    v24, _, _ = displacement_from_view(_view(), p, DisplacementSettings())
    s8 = DisplacementSettings(source=Source(session="8h"))
    v8, _, r8 = displacement_from_view(_view(per_day=3), p, s8)
    assert r8 is None
    assert v24["ends"]["start"]["north"]["n"] == 5
    assert v8["ends"]["start"]["north"]["n"] == 15
    # the daily factors are measured at the session step, not left NaN
    assert all(math.isfinite(k) for k in v8["ends"]["k_day"])
    for i in range(3):
        diff = v8["displacement"][i] - v24["displacement"][i]
        assert abs(diff) < 3 * math.hypot(v8["sigma"][i], v24["sigma"][i])


def test_one_day_end_blunder_is_disputed() -> None:
    v = _view(noise=1.0, seed=3)
    p = (2024.2, 2024.2 + 100 * DAY)
    last = np.flatnonzero(v.t < p[1])[-1]
    v.y[0, last] += 40.0
    s = DisplacementSettings(ends_days=1)
    _, flags, refused = displacement_from_view(v, p, s)
    assert refused is None
    assert "end_disputed:end:N" in flags
    assert not any(f.startswith("end_disputed:start") for f in flags)


def test_n5_and_n1_agree_on_quiet_data() -> None:
    v = _view(noise=0.5, seed=4)
    p = (2024.2, 2024.2 + 150 * DAY)
    a, _, _ = displacement_from_view(v, p, DisplacementSettings())
    b, flags, _ = displacement_from_view(v, p, DisplacementSettings(ends_days=1))
    assert not any(f.startswith("end_disputed") for f in flags)
    for i in range(3):
        assert abs(a["displacement"][i] - b["displacement"][i]) < 3 * b["sigma"][i]


def test_step_in_end_window_refuses_inside_period_flags() -> None:
    v = _view()
    p = (2024.2, 2024.2 + 100 * DAY)
    _, _, refused = displacement_from_view(
        v, p, DisplacementSettings(), step_epochs=[2024.2 + 2 * DAY]
    )
    assert refused is not None and refused.startswith("step_in_end_window@")
    _, flags, refused = displacement_from_view(
        v, p, DisplacementSettings(), step_epochs=[2024.2 + 50 * DAY]
    )
    assert refused is None and "steps_in_period" in flags


def test_end_window_closes_at_the_last_epoch() -> None:
    v = _view(days=100)
    p = (2024.0, 2024.0 + 120 * DAY)  # 20 days past the data
    vals, flags, refused = displacement_from_view(v, p, DisplacementSettings())
    assert refused is None
    assert vals["ends"]["end"]["north"]["n"] == 5
    assert any(f.startswith("end_early:") for f in flags)
    # a one-day lag (today's solution not out yet) is not flagged
    p2 = (2024.0, v.t[-1] + 0.5 * DAY + 1.0 * DAY)
    _, flags2, _ = displacement_from_view(v, p2, DisplacementSettings())
    assert not any(f.startswith("end_early") for f in flags2)


def test_refusals() -> None:
    v = _view(days=50)
    s = DisplacementSettings()
    assert displacement_from_view(v, (2030.0, 2030.5), s)[2] == (
        "no epochs in the period (last data 2024-02-19)"
    )
    short = displacement_from_view(v, (2024.0, 2024.0 + 7 * DAY), s)[2]
    assert short and "shorter than two" in short
    v.y[1, :10] = np.nan
    assert displacement_from_view(v, (2024.0, 2024.1), s)[2] == (
        "no east epochs in the start window"
    )
    v.y[:, :10] = np.nan  # no component: the window dates and the nearest data
    msg = displacement_from_view(v, (2024.0, 2024.1), s)[2]
    assert msg and msg.startswith(
        "no epochs in the start window 2024-01-01..2024-01-05"
    )
    assert "first data after it 2024-01-11" in msg


def test_settings_validate() -> None:
    with pytest.raises(ValueError):
        DisplacementSettings(ends_days=0)
    with pytest.raises(ValueError):
        Source(view="smooth")
    with pytest.raises(ValueError):
        Source(frame="itrf99")
    assert Source(frame="igb14").pos_id == "imo.24hr_itrf14"
    assert Source(frame="itrf20", session="8h").pos_id == "imo.08hr_itrf20"


def test_resolve_period() -> None:
    today = dt.date(2026, 10, 26)
    p = resolve_period("20231110:20240110")
    assert p.dates == ("2023-11-10", "2024-01-10")
    assert p.bounds == window_bounds(dt.date(2023, 11, 10), dt.date(2024, 1, 10))
    assert resolve_period("2023-11-10:", today=today).last_day == today
    assert resolve_period(since="20240101", end="20240131").dates[1] == "2024-01-31"
    assert resolve_period(last=7, today=today).first_day == dt.date(2026, 10, 20)
    with pytest.raises(ValueError):
        resolve_period("20231110")
    with pytest.raises(ValueError):
        resolve_period("20240110:20231110")
    with pytest.raises(ValueError):
        resolve_period()


def _patch_load(monkeypatch: pytest.MonkeyPatch, views: dict[str, StationView]) -> None:
    def fake(sta, start, end, source, *, doc=None):
        if sta not in views:
            raise ValueError(f"no data for station {sta}")
        return dataclasses.replace(views[sta], station=sta)

    monkeypatch.setattr(pdisp, "load", fake)
    monkeypatch.setattr(pdisp, "station_coords", lambda sta: (-22.0, 64.0))
    monkeypatch.setattr(Source, "params_doc", lambda self: None)
    import geo_dataread.gps_views as gv

    monkeypatch.setattr(
        gv, "station_step_epochs", lambda sta, steps=None: (np.zeros(0), None)
    )


def test_records_carry_provenance_and_refusals(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_load(monkeypatch, {"AAAA": _view(), "BBBB": _view(seed=7)})
    rs = estimate_displacements(["bbbb", "AAAA", "ZZZZ"], (2024.2, 2024.5))
    assert [r.station for r in rs] == ["AAAA", "BBBB", "ZZZZ"]
    assert len(rs.accepted()) == 2
    assert rs.records[2].refused == "read: no data for station ZZZZ"
    for r in rs:
        d = r.to_dict()
        for key in PROVENANCE:
            assert d[key] not in (None, ""), key
    # round trip through JSON
    back = DisplacementRecord.from_dict(json.loads(json.dumps(rs.records[0].to_dict())))
    assert back.displacement == pytest.approx(rs.records[0].displacement)


def test_files_are_byte_stable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_load(monkeypatch, {"AAAA": _view(), "BBBB": _view(seed=7)})
    a = estimate_displacements(["AAAA", "BBBB"], (2024.2, 2024.5))
    b = estimate_displacements(["BBBB", "AAAA"], (2024.2, 2024.5))
    for ext in (".csv", ".jsonl"):
        pa, pb = tmp_path / f"a{ext}", tmp_path / f"b{ext}"
        write(a, str(pa))
        write(b, str(pb))
        assert pa.read_bytes() == pb.read_bytes()
    text = (tmp_path / "a.csv").read_text()
    assert text.startswith("# gpsprod displacement schema 1\n")
    assert "\nstation,lon,lat,start,end,dN,dE,dU" in text
    # metres: the numbers / 1000, the unit says so
    write(a, str(tmp_path / "m.csv"), units="m")
    mm = a.to_frame()
    m = [
        line.split(",")
        for line in (tmp_path / "m.csv").read_text().splitlines()
        if line.startswith("AAAA")
    ]
    assert float(m[0][5]) == pytest.approx(float(mm.loc[0, "dN"]) / 1000, abs=1e-6)
    assert m[0][11] == "m"
    with pytest.raises(ValueError, match="unknown format"):
        write(a, str(tmp_path / "x.txt"))
    assert "AAAA" in table(a.records)


def test_relative_to_cancels_common_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    common = _view(rate=(30.0, 30.0, 30.0), noise=0.0)
    a = _view(rate=(5.0, 0.0, 0.0), seed=1, noise=0.3)
    b = _view(rate=(0.0, 0.0, 0.0), seed=2, noise=0.3)
    a = dataclasses.replace(a, y=a.y + common.y)
    b = dataclasses.replace(b, y=b.y + common.y)

    def fake_read(sta, start, end, source, *, doc):
        return dataclasses.replace({"AAAA": a, "BBBB": b}[sta], station=sta)

    monkeypatch.setattr(loader, "_read_one", fake_read)
    src = Source(relative_to="bbbb")
    view = loader.load("aaaa", 2024.0, 2025.0, src)
    assert view.attrs["baseline"]["relative_to"] == "BBBB"
    vals, _, refused = displacement_from_view(
        view, (2024.2, 2024.7), DisplacementSettings(source=src)
    )
    assert refused is None
    span = 0.5
    assert abs(vals["displacement"][0] - 5.0 * span) < 0.5
    assert abs(vals["displacement"][1]) < 0.5


def test_cli_displacement_and_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_load(monkeypatch, {"AAAA": _view(start=2024.0, days=400)})
    out = tmp_path / "d.csv"
    rc = cli.main(
        ["displacement", "AAAA", "--period", "20240201:20240701", "-q", "-o", str(out)]
    )
    assert rc == 0 and out.read_text().count("\nAAAA,") == 1
    rc = cli.main(["displacement", "AAAA", "--period", "20240201:20240701", "-q"])
    assert "AAAA" in capsys.readouterr().out
    rc = cli.displacemnts_alias(["AAAA", "-s", "20240201", "-e", "20240701"])
    captured = capsys.readouterr()
    assert (
        rc == 0
        and "gpsprod displacement AAAA --period 20240201:20240701" in captured.err
    )
    with pytest.raises(SystemExit):
        cli.main(
            [
                "displacement",
                "AAAA",
                "--start",
                "20240101",
                "--days",
                "3",
                "--end",
                "20240110",
            ]
        )
    with pytest.raises(SystemExit):
        cli.main(
            [
                "displacement",
                "AAAA",
                "--ref",
                "background",
                "--keep-seasonal",
                "--last",
                "3",
            ]
        )


# --------------------------------------------------------------------------
# step 2: event periods, step, rate, conversions, help
# --------------------------------------------------------------------------


def _events():
    from geo_dataread.events import Event

    return [
        Event(
            "reykjanes-2023-11-10-dike",
            "dike",
            dt.date(2023, 11, 10),
            dt.date(2023, 11, 10),
            "reykjanes",
        ),
        Event(
            "sundhnukur-2023-12-eruption",
            "eruption",
            dt.date(2023, 12, 18),
            dt.date(2023, 12, 21),
            "reykjanes",
        ),
        Event(
            "sundhnukur-2024-01-eruption",
            "eruption",
            dt.date(2024, 1, 14),
            dt.date(2024, 1, 16),
            "reykjanes",
        ),
    ]


def test_event_periods() -> None:
    ev = _events()
    today = dt.date(2026, 10, 26)
    p = resolve_period("2023-11-10-dike:2023-12-eruption", events=ev, today=today)
    assert p.dates == ("2023-11-11", "2023-12-17")
    assert p.names == ("reykjanes-2023-11-10-dike", "sundhnukur-2023-12-eruption")
    assert resolve_period("after:2023-11-10-dike", events=ev, today=today).dates == (
        "2023-11-11",
        "2026-10-26",
    )
    assert resolve_period("20231001:2023-11-10-dike", events=ev).dates == (
        "2023-10-01",
        "2023-11-09",
    )
    assert resolve_period("2023-12-eruption", events=ev).dates == (
        "2023-12-18",
        "2023-12-21",
    )
    with pytest.raises(ValueError, match="names 2 events"):
        resolve_period("eruption:", events=ev)
    with pytest.raises(ValueError, match="no event"):
        resolve_period("2022-dike:", events=ev)


def test_step_recovers_a_dike_and_refuses_thin_sides() -> None:
    from geo_dataread.products.displacement import step_from_view

    v = _view(rate=(10.0, 10.0, 10.0), noise=0.5, seed=8)
    onset = 2024.3
    jump = np.array([50.0, -80.0, 30.0])
    v.y[:, v.t > onset] += jump[:, None]
    ev = (onset, onset + DAY, "dike")
    vals, flags, refused = step_from_view(
        v, (2024.0, 2024.6), ev, DisplacementSettings(method="step")
    )
    assert refused is None
    for i in range(3):
        # the jump plus one day of the 10 mm/yr background (the event day)
        assert (
            abs(vals["displacement"][i] - (jump[i] + 10.0 * DAY)) < 3 * vals["sigma"][i]
        )
    assert vals["step"]["event"] == "dike"
    assert vals["step"]["lines"]["before"]["north"]["n"] == 30
    gappy = dataclasses.replace(v, y=v.y.copy())
    gappy.y[:, (v.t > onset) & (v.t < onset + 20 * DAY)] = np.nan
    _, _, refused = step_from_view(
        gappy, (2024.0, 2024.6), ev, DisplacementSettings(method="step")
    )
    assert refused is not None and refused.startswith("step: after cover=")
    _, _, refused = step_from_view(
        v, (2024.0, 2024.2), ev, DisplacementSettings(method="step")
    )
    assert refused and "not inside the period" in refused


def test_event_bounds() -> None:
    from geo_dataread.products.displacement import event_bounds

    ev, areas = _events(), {"reykjanes": frozenset({"THOB"})}
    period = window_bounds(dt.date(2023, 11, 1), dt.date(2023, 12, 10))
    s = DisplacementSettings(method="step")
    got = event_bounds("THOB", period, s, events=ev, areas=areas)
    assert not isinstance(got, str) and got[2] == "reykjanes-2023-11-10-dike"
    assert got[1] - got[0] == pytest.approx(DAY, rel=1e-3)
    wide = window_bounds(dt.date(2023, 11, 1), dt.date(2024, 2, 1))
    msg = event_bounds("THOB", wide, s, events=ev, areas=areas)
    assert isinstance(msg, str) and "3 dikes/eruptions" in msg
    assert isinstance(event_bounds("HOFN", period, s, events=ev, areas=areas), str)
    at = event_bounds(
        "HOFN",
        period,
        DisplacementSettings(method="step", at="20231110"),
        events=None,
        areas=None,
    )
    assert not isinstance(at, str) and at[2] == "2023-11-10"
    eid = event_bounds(
        "HOFN",
        wide,
        DisplacementSettings(method="step", event="2024-01"),
        events=ev,
        areas=areas,
    )
    assert not isinstance(eid, str) and eid[2] == "sundhnukur-2024-01-eruption"


def _vrec(rate=(10.0, -20.0, 4.0), sigma=(1.0, 2.0, 0.5), window=(2024.0, 2024.5)):
    from geo_dataread.velocity import VelocityRecord

    nan3 = (math.nan,) * 3
    return VelocityRecord(
        station="SYNT",
        lon=-22.0,
        lat=64.0,
        frame="itrf2014",
        plate="EURA",
        ref="plate-periodic",
        ref_requested="plate-periodic",
        series="imo.24hr_itrf14",
        view="declared",
        uncert=10.0,
        window=window,
        span_used=window,
        n_epochs=(180, 180, 180),
        largest_gap=1.0,
        method="wls",
        model="linear",
        rate=rate,
        sigma=sigma,
        sigma_formal=nan3,
        sigma_method="empirical_k",
        k=(3.0, 3.0, 3.0),
        k_source={},
        corr_en=0.0,
        record=None,
        steps_in_window=(),
        outlier_abort=False,
        n_flagged=0,
        flags=("nonlinear:N",),
        refused=None,
        software={},
        input_fingerprint="abc",
    )


def test_conversions_round_trip() -> None:
    from geo_dataread.products.displacement import to_displacement, to_velocity

    v = _vrec()
    d = to_displacement(v)
    assert d.displacement == pytest.approx((5.0, -10.0, 2.0))
    assert d.sigma == pytest.approx((0.5, 1.0, 0.25))
    assert d.method == "rate" and d.flags == ("nonlinear:N",) and d.rate is not None
    back = to_velocity(d)
    assert back.rate == pytest.approx(v.rate, rel=1e-12)
    assert back.sigma == pytest.approx(v.sigma, rel=1e-12)
    assert back.method == "from_displacement:rate"
    # a refused velocity stays refused
    from dataclasses import replace

    r = to_displacement(replace(v, refused="n_obs=2<5", rate=(math.nan,) * 3))
    assert r.refused == "n_obs=2<5" and not r.accepted


def test_ends_to_velocity_uses_the_window_centres() -> None:
    from geo_dataread.products.displacement import displacement_span

    v = _view(noise=0.2, seed=9)
    p = (2024.2, 2024.2 + 200 * DAY)
    vals, _, _ = displacement_from_view(v, p, DisplacementSettings())
    rec = DisplacementRecord(
        station="SYNT",
        lon=0.0,
        lat=0.0,
        period=p,
        period_dates=("a", "b"),
        frame="itrf2014",
        series="x",
        session="24h",
        ref="plate-periodic",
        ref_requested="plate-periodic",
        view="declared",
        method="ends",
        sigma_method="s",
        **vals,
    )
    from geo_dataread.products.displacement import to_velocity

    assert displacement_span(rec) == pytest.approx(195 * DAY, rel=1e-3)
    assert to_velocity(rec).rate[0] == pytest.approx(20.0, abs=0.5)


def test_rate_method_is_the_velocity_product(monkeypatch: pytest.MonkeyPatch) -> None:
    import geo_dataread.velocity as vel

    seen = {}

    def fake(sta, start, end, settings, *, doc=None, coords=None):
        seen["args"] = (sta, start, end, settings.ref, settings.view)
        return _vrec(window=(start, end))

    monkeypatch.setattr(vel, "estimate_station_velocity", fake)
    from geo_dataread.products.displacement import estimate_station_displacement

    rec = estimate_station_displacement(
        "SYNT", (2024.0, 2024.5), DisplacementSettings(method="rate"), coords=(0.0, 0.0)
    )
    assert seen["args"] == ("SYNT", 2024.0, 2024.5, "plate-periodic", "declared")
    assert rec.displacement == pytest.approx((5.0, -10.0, 2.0))
    assert rec.rate and rec.rate["input_fingerprint"] == "abc"
    bad = estimate_station_displacement(
        "SYNT",
        (2024.0, 2024.5),
        DisplacementSettings(method="rate", source=Source(ref="itrf")),
        coords=(0.0, 0.0),
    )
    assert bad.refused and bad.refused.startswith("rate: ref itrf")


def test_help_lists_every_option_and_examples(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        cli.main(["displacement", "--help"])
    out = capsys.readouterr().out
    for opt in (
        "--method",
        "--ends",
        "--neighbour-days",
        "--event",
        "--at",
        "--step-window",
        "--as-velocity",
        "--relative-to",
        "--keep-seasonal",
        "--units",
        "--region",
    ):
        assert opt in out, opt
    assert "examples:" in out and "periods (--period):" in out and "methods:" in out
    p = cli.build_parser()
    sub = p._subparsers._group_actions[0].choices["displacement"]  # type: ignore[union-attr]
    for action in sub._actions:
        if action.dest != "help":
            assert action.help, action.dest


def test_as_velocity_gmt_rows_are_whole(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_load(monkeypatch, {"AAAA": _view(), "BBBB": _view(seed=7)})
    out = tmp_path / "vel" / "set"
    rc = cli.main(
        ["displacement", "AAAA", "BBBB", "--period", "20240201:20240701",
         "--as-velocity", "-o", str(out), "-q"]
    )  # fmt: skip
    assert rc == 0
    rows = [
        line.split()
        for line in (tmp_path / "vel" / "set.gmt").read_text().splitlines()
        if not line.startswith("#")
    ]
    assert len(rows) == 2
    for r in rows:
        assert len(r) == 8 and "nan" not in r


def test_a_station_without_a_plate_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_load(monkeypatch, {})

    def no_plate(sta, start, end, source, *, doc=None):
        raise KeyError(f"station {sta} has no plate in the platefile")

    monkeypatch.setattr(pdisp, "load", no_plate)
    rs = estimate_displacements(["ALGO"], (2024.2, 2024.5))
    assert rs.records[0].refused and "no plate" in rs.records[0].refused


def test_metres_say_the_nested_units() -> None:
    v = _view()
    vals, _, _ = displacement_from_view(v, (2024.2, 2024.5), DisplacementSettings())
    rec = DisplacementRecord(
        station="SYNT",
        lon=0.0,
        lat=0.0,
        period=(2024.2, 2024.5),
        period_dates=("a", "b"),
        frame="itrf2014",
        series="x",
        session="24h",
        ref="plate-periodic",
        ref_requested="plate-periodic",
        view="declared",
        method="ends",
        sigma_method="s",
        **vals,
    )
    m = rec.scaled("m")
    assert m.units == "m" and m.ends is not None and m.ends["units"] == "mm"
    assert m.displacement[0] == pytest.approx(rec.displacement[0] / 1000)


def test_points_is_end_point_minus_start_point() -> None:
    v = _view(noise=1.0, seed=10)
    p = (2024.2, 2024.2 + 20 * DAY)
    vals, flags, refused = displacement_from_view(
        v, p, DisplacementSettings(method="points")
    )
    assert refused is None
    inside = np.flatnonzero((v.t >= p[0]) & (v.t < p[1]))
    i0, i1 = inside[0], inside[-1]
    for c in range(3):
        assert vals["displacement"][c] == pytest.approx(v.y[c, i1] - v.y[c, i0])
    assert vals["ends"]["start"]["north"]["n"] == 1 and vals["ends"]["days"] == 0
    assert not any(f.startswith(("start_at", "end_at")) for f in flags)


def test_points_takes_the_nearest_epochs_and_names_them() -> None:
    v = _view(noise=1.0, seed=11)
    p = (2024.2, 2024.2 + 20 * DAY)
    inside = np.flatnonzero((v.t >= p[0]) & (v.t < p[1]))
    v.y[:, inside[:2]] = np.nan  # the first two days missing
    v.y[0, inside[-1]] = np.nan  # the last day lacks north: not a whole epoch
    vals, flags, refused = displacement_from_view(
        v, p, DisplacementSettings(method="points")
    )
    assert refused is None
    assert any(f.startswith("start_at:") for f in flags)
    assert any(f.startswith("end_at:") for f in flags)
    assert vals["displacement"][1] == pytest.approx(
        v.y[1, inside[-2]] - v.y[1, inside[2]]
    )
    one = displacement_from_view(
        v, (p[0], p[0] + 3 * DAY), DisplacementSettings(method="points")
    )
    assert one[2] == "points: fewer than two epochs in the period"


def test_end_check_looks_outside_the_period() -> None:
    """A fast but steady motion inside the period, quiet outside: the ends
    agree with their outer neighbours, so nothing is disputed (a line
    through the inside days would cross the motion)."""
    rng = np.random.default_rng(12)
    t = 2024.0 + (np.arange(120) + 0.5) * DAY
    p = (2024.0 + 50 * DAY, 2024.0 + 60 * DAY)
    y = np.zeros((3, t.size))
    ramp = np.clip((t - p[0]) / (p[1] - p[0]), 0, 1) * 300.0
    y += ramp + rng.normal(0, 1, (3, t.size))
    # a quiet station whose ends sit on a 300 mm ramp: the inside line would
    # dispute them; the outside one (before/after the ramp) does not
    v = StationView("SYNT", t, y, np.ones_like(y), "plate-periodic")
    period = (p[0] - 0.5 * DAY, p[1] + 1.5 * DAY)
    vals, flags, refused = displacement_from_view(
        v, period, DisplacementSettings(method="points")
    )
    assert refused is None
    assert not any(f.startswith("end_disputed") for f in flags), flags
    assert vals["displacement"][0] == pytest.approx(300.0, abs=5)


def test_start_end_days() -> None:
    today = dt.date(2026, 10, 26)
    assert resolve_period(start="20231109", end="20231112").dates == (
        "2023-11-09",
        "2023-11-12",
    )
    assert resolve_period(start="20231109", days=4).dates == (
        "2023-11-09",
        "2023-11-12",
    )
    assert resolve_period(days=30, end="20241031").dates == ("2024-10-02", "2024-10-31")
    assert resolve_period(days=1, today=today).dates == ("2026-10-26", "2026-10-26")
    assert resolve_period(start="20261001", today=today).dates[1] == "2026-10-26"
    for bad in (
        {"start": "20231109", "days": 4, "end": "20231112"},
        {"spec": "20231109:20231112", "start": "20231109"},
        {"start": "20231109", "since": "20231109"},
        {"days": 0},
        {"end": "20231112"},
    ):
        with pytest.raises(ValueError):
            resolve_period(**bad)  # type: ignore[arg-type]


def test_cli_start_days_and_old_names(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    _patch_load(monkeypatch, {"AAAA": _view(start=2024.0, days=400)})
    for argv in (
        ["--start", "20240201", "--days", "30"],
        ["--since", "20240201", "--end", "20240301"],
        ["--last", "30", "--end", "20240302"],
    ):
        assert cli.main(["displacement", "AAAA", "-q", *argv]) == 0
        assert "AAAA" in capsys.readouterr().out


def test_a_step_refuses_only_a_window_that_straddles_it() -> None:
    v = _view()
    p = (2024.2, 2024.2 + 100 * DAY)
    first = v.t[np.flatnonzero(v.t >= p[0])[0]]
    # a step after the first epoch of a one-day window: one solution, fine
    one = DisplacementSettings(ends_days=1)
    _, flags, refused = displacement_from_view(
        v, p, one, step_epochs=[first + 0.1 * DAY]
    )
    assert refused is None and "steps_in_period" in flags
    # the same step inside a five-day window: its mean would straddle it
    _, _, refused = displacement_from_view(
        v, p, DisplacementSettings(), step_epochs=[first + 0.1 * DAY]
    )
    assert refused is not None and refused.startswith("step_in_end_window@")
