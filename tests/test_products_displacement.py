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
    assert (
        displacement_from_view(v, (2030.0, 2030.5), s)[2] == "no epochs in the period"
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
        cli.main(["displacement", "AAAA", "--since", "20240101", "--last", "3"])
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
