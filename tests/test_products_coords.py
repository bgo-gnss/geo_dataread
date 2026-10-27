"""gpsprod step 4: coordinates at an epoch (observed, model, apr, baseline)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from geo_dataread.products import cli
from geo_dataread.products import coords as pc
from geo_dataread.products.files import write

DAY = 1 / 365.25
BASE = {"AAAA": np.array([2602350.0, -1074405.0, 5704035.0]),
        "BBBB": np.array([2603999.0, -1075381.0, 5703317.0])}  # fmt: skip


def _series(
    sta: str, n: int = 200, per_day: int = 1, noise: float = 0.002, seed: int = 0
):
    rng = np.random.default_rng(seed + len(sta) + ord(sta[0]))
    t = 2024.0 + (np.arange(n * per_day) + 0.5) * DAY / per_day
    xyz = BASE[sta] + rng.normal(0, noise, (t.size, 3))
    cov = np.zeros((t.size, 3, 3))
    cov[:, np.arange(3), np.arange(3)] = noise**2
    return t, xyz, cov


def _patch(monkeypatch: pytest.MonkeyPatch, series=_series, steps=()) -> None:
    monkeypatch.setattr(pc, "_pos_series", lambda sta, src: series(sta))
    monkeypatch.setattr(
        pc, "_view_epochs", lambda sta, a, b, src, doc: (series(sta)[0], 1.0)
    )
    monkeypatch.setattr(
        pc,
        "_model_shift",
        lambda sta, src, rec, a, b: (np.array([0.01, 0.02, 0.0]) * (b - a), []),
    )
    import geo_dataread.gps_views as gv

    monkeypatch.setattr(
        gv,
        "station_step_epochs",
        lambda sta, steps=None, _s=steps: (np.asarray(_s, float), None),
    )


def test_observed_mean_of_the_window() -> None:
    t, xyz, cov = _series("AAAA")
    epoch = 2024.2
    m = pc.observed_mean(t, xyz, cov, epoch, 5)
    keep = np.abs(t - epoch) <= 5 * DAY
    assert m is not None and m["n"] == keep.sum() == 10  # 2024.2 is not a solution time
    np.testing.assert_allclose(m["xyz"], xyz[keep].mean(axis=0), atol=1e-9)
    assert abs(m["t_obs"] - epoch) < DAY
    one = pc.observed_mean(t, xyz, cov, float(t[40]), 0)
    assert one is not None and one["n"] == 1
    np.testing.assert_allclose(one["xyz"], xyz[40])
    assert pc.observed_mean(t, xyz, cov, 2030.0, 5) is None


def test_cut_at_steps_keeps_the_epoch_side() -> None:
    t = 2024.0 + (np.arange(30) + 0.5) * DAY
    epoch = float(t[15])
    keep, flags = pc.cut_at_steps(
        t, epoch, pc.window_half(5), [float(t[12]) + 0.1 * DAY]
    )
    assert (
        keep.sum() == 8
        and (t[keep] > t[12]).all()
        and flags[0].startswith("window_cut_at_step:")
    )
    keep, flags = pc.cut_at_steps(t, epoch, pc.window_half(5), [2030.0])
    assert keep.sum() == 11 and not flags


def test_record_observed(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch)
    rec = pc.estimate_station_coords("aaaa", 2024.2, epoch_date="2024-03-14")
    assert rec.accepted and rec.method == "observed" and rec.n_used == 10
    np.testing.assert_allclose(rec.xyz, BASE["AAAA"], atol=0.003)
    assert rec.llh[0] == pytest.approx(63.88, abs=0.01)
    assert all(0 < s < 0.002 for s in rec.sigma_neu)
    d = json.loads(json.dumps(rec.to_dict()))
    assert (
        d["units"] == "m" and d["kind"] == "coords" and d["epoch_date"] == "2024-03-14"
    )


def test_gap_extrapolates_and_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    def gappy(sta):
        t, xyz, cov = _series(sta)
        keep = (t < 2024.1) | (t > 2024.3)
        return t[keep], xyz[keep], cov[keep]

    _patch(monkeypatch, series=gappy, steps=[2024.15])
    rec = pc.estimate_station_coords("AAAA", 2024.2)
    assert rec.accepted
    assert any(f.startswith("extrapolated:") for f in rec.flags)
    assert "steps_between" in rec.flags and "sigma:observed_only" in rec.flags
    # the model shift was applied from the anchor's epoch
    assert rec.t_obs < 2024.1  # anchored on the window before the gap


def test_model_method_and_window_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, steps=[2024.2 - 2 * DAY])
    obs = pc.estimate_station_coords("AAAA", 2024.2)
    assert (
        any(f.startswith("window_cut_at_step") for f in obs.flags) and obs.n_used < 10
    )
    mod = pc.estimate_station_coords("AAAA", 2024.2, pc.CoordSettings(method="model"))
    assert mod.accepted and "sigma:observed_only" in mod.flags


def test_fast_motion_is_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    def moving(sta):
        t, xyz, cov = _series(sta)
        return t, xyz + np.outer(t - 2024.0, [3.0, 0.0, 0.0]), cov  # 3 m/yr: 8 mm/day

    _patch(monkeypatch, series=moving)
    rec = pc.estimate_station_coords("AAAA", 2024.2)
    assert any(f.startswith("fast_motion:") for f in rec.flags)


def test_relative_to_is_the_baseline_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch)
    from geo_dataread.products.loader import Source

    s = pc.CoordSettings(source=Source(view="cleaned", relative_to="bbbb"))
    rec = pc.estimate_station_coords("AAAA", 2024.2, s)
    assert rec.relative_to == "BBBB" and rec.dneu is not None
    np.testing.assert_allclose(rec.xyz, BASE["AAAA"] - BASE["BBBB"], atol=0.003)
    assert math.hypot(*rec.dneu) == pytest.approx(math.hypot(*rec.xyz), rel=1e-6)


def test_refusals(monkeypatch: pytest.MonkeyPatch) -> None:
    def nothing(sta):
        raise FileNotFoundError(f"no .pos time series for station '{sta}'")

    monkeypatch.setattr(pc, "_pos_series", lambda sta, src: nothing(sta))
    rec = pc.estimate_station_coords("ZZZZ", 2024.2)
    assert rec.refused and rec.refused.startswith("read:")
    from geo_dataread.products.loader import Source

    apr = pc.estimate_station_coords(
        "AAAA", 2024.2, pc.CoordSettings(method="apr", source=Source(frame="itrf08"))
    )
    assert apr.refused and "no GAMIT apriori" in apr.refused


def test_files_and_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    _patch(monkeypatch)
    monkeypatch.setattr(pc, "station_event_edges", lambda names: {})
    from geo_dataread.products.loader import Source

    monkeypatch.setattr(Source, "params_doc", lambda self: None)
    rs = pc.estimate_coords(["BBBB", "AAAA"], 2024.2, epoch_date="2024-03-14")
    assert [r.station for r in rs] == ["AAAA", "BBBB"]
    out = tmp_path / "c.csv"
    write(rs, str(out))
    text = out.read_text()
    assert "# gpsprod coords schema 1" in text and "\nstation,epoch,X,Y,Z," in text
    assert cli.main(["coords", "AAAA", "--epoch", "20240314", "-q"]) == 0
    assert "AAAA" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["coords", "--help"])
    help_text = capsys.readouterr().out
    for opt in (
        "--epoch",
        "--method",
        "--window-days",
        "--view",
        "--apr",
        "--eq",
        "--relative-to",
        "--uncert",
        "examples:",
        "flags:",
        "fast_motion",
    ):
        assert opt in help_text, opt  # fmt: skip
    sub = cli.build_parser()._subparsers._group_actions[0].choices["coords"]  # type: ignore[union-attr]
    for action in sub._actions:
        if action.dest != "help":
            assert action.help, action.dest
