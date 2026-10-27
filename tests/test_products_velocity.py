"""gpsprod step 3: the velocity product inside gpsprod (session, relative_to, CLI)."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from geo_dataread import velocity as vel
from geo_dataread.products import cli
from geo_dataread.products import velocity_core as core
from geo_dataread.products.loader import StationView

DAY = 1 / 365.25


def test_the_old_module_is_the_core() -> None:
    assert vel is core
    from geo_dataread.velocity import _coords, main, select_stations, write_velocity_set

    assert write_velocity_set.__module__ == "geo_dataread.products.velocity_files"
    assert (
        main.__module__
        == select_stations.__module__
        == "geo_dataread.products.velocity_cli"
    )
    assert _coords is core._coords


def test_later_fields_stay_out_of_json_and_fingerprint_at_defaults() -> None:
    s = core.VelocitySettings()
    assert "session" not in core._fingerprint_settings(s)
    assert "relative_to" not in core._fingerprint_settings(s)
    s8 = core.VelocitySettings(session="8h", relative_to="SKSH")
    fp = core._fingerprint_settings(s8)
    assert fp["session"] == "8h" and fp["relative_to"] == "SKSH"
    with pytest.raises(ValueError):
        core.VelocitySettings(session="12h")


def _rec(**kw):
    base = core.VelocityRecord(
        station="A", lon=0.0, lat=0.0, frame="itrf2014", plate=None, ref="plate",
        ref_requested="plate", series="imo.24hr_itrf14", view="declared", uncert=10.0,
        window=(2024.0, 2024.5), span_used=None, n_epochs=(1, 1, 1), largest_gap=None,
        method="wls", model="linear", rate=(1.0, 2.0, 3.0), sigma=(1.0, 1.0, 1.0),
        sigma_formal=(1.0, 1.0, 1.0), sigma_method="default_k", k=(5.0, 5.0, 5.0),
        k_source={}, corr_en=0.0, record=None, steps_in_window=(), outlier_abort=False,
        n_flagged=0, flags=(), refused=None, software={}, input_fingerprint="x",
    )  # fmt: skip
    return dataclasses.replace(base, **kw)


def test_record_json_round_trips_the_new_fields() -> None:
    d = _rec().to_dict()
    assert "session" not in d and "relative_to" not in d
    d8 = _rec(session="8h", relative_to="SKSH").to_dict()
    assert d8["session"] == "8h" and d8["relative_to"] == "SKSH"
    back = core.VelocityRecord.from_dict(json.loads(json.dumps(d8)))
    assert back.session == "8h" and back.relative_to == "SKSH"
    assert core.VelocityRecord.from_dict(json.loads(json.dumps(d))).session == "24h"


def test_a_baseline_velocity_reads_through_gpsprod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import geo_dataread.products.loader as loader

    rng = np.random.default_rng(1)
    t = 2024.0 + (np.arange(120) + 0.5) * DAY
    common = 50.0 * (t - 2024.0)
    a = np.vstack([common + 10.0 * (t - 2024.0)] * 3) + rng.normal(0, 0.5, (3, t.size))
    b = np.vstack([common] * 3) + rng.normal(0, 0.5, (3, t.size))
    views = {
        "AAAA": StationView("AAAA", t, a, np.full_like(a, 0.5), "plate"),
        "BBBB": StationView("BBBB", t, b, np.full_like(b, 0.5), "plate"),
    }
    monkeypatch.setattr(loader, "_read_one", lambda sta, *a, **k: views[sta])
    monkeypatch.setattr(core, "_data_end", lambda *a, **k: None)
    s = core.VelocitySettings(ref="plate", relative_to="bbbb", k_table="none")
    r = core.estimate_station_velocity(
        "AAAA", 2024.05, 2024.3, s, doc={"stations": {}}, coords=(0.0, 0.0)
    )
    assert r.accepted, r.refused
    np.testing.assert_allclose(r.rate, (10.0, 10.0, 10.0), atol=3.0)
    assert r.relative_to == "BBBB"
    assert "k:baseline_unmeasured" in r.flags and r.sigma_method == "default_k"


def test_an_eight_hour_velocity_uses_its_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import geo_dataread.products.loader as loader

    rng = np.random.default_rng(2)
    t = 2024.0 + (np.arange(3 * 60) + 0.5) * DAY / 3
    y = np.vstack([20.0 * (t - 2024.0)] * 3) + rng.normal(0, 1.0, (3, t.size))
    seen = {}

    def read(sta, start, end, source, *, doc):
        seen["pos_id"] = source.pos_id
        return StationView(sta, t, y, np.ones_like(y), "plate")

    monkeypatch.setattr(loader, "_read_one", read)
    monkeypatch.setattr(core, "_data_end", lambda *a, **k: None)
    s = core.VelocitySettings(ref="plate", session="8h", frame="itrf20")
    r = core.estimate_station_velocity(
        "AAAA", 2024.0, 2024.15, s, doc={"stations": {}}, coords=(0.0, 0.0)
    )
    assert seen["pos_id"] == "imo.08hr_itrf20"
    assert r.accepted and r.session == "8h" and r.series == "imo.08hr_itrf20"
    assert "k:session_unmeasured" in r.flags
    np.testing.assert_allclose(r.rate, (20.0, 20.0, 20.0), atol=2.0)


def test_velocity_help_documents_every_option(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        cli.main(["velocity", "--help"])
    out = capsys.readouterr().out
    for opt in (
        "--pre-window",
        "--no-record",
        "--allow-steps",
        "--min-obs",
        "--min-cover",
        "--no-short-tier",
        "--short-snr",
        "--default-k",
        "--k-table",
        "--save",
        "--relative-to",
        "--session",
        "--start",
        "--days",
    ):
        assert opt in out, opt  # fmt: skip
    assert "examples:" in out and "flags:" in out
    p = cli.build_parser()
    sub = p._subparsers._group_actions[0].choices["velocity"]  # type: ignore[union-attr]
    for action in sub._actions:
        if action.dest != "help":
            assert action.help, action.dest


def test_k_table_and_show_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    import geo_dataread.products.velocity_cli as vcli

    got = []
    monkeypatch.setattr(vcli, "main", lambda argv: got.append(argv) or 0)
    assert cli.main(["show", "set.jsonl", "--all"]) == 0
    assert cli.main(["k-table", "--frame", "itrf14"]) == 0
    assert got == [["show", "set.jsonl", "--all"], ["k-table", "--frame", "itrf14"]]
