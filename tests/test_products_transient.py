"""gpsprod transient: the rate on a date from a fitted recharge curve."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import geo_dataread.products.transient as tr
from geo_dataread.products.velocity_core import VelocityRecord, VelocitySettings

DAY = 1.0 / 365.25
T0 = 2025.6
TAU = 0.24
# per station: (steady rate N/E/U, transient amplitude phi N/E/U) [mm/yr, mm]
TRUTH = {
    "AAAA": ((80.0, -30.0, 110.0), (60.0, -50.0, 130.0)),
    "BBBB": ((-10.0, -60.0, 115.0), (-25.0, -110.0, 140.0)),
    "CCCC": ((-25.0, 45.0, 70.0), (-50.0, 60.0, 40.0)),
    "DDDD": ((3.0, -10.0, 4.0), (0.0, 0.0, 0.0)),  # far: no transient
}
STEP = {"EEEE": 2026.1}  # a station with a declared step of 20 mm


def _series(sta: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(sum(ord(c) for c in sta))
    t = T0 + (np.arange(int(1.2 * 365.25)) + 0.5) * DAY
    v, phi = TRUTH.get(sta, TRUTH["AAAA"])
    dt_ = t - T0
    y = np.vstack([v[c] * dt_ + phi[c] * (1 - np.exp(-dt_ / TAU)) for c in range(3)])
    if sta in STEP:
        y += np.where(t > STEP[sta], 20.0, 0.0)
    y += rng.normal(0.0, 1.5, y.shape)
    s = np.full(y.shape, 1.5)
    keep = rng.random(t.size) > 0.05
    return t[keep], y[:, keep], s[:, keep]


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> None:
    def read(
        sta: str, start: float, end: float, settings: Any, *, terms: Any, doc: Any
    ):
        if sta == "SHRT":
            t, y, s = _series("AAAA")
            k = t < T0 + 40 * DAY
            return t[k], y[:, k], s[:, k], {}
        t, y, s = _series(sta)
        k = (t >= start) & (t < end)
        return t[k], y[:, k], s[:, k], {}

    import geo_dataread.gps_views as gv
    import geo_dataread.station_catalog as sc

    monkeypatch.setattr(tr, "_read", read)
    monkeypatch.setattr(tr, "_params_doc", lambda settings: {"stations": {}})
    monkeypatch.setattr(tr, "_coords", lambda sta: (-22.4, 63.9))
    monkeypatch.setattr(tr, "_data_end", lambda *a, **k: None)
    monkeypatch.setattr(tr, "_software", lambda: {"geo_dataread": "t"})
    monkeypatch.setattr(sc, "station_plate", lambda sta: "NOAM")
    monkeypatch.setattr(
        gv,
        "station_step_epochs",
        lambda sta, steps=None: (
            np.array([STEP[sta]]) if sta in STEP else np.zeros(0),
            "x",
        ),
    )


def _rate_truth(sta: str, at: float) -> np.ndarray:
    v, phi = TRUTH[sta]
    return np.array(
        [v[c] + phi[c] / TAU * math.exp(-(at - T0) / TAU) for c in range(3)]
    )


T1 = T0 + 1.2
AT = [T0 + 0.1, T0 + 1.1]


def test_shared_decay_and_rates(fake: None) -> None:
    recs, decay = tr.estimate_transients(list(TRUTH), T0, T1, AT)
    assert decay.source == "shared"
    assert abs(decay.tau - TAU) < 4 * decay.tau_sigma
    assert decay.stations == tuple(sorted(TRUTH))
    assert len(recs) == len(TRUTH) * len(AT)
    for r in recs:
        assert r.accepted and r.kind == "transient" and r.model == "exp_linear"
        assert r.at in AT
        err = np.abs(np.array(r.rate) - _rate_truth(r.station, r.at))
        assert np.all(err < 4 * np.array(r.sigma) + 1.0), (r.station, err, r.sigma)
        assert "sigma:lag1" in r.flags and r.sigma_method == "lag1_residual"
        assert all(k >= 1.0 for k in r.k)
        assert all(a >= b for a, b in zip(r.sigma, r.sigma_formal, strict=True))
        assert r.transient is not None
        np.testing.assert_allclose(r.transient["v"], TRUTH[r.station][0], atol=8)


def test_station_depends_only_on_decay(fake: None) -> None:
    """With the run's decay given, a station alone gives the same numbers."""
    recs, decay = tr.estimate_transients(list(TRUTH), T0, T1, AT)
    alone, d2 = tr.estimate_transients(["CCCC"], T0, T1, AT, decay=decay)
    assert d2 is decay
    want = [r for r in recs if r.station == "CCCC"]
    for a, b in zip(alone, want, strict=True):
        assert a.rate == b.rate and a.sigma == b.sigma
        assert a.k_source["decay_fingerprint"] == decay.fingerprint


def test_decay_file_round_trip(fake: None, tmp_path: Path) -> None:
    _, decay = tr.estimate_transients(list(TRUTH), T0, T1, AT)
    p = tr.write_decay(decay, tmp_path / "x.decay.json")
    back = tr.read_decay(p)
    assert back.tau == decay.tau and back.stations == decay.stations
    assert back.fingerprint == decay.fingerprint
    assert back.source == str(p)
    assert json.loads(p.read_text())["half_life_days"] == pytest.approx(
        math.log(2) * decay.tau * 365.25, abs=1e-3
    )
    assert p.read_bytes() == tr.write_decay(back, tmp_path / "y.json").read_bytes()


def test_fixed_half_life_and_t0_guard(fake: None) -> None:
    d = tr.Decay(t0=T0, tau=tr.tau_from_half_life(60.0), tau_sigma=0.0)
    recs, out = tr.estimate_transients(["AAAA"], T0, T1, AT, decay=d)
    assert out is d and all(r.accepted for r in recs)
    assert tr.half_life_days(d.tau) == pytest.approx(60.0)
    with pytest.raises(ValueError, match="t0"):
        tr.estimate_transients(["AAAA"], T0 + 0.1, T1, AT, decay=d)


def test_refusals_and_steps(fake: None) -> None:
    recs, _ = tr.estimate_transients([*TRUTH, "SHRT", "EEEE"], T0, T1, [AT[1]])
    by = {r.station: r for r in recs}
    assert by["SHRT"].refused and "short" in by["SHRT"].refused
    assert by["EEEE"].refused and by["EEEE"].refused.startswith("steps_in_window")
    settings = VelocitySettings(allow_steps=True)
    recs, decay = tr.estimate_transients([*TRUTH, "EEEE"], T0, T1, [AT[1]], settings)
    e = next(r for r in recs if r.station == "EEEE")
    assert e.accepted and "steps_in_window" in e.flags
    assert e.transient is not None and e.transient["steps"] == [STEP["EEEE"]]
    assert "EEEE" not in decay.stations  # a stepped station sits out of tau
    err = np.abs(np.array(e.rate) - _rate_truth("AAAA", AT[1]))
    assert np.all(err < 4 * np.array(e.sigma) + 1.0)


def test_default_at_is_newest_data_and_extrapolation_flag(fake: None) -> None:
    recs, _ = tr.estimate_transients(["AAAA", "BBBB"], T0, T1)
    last = max(_series(s)[0].max() for s in ("AAAA", "BBBB"))
    assert {r.at for r in recs} == {last}
    recs, _ = tr.estimate_transients(["AAAA"], T0, T1, [last + 10 * DAY])
    assert any(f.startswith("extrapolated:") for f in recs[0].flags)


def test_json_fields_only_on_transient_records(fake: None) -> None:
    recs, _ = tr.estimate_transients(["AAAA", "BBBB"], T0, T1, [AT[1]])
    d = recs[0].to_dict()
    assert d["at"] == AT[1] and d["transient"]["tau"] > 0
    back = VelocityRecord.from_dict(json.loads(json.dumps(d)))
    assert back.at == AT[1] and back.transient == json.loads(json.dumps(d))["transient"]
    plain = VelocityRecord.from_dict({**d, "at": None, "transient": None}).to_dict()
    assert "at" not in plain and "transient" not in plain


def test_cli_help_and_dates() -> None:
    from geo_dataread.products import transient_cli
    from geo_dataread.products.cli import build_parser

    p = build_parser()
    args = p.parse_args(
        ["transient", "SENG", "--start", "20250806", "--at", "20260101,20261025",
         "--half-life", "60"]
    )  # fmt: skip
    assert args.command == "transient" and args.half_life == 60
    assert [d.isoformat() for d in transient_cli._dates(args.at)] == [
        "2026-01-01", "2026-10-25",
    ]  # fmt: skip
    with pytest.raises(SystemExit):
        p.parse_args(["transient", "SENG", "--half-life", "60", "--decay", "x.json"])
    sub = next(a for a in p._actions if a.dest == "command")
    text = sub.choices["transient"].format_help()  # type: ignore[attr-defined]
    for word in (
        "--half-life",
        "--decay",
        "tau_differs",
        "after:EVENT",
        "extrapolated",
    ):
        assert word in text
