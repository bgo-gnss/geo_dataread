"""gpsprod --common-mode: the daily network common mode on the read path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import geo_dataread.gps_views as gv
from geo_dataread.products import Source, cli, loader
from geo_dataread.products.velocity_core import (
    VelocitySettings,
    _fingerprint_settings,
    common_mode_flag,
)

T = np.array([2024.0014, 2024.0041, 2024.0068])  # three noons


@pytest.fixture
def fake_read(monkeypatch: pytest.MonkeyPatch) -> None:
    def read_gps_view(sta, start, end, **kw):
        df = pd.DataFrame(
            {"yearf": T, **{c: [10.0, 20.0, 30.0] for c in ("north", "east", "up")},
             **{f"D{c}": [1.0] * 3 for c in ("north", "east", "up")}}
        )  # fmt: skip
        df.attrs["gps_view"] = {}
        return df

    # the common mode on the first two days only (NaN: not estimated)
    cm = np.array([[1.0, 2.0, np.nan], [-1.0, -2.0, np.nan], [3.0, 4.0, np.nan]])
    monkeypatch.setattr(gv, "read_gps_view", read_gps_view)
    monkeypatch.setattr(gv, "resolve_common_mode", lambda d: (T, cm, "/pos/cm.csv"))
    monkeypatch.setattr(Source, "directory", lambda self: "/pos")


def _read(**kw) -> loader.StationView:
    return loader._read_one("AAAA", 2024.0, 2024.1, Source(ref="plate", view="raw", **kw), doc=None)  # fmt: skip


def test_off_by_default_leaves_the_series(fake_read: None) -> None:
    v = _read()
    assert np.array_equal(v.y[0], [10.0, 20.0, 30.0])
    assert not any(f.startswith("common_mode") for f in v.flags)


def test_subtracted_on_the_days_it_was_estimated(fake_read: None) -> None:
    v = _read(common_mode=True)
    assert np.allclose(v.y[0], [9.0, 18.0, 30.0])  # day 3: no estimate, kept
    assert np.allclose(v.y[1], [11.0, 22.0, 30.0])
    assert np.allclose(v.y[2], [7.0, 16.0, 30.0])
    assert "common_mode" in v.flags
    assert v.attrs["common_mode"] == {"source": "/pos/cm.csv", "n_days": 2}
    assert v.y_raw is not None and np.allclose(v.y_raw, v.y)  # the view's input


@pytest.mark.parametrize("kw", [{"frame": "itrf20"}, {"session": "8h"}])
def test_other_solutions_have_none(fake_read: None, kw: dict) -> None:
    v = _read(common_mode=True, **kw)
    assert np.array_equal(v.y[0], [10.0, 20.0, 30.0])
    assert "common_mode:none" in v.flags


def test_no_file_flags_none(monkeypatch: pytest.MonkeyPatch, fake_read: None) -> None:
    monkeypatch.setattr(gv, "resolve_common_mode", lambda d: (np.zeros(0), np.zeros((3, 0)), None))  # fmt: skip
    assert "common_mode:none" in _read(common_mode=True).flags


def test_cli_defaults_per_product() -> None:
    p = cli.build_parser()
    disp = p.parse_args(["displacement", "X", "--days", "3"])
    assert cli._source(disp).common_mode is True
    off = p.parse_args(["displacement", "X", "--days", "3", "--common-mode", "off"])
    assert cli._source(off).common_mode is False
    rate = p.parse_args(["displacement", "X", "--days", "3", "--method", "rate"])
    assert cli._source(rate).common_mode is False  # = gpsprod velocity
    vel = p.parse_args(["velocity", "X", "--days", "30"])
    assert cli.velocity_settings(vel).common_mode is False
    vel_cm = p.parse_args(
        ["velocity", "X", "--days", "30", "--common-mode", "subtract"]
    )
    assert cli.velocity_settings(vel_cm).common_mode is True
    assert p.parse_args(["transient", "X", "--days", "30"]).common_mode == "subtract"
    assert p.parse_args(["series", "X"]).common_mode == "off"


def test_velocity_fingerprint_unchanged_at_the_default() -> None:
    assert "common_mode" not in _fingerprint_settings(VelocitySettings())
    assert _fingerprint_settings(VelocitySettings(common_mode=True))["common_mode"]


def test_record_flag() -> None:
    on = VelocitySettings(common_mode=True)
    assert common_mode_flag(VelocitySettings(), {"common_mode": {"n_days": 1}}) is None
    assert common_mode_flag(on, {"common_mode": {"n_days": 1}}) == "common_mode"
    assert common_mode_flag(on, {"common_mode": None}) == "common_mode:none"
    assert common_mode_flag(on, {"product_flags": ["common_mode"]}) == "common_mode"
