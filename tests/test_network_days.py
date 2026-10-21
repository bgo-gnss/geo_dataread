"""Network days: the list file, its resolver and the cleaned view's mask."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from geo_dataread import gps_views
from geo_dataread import network_days as nd
from gps_analysis import OutlierParams

RUNNING = OutlierParams(method="running")


def _series(n: int = 400, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-01-01", periods=n, freq="D")
    return nd.noon_yearf(days), rng.normal(0.0, 1.0, (3, n))


def test_days_and_noon_yearf_round_trip() -> None:
    days = pd.date_range("2023-12-30", "2024-01-02", freq="D").append(
        pd.DatetimeIndex(["2024-12-31", "2026-09-15"])
    )
    back = nd._days(nd.noon_yearf(days))
    assert [str(d) for d in back] == [f"{d:%Y-%m-%d}" for d in days]


def test_doy365_epochs_land_on_their_day() -> None:
    """The .pos yearf is doy/365: HRAC's −42 mm epoch is 2026-09-15."""
    assert str(nd._days(np.array([2026.7055]))[0]) == "2026-09-15"


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-09-11", "2026-09-15", "2023-07-14"]),
            "status": ["decided", "decided", "partial"],
            "share": [1.0, 0.98, 0.45],
            "n_bad": [165, 166, 49],
            "n_voting": [165, 169, 108],
            "n_bad_north": [165, 1, 1],
            "n_bad_east": [165, 166, 0],
            "n_bad_up": [6, 0, 48],
        }
    )


def test_write_read_and_decided_epochs(tmp_path) -> None:
    path = nd.write_network_days(
        _table(), nd.network_days_path(tmp_path), n_stations=190
    )
    assert path.name == "network_days.itrf14.csv"
    assert path.read_text().startswith("# gps-network-days frame=itrf14 ")
    back = nd.read_network_days(path)
    assert back["status"].tolist() == ["decided", "decided", "partial"]
    ep = nd.network_epochs(path)
    assert len(ep) == 2 and nd._days(np.array(ep)).astype(str).tolist() == [
        "2026-09-11",
        "2026-09-15",
    ]
    assert len(nd.network_epochs(path, statuses=("decided", "partial"))) == 3
    assert not list(tmp_path.glob("*.tmp"))  # the atomic write leaves nothing


def test_resolver_switch(tmp_path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert gps_views.resolve_network_epochs(tmp_path) == ((), None)  # no list
    assert gps_views.resolve_network_epochs(False) == ((), None)
    assert gps_views.resolve_network_epochs([2026.5]) == ((2026.5,), "explicit")
    path = nd.write_network_days(_table(), nd.network_days_path(tmp_path))
    for arg in (tmp_path, path):
        ep, src = gps_views.resolve_network_epochs(arg)
        assert len(ep) == 2 and src == str(path)


def test_network_days_are_taken_out_then_flagged_in_every_component() -> None:
    t, y = _series()
    y[1, 200] += 8.0  # modest on this station: the network decides it
    flags, prov = gps_views.detect_view_outliers(
        t, y, outlier_params=RUNNING, network_epochs=t[[200]]
    )
    assert flags[:, 200].all() and prov["n_network"] == 1
    assert prov["network"][:, 200].all() and prov["network"].sum() == 3


def test_network_mode_keep_marks_only() -> None:
    t, y = _series()
    flags, prov = gps_views.detect_view_outliers(
        t, y, outlier_params=RUNNING, network_epochs=t[[200]], network_mode="keep"
    )
    assert not flags[:, 200].any() and prov["network"][:, 200].all()
    with pytest.raises(ValueError, match="network_mode"):
        gps_views.detect_view_outliers(
            t, y, network_epochs=t[[200]], network_mode="hide"
        )


def test_a_bad_day_no_longer_sways_its_neighbours() -> None:
    """A 40 mm network day next to a 9 mm blunder: out of detection, the
    blunder's side lines are clean and it is still flagged."""
    t, y = _series()
    y[1, 200] += 40.0
    y[1, 203] += 9.0
    flags, _ = gps_views.detect_view_outliers(
        t, y, outlier_params=RUNNING, network_epochs=t[[200]]
    )
    assert flags[1, 203]


def test_network_days_are_flagged_even_when_detection_fails() -> None:
    t, y = _series()
    bad = OutlierParams(method="running", max_flag_fraction=1e-9)
    y[:, 300] += 40.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flags, prov = gps_views.detect_view_outliers(
            t, y, outlier_params=bad, network_epochs=t[[100]]
        )
    assert prov["outlier_abort"] is True
    assert flags[:, 100].all() and flags.sum() == 3
