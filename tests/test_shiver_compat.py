"""Pin the ``geo_dataread.sil_read`` calls made by ``UR/shiver`` (git.vedur.is).

``shiver`` (a 2020–2021 seismic-analysis collection; the Mpgv page on
``hraun.vedur.is/ja/Mpgv/`` comes from the same student project) imports the
legacy module directly, so its exact call shapes must keep working through the
T6 shim and the later T7 alert move (PLAN-seismicity.md §9d, B1):

- ``shiver/u_SIL.py`` ``silcat_by_conf``: ``read_sil_data`` with
  ``fread="events.fps"``, ``rcolumns="all"``, ``cum_moment`` and the
  ``min_``/``max_`` limits; its file+archive merge reads a saved frame with
  ``frfile=True, fname=...`` and appends the archive from the last day.
- ``scripts/tests_alertmap.py``: ``find_alert(event_id, sil_path=...,
  pre_path=..., eps=30, logger=...)`` and ``X.MW_pgv.median()``; the column
  comes from ``dist_pgv.apply(M_PGV, axis=1)`` over ``[dist, pgv]`` rows.

Shiver revision checked: master 2021-02-03 ("bc2mseed works now ...").
"""

from __future__ import annotations

import inspect
import math
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from geo_dataread import sil_read

FIXTURES = Path(__file__).parent / "fixtures" / "sil"

#: The column set ``silcat_by_conf`` documents for ``fread="events.fps"``.
SHIVER_FPS_COLUMNS = [
    "latitude",
    "Dlatitude",
    "longitude",
    "Dlongitude",
    "depth",
    "Ddepth",
    "MLW",
    "seismic_moment",
]


def _silcat_kwargs(**over: Any) -> dict[str, Any]:
    """The keyword set ``shiver.u_SIL.silcat_by_conf`` builds (archive branch)."""
    kw: dict[str, Any] = {
        "start": "2024-08-22",
        "end": "2024-08-23",
        "fname": None,
        "frfile": False,
        "base_path": str(FIXTURES),
        "fread": "events.fps",
        "rcolumns": "all",
        "cum_moment": False,
        "min_lat": None,
        "max_lat": None,
        "min_lon": None,
        "max_lon": None,
        "min_depth": None,
        "max_depth": None,
        "min_mlw": None,
        "max_mlw": None,
        "logging_level": None,
    }
    kw.update(over)
    return kw


def _read(**kw: Any) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        out: pd.DataFrame = sil_read.read_sil_data(**kw)
    return out


def test_silcat_by_conf_call_returns_documented_columns() -> None:
    df = _read(**_silcat_kwargs())
    assert list(df.columns) == SHIVER_FPS_COLUMNS
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) > 0
    assert df.index.is_monotonic_increasing


def test_silcat_by_conf_cum_moment_and_limits() -> None:
    df = _read(**_silcat_kwargs(cum_moment=True, min_mlw=1.0, max_depth=10.0))
    assert list(df.columns) == [*SHIVER_FPS_COLUMNS, "cum_moment"]
    assert (df["MLW"] >= 1.0).all()
    assert (df["depth"] <= 10.0).all()
    assert df["cum_moment"].is_monotonic_increasing


def test_file_then_archive_merge(tmp_path: Path) -> None:
    """The ``u_SIL`` merge: saved frame (``frfile``) + archive from the last day."""
    full = _read(**_silcat_kwargs())
    pik = tmp_path / "silcat.pik"
    full.to_pickle(pik)
    x = _read(**_silcat_kwargs(fname=str(pik), frfile=True, base_path=None, end=None))
    assert list(x.columns) == SHIVER_FPS_COLUMNS
    assert len(x) == len(full)
    s_lastday = x.index[-1].strftime("%Y-%m-%d")
    y = _read(**_silcat_kwargs(start=s_lastday, end="2024-08-23"))
    assert list(y.columns) == SHIVER_FPS_COLUMNS


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("find_alert", ["event_id", "sil_path", "pre_path", "eps", "logger"]),
        ("M_PGV", ["x", "mu"]),
        ("pgverror", ["r", "M"]),
    ],
)
def test_alert_signatures_pinned(name: str, params: list[str]) -> None:
    func = getattr(sil_read, name)
    assert list(inspect.signature(func).parameters)[: len(params)] == params


@pytest.mark.parametrize(
    "name",
    [
        "M_PGV",
        "pgverror",
        "read_evb",
        "read_alert",
        "downl_PGV_hraun",
        "find_alert",
        "read_station_loc",
        "check_event_fit",
        "find_alertevent",
    ],
)
def test_nine_alert_names_exist(name: str) -> None:
    assert callable(getattr(sil_read, name))


def test_m_pgv_row_apply_value() -> None:
    """``dist_pgv.apply(M_PGV, axis=1)`` over ``[dist, pgv]`` (VÍ 2009-012 eq. C)."""
    rows = pd.DataFrame({"dist": [50.0], "pgv": [1000.0]})
    got = rows.apply(sil_read.M_PGV, axis=1).iloc[0]
    want = (1.69 * math.log10(50.0) + math.log10(1e-6 * 1000.0) + 4.96) / 1.05
    assert got == pytest.approx(want, abs=1e-12)
