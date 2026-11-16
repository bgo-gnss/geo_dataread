"""The deprecated ``geo_dataread.sil_read`` shim (ticket T6, PLAN-seismicity.md).

Pins what the research scripts in ``gps_data_analyses`` use
(``tmp_multiploting/beta_multiplot*.py``, ``oraefajokull/beta_multiplot.py``):

    Seis = read_sil_data(start=dt_start, end=dt_end, base_path=..., fread="events.fps",
                         logging_level=logging.INFO, min_lat=..., max_lat=...,
                         min_lon=..., max_lon=...)
    Seis.index, Seis.MLW, Seis.cum_moment, max(Seis.MLW), max(Seis.cum_moment)

plus the legacy column sets and index layout, the deprecation warnings and
``MW`` with the legacy 6.03. Fixtures: ``tests/fixtures/sil`` (no mount).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from geo_dataread import sil_read

FIXTURES = Path(__file__).parent / "fixtures" / "sil"

FPS_COLUMNS = [
    "latitude",
    "Dlatitude",
    "longitude",
    "Dlongitude",
    "depth",
    "Ddepth",
    "MLW",
    "seismic_moment",
    "cum_moment",
]
DEFAULT_COLUMNS = [
    "latitude",
    "longitude",
    "depth",
    "MLW",
    "Q",
    "id",
    "seismic_moment",
    "MW",
    "cum_moment",
]
LIB_MAG_COLUMNS = [
    "latitude",
    "Dlatitude",
    "longitude",
    "Dlongitude",
    "depth",
    "Ddepth",
    "MLW",
    "ML",
]

#: Every fixture day: (reviewed, automatic-not-reviewed) rows of the new reader.
DAYS = {
    "1993-01-29": (0, 5),
    "1995-06-15": (4, 1),
    "2022-12-12": (4, 1),
    "2024-08-22": (8, 2),
    "2025-05-03": (6, 1),
    "2026-09-10": (0, 6),
    "2026-09-11": (0, 0),
}


def read(day: str, **kw: Any) -> pd.DataFrame:
    """read_sil_data over one fixture day, the way the research scripts call it."""
    start = datetime.fromisoformat(day)
    kw.setdefault("base_path", str(FIXTURES) + "/")
    kw.setdefault("logging_level", logging.INFO)
    with pytest.warns(DeprecationWarning, match="read_sil_catalog") as rec:
        out: pd.DataFrame = sil_read.read_sil_data(
            start=start, end=start + pd.Timedelta(days=1), **kw
        )
    assert [w.filename for w in rec] == [__file__]  # one warning, at the caller
    return out


# --------------------------------------------------------------------------
# read_sil_data: layout
# --------------------------------------------------------------------------


@pytest.mark.parametrize("day", list(DAYS))
def test_events_fps_layout(day: str) -> None:
    df = read(day, fread="events.fps")
    assert list(df.columns) == FPS_COLUMNS
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "date_time"
    assert df.index.tz is None
    assert df.index.is_monotonic_increasing
    assert len(df) == DAYS[day][0]
    assert all(df.index.normalize() == pd.Timestamp(day))
    for col in FPS_COLUMNS:
        assert df[col].dtype == np.float64, col


@pytest.mark.parametrize("day", list(DAYS))
def test_default_layout(day: str) -> None:
    df = read(day, aut_per="all")
    assert list(df.columns) == DEFAULT_COLUMNS
    assert df.index.name == "date_time"
    assert df.index.tz is None
    assert df.index.is_monotonic_increasing
    assert len(df) == sum(DAYS[day])
    assert (df["Q"] == sil_read.REVIEWED_Q).sum() == DAYS[day][0]
    assert df["id"].map(type).eq(str).all()


def test_lib_mag_layout() -> None:
    df = read("2024-08-22", fread="lib.mag")
    assert list(df.columns) == LIB_MAG_COLUMNS
    assert len(df) == 8


def test_known_events_fps_rows() -> None:
    df = read("2024-08-22", fread="events.fps")
    first = df.iloc[0]
    assert df.index[0] == pd.Timestamp("2024-08-22 00:10:04.454")  # UTC, naive
    assert first["latitude"] == 63.88752
    assert first["longitude"] == -22.37545
    assert first["depth"] == 2.111
    assert first["Ddepth"] == 0.6
    # Mlw at 0.01 from lib.mag/events.lib (events.fps writes 0.20)
    assert first["MLW"] == 0.25
    assert first["seismic_moment"] == 1.77e10
    assert math.isnan(first["Dlatitude"]) and math.isnan(first["Dlongitude"])
    big = df.loc[pd.Timestamp("2024-08-22 11:39:04.146")]
    assert (big["MLW"], big["seismic_moment"]) == (1.75, 5.65e11)


def test_events_fps_ddepth_from_lib_mag_1995() -> None:
    """events.fps writes Ddepth 242.5 as 42.5; the shim takes lib.mag's value."""
    df = read("1995-06-15", fread="events.fps")
    assert df.index[0] == pd.Timestamp("1995-06-15 01:07:33.724")
    assert df["Ddepth"].iloc[0] == 242.5


def test_cum_moment_is_running_total_without_nan() -> None:
    """The scripts do ``max(Seis.cum_moment)`` with the builtin max: no NaN."""
    for fread in ("events.fps", "default"):
        df = read("2024-08-22", fread=fread, aut_per="all")
        cum = df["cum_moment"].to_numpy()
        assert not np.isnan(cum).any()
        assert (np.diff(cum) >= 0).all()
        assert cum[-1] == pytest.approx(np.nansum(df["seismic_moment"]))
    assert max(df["MLW"]) == 1.75
    assert max(df["cum_moment"]) == pytest.approx(9.341e11)


def test_default_maps_automatic_rows() -> None:
    df = read("2024-08-22", aut_per="all")
    aut = df[df["Q"] != sil_read.REVIEWED_Q]
    assert list(aut["Q"]) == [10.0, 10.0]
    assert list(aut["MLW"]) == [-1.9, -2.1]  # automatic ML, as before
    assert aut["seismic_moment"].isna().all() and aut["MW"].isna().all()
    rev = df[df["Q"] == sil_read.REVIEWED_Q]
    assert rev["id"].iloc[0] == "/2024/aug/22/00:05:00/10:05:430"
    expected = 2.0 / 3.0 * np.log10(rev["seismic_moment"]) - 6.03
    np.testing.assert_allclose(rev["MW"], expected, rtol=0, atol=1e-12)


def test_default_aut_per_limits_automatic_events() -> None:
    start = datetime(2024, 8, 22)
    kw: dict[str, Any] = {"base_path": str(FIXTURES), "start": start}
    with pytest.warns(DeprecationWarning):
        near = sil_read.read_sil_data(end=datetime(2024, 8, 23), aut_per=30, **kw)
        far = sil_read.read_sil_data(end=datetime(2024, 10, 1), aut_per=30, **kw)
        none = sil_read.read_sil_data(end=datetime(2024, 8, 23), aut=False, **kw)
    assert len(near) == 10
    assert len(far) == 8
    assert len(none) == 8
    assert (far["Q"] == sil_read.REVIEWED_Q).all()


def test_start_as_days_before_end() -> None:
    with pytest.warns(DeprecationWarning):
        df = sil_read.read_sil_data(
            start=1, end="2024-08-23", base_path=str(FIXTURES), fread="events.fps"
        )
    assert len(df) == 8


def test_box_filter_is_inclusive_and_max_lon_alone_works() -> None:
    df = read("2024-08-22", fread="events.fps", min_lat=63.87, max_lon=-22.38)
    # 63.87000 is kept (legacy bounds were exclusive); max_lon works without
    # min_lon (legacy ignored it).
    assert list(df["latitude"]) == [63.87, 63.88285, 63.88013]
    assert (df["longitude"] <= -22.38).all()
    # cum_moment is computed after the cut
    assert df["cum_moment"].iloc[-1] == pytest.approx(df["seismic_moment"].sum())


def test_research_script_call() -> None:
    """The call of beta_multiplot*.py (Öræfajökull box), on fixture data."""
    constr = {
        "min_lat": 63.576,
        "max_lat": 63.690,
        "min_lon": -19.269,
        "max_lon": -19.035,
    }
    df = read("2024-08-22", fread="events.fps", **constr)
    assert list(df.columns) == FPS_COLUMNS and df.empty


def test_magnitude_and_q_filters() -> None:
    df = read("2024-08-22", aut_per="all", min_mlw=0.6, max_Q=99.99)
    assert (df["MLW"] >= 0.6).all() and len(df) == 6
    df = read("2024-08-22", aut_per="all", max_Q=50)
    assert list(df["Q"]) == [10.0, 10.0]


def test_rcolumns() -> None:
    df = read("2024-08-22", fread="events.fps", rcolumns=["MLW", "latitude", "nope"])
    assert list(df.columns) == ["MLW", "latitude"]  # order kept, no cum_moment
    df = read("2024-08-22", fread="events.fps", rcolumns="all")
    assert list(df.columns) == FPS_COLUMNS


def test_dna_ignores_placeholder_columns() -> None:
    df = read("2024-08-22", aut_per="all", dna=True)
    assert len(df) == 8  # automatic rows have no moment/MW
    # only the returned columns count: 07:54 has no ML, but ML is not returned
    assert pd.Timestamp("2024-08-22 07:54:42.806") in df.index
    df = read("2024-08-22", fread="events.fps", dna=True)
    assert len(df) == 8  # Dlatitude/Dlongitude are always NaN


def test_unknown_keyword_and_unsupported_fread() -> None:
    with pytest.warns(DeprecationWarning), pytest.raises(TypeError, match="min_lt"):
        sil_read.read_sil_data(start="2024-08-22", base_path=str(FIXTURES), min_lt=1.0)
    for fread in ("events.lib", "cleanaut.mag"):
        with (
            pytest.warns(DeprecationWarning),
            pytest.raises(NotImplementedError, match="read_sil_file"),
        ):
            sil_read.read_sil_data(
                start="2024-08-22", base_path=str(FIXTURES), fread=fread
            )


def test_frfile_reads_and_cuts(tmp_path: Path) -> None:
    src = read("2024-08-22", fread="events.fps", cum_moment=False)
    path = tmp_path / "sil-data.pik"
    src.to_pickle(path)
    with pytest.warns(DeprecationWarning):
        df = sil_read.read_sil_data(
            start="2024-08-22 07:00",
            end="2024-08-23",
            frfile=True,
            fname=str(path),
            fread="events.fps",
            max_lon=-22.38,
        )
    assert list(df.index.strftime("%H:%M")) == ["07:36", "07:54", "21:26"]
    assert "cum_moment" in df


def test_base_path_required(monkeypatch: pytest.MonkeyPatch) -> None:
    import geo_dataread.seismic.sil_catalog as sc

    monkeypatch.delenv("GPS_SIL_DIR", raising=False)
    monkeypatch.setattr(sc, "_config_sil_dir", lambda: None)
    with pytest.warns(DeprecationWarning), pytest.raises(FileNotFoundError):
        sil_read.read_sil_data(start="2024-08-22", end="2024-08-23")


def test_base_path_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPS_SIL_DIR", str(FIXTURES))
    with pytest.warns(DeprecationWarning):
        df = sil_read.read_sil_data(
            start="2024-08-22", end="2024-08-23", fread="events.fps"
        )
    assert len(df) == 8


# --------------------------------------------------------------------------
# seism_subset
# --------------------------------------------------------------------------


@pytest.fixture
def legacy() -> pd.DataFrame:
    return read("2024-08-22", aut_per="all")


def test_seism_subset_warns_and_keeps_layout(legacy: pd.DataFrame) -> None:
    with pytest.warns(DeprecationWarning, match="geo_dataread.seismic.subset") as rec:
        out = sil_read.seism_subset(legacy, min_depth=5.0)
    assert rec[0].filename == __file__
    assert list(out.columns) == list(legacy.columns)
    assert out.index.name == "date_time" and out.index.tz is None
    assert list(out["depth"]) == [5.344, 6.124, 5.2, 5.2, 6.073]


def test_seism_subset_time_bounds_inclusive(legacy: pd.DataFrame) -> None:
    t = legacy.index[3]
    with pytest.warns(DeprecationWarning):
        out = sil_read.seism_subset(legacy, start=t, end=t)
    assert list(out.index) == [t]


def test_seism_subset_empty_result_does_not_raise(legacy: pd.DataFrame) -> None:
    with pytest.warns(DeprecationWarning):
        out = sil_read.seism_subset(legacy, min_mlw=9.0)
    assert out.empty and list(out.columns) == list(legacy.columns)
    with pytest.warns(DeprecationWarning):
        again = sil_read.seism_subset(out, max_lon=0.0)
    assert again.empty


def test_seism_subset_ml_fallback_and_input_untouched(legacy: pd.DataFrame) -> None:
    frame = legacy.rename(columns={"MLW": "ML"})
    before = frame.copy()
    with pytest.warns(DeprecationWarning):
        out = sil_read.seism_subset(frame, max_mlw=0.0)
    assert list(out["ML"]) == [-1.9, -2.1]
    pd.testing.assert_frame_equal(frame, before)


def test_seism_subset_without_q_ignores_q_bounds(legacy: pd.DataFrame) -> None:
    with pytest.warns(DeprecationWarning):
        out = sil_read.seism_subset(legacy.drop(columns="Q"), min_Q=50)
    assert len(out) == len(legacy)


# --------------------------------------------------------------------------
# MW
# --------------------------------------------------------------------------


def test_mw_uses_legacy_constant() -> None:
    with pytest.warns(DeprecationWarning, match="moment_magnitude") as rec:
        mw = sil_read.MW(1.77e10)
    assert rec[0].filename == __file__  # attributed to the caller (stacklevel)
    assert isinstance(mw, float)
    assert mw == pytest.approx(2.0 / 3.0 * math.log10(1.77e10) - 6.03, abs=1e-12)
    # PLAN §2.1: an event with Mlw 2.02 has MW 1.99
    with pytest.warns(DeprecationWarning):
        assert sil_read.MW(10 ** (1.5 * (1.99 + 6.03))) == pytest.approx(1.99)


def test_mw_array_and_series() -> None:
    with pytest.warns(DeprecationWarning):
        arr = sil_read.MW(np.array([1e9, 0.0, np.nan]))
        ser = sil_read.MW(pd.Series([1e9, 1e12], index=["a", "b"], name="m0"))
    assert isinstance(arr, np.ndarray)
    assert arr[0] == pytest.approx(6.0 - 6.03) and np.isnan(arr[1:]).all()
    assert isinstance(ser, pd.Series) and list(ser.index) == ["a", "b"]
    assert ser["b"] == pytest.approx(8.0 - 6.03)


# --------------------------------------------------------------------------
# Retired functions
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "args", "replacement"),
    [
        ("read_sil_file", (str(FIXTURES / "2024/aug/22/events.fps"),), "read_sil_file"),
        ("read_automatic", (str(FIXTURES / "2024/aug/22"),), "read_sil_day"),
        ("removedubl", (pd.DataFrame(),), "read_sil_catalog"),
    ],
)
def test_retired_functions_raise_naming_replacement(
    name: str, args: tuple[Any, ...], replacement: str
) -> None:
    with (
        pytest.warns(DeprecationWarning, match=replacement),
        pytest.raises(NotImplementedError, match=replacement),
    ):
        getattr(sil_read, name)(*args)


def test_module_has_docstring() -> None:
    assert sil_read.__doc__ and "read_sil_catalog" in sil_read.__doc__
