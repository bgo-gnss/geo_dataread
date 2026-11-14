"""The nine legacy ``sil_read`` alert names over the T7c code (B1).

Shapes are the legacy ones; ``tests/test_shiver_compat.py`` pins shiver's
calls. No network, no mount: sources point at ``tests/fixtures/alert``.
"""

from __future__ import annotations

import datetime as dt
import math
import os
import warnings
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geo_dataread import sil_read
from geo_dataread.seismic import alert_source as src
from geo_dataread.seismic import quakes_api as qa
from geo_dataread.seismic import sil_alert as sa

ALERT = Path(__file__).parent / "fixtures" / "alert"
SIL = ALERT / "sil"
GOLDEN_ID = "/2026/nov/12/21:50:00/53:43:797"
GOLDEN_EVB = str(SIL) + GOLDEN_ID + ".evb"
GOLDEN_EVENT = str(ALERT / "2026/nov/12/076.event")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for var in (
        src.ENV_ALERTMAP_DIR,
        src.ENV_ALERTMAP_URL,
        src.ENV_ALERTMAP_CACHE,
        src.ENV_ALERTMAP_FALLBACK,
        "GPS_SIL_DIR",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(src, "_config_alertmap_dir", lambda: None)

    def _blocked(self: object, url: str, **_: object) -> None:
        raise AssertionError(f"network access in an offline test: {url}")

    monkeypatch.setattr(qa.UrllibSession, "get", _blocked)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        yield


def test_every_name_warns() -> None:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sil_read.read_alert(GOLDEN_EVENT)
        sil_read.read_evb(GOLDEN_EVB)
        sil_read.pgverror(10.0, 5.0)
        sil_read.M_PGV([50.0, 1000.0])
    assert sum(issubclass(w.category, DeprecationWarning) for w in rec) == 4
    assert all("T7c" in str(w.message) for w in rec)


def test_pgverror_and_m_pgv_values_and_types() -> None:
    want = (math.log10(10 + 0.00299 * 10 ** (0.621 * 5)) - 1.0) / 1.0
    got = sil_read.pgverror(10.0, 5.0)
    assert np.ndim(got) == 0 and got == pytest.approx(want, rel=1e-12)
    r = pd.Series([10.0, 50.0], index=["a", "b"])
    s = sil_read.pgverror(r, 5.0)
    assert isinstance(s, pd.Series) and list(s.index) == ["a", "b"]
    arr = sil_read.pgverror(np.array([10.0, 50.0]), 4.0)
    assert isinstance(arr, np.ndarray) and arr.shape == (2,)
    m = sil_read.M_PGV(pd.Series([50.0, 1000.0], index=["dist", "pgv"]))
    assert m == pytest.approx(
        (1.69 * math.log10(50.0) + math.log10(1e-3) + 4.96) / 1.05, abs=1e-12
    )
    assert math.isnan(sil_read.M_PGV([0.0, 1000.0]))
    assert sil_read.M_PGV([50.0, 1.0], mu=1.0) == pytest.approx(
        (1.69 * math.log10(50.0) + 4.96) / 1.05
    )


def test_read_alert_legacy_shape() -> None:
    df = sil_read.read_alert(GOLDEN_EVENT)
    assert list(df.columns) == ["lon", "lat", "time", "pgv", "fb"]
    assert df.index.name == "sta" and len(df) == 66 and df.index.is_unique
    assert all(df[c].dtype == np.float64 for c in df.columns)
    assert df.loc["von", "time"] == pytest.approx(2.47)
    assert df.loc["von", "pgv"] == pytest.approx(3432.93)


def test_read_evb_legacy_shape() -> None:
    df = sil_read.read_evb(GOLDEN_EVB)
    assert list(df.columns) == [
        "arr.", "time", "res.", "weight", "dist", "azi", "az.obs", "err", "number",
    ]  # fmt: skip
    assert df.index.name == "sta" and len(df) == 33 and df.index.is_unique  # p only
    assert isinstance(df["time"].iloc[0], dt.time)
    assert df.loc["dyn", "dist"] == pytest.approx(16.3)
    ps = sil_read.read_evb(str(SIL / "2024/aug/22/21:55:00/59:59:641.evb"))
    assert not ps.index.is_unique and set(ps["arr."]) == {"p", "s"}  # shared index


def test_check_event_fit_legacy_contract() -> None:
    pgv = sil_read.read_alert(GOLDEN_EVENT)
    ok, out = sil_read.check_event_fit(
        [64.653, -17.481],
        dt.datetime(2026, 11, 12, 21, 53, 40, 500000),
        dt.datetime(2026, 11, 12, 21, 53, 45),
        pgv,
    )
    assert ok is True
    assert "distance" in pgv.columns  # legacy side effect kept
    assert out["fit"].iloc[0] == "61/66 (92%) "
    assert out.loc["von", "time"] == pd.Timestamp("2026-11-12 21:53:47.47")
    assert out["distance"].between(0, 400).all()
    small = sil_read.read_alert(str(ALERT / "2026/nov/12/001.event"))
    ok2, out2 = sil_read.check_event_fit(
        (64.653, -17.481),
        dt.datetime(2026, 11, 12, 21, 53, 40),
        dt.datetime(2026, 11, 12, 0, 0, 57),
        small,
    )
    assert ok2 is False and "fit" not in out2.columns


def test_find_alert_golden_shape() -> None:
    df = sil_read.find_alert(
        GOLDEN_ID, sil_path=str(SIL), source=src.DirectoryAlertSource(ALERT)
    )
    assert list(df.columns) == ["dist", "pgv", "MW_pgv"]
    assert df.index.name == "sta" and df.index.is_unique
    assert df["dist"].is_monotonic_increasing
    assert len(df) == 29
    assert df.MW_pgv.median() == pytest.approx(4.134124, abs=1e-6)
    expect = df.apply(sil_read.M_PGV, axis=1)
    np.testing.assert_allclose(df["MW_pgv"], expect, rtol=0, atol=1e-12)


def test_find_alert_one_row_per_station_vs_legacy_join() -> None:
    """The legacy join repeated stations with an S pick; the shim does not.

    The golden evb has p picks only, so there the two agree exactly; the
    duplication is shown on a synthetic frame with one doubled station.
    """
    df = sil_read.find_alert(
        GOLDEN_ID, sil_path=str(SIL), source=src.DirectoryAlertSource(ALERT)
    )
    evb = sil_read.read_evb(GOLDEN_EVB)
    pgv = sil_read.read_alert(GOLDEN_EVENT)
    common = list(set(pgv.index) & set(evb.index))
    old = evb.loc[common, ["dist"]].join(pgv.loc[common, ["pgv"]])
    old["MW_pgv"] = old.apply(sil_read.M_PGV, axis=1)
    assert len(old) == len(df)
    assert old.MW_pgv.median() == pytest.approx(df.MW_pgv.median(), abs=1e-12)
    doubled = pd.concat([evb, evb.loc[["dyn"]].assign(**{"arr.": "s"})])
    joined = doubled.loc[common, ["dist"]].join(pgv.loc[common, ["pgv"]])
    assert len(joined) == len(df) + 1


def test_find_alert_none_and_errors(tmp_path: Path) -> None:
    empty = tmp_path / "alert"
    (empty / "2026/nov/12").mkdir(parents=True)
    assert (
        sil_read.find_alert(
            GOLDEN_ID, sil_path=str(SIL), source=src.DirectoryAlertSource(empty)
        )
        is None
    )
    with pytest.raises(FileNotFoundError):  # no evb: raises, as before
        sil_read.find_alert(GOLDEN_ID, sil_path=str(tmp_path))
    with pytest.raises(ValueError, match="inside the data directory"):
        sil_read.find_alert(GOLDEN_ID, sil_path=str(SIL), pre_path=str(SIL / "x"))
    with pytest.raises(ValueError, match="looks like a data tree"):
        sil_read.find_alert(GOLDEN_ID, sil_path=str(SIL), pre_path=str(ALERT))


def test_find_alert_source_failures_raise(tmp_path: Path) -> None:
    """A source that cannot be read is an error, never a silent None (B2)."""
    root = tmp_path / "alert"
    root.mkdir()
    (root / "2026").symlink_to(tmp_path / "unreachable")
    with pytest.raises(src.AlertYearUnavailable):
        sil_read.find_alert(
            GOLDEN_ID, sil_path=str(SIL), source=src.DirectoryAlertSource(root)
        )
    with pytest.raises(src.AlertYearUnavailable):
        sil_read.find_alertevent(GOLDEN_ID, sil_path=str(SIL), pre_path=str(root))


def test_find_alert_pre_path_is_http_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[src.HttpAlertSource] = []
    real = src.read_alert_pgv

    def spy(*a: object, **kw: object) -> object:
        s = kw["source"]
        assert isinstance(s, src.HttpAlertSource)
        seen.append(s)
        return real(*a, **{**kw, "source": src.DirectoryAlertSource(ALERT)})  # type: ignore[arg-type]

    monkeypatch.setattr(src, "read_alert_pgv", spy)
    out = sil_read.find_alert(GOLDEN_ID, sil_path=str(SIL), pre_path=str(tmp_path))
    assert out is not None and seen[0].cache_dir == tmp_path.resolve()
    # The legacy default "/tmp" means the configured source.
    monkeypatch.setenv(src.ENV_ALERTMAP_DIR, str(ALERT))
    monkeypatch.setattr(src, "read_alert_pgv", real)
    assert len(sil_read.find_alert(GOLDEN_ID, sil_path=str(SIL))) == 29


def test_find_alert_resolves_sil_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPS_SIL_DIR", str(SIL))
    monkeypatch.setenv(src.ENV_ALERTMAP_DIR, str(ALERT))
    df = sil_read.find_alert(GOLDEN_ID)  # legacy default /mnt_data/sildata
    assert df is not None and len(df) == 29


def test_find_alertevent_with_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    out = sil_read.find_alertevent(
        GOLDEN_ID,
        sil_path=str(SIL),
        pre_path=str(ALERT),
        time=dt.datetime(2026, 11, 12, 21, 53, 40, 500000),
        location=[64.653, -17.481],
    )
    assert out["event"].iloc[0] == "076" and out["fit"].iloc[0] == "61/66 (92%) "
    assert {"lon", "lat", "time", "pgv", "fb", "distance", "fit", "event"} == set(
        out.columns
    )
    plain = sil_read.find_alertevent(GOLDEN_ID, sil_path=str(SIL), pre_path=str(ALERT))
    assert (
        list(plain.columns) == ["lon", "lat", "time", "pgv", "fb"] and len(plain) == 66
    )
    # Legacy defaults resolve from configuration.
    monkeypatch.setenv("GPS_SIL_DIR", str(SIL))
    monkeypatch.setenv(src.ENV_ALERTMAP_DIR, str(ALERT))
    assert len(sil_read.find_alertevent(GOLDEN_ID)) == 66


def test_find_alertevent_missing_day_dir(tmp_path: Path) -> None:
    root = tmp_path / "alert"
    (root / "2026/nov").mkdir(parents=True)  # the year, but no day directory
    out = sil_read.find_alertevent(GOLDEN_ID, sil_path=str(SIL), pre_path=str(root))
    assert isinstance(out, pd.DataFrame) and out.empty


def test_find_alertevent_none(tmp_path: Path) -> None:
    root = tmp_path / "alert"
    (root / "2026/nov/12").mkdir(parents=True)
    out = sil_read.find_alertevent(
        GOLDEN_ID,
        sil_path=str(SIL),
        pre_path=str(root),
        time=dt.datetime(2026, 11, 12, 21, 53, 40),
        location=[64.653, -17.481],
    )
    assert isinstance(out, pd.DataFrame) and out.empty


def test_downl_pgv_hraun(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(src.ENV_ALERTMAP_DIR, str(ALERT))
    p = sil_read.downl_PGV_hraun(inpdate="/2026/nov/12")
    assert p == str(ALERT / "2026/nov/12") + "/"
    out = tmp_path / "dl"
    (out).mkdir()
    (out / "999.event").write_text("old")
    p2 = sil_read.downl_PGV_hraun(
        dt.date(2026, 11, 12), tmp_path=str(out), rm_files=True
    )
    assert p2 == str(out) + "/" and not p2.endswith("//")
    assert sorted(f.name for f in out.glob("*.event")) == ["001.event", "076.event"]
    assert sil_read.downl_PGV_hraun(GOLDEN_ID) == p
    with pytest.raises(ValueError, match="cannot read the date"):
        sil_read.downl_PGV_hraun("yesterday")
    with pytest.raises(ValueError, match="inside the data directory"):
        sil_read.downl_PGV_hraun(
            "/2026/nov/12", tmp_path=str(ALERT / "x"), rm_files=True
        )


def test_downl_pgv_hraun_no_locale_change(monkeypatch: pytest.MonkeyPatch) -> None:
    import locale

    before = locale.setlocale(locale.LC_TIME)
    monkeypatch.setenv(src.ENV_ALERTMAP_DIR, str(ALERT))
    sil_read.downl_PGV_hraun("/2026/nov/12")
    assert locale.setlocale(locale.LC_TIME) == before


def test_read_station_loc(tmp_path: Path) -> None:
    f = tmp_path / "net.dat"
    f.write_text("hei  64.19978 -21.23604 162.0\ngha  65.84346 -16.66291 396.0\nbad\n")
    df = sil_read.read_station_loc(str(f))
    assert list(df.columns) == ["sta", "latitude", "longitude", "height"]
    assert df["sta"].tolist() == ["hei", "gha"]
    assert df["latitude"].iloc[0] == pytest.approx(64.19978)
    assert "liljajo" not in (sil_read.read_station_loc.__doc__ or "")


def test_no_bare_except_or_setlocale_in_shim() -> None:
    text = Path(sil_read.__file__).read_text(encoding="utf-8")
    assert "except:" not in text
    assert "setlocale" not in text
    assert "print(" not in text


def test_sa_reexports_unchanged() -> None:
    # the shim's evb/alert frames come from the T7b parsers
    assert sa.read_alert_event(GOLDEN_EVENT).header.alert_id == 76
    assert os.path.isfile(GOLDEN_EVB)
