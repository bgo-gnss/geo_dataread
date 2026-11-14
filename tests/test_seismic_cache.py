"""Local Parquet cache of finished SIL days (ticket T4 of PLAN-seismicity.md).

Every test works on a copy of ``tests/fixtures/sil`` in ``tmp_path`` and a
cache directory in ``tmp_path``; the real cache is never touched (the
autouse fixture points ``$GPS_SIL_CACHE`` and ``$XDG_CACHE_HOME`` into
``tmp_path``). ``today`` is pinned, so the 2026-09 fixture days are finished
with ``recent_days=30`` and recent with ``recent_days=90``. The one
real-data timing test skips without ``$GPS_SIL_DIR``.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import warnings
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from geo_dataread.seismic import cache as c  # noqa: E402
from geo_dataread.seismic import schema as sch  # noqa: E402
from geo_dataread.seismic import sil_catalog as sc  # noqa: E402
from geo_dataread.seismic.mechanisms import (  # noqa: E402
    EXTRA_COLUMNS,
    read_sil_day_mechanisms,
)
from geo_dataread.seismic.schema import (  # noqa: E402
    ATTRS_KEY,
    TIME_DTYPE,
    column_names,
    validate,
    validate_mechanisms,
)

FIXTURES = Path(__file__).parent / "fixtures" / "sil"
TODAY = date(2026, 11, 14)

#: Every fixture day directory (all finished at recent_days=30).
FIXTURE_DAYS = (
    date(1993, 1, 29),
    date(1995, 6, 15),
    date(2022, 12, 12),
    date(2024, 8, 22),
    date(2025, 5, 3),
    date(2026, 9, 10),
    date(2026, 9, 11),
)

WINDOWS = [
    ("2024-08-22", "2024-08-23"),  # one day
    ("2024-08-22T10:00", "2024-08-22T15:00"),  # part of a day
    ("2024-08-22", "2024-08-22"),  # empty window
    ("2000-01-01", "2001-01-01"),  # a year with no directories
    ("1993-01-01", "1996-01-01"),  # old layouts, across years
    ("2026-09-10", "2026-09-12"),  # automatic-only + empty day
    ("1990-01-01", "2027-01-01"),  # everything
]


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPS_SIL_CACHE", str(tmp_path / "env-cache"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("GPS_SIL_DIR", raising=False)


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    dst = tmp_path / "sil"
    shutil.copytree(FIXTURES, dst, ignore=shutil.ignore_patterns("README.md"))
    return dst


@pytest.fixture()
def cdir(tmp_path: Path) -> Path:
    return tmp_path / "cache"


def cached(tree: Path, cdir: Path, start: Any, end: Any, **kw: Any) -> Any:
    kw.setdefault("today", TODAY)
    return c.read_sil_catalog_cached(start, end, base=tree, cache_dir=cdir, **kw)


def mechs(tree: Path, cdir: Path, start: Any, end: Any, **kw: Any) -> pd.DataFrame:
    kw.setdefault("today", TODAY)
    return c.read_sil_mechanisms_cached(start, end, base=tree, cache_dir=cdir, **kw)


def stats(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = df.attrs[ATTRS_KEY]["cache"]
    return out


def parquet_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.parquet"))


def manifest_days(path: Path) -> set[str]:
    days: dict[str, Any] = pd.read_parquet(path).attrs[c.MANIFEST_KEY]["days"]
    return set(days)


# --------------------------------------------------------------------------
# Location
# --------------------------------------------------------------------------


def test_cache_dir_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert c.sil_cache_dir(tmp_path / "arg") == tmp_path / "arg"
    assert c.sil_cache_dir() == tmp_path / "env-cache"
    monkeypatch.delenv("GPS_SIL_CACHE")
    assert c.sil_cache_dir() == tmp_path / "xdg" / "gps" / "seismicity"
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert c.sil_cache_dir() == Path.home() / ".cache" / "gps" / "seismicity"


def test_env_cache_is_used_when_no_dir_given(tree: Path, tmp_path: Path) -> None:
    c.read_sil_catalog_cached("2024-08-22", "2024-08-23", base=tree, today=TODAY)
    files = parquet_files(tmp_path / "env-cache")
    assert [p.name for p in files] == ["2024.parquet"]
    assert files[0].parent.name == "events"
    assert files[0].parents[1].name.startswith(f"v{sch.SCHEMA_VERSION}-")
    assert files[0].parents[2].name == "sil-files"


def test_cache_inside_archive_refused(tree: Path) -> None:
    with pytest.raises(ValueError, match="inside the SIL archive"):
        cached(tree, tree / "cache", "2024-08-22", "2024-08-23")
    with pytest.raises(ValueError, match="inside the SIL archive"):
        mechs(tree, tree, "2024-08-22", "2024-08-23")
    assert not (tree / "cache").exists()


# --------------------------------------------------------------------------
# Events: parity with read_sil_catalog, cold and warm
# --------------------------------------------------------------------------


def _same_report(a: sc.SilReadReport, b: sc.SilReadReport) -> None:
    assert dict(a.counts) == dict(b.counts)
    assert a.issues == b.issues
    assert a.fallback_dt_s == pytest.approx(b.fallback_dt_s)


@pytest.mark.parametrize(("start", "end"), WINDOWS)
def test_events_equal_archive_read_cold_and_warm(
    tree: Path, cdir: Path, start: str, end: str
) -> None:
    ref, ref_rep = sc.read_sil_catalog(start, end, base=tree)
    cold, cold_rep = cached(tree, cdir, start, end)
    warm, warm_rep = cached(tree, cdir, start, end)
    for got, rep in ((cold, cold_rep), (warm, warm_rep)):
        pd.testing.assert_frame_equal(got, ref)
        _same_report(rep, ref_rep)
        assert validate(got).ok, validate(got).summary()
        assert got.index.dtype == TIME_DTYPE and got.index.name == "time"
    assert stats(warm)["days_read"] == 0
    assert stats(warm)["days_written"] == 0
    assert stats(warm)["days_cached"] == stats(cold)["days_read"]


def test_warm_dtypes_and_attrs_round_trip(tree: Path, cdir: Path) -> None:
    ref, _ = sc.read_sil_catalog("1990-01-01", "2027-01-01", base=tree)
    cached(tree, cdir, "1990-01-01", "2027-01-01")
    warm, _ = cached(tree, cdir, "1990-01-01", "2027-01-01")
    assert list(warm.columns) == list(ref.columns)
    assert dict(warm.dtypes) == dict(ref.dtypes)
    for col in ("source", "magnitude_type", "evaluation_mode", "event_type"):
        assert list(warm[col].dtype.categories) == list(ref[col].dtype.categories)
    assert validate(warm, strict=True).ok
    a, b = dict(ref.attrs[ATTRS_KEY]), dict(warm.attrs[ATTRS_KEY])
    assert b.pop("cache")["used"] is True
    for d in (a, b):
        d.pop("fetched_at")
        d["report"].pop("seconds")
    assert a == b
    assert b["schema_version"] == sch.SCHEMA_VERSION
    assert b["source"] == "sil-files"


def test_off_day_padding_matches(tree: Path, cdir: Path) -> None:
    """A window ending at a day boundary reads the next directory too."""
    for start, end in (("2024-08-23", "2024-08-24"), ("2024-08-21", "2024-08-22")):
        ref, rep = sc.read_sil_catalog(start, end, base=tree)
        got, grep = cached(tree, cdir, start, end)
        pd.testing.assert_frame_equal(got, ref)
        _same_report(grep, rep)


def test_finished_days_survive_archive_loss(tree: Path, cdir: Path) -> None:
    ref, _ = sc.read_sil_catalog("1990-01-01", "2027-01-01", base=tree)
    cached(tree, cdir, "1990-01-01", "2027-01-01")
    shutil.rmtree(tree)
    tree.mkdir()
    warm, rep = cached(tree, cdir, "1990-01-01", "2027-01-01")
    pd.testing.assert_frame_equal(warm, ref)
    assert stats(warm)["days_cached"] == len(FIXTURE_DAYS)
    assert stats(warm)["days_read"] == 0
    assert rep.get("days_read") == len(FIXTURE_DAYS)


# --------------------------------------------------------------------------
# Recent days
# --------------------------------------------------------------------------


def _truncate_aut(day: Path) -> None:
    for name in ("aut.mag", "cleanaut.mag"):
        p = day / name
        if p.exists():
            lines = p.read_text().splitlines(keepends=True)
            p.write_text("".join(lines[:1]))


def test_recent_days_are_never_cached(tree: Path, cdir: Path) -> None:
    w = ("2026-09-10", "2026-09-12")
    first, _ = cached(tree, cdir, *w, recent_days=90)
    assert stats(first)["days_recent"] == 2
    assert stats(first)["days_written"] == 0
    assert not parquet_files(cdir)
    _truncate_aut(tree / "2026" / "sep" / "10")
    again, _ = cached(tree, cdir, *w, recent_days=90)
    assert len(again) < len(first)
    ref, _ = sc.read_sil_catalog(*w, base=tree)
    pd.testing.assert_frame_equal(again, ref)


def test_finished_days_are_trusted_until_refresh(tree: Path, cdir: Path) -> None:
    w = ("2026-09-10", "2026-09-12")
    first, _ = cached(tree, cdir, *w, recent_days=30)
    assert stats(first)["days_written"] == 2
    _truncate_aut(tree / "2026" / "sep" / "10")
    stale, _ = cached(tree, cdir, *w, recent_days=30)
    pd.testing.assert_frame_equal(stale, first)
    fresh, _ = cached(tree, cdir, *w, recent_days=30, refresh=True)
    assert len(fresh) < len(first)
    assert stats(fresh)["days_read"] == 2 and stats(fresh)["days_written"] == 2
    after, _ = cached(tree, cdir, *w, recent_days=30)
    pd.testing.assert_frame_equal(after, fresh)


def test_today_defaults_to_utc_now(tree: Path, cdir: Path) -> None:
    got, _ = c.read_sil_catalog_cached(
        "2024-08-22", "2024-08-23", base=tree, cache_dir=cdir
    )
    assert stats(got)["days_written"] == 1


def test_bad_recent_days_and_window(tree: Path, cdir: Path) -> None:
    with pytest.raises(ValueError, match="recent_days"):
        cached(tree, cdir, "2024-08-22", "2024-08-23", recent_days=0)
    with pytest.raises(ValueError, match="recent_days"):
        mechs(tree, cdir, "2024-08-22", "2024-08-23", recent_days=-1)
    with pytest.raises(ValueError, match="before start"):
        cached(tree, cdir, "2024-08-23", "2024-08-22")
    with pytest.raises(ValueError, match="before start"):
        mechs(tree, cdir, "2024-08-23", "2024-08-22")


def test_no_archive_path(cdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sc, "_config_sil_dir", lambda: None)
    with pytest.raises(FileNotFoundError, match="GPS_SIL_DIR"):
        c.read_sil_catalog_cached("2024-08-22", "2024-08-23", cache_dir=cdir)


def test_archive_from_env(
    tree: Path, cdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPS_SIL_DIR", str(tree))
    got, _ = c.read_sil_catalog_cached(
        "2024-08-22", "2024-08-23", cache_dir=cdir, today=TODAY
    )
    assert len(got) == 10
    assert got.attrs[ATTRS_KEY]["location"] == str(tree)


# --------------------------------------------------------------------------
# Key and invalidation
# --------------------------------------------------------------------------


def test_schema_version_change_invalidates(
    tree: Path, cdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cached(tree, cdir, "2024-08-22", "2024-08-23")
    before = {p.parents[1].name for p in parquet_files(cdir)}
    monkeypatch.setattr(sch, "SCHEMA_VERSION", sch.SCHEMA_VERSION + 1)
    got, _ = cached(tree, cdir, "2024-08-22", "2024-08-23")
    assert stats(got)["days_cached"] == 0 and stats(got)["days_read"] == 1
    after = {p.parents[1].name for p in parquet_files(cdir)}
    new = after - before
    assert len(new) == 1 and next(iter(new)).startswith(f"v{sch.SCHEMA_VERSION}-")


def test_cache_format_change_invalidates(
    tree: Path, cdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cached(tree, cdir, "2024-08-22", "2024-08-23")
    monkeypatch.setattr(c, "CACHE_FORMAT", c.CACHE_FORMAT + 1)
    got, _ = cached(tree, cdir, "2024-08-22", "2024-08-23")
    assert stats(got)["days_read"] == 1
    assert len({p.parents[1] for p in parquet_files(cdir)}) == 2


def test_other_archive_gets_its_own_key(tree: Path, cdir: Path, tmp_path: Path) -> None:
    other = tmp_path / "other"
    shutil.copytree(tree, other)
    _truncate_aut(other / "2024" / "aug" / "22")
    a, _ = cached(tree, cdir, "2024-08-22", "2024-08-23")
    b, _ = cached(other, cdir, "2024-08-22", "2024-08-23")
    assert stats(b)["days_read"] == 1
    assert len(b) < len(a)
    assert stats(a)["cache_dir"] != stats(b)["cache_dir"]


def test_corrupt_year_file_is_a_miss_and_rewritten(tree: Path, cdir: Path) -> None:
    ref, _ = cached(tree, cdir, "2024-08-22", "2024-08-23")
    (path,) = parquet_files(cdir)
    path.write_bytes(b"not a parquet file")
    with pytest.warns(UserWarning, match="unreadable"):
        got, _ = cached(tree, cdir, "2024-08-22", "2024-08-23")
    pd.testing.assert_frame_equal(got, ref)
    assert stats(got)["files_rejected"] == 1 and stats(got)["days_written"] == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        again, _ = cached(tree, cdir, "2024-08-22", "2024-08-23")
    assert stats(again)["days_cached"] == 1


def test_foreign_key_in_file_is_a_miss(tree: Path, cdir: Path) -> None:
    cached(tree, cdir, "2024-08-22", "2024-08-23")
    (path,) = parquet_files(cdir)
    df = pd.read_parquet(path)
    df.attrs[c.MANIFEST_KEY]["key"]["schema_version"] = -1
    df.to_parquet(path)
    with pytest.warns(UserWarning, match="another key"):
        got, _ = cached(tree, cdir, "2024-08-22", "2024-08-23")
    assert stats(got)["files_rejected"] == 1 and stats(got)["days_read"] == 1


def test_row_count_mismatch_is_a_miss(tree: Path, cdir: Path) -> None:
    ref, _ = cached(tree, cdir, "2024-08-22", "2024-08-23")
    (path,) = parquet_files(cdir)
    df = pd.read_parquet(path)
    attrs = df.attrs
    df = df.iloc[:3]
    df.attrs = attrs
    df.to_parquet(path)
    got, _ = cached(tree, cdir, "2024-08-22", "2024-08-23")
    pd.testing.assert_frame_equal(got, ref)
    assert stats(got)["days_read"] == 1 and stats(got)["days_written"] == 1


# --------------------------------------------------------------------------
# Bypass
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        {"use_cache": False},
        {"automatic": False},
        {"match_tol": sc.MatchTolerance(1.0, 5.0)},
        {"match_tol_no_id": sc.MatchTolerance(1.0, 5.0)},
    ],
)
def test_non_default_reads_the_archive(
    tree: Path, cdir: Path, kw: dict[str, Any]
) -> None:
    args = {k: v for k, v in kw.items() if k != "use_cache"}
    ref, _ = sc.read_sil_catalog("2024-08-22", "2024-08-23", base=tree, **args)
    got, _ = cached(tree, cdir, "2024-08-22", "2024-08-23", **kw)
    assert stats(got) == {"used": False}
    pd.testing.assert_frame_equal(got, ref)
    assert not cdir.exists()


def test_missing_pyarrow_is_a_clear_error(
    tree: Path, cdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "pyarrow" else real(name, *a),
    )
    with pytest.raises(ImportError, match="pyarrow"):
        cached(tree, cdir, "2024-08-22", "2024-08-23")
    got, _ = cached(tree, cdir, "2024-08-22", "2024-08-23", use_cache=False)
    assert len(got) == 10


def test_write_false_reads_without_writing(tree: Path, cdir: Path) -> None:
    got, _ = cached(tree, cdir, "2024-08-22", "2024-08-23", write=False)
    assert stats(got)["days_read"] == 1 and stats(got)["days_written"] == 0
    assert not parquet_files(cdir)


# --------------------------------------------------------------------------
# Writes: merge, atomicity, concurrency
# --------------------------------------------------------------------------


def _extra_2024_days(tree: Path, days: list[int]) -> None:
    for d in days:
        shutil.copytree(
            tree / "2026" / "sep" / "11", tree / "2024" / "aug" / f"{d:02d}"
        )


def test_year_file_merges_days(tree: Path, cdir: Path) -> None:
    _extra_2024_days(tree, [10])
    cached(tree, cdir, "2024-08-22", "2024-08-23")
    cached(tree, cdir, "2024-08-10", "2024-08-11")
    (path,) = parquet_files(cdir)
    assert manifest_days(path) == {"20240822", "20240810"}
    got, _ = cached(tree, cdir, "2024-08-01", "2024-09-01")
    assert stats(got)["days_read"] == 0
    assert stats(got)["days_cached"] == 2
    assert len(got) == 10


def test_failed_write_leaves_old_file_and_no_temp(
    tree: Path, cdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _extra_2024_days(tree, [10])
    cached(tree, cdir, "2024-08-22", "2024-08-23")
    (path,) = parquet_files(cdir)
    before = path.read_bytes()

    def boom(self: pd.DataFrame, *a: Any, **k: Any) -> None:
        Path(a[0]).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(OSError, match="disk full"):
        cached(tree, cdir, "2024-08-10", "2024-08-11")
    assert path.read_bytes() == before
    assert not [p for p in path.parent.iterdir() if p.suffix == ".tmp"]


def test_concurrent_writers_lose_nothing(
    tree: Path, cdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    days = [3, 6, 9, 12, 15, 18]
    _extra_2024_days(tree, days)
    write = c._atomic_parquet

    def slow(df: pd.DataFrame, path: Path) -> None:
        time.sleep(0.05)  # widen the window in which writers overlap
        write(df, path)

    monkeypatch.setattr(c, "_atomic_parquet", slow)
    errors: list[BaseException] = []

    def run(d: int) -> None:
        try:
            cached(tree, cdir, f"2024-08-{d:02d}", f"2024-08-{d + 1:02d}")
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(d,)) for d in days]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    (path,) = parquet_files(cdir)
    assert manifest_days(path) == {f"202408{d:02d}" for d in days}
    assert not [p for p in path.parent.iterdir() if p.suffix == ".tmp"]


# --------------------------------------------------------------------------
# Mechanisms
# --------------------------------------------------------------------------


def _mech_reference(tree: Path, start: str, end: str, extras: bool) -> pd.DataFrame:
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    parts = []
    for d in FIXTURE_DAYS:
        f, _ = read_sil_day_mechanisms(sc.day_dir(tree, d), extras=extras)
        parts.append(f.loc[(f.index >= lo) & (f.index < hi)])
    df = pd.concat([p for p in parts if len(p)])
    return sch.coerce_mechanisms(df)


@pytest.mark.parametrize("extras", [False, True])
def test_mechanisms_equal_day_reads_cold_and_warm(
    tree: Path, cdir: Path, extras: bool
) -> None:
    w = ("1990-01-01", "2027-01-01")
    ref = _mech_reference(tree, *w, extras=extras)
    cold = mechs(tree, cdir, *w, extras=extras)
    warm = mechs(tree, cdir, *w, extras=extras)
    for got in (cold, warm):
        assert got.attrs[ATTRS_KEY]["frame"] == "mechanisms"
        got = got.copy()
        got.attrs = {}
        pd.testing.assert_frame_equal(got, ref)
        assert validate_mechanisms(got).ok
    assert stats(warm)["days_read"] == 0
    assert stats(warm)["days_cached"] == len(FIXTURE_DAYS)
    cols = list(column_names("mechanisms")) + (list(EXTRA_COLUMNS) if extras else [])
    assert list(warm.columns) == cols


def test_mechanisms_window_cut_and_report(tree: Path, cdir: Path) -> None:
    whole = mechs(tree, cdir, "2024-08-22", "2024-08-23")
    part = mechs(tree, cdir, "2024-08-22T00:00", "2024-08-22T12:00")
    assert 0 < len(part) < len(whole)
    assert (part.index < pd.Timestamp("2024-08-22T12:00", tz="UTC")).all()
    rep = whole.attrs[ATTRS_KEY]["report"]
    assert rep["rows_by_method"] == {"sil_fps": len(whole)}
    assert rep["window_rows"] == len(whole)


def test_mechanisms_empty_window_is_typed(tree: Path, cdir: Path) -> None:
    for _ in range(2):  # cold, then warm on a cached empty day
        got = mechs(tree, cdir, "2026-09-11", "2026-09-12")
        assert len(got) == 0
        assert validate_mechanisms(got).ok
        assert got.index.dtype == TIME_DTYPE
    got = mechs(tree, cdir, "2026-09-11", "2026-09-12", extras=True)
    full = c.reader_empty("mechanisms")
    assert dict(got.dtypes) == dict(full.dtypes)


def test_mechanisms_bypass(tree: Path, cdir: Path) -> None:
    a = mechs(tree, cdir, "2024-08-22", "2024-08-23", use_cache=False)
    b = mechs(tree, cdir, "2024-08-22", "2024-08-23", plane_tol_deg=10.0)
    assert stats(a)["used"] is False and stats(b)["used"] is False
    assert not cdir.exists()
    assert len(a) == len(b) == 8


def test_events_and_mechanisms_are_separate_files(tree: Path, cdir: Path) -> None:
    cached(tree, cdir, "2024-08-22", "2024-08-23")
    mechs(tree, cdir, "2024-08-22", "2024-08-23")
    kinds = sorted(p.parent.name for p in parquet_files(cdir))
    assert kinds == ["events", "mechanisms"]


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def test_build_then_reads_are_warm(tree: Path, cdir: Path) -> None:
    lines: list[str] = []
    s = c.build_sil_cache(
        "1990-01-01",
        "2027-01-01",
        base=tree,
        cache_dir=cdir,
        today=TODAY,
        progress=lines.append,
    )
    for k in c.KINDS:
        assert s.stats[k].days_written == len(FIXTURE_DAYS)
        assert s.stats[k].days_recent == 0
    assert len(lines) == 2 * (2026 - 1990 + 1)
    assert s.as_dict()["stats"]["events"]["days_written"] == len(FIXTURE_DAYS)
    again = c.build_sil_cache(
        "1990-01-01", "2027-01-01", base=tree, cache_dir=cdir, today=TODAY
    )
    assert again.stats["events"].days_written == 0
    assert again.stats["events"].days_cached == len(FIXTURE_DAYS)
    ref, _ = sc.read_sil_catalog("1990-01-01", "2027-01-01", base=tree)
    got, _ = cached(tree, cdir, "1990-01-01", "2027-01-01")
    pd.testing.assert_frame_equal(got, ref)
    assert stats(got)["days_read"] == 0
    m = mechs(tree, cdir, "1990-01-01", "2027-01-01")
    assert stats(m)["days_read"] == 0


def test_build_skips_recent_days(tree: Path, cdir: Path) -> None:
    s = c.build_sil_cache(
        "2026-01-01",
        "2027-01-01",
        base=tree,
        cache_dir=cdir,
        today=TODAY,
        recent_days=90,
        kinds=("events",),
    )
    assert s.stats["events"].days_written == 0
    assert s.stats["events"].days_read == 0
    assert set(s.stats) == {"events"}


def test_build_errors(tree: Path, cdir: Path) -> None:
    with pytest.raises(ValueError, match="unknown kind"):
        c.build_sil_cache(
            "2024-01-01", "2025-01-01", base=tree, cache_dir=cdir, kinds=("x",)  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="before start"):
        c.build_sil_cache("2025-01-01", "2024-01-01", base=tree, cache_dir=cdir)
    with pytest.raises(ValueError, match="inside the SIL archive"):
        c.build_sil_cache("2024-01-01", "2025-01-01", base=tree, cache_dir=tree)
    empty = c.build_sil_cache("2024-01-01", "2024-01-01", base=tree, cache_dir=cdir)
    assert empty.stats["events"].days_read == 0


# --------------------------------------------------------------------------
# Real archive: timing (skips without $GPS_SIL_DIR)
# --------------------------------------------------------------------------

#: Read at import, before the autouse fixture clears ``$GPS_SIL_DIR``.
REAL_SIL_DIR = os.environ.get("GPS_SIL_DIR")
#: The year timed (``$GPS_SIL_TIMING_YEAR``, default 2023: the largest of the
#: T3 checks, 42.6k reviewed + 71.7k automatic events).
TIMING_YEAR = int(os.environ.get("GPS_SIL_TIMING_YEAR", "2023"))


@pytest.mark.skipif(
    not REAL_SIL_DIR or not Path(REAL_SIL_DIR).is_dir(),
    reason="SIL archive not given ($GPS_SIL_DIR)",
)
def test_timing_full_real_year(tmp_path: Path) -> None:
    """Cold (archive + write) vs warm (cache) read of one real year of events.

    Mechanisms are timed on January of that year only (about 11 ms per fps
    row cold, T3b) unless ``$GPS_SIL_TIMING_MECHANISMS=year``.
    """
    assert REAL_SIL_DIR is not None
    base, cdir = Path(REAL_SIL_DIR), tmp_path / "cache"
    w = (f"{TIMING_YEAR}-01-01", f"{TIMING_YEAR + 1}-01-01")
    t0 = time.perf_counter()
    cold, cold_rep = c.read_sil_catalog_cached(*w, base=base, cache_dir=cdir)
    t_cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    warm, warm_rep = c.read_sil_catalog_cached(*w, base=base, cache_dir=cdir)
    t_warm = time.perf_counter() - t0
    pd.testing.assert_frame_equal(warm, cold)
    assert dict(warm_rep.counts) == dict(cold_rep.counts)
    assert validate(warm).ok
    assert stats(warm)["days_read"] == 0
    size = sum(p.stat().st_size for p in parquet_files(cdir))
    mw = (
        w
        if os.environ.get("GPS_SIL_TIMING_MECHANISMS") == "year"
        else (w[0], f"{TIMING_YEAR}-02-01")
    )
    t0 = time.perf_counter()
    mcold = c.read_sil_mechanisms_cached(*mw, base=base, cache_dir=cdir)
    m_cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    mwarm = c.read_sil_mechanisms_cached(*mw, base=base, cache_dir=cdir)
    m_warm = time.perf_counter() - t0
    pd.testing.assert_frame_equal(mwarm, mcold)
    # The archive itself holds fps rows whose plane 2 is not plane 1's
    # auxiliary (2023: 2 rows, 22°, both 180/39/-79 + 14/52/-81); that is a
    # reader/data finding (T3b), not a cache one, so only other errors fail.
    errors = [
        i
        for i in validate_mechanisms(mwarm).issues
        if i.severity == "error" and i.kind != "planes_inconsistent"
    ]
    assert not errors, errors
    print(
        f"\nSIL cache timing {TIMING_YEAR}: events {len(warm)} rows, cold "
        f"{t_cold:.1f} s, warm {t_warm:.2f} s, {size / 1e6:.1f} MB; "
        f"mechanisms {mw[0]}..{mw[1]} {len(mwarm)} rows, cold {m_cold:.1f} s, "
        f"warm {m_warm:.2f} s"
    )
    assert t_warm < t_cold
