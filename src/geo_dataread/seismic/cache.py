"""Local Parquet cache of finished SIL days (ticket T4, PLAN-seismicity.md).

The SIL archive is slow to read over NFS (a year of events is tens of
seconds; a year of fault planes minutes, T3b), and old days do not change.
This module keeps the assembled frames of *finished* day directories in
per-year Parquet files, separately for events (:func:`read_sil_day`) and
mechanisms (:func:`read_sil_day_mechanisms`):

.. code-block:: text

    <root>/sil-files/v<schema>-<hash>/events/<yyyy>.parquet
    <root>/sil-files/v<schema>-<hash>/mechanisms/<yyyy>.parquet

``<root>`` is ``$GPS_SIL_CACHE``, else ``$XDG_CACHE_HOME/gps/seismicity``,
else ``~/.cache/gps/seismicity`` (:func:`sil_cache_dir`). The key directory
carries the schema version (:data:`geo_dataread.seismic.schema.SCHEMA_VERSION`)
and a hash of the source name, :data:`CACHE_FORMAT` and the archive root, so a
schema change, a change of the assembly logic (bump :data:`CACHE_FORMAT`) or a
different archive never serves stale rows; the old directories are simply no
longer read (delete them by hand).

Rules:

* **Finished days only.** A day is cached when it is more than
  ``recent_days`` (default :data:`DEFAULT_RECENT_DAYS`) before ``today``
  (UTC). The recent days are always read from the archive and never written.
* **Per day directory.** Each year file holds the full frame of each day
  directory read (events whose time falls on the next day included, the
  helper column :data:`DAY_COLUMN` names the directory) and, in its
  ``attrs``, a manifest of every day handled with its read report. A day with
  no events is in the manifest and has no rows. A missing day directory is
  never cached (it is one ``stat``, and a not-yet-mounted archive must not
  freeze a year as empty).
* **Default parameters only.** Events are cached for ``automatic=True`` and the
  default match tolerances, mechanisms with ``extras=True`` (columns chosen on
  read) and the default plane tolerance. Any other call reads the archive.
* **Same semantics as** :func:`read_sil_catalog`: half-open ``[start, end)``,
  one padding day directory each side, the cut by origin time, one final
  :func:`coerce_catalog`. A cached read equals an archive read
  (``assert_frame_equal``), and its report equals the archive read's counts.
* **Atomic writes.** A year file is written to a temporary file in the same
  directory and renamed over the old one, under an exclusive ``flock`` on a
  sibling ``.lock`` file; the writer re-reads the file under the lock and
  merges, so two processes on one machine lose nothing. Readers take no lock.
  An unreadable year file, or one whose stored key does not match, is a miss
  (with a :class:`UserWarning`) and is rewritten.
* **Never under the archive.** A cache directory inside the archive root is
  refused.

Parquet needs ``pyarrow`` (present in the ``gpslibrary`` env, not declared by
the package).
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import tempfile
import time as _time
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

from geo_dataread.seismic import schema as _schema
from geo_dataread.seismic.mechanisms import (
    EXTRA_COLUMNS,
    mechanisms_from_blocks,
    read_sil_day_mechanisms,
)
from geo_dataread.seismic.schema import (
    DEFAULT_PLANE_TOL_DEG,
    INDEX_NAME,
    STRING_DTYPE,
    TIME_DTYPE,
    catalog_provenance,
    coerce_catalog,
    coerce_mechanisms,
    column_names,
    with_provenance,
)
from geo_dataread.seismic.sil_catalog import (
    DEFAULT_MATCH_TOL,
    DEFAULT_MATCH_TOL_NO_ID,
    SOURCE,
    MatchTolerance,
    SilReadReport,
    day_dir,
    read_sil_catalog,
    read_sil_day,
    sil_dir,
)

log = logging.getLogger(__name__)

#: Environment variable overriding the cache root.
ENV_SIL_CACHE: Final = "GPS_SIL_CACHE"

#: Days before ``today`` that are always read from the archive.
DEFAULT_RECENT_DAYS: Final = 30

#: Bump when the assembly logic changes the cached frames without a schema
#: version change (it is part of the key hash).
CACHE_FORMAT: Final = 1

#: Helper column of the year files: the day directory as ``yyyymmdd``.
DAY_COLUMN: Final = "_sil_day"

#: ``attrs`` key of the year-file manifest.
MANIFEST_KEY: Final = "sil_cache"

Kind = Literal["events", "mechanisms"]
KINDS: Final[tuple[Kind, ...]] = ("events", "mechanisms")


# --------------------------------------------------------------------------
# Location and key
# --------------------------------------------------------------------------


def sil_cache_dir(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """The cache root: ``cache_dir``, ``$GPS_SIL_CACHE``, else
    ``$XDG_CACHE_HOME/gps/seismicity`` (``~/.cache/gps/seismicity``)."""
    if cache_dir:
        return Path(cache_dir).expanduser()
    env = os.environ.get(ENV_SIL_CACHE)
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(xdg).expanduser() / "gps" / "seismicity"


def _cache_key(root: Path) -> dict[str, Any]:
    return {
        "cache_format": CACHE_FORMAT,
        "schema_version": int(_schema.SCHEMA_VERSION),
        "source": SOURCE,
        "base": os.path.abspath(root),
    }


def _key_dir(cache_root: Path, key: Mapping[str, Any]) -> Path:
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    return cache_root / SOURCE / f"v{key['schema_version']}-{digest[:12]}"


def _check_outside(cache_root: Path, archive: Path) -> None:
    try:
        inside = Path(os.path.abspath(cache_root)).is_relative_to(
            os.path.abspath(archive)
        )
    except ValueError:  # pragma: no cover - different drives
        inside = False
    if inside:
        raise ValueError(
            f"cache directory {cache_root} is inside the SIL archive {archive}; "
            "the cache never writes under the data mount"
        )


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------


@dataclasses.dataclass
class CacheStats:
    """What a cached read or build did, per day directory.

    ``days_cached`` were served from the cache, ``days_read`` from the
    archive (of which ``days_recent`` inside the recent window, never
    cached), ``days_written`` were added to the cache, ``days_missing`` had
    no directory; ``years_written`` year files were rewritten,
    ``files_rejected`` were unreadable or carried another key.
    """

    kind: str
    cache_dir: str = ""
    days_cached: int = 0
    days_read: int = 0
    days_recent: int = 0
    days_written: int = 0
    days_missing: int = 0
    years_written: int = 0
    files_rejected: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe form (rides in ``attrs["catalog"]["cache"]``)."""
        d = dataclasses.asdict(self)
        d["seconds"] = round(self.seconds, 3)
        return d

    def add(self, other: CacheStats) -> None:
        """Accumulate ``other`` (same kind) into this one."""
        for f in dataclasses.fields(self):
            if f.name in ("kind", "cache_dir"):
                continue
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))


# --------------------------------------------------------------------------
# Day readers
# --------------------------------------------------------------------------


def _day_int(d: date) -> int:
    return d.year * 10000 + d.month * 100 + d.day


def _read_events_day(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df, rep = read_sil_day(path)
    df.attrs = {}
    meta = {
        "rows": len(df),
        "counts": {k: int(v) for k, v in rep.counts.items()},
        "issues": list(rep.issues),
        "fallback_dt_s": [float(x) for x in rep.fallback_dt_s],
    }
    return df, meta


def _read_mechanisms_day(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, rep = read_sil_day_mechanisms(path, extras=True)
    frame.attrs = {}
    meta = {
        "rows": len(frame),
        "has_fps": rep.has_fps,
        "fps_rows": rep.fps_rows,
        "no_solution": rep.no_solution,
        "rows_by_method": {str(k): int(v) for k, v in rep.rows_by_method.items()},
        "blocks_read": rep.blocks_read,
        "blocks_missing": len(rep.blocks_missing),
        "block_mismatch": len(rep.block_mismatch),
        "max_block_mismatch_deg": float(rep.max_block_mismatch_deg),
        "unused_blocks": len(rep.unused_blocks),
        "block_issues": len(rep.block_issues),
        "no_block": rep.no_block,
    }
    return frame, meta


_DAY_READERS: Final[
    Mapping[str, Callable[[Path], tuple[pd.DataFrame, dict[str, Any]]]]
] = {"events": _read_events_day, "mechanisms": _read_mechanisms_day}


# --------------------------------------------------------------------------
# Year files
# --------------------------------------------------------------------------


@dataclasses.dataclass
class _Day:
    """One day directory's frame (``None`` = missing) and its read report."""

    day: date
    frame: pd.DataFrame | None
    meta: dict[str, Any]
    cached: bool


def _year_path(kdir: Path, kind: Kind, year: int) -> Path:
    return kdir / kind / f"{year:04d}.parquet"


def _require_pyarrow() -> None:
    if importlib.util.find_spec("pyarrow") is None:
        raise ImportError(
            "the SIL cache needs pyarrow (Parquet); install it, or read with "
            "use_cache=False"
        )


def _load_manifest(
    path: Path, key: Mapping[str, Any], stats: CacheStats, *, quiet: bool = False
) -> dict[str, Any] | None:
    """The manifest days of a year file, ``None`` when absent or rejected.

    A rejection warns unless ``quiet`` (the writer re-reads under its lock;
    ``warnings.catch_warnings`` is process-wide and not thread-safe).
    """
    if not path.exists():
        return None
    try:
        pq = importlib.import_module("pyarrow.parquet")
        meta = pq.read_schema(path).metadata or {}
        raw = meta.get(b"PANDAS_ATTRS")
        manifest = json.loads(raw)[MANIFEST_KEY] if raw else None
    except Exception as exc:  # noqa: BLE001 - corrupt file = miss
        if not quiet:
            warnings.warn(
                f"SIL cache: unreadable {path} ({exc}); re-reading", stacklevel=3
            )
        stats.files_rejected += 1
        return None
    if not isinstance(manifest, dict) or manifest.get("key") != dict(key):
        if not quiet:
            warnings.warn(
                f"SIL cache: {path} has another key; re-reading", stacklevel=3
            )
        stats.files_rejected += 1
        return None
    days = manifest.get("days")
    return days if isinstance(days, dict) else None


def _load_rows(path: Path, days: Sequence[int]) -> pd.DataFrame:
    df = pd.read_parquet(path, filters=[(DAY_COLUMN, "in", list(days))])
    if not isinstance(df, pd.DataFrame):  # pragma: no cover - defensive
        raise TypeError(f"{path} did not read as a DataFrame")
    return df


def _split_days(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    if not len(df):
        return out
    codes = df[DAY_COLUMN].to_numpy()
    body = df.drop(columns=[DAY_COLUMN])
    body.attrs = {}
    for code in np.unique(codes):
        out[int(code)] = body.loc[codes == code]
    return out


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _stack(pieces: list[tuple[int, pd.DataFrame]]) -> pd.DataFrame:
    """Concatenate day frames in day order, with the :data:`DAY_COLUMN`."""
    pieces = sorted(pieces, key=lambda p: p[0])
    nonempty = [(d, f) for d, f in pieces if len(f)]
    use = nonempty or pieces[:1]
    frames = []
    for code, frame in use:
        f = frame.copy()
        f.attrs = {}
        f[DAY_COLUMN] = np.full(len(f), code, dtype="int32")
        frames.append(f)
    return pd.concat(frames)


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=True)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _write_year(
    path: Path,
    key: Mapping[str, Any],
    new: Mapping[int, tuple[pd.DataFrame, dict[str, Any]]],
    stats: CacheStats,
) -> None:
    """Merge ``new`` days into the year file at ``path`` (locked, atomic).

    The file is re-read under the lock, so days another process wrote since
    this one looked are kept. An unreadable old file is replaced.
    """
    with _locked(path):
        old_days = (
            _load_manifest(path, key, CacheStats(kind=stats.kind), quiet=True) or {}
        )
        manifest: dict[str, Any] = {}
        pieces: list[tuple[int, pd.DataFrame]] = []
        keep = [int(k) for k in old_days if int(k) not in new]
        old: dict[int, pd.DataFrame] = {}
        if keep:
            try:
                old = _split_days(_load_rows(path, keep))
            except Exception as exc:  # noqa: BLE001 - corrupt: drop old days
                log.warning("SIL cache: dropping unreadable %s (%s)", path, exc)
                keep = []
        for code in keep:
            meta = old_days[str(code)]
            frame = old.get(code)
            n = len(frame) if frame is not None else 0
            if n != int(meta.get("rows", 0)):
                continue  # rows lost: forget the day, it is re-read later
            manifest[str(code)] = meta
            if frame is not None:
                pieces.append((code, frame))
        for code, (frame, meta) in new.items():
            manifest[str(code)] = meta
            pieces.append((code, frame))
        df = _stack(pieces)
        df.attrs = {MANIFEST_KEY: {"key": dict(key), "days": manifest}}
        _atomic_parquet(df, path)
    stats.years_written += 1
    stats.days_written += len(new)


# --------------------------------------------------------------------------
# Gathering days
# --------------------------------------------------------------------------


def _today(today: date | None) -> date:
    return today if today is not None else datetime.now(UTC).date()


def _check_recent(recent_days: int) -> None:
    if int(recent_days) < 1:
        raise ValueError(f"recent_days must be >= 1, got {recent_days}")


def _gather(
    kind: Kind,
    root: Path,
    days: Sequence[date],
    *,
    cache_root: Path | None,
    today: date,
    recent_days: int,
    refresh: bool,
    write: bool,
    keep_frames: bool,
    stats: CacheStats,
) -> list[_Day]:
    """Every day of ``days`` (in order): from the cache when finished and
    cached, else from the archive (writing finished days back)."""
    reader = _DAY_READERS[kind]
    cutoff = today - timedelta(days=int(recent_days))
    key = _cache_key(root)
    kdir = _key_dir(cache_root, key) if cache_root is not None else None
    if kdir is not None:
        stats.cache_dir = str(kdir)
    by_year: dict[int, list[date]] = {}
    for d in days:
        by_year.setdefault(d.year, []).append(d)
    out: list[_Day] = []
    for year, ydays in sorted(by_year.items()):
        path = _year_path(kdir, kind, year) if kdir is not None else None
        manifest: dict[str, Any] = {}
        if path is not None and not refresh:
            manifest = _load_manifest(path, key, stats) or {}
        want = [
            _day_int(d) for d in ydays if d < cutoff and str(_day_int(d)) in manifest
        ]
        cached: dict[int, pd.DataFrame] = {}
        if want and path is not None:
            try:
                cached = _split_days(_load_rows(path, want)) if keep_frames else {}
                bad = [
                    c
                    for c in want
                    if keep_frames
                    and len(cached.get(c, ())) != int(manifest[str(c)].get("rows", 0))
                ]
            except Exception as exc:  # noqa: BLE001 - corrupt file = miss
                warnings.warn(
                    f"SIL cache: unreadable {path} ({exc}); re-reading",
                    stacklevel=2,
                )
                stats.files_rejected += 1
                cached, bad = {}, list(want)
            for c in bad:
                manifest.pop(str(c), None)
        new: dict[int, tuple[pd.DataFrame, dict[str, Any]]] = {}
        for d in ydays:
            code = _day_int(d)
            finished = d < cutoff
            if finished and str(code) in manifest and path is not None:
                frame = cached.get(code)
                if frame is None and keep_frames:
                    frame = reader_empty(kind)
                stats.days_cached += 1
                out.append(_Day(d, frame, manifest[str(code)], True))
                continue
            ddir = day_dir(root, d)
            if not ddir.is_dir():
                stats.days_missing += 1
                out.append(_Day(d, None, {}, False))
                continue
            frame, meta = reader(ddir)
            stats.days_read += 1
            if not finished:
                stats.days_recent += 1
            elif path is not None and write:
                new[code] = (frame, meta)
            out.append(_Day(d, frame if keep_frames else None, meta, False))
        if new and path is not None:
            _write_year(path, key, new, stats)
    return out


def reader_empty(kind: Kind) -> pd.DataFrame:
    """An empty day frame of ``kind``, typed as the day readers type it."""
    if kind == "events":
        df = coerce_catalog(
            pd.DataFrame({INDEX_NAME: pd.DatetimeIndex([], tz="UTC")}).set_index(
                INDEX_NAME
            ),
            source=SOURCE,
        )
    else:
        none = pd.DataFrame(
            {
                INDEX_NAME: pd.Series([], dtype=TIME_DTYPE),
                "id": pd.Series([], dtype=STRING_DTYPE),
            }
        )
        df = mechanisms_from_blocks(none, {}, extras=True)
    df.attrs = {}
    return df


# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------


def _utc(t: date | datetime | str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _window(
    start: date | datetime | str | pd.Timestamp,
    end: date | datetime | str | pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp, list[date]]:
    """``[lo, hi)`` and the day directories :func:`read_sil_catalog` reads."""
    lo, hi = _utc(start), _utc(end)
    if hi < lo:
        raise ValueError(f"end {hi} is before start {lo}")
    first = (lo - pd.Timedelta(days=1)).date()
    last = (max(hi - pd.Timedelta(1, "ns"), lo) + pd.Timedelta(days=1)).date()
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    return lo, hi, days


def _cut(df: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    t = pd.DatetimeIndex(df.index)
    keep = np.asarray((t >= lo) & (t < hi), dtype=bool)
    return df.loc[keep]


def _event_filters() -> dict[str, Any]:
    return {
        "automatic": True,
        "match_tol": dataclasses.asdict(DEFAULT_MATCH_TOL),
        "match_tol_no_id": dataclasses.asdict(DEFAULT_MATCH_TOL_NO_ID),
    }


def _cache_root(
    use_cache: bool, cache_dir: str | os.PathLike[str] | None, root: Path
) -> Path | None:
    if not use_cache:
        return None
    _require_pyarrow()
    croot = sil_cache_dir(cache_dir)
    _check_outside(croot, root)
    return croot


# --------------------------------------------------------------------------
# Public readers
# --------------------------------------------------------------------------


def read_sil_catalog_cached(
    start: date | datetime | str | pd.Timestamp,
    end: date | datetime | str | pd.Timestamp,
    *,
    base: str | os.PathLike[str] | None = None,
    automatic: bool = True,
    match_tol: MatchTolerance = DEFAULT_MATCH_TOL,
    match_tol_no_id: MatchTolerance = DEFAULT_MATCH_TOL_NO_ID,
    cache_dir: str | os.PathLike[str] | None = None,
    recent_days: int = DEFAULT_RECENT_DAYS,
    today: date | None = None,
    use_cache: bool = True,
    refresh: bool = False,
    write: bool = True,
) -> tuple[pd.DataFrame, SilReadReport]:
    """:func:`read_sil_catalog` through the cache; same arguments and result.

    ``[start, end)`` half-open, naive times UTC, archive root as
    :func:`sil_dir`. Finished day directories (more than ``recent_days``
    before ``today``, default today UTC) come from the cache under
    ``cache_dir`` (:func:`sil_cache_dir`) when present, else from the archive
    and are then written to it (``write=False`` reads without writing);
    ``refresh=True`` ignores and rewrites what is cached. The frame and the
    report's counts equal :func:`read_sil_catalog`'s; ``report.seconds`` and
    ``attrs["catalog"]["fetched_at"]`` are this call's. The cache figures
    are in ``attrs["catalog"]["cache"]`` (:class:`CacheStats`).

    ``automatic=False`` or non-default tolerances, and ``use_cache=False``,
    read the archive through :func:`read_sil_catalog` (``cache.used`` false).
    """
    _check_recent(recent_days)
    if (
        not use_cache
        or not automatic
        or match_tol != DEFAULT_MATCH_TOL
        or match_tol_no_id != DEFAULT_MATCH_TOL_NO_ID
    ):
        df, rep = read_sil_catalog(
            start,
            end,
            base=base,
            automatic=automatic,
            match_tol=match_tol,
            match_tol_no_id=match_tol_no_id,
        )
        df.attrs[_schema.ATTRS_KEY]["cache"] = {"used": False}
        return df, rep
    t0 = _time.perf_counter()
    root = sil_dir(base)
    lo, hi, days = _window(start, end)
    croot = _cache_root(use_cache, cache_dir, root)
    stats = CacheStats(kind="events")
    got = _gather(
        "events",
        root,
        days,
        cache_root=croot,
        today=_today(today),
        recent_days=recent_days,
        refresh=refresh,
        write=write,
        keep_frames=True,
        stats=stats,
    )
    report = SilReadReport()
    frames: list[pd.DataFrame] = []
    win_rev = win_aut = 0
    for g in got:
        if g.frame is None:
            continue
        report = report + SilReadReport(
            dict(g.meta.get("counts", {})),
            tuple(g.meta.get("issues", ())),
            tuple(g.meta.get("fallback_dt_s", ())),
        )
        if not len(g.frame):
            continue
        cut = _cut(g.frame, lo, hi)
        is_aut = (cut["evaluation_mode"] == "automatic").to_numpy()
        win_rev += int((~is_aut).sum())
        win_aut += int(is_aut.sum())
        frames.append(cut)
    extra: dict[str, int] = {
        "days_missing": stats.days_missing,
        "window_reviewed": win_rev,
        "window_automatic": win_aut,
    }
    stats.seconds = _time.perf_counter() - t0
    report = report + SilReadReport(
        {k: v for k, v in extra.items() if v}, seconds=stats.seconds
    )
    nonempty = [f for f in frames if len(f)]
    if nonempty:
        df = pd.concat(nonempty)
    else:
        df = pd.DataFrame({INDEX_NAME: pd.DatetimeIndex([], tz="UTC")}).set_index(
            INDEX_NAME
        )
    df.attrs = {}
    out = coerce_catalog(df, source=SOURCE)
    prov = catalog_provenance(
        source=SOURCE,
        location=str(root),
        start=lo.to_pydatetime(),
        end=hi.to_pydatetime(),
        filters=_event_filters(),
        extra={
            "report": report.as_dict(),
            "cache": {"used": True, **stats.as_dict()},
        },
    )
    return with_provenance(out, prov), report


def read_sil_mechanisms_cached(
    start: date | datetime | str | pd.Timestamp,
    end: date | datetime | str | pd.Timestamp,
    *,
    base: str | os.PathLike[str] | None = None,
    extras: bool = False,
    plane_tol_deg: float = DEFAULT_PLANE_TOL_DEG,
    cache_dir: str | os.PathLike[str] | None = None,
    recent_days: int = DEFAULT_RECENT_DAYS,
    today: date | None = None,
    use_cache: bool = True,
    refresh: bool = False,
    write: bool = True,
) -> pd.DataFrame:
    """The ``mechanisms`` frame of ``[start, end)`` (by origin time).

    The period counterpart of :func:`read_sil_day_mechanisms` with the window
    rules of :func:`read_sil_catalog` (one padding day directory each side,
    cut by origin time, sorted stably by time): the concatenated day frames,
    cut, coerced, with ``extras`` columns only on request. Cache rules as
    :func:`read_sil_catalog_cached`; a non-default ``plane_tol_deg`` or
    ``use_cache=False`` reads the archive. Provenance in
    ``attrs["catalog"]`` (``frame="mechanisms"``) with the summed day
    reports under ``report`` and the cache figures under ``cache``.
    """
    _check_recent(recent_days)
    t0 = _time.perf_counter()
    root = sil_dir(base)
    lo, hi, days = _window(start, end)
    cached_ok = use_cache and plane_tol_deg == DEFAULT_PLANE_TOL_DEG
    stats = CacheStats(kind="mechanisms")
    if cached_ok:
        got = _gather(
            "mechanisms",
            root,
            days,
            cache_root=_cache_root(True, cache_dir, root),
            today=_today(today),
            recent_days=recent_days,
            refresh=refresh,
            write=write,
            keep_frames=True,
            stats=stats,
        )
    else:
        got = []
        for d in days:
            ddir = day_dir(root, d)
            if not ddir.is_dir():
                stats.days_missing += 1
                continue
            frame, rep = read_sil_day_mechanisms(
                ddir, extras=True, plane_tol_deg=plane_tol_deg
            )
            stats.days_read += 1
            got.append(_Day(d, frame, {"rows": len(frame)}, False))
    frames = [_cut(g.frame, lo, hi) for g in got if g.frame is not None]
    nonempty = [f for f in frames if len(f)]
    df = pd.concat(nonempty) if nonempty else reader_empty("mechanisms")
    df.attrs = {}
    out = coerce_mechanisms(df, source=SOURCE)
    keep = list(column_names("mechanisms")) + (list(EXTRA_COLUMNS) if extras else [])
    out = out[keep]
    stats.seconds = _time.perf_counter() - t0
    summary = _sum_mech_meta([g.meta for g in got if g.frame is not None])
    summary["window_rows"] = len(out)
    prov = catalog_provenance(
        source=SOURCE,
        location=str(root),
        start=lo.to_pydatetime(),
        end=hi.to_pydatetime(),
        filters={"extras": extras, "plane_tol_deg": plane_tol_deg},
        frame="mechanisms",
        extra={"report": summary, "cache": {"used": cached_ok, **stats.as_dict()}},
    )
    return with_provenance(out, prov)


def _sum_mech_meta(metas: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total: dict[str, Any] = {"days": len(metas), "rows_by_method": {}}
    for m in metas:
        for k, v in m.items():
            if k == "rows_by_method":
                for meth, n in v.items():
                    total["rows_by_method"][meth] = total["rows_by_method"].get(
                        meth, 0
                    ) + int(n)
            elif k == "max_block_mismatch_deg":
                total[k] = max(float(total.get(k, 0.0)), float(v))
            elif k == "has_fps":
                total["days_with_fps"] = total.get("days_with_fps", 0) + int(bool(v))
            elif isinstance(v, int | float):
                total[k] = total.get(k, 0) + v
    return total


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BuildSummary:
    """:func:`build_sil_cache`'s result: one :class:`CacheStats` per kind."""

    start: str
    end: str
    stats: Mapping[str, CacheStats]

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe summary."""
        return {
            "start": self.start,
            "end": self.end,
            "stats": {k: v.as_dict() for k, v in self.stats.items()},
        }


def build_sil_cache(
    start: date | datetime | str | pd.Timestamp,
    end: date | datetime | str | pd.Timestamp,
    *,
    base: str | os.PathLike[str] | None = None,
    kinds: Sequence[Kind] = KINDS,
    cache_dir: str | os.PathLike[str] | None = None,
    recent_days: int = DEFAULT_RECENT_DAYS,
    today: date | None = None,
    refresh: bool = False,
    progress: Callable[[str], None] | None = None,
) -> BuildSummary:
    """Fill the cache for every finished day directory of ``[start, end)``.

    Works year by year (one year in memory, one write per year and kind), so
    it can be stopped and restarted: a rerun skips the days already cached
    (``refresh=True`` rereads them). Days inside the recent window are not
    read at all. ``progress`` gets one line per year and kind. For the whole
    archive: ``build_sil_cache("1990-01-01", date.today())``.
    """
    _check_recent(recent_days)
    for k in kinds:
        if k not in KINDS:
            raise ValueError(f"unknown kind {k!r}; expected one of {KINDS}")
    root = sil_dir(base)
    lo, hi = _utc(start), _utc(end)
    if hi < lo:
        raise ValueError(f"end {hi} is before start {lo}")
    croot = _cache_root(True, cache_dir, root)
    day0 = lo.date()
    last = (hi - pd.Timedelta(1, "ns")).date() if hi > lo else day0 - timedelta(1)
    cutoff = _today(today) - timedelta(days=int(recent_days))
    last = min(last, cutoff - timedelta(days=1))
    totals: dict[str, CacheStats] = {k: CacheStats(kind=k) for k in kinds}
    year = day0.year
    while day0 <= last and year <= last.year:
        y0 = max(day0, date(year, 1, 1))
        y1 = min(last, date(year, 12, 31))
        days = [y0 + timedelta(days=i) for i in range((y1 - y0).days + 1)]
        for k in kinds:
            st = CacheStats(kind=k)
            t0 = _time.perf_counter()
            _gather(
                k,
                root,
                days,
                cache_root=croot,
                today=_today(today),
                recent_days=recent_days,
                refresh=refresh,
                write=True,
                keep_frames=False,
                stats=st,
            )
            st.seconds = _time.perf_counter() - t0
            totals[k].cache_dir = st.cache_dir
            totals[k].add(st)
            if progress is not None:
                progress(
                    f"{year} {k}: {st.days_written} written, {st.days_cached} "
                    f"cached, {st.days_missing} missing, {st.seconds:.1f} s"
                )
        year += 1
    return BuildSummary(str(lo.isoformat()), str(hi.isoformat()), totals)


__all__ = [
    "CACHE_FORMAT",
    "DAY_COLUMN",
    "DEFAULT_RECENT_DAYS",
    "ENV_SIL_CACHE",
    "KINDS",
    "MANIFEST_KEY",
    "BuildSummary",
    "CacheStats",
    "build_sil_cache",
    "read_sil_catalog_cached",
    "read_sil_mechanisms_cached",
    "reader_empty",
    "sil_cache_dir",
]
