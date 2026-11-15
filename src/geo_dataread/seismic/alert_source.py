"""Where the alertMap files come from, and the PGV magnitude of a SIL event (T7c).

PLAN-seismicity.md §7 (T7c) and §9c (B2, B3, B7); formats in
``docs/seismic/sil_alert_discovery.md``. The parsers, the relation, the
estimator and the pure association are in :mod:`geo_dataread.seismic.sil_alert`.

**Sources** (:class:`AlertSource`). The alertMap tree is
``<root>/<yyyy>/<mon>/<dd>/NNN.event`` (``mon`` = lower-case English
three-letter month), with ``NNN.html``, ``NNN.list``, the ``.jpg`` maps and
a day ``index.html`` next to them.

* :class:`DirectoryAlertSource`: one or more local roots (a mount of hellir
  ``/mnt/sdc1/alert`` and, e.g., an archive mount holding ``<yyyy>/…`` of
  frumgogn04, e.g. ``GPS_ALERTMAP_DIR=/mnt_data/alert:/mnt_data/alert_archive``).
  For each year the first root whose ``<yyyy>`` resolves to a directory wins
  (a dangling symlink in the primary root never does). A directory that
  exists but may not be read (archive ``2011/may``) raises
  :class:`AlertAccessDenied`, never an empty day. A year that no root has (on the laptop 2016–2020 are
  symlinks into ``/net/frumgogn04.vedur.is/…``, which is not reachable)
  raises :class:`AlertYearUnavailable` naming every root and why, unless an
  explicit ``fallback`` source (HTTP) is given. Days are listed from the
  directory (``NNN.event`` files).
* :class:`HttpAlertSource`: ``http://sil.vedur.is/alert/`` (the default,
  same content as ``hraun.vedur.is/ja/alert/``). Days are listed from the
  day ``index.html`` (its ``NNN.html`` links), **never by probing numbers**
  (B2). Downloads of finished days (older than ``recent_days``, default 2:
  ``.event`` files are rewritten up to about 2 h after the detection) are
  kept in an XDG cache, ``$GPS_ALERTMAP_CACHE``, else
  ``<sil cache root>/alertmap/<host>/`` (:func:`alertmap_cache_dir`), with
  the archive layout. Nothing is ever written into a data directory.

**Choice of source** (:func:`resolve_alert_source`): an argument, else
``$GPS_ALERTMAP_DIR`` (an :data:`os.pathsep` list of roots), else
``postprocess.cfg`` ``[PATHS] alertmap_dir`` → directory; otherwise HTTP
(``$GPS_ALERTMAP_URL`` or the default URL). A configured root that does not
exist **raises** (:class:`AlertSourceUnavailable`, saying how to switch):
there is no silent fallback. ``$GPS_ALERTMAP_FALLBACK=http`` (or
``http_fallback=True``) makes a directory source read the years none of its
roots has over HTTP.

**Day index times are not used.** The ``index.html`` row time of a
detection can be stale: in a sample of 2021–2026 days 5 of ~120 rows
disagreed with the ``.event`` header (by up to 2 h 40 min; on 2025-01-16
#054 the header, the ``.html`` title and the file time agree and the index
does not). The index gives the list of ids only; association reads every
``.event`` of the days in the window.

**Orchestration**: :func:`read_alert_pgv` takes a SIL event id
(``/2026/nov/12/21:50:00/53:43:797``) or an :class:`Origin`, reads the evb
from the SIL archive (:func:`geo_dataread.seismic.sil_catalog.sil_dir`),
reads the detections of the days the association window touches (the
previous or next day near midnight), associates
(:func:`~geo_dataread.seismic.sil_alert.associate_alert`) and estimates
(:func:`~geo_dataread.seismic.sil_alert.estimate_pgv_magnitude`) with the
``"page"`` or ``"legacy"`` method.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
import re
import tempfile
import time as _time
import urllib.parse
from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final, Literal, Protocol, TypeAlias, runtime_checkable

import pandas as pd

from geo_dataread.seismic.cache import sil_cache_dir
from geo_dataread.seismic.quakes_api import Session, UrllibSession
from geo_dataread.seismic.sil_alert import (
    AlertAssociation,
    AlertEvent,
    AlertFormatError,
    AlertMethod,
    EvbFile,
    FitSettings,
    Origin,
    PgvResult,
    PgvSettings,
    associate_alert,
    association_window,
    estimate_pgv_magnitude,
    parse_alert_day_index,
    parse_alert_event,
    read_evb,
    Rank,
    resolve_method,
)
from geo_dataread.seismic.sil_catalog import MONTHS, sil_dir

logger = logging.getLogger(__name__)

__all__ = [
    "ALERT_RECENT_DAYS",
    "CACHE_MARKER",
    "CONFIG_KEY_ALERTMAP_DIR",
    "DEFAULT_ALERTMAP_URL",
    "ENV_ALERTMAP_CACHE",
    "ENV_ALERTMAP_DIR",
    "ENV_ALERTMAP_FALLBACK",
    "ENV_ALERTMAP_URL",
    "AlertAccessDenied",
    "AlertDayNotFound",
    "AlertEventNotFound",
    "AlertPgv",
    "AlertSource",
    "AlertSourceError",
    "AlertSourceUnavailable",
    "AlertYearUnavailable",
    "DirectoryAlertSource",
    "HttpAlertSource",
    "SourceChoice",
    "alert_day_path",
    "alertmap_cache_dir",
    "alertmap_dirs",
    "alertmap_url",
    "evb_path",
    "read_alert_day",
    "read_alert_pgv",
    "resolve_alert_source",
    "sil_event_day",
]

#: ``$GPS_ALERTMAP_DIR``: alertMap root(s), an :data:`os.pathsep` list.
ENV_ALERTMAP_DIR: Final = "GPS_ALERTMAP_DIR"
#: ``$GPS_ALERTMAP_URL``: base URL of the HTTP source.
ENV_ALERTMAP_URL: Final = "GPS_ALERTMAP_URL"
#: ``$GPS_ALERTMAP_CACHE``: cache root of the HTTP source.
ENV_ALERTMAP_CACHE: Final = "GPS_ALERTMAP_CACHE"
#: ``$GPS_ALERTMAP_FALLBACK``: ``http`` lets a directory source read the
#: years none of its roots has over HTTP (off by default).
ENV_ALERTMAP_FALLBACK: Final = "GPS_ALERTMAP_FALLBACK"
#: ``postprocess.cfg`` ``[PATHS]`` key of the alertMap root(s).
CONFIG_KEY_ALERTMAP_DIR: Final = "alertmap_dir"
#: The HTTP source host (internal; ``http://hraun.vedur.is/ja/alert/`` serves
#: the same files).
DEFAULT_ALERTMAP_URL: Final = "http://sil.vedur.is/alert/"
#: Days younger than this (UTC) are never cached. ``.event`` files are
#: rewritten up to ~2 h after the detection (Sep–Oct 2026: max 1.9 h).
ALERT_RECENT_DAYS: Final = 2
#: File at the root of an HTTP cache tree (marks it as a cache, not data).
CACHE_MARKER: Final = ".geo_dataread-alertmap-cache"
#: User-Agent of the HTTP source.
USER_AGENT: Final = "geo_dataread-alertmap"

#: How :func:`resolve_alert_source` picks a source.
SourceChoice: TypeAlias = Literal["auto", "directory", "http"]

_EVENT_FILE: Final = re.compile(r"^(\d{3,})\.event$")
_SIL_ID: Final = re.compile(
    r"^/?(\d{4})/([a-z]{3})/(\d{2})/(\d{2}):(\d{2}):(\d{2})/(\d{2}):(\d{2}):(\d{3})$"
)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class AlertSourceError(RuntimeError):
    """An alertMap source could not deliver what was asked."""


class AlertSourceUnavailable(AlertSourceError, FileNotFoundError):
    """A configured source cannot be used at all (missing mount, no network)."""


class AlertYearUnavailable(AlertSourceUnavailable):
    """No root of a directory source has this year (e.g. a dangling symlink)."""


class AlertAccessDenied(AlertSourceUnavailable, PermissionError):
    """A directory exists but may not be read (e.g. archive ``2011/may``)."""


class AlertDayNotFound(AlertSourceError, LookupError):
    """The source has the year but not this day (no directory, index 404)."""


class AlertEventNotFound(AlertSourceError, LookupError):
    """A listed ``NNN.event`` is missing."""


# --------------------------------------------------------------------------
# Paths, ids, configuration
# --------------------------------------------------------------------------


def alert_day_path(day: date) -> str:
    """``yyyy/mon/dd`` of ``day`` (fixed English months, no ``locale``)."""
    return f"{day.year:04d}/{MONTHS[day.month - 1]}/{day.day:02d}"


def sil_event_day(event_id: str) -> date:
    """The day of a SIL event id ``/yyyy/mon/dd/HH:MM:SS/MM:SS:mmm``."""
    m = _SIL_ID.match(event_id.strip())
    if not m or m.group(2) not in MONTHS:
        raise ValueError(
            f"not a SIL event id {event_id!r} (expected /yyyy/mon/dd/HH:MM:SS/MM:SS:mmm)"
        )
    return date(int(m.group(1)), MONTHS.index(m.group(2)) + 1, int(m.group(3)))


def evb_path(event_id: str, sil_path: str | os.PathLike[str] | None = None) -> Path:
    """``<sil_dir>/<id>.evb``; ``sil_path`` as :func:`sil_dir` resolves it.

    The id's leading ``/`` is stripped (the legacy ``os.path.join`` with an
    absolute second part dropped ``sil_path``).
    """
    sil_event_day(event_id)
    return sil_dir(sil_path) / (event_id.strip().lstrip("/") + ".evb")


def _config_alertmap_dir() -> str | None:
    try:
        from gps_parser import ConfigParser

        value = ConfigParser().getPostProcessDir(CONFIG_KEY_ALERTMAP_DIR)
    except Exception:  # noqa: BLE001 - no gpsconfig, or the key is not deployed
        return None
    return str(value) if value else None


def _split_roots(value: str) -> tuple[Path, ...]:
    return tuple(Path(p).expanduser() for p in value.split(os.pathsep) if p.strip())


def alertmap_dirs(
    value: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None = None,
) -> tuple[Path, ...]:
    """The configured alertMap roots, or ``()`` when none is configured.

    ``value`` (a path, an :data:`os.pathsep` list or a sequence), else
    ``$GPS_ALERTMAP_DIR``, else ``postprocess.cfg`` ``[PATHS] alertmap_dir``.
    Existence is not checked here (:class:`DirectoryAlertSource` does).
    """
    if value is not None and not isinstance(value, str | os.PathLike):
        return tuple(Path(v).expanduser() for v in value)
    if value:
        return _split_roots(os.fspath(value))
    env = os.environ.get(ENV_ALERTMAP_DIR, "")
    if env.strip():
        return _split_roots(env)
    cfg = _config_alertmap_dir()
    return _split_roots(cfg) if cfg else ()


def alertmap_url(url: str | None = None) -> str:
    """``url``, else ``$GPS_ALERTMAP_URL``, else :data:`DEFAULT_ALERTMAP_URL`;
    always with a trailing ``/``."""
    base = url or os.environ.get(ENV_ALERTMAP_URL, "").strip() or DEFAULT_ALERTMAP_URL
    return base if base.endswith("/") else base + "/"


def alertmap_cache_dir(
    cache_dir: str | os.PathLike[str] | None = None, *, url: str | None = None
) -> Path:
    """The HTTP cache root: ``cache_dir`` as given, else ``$GPS_ALERTMAP_CACHE``,
    else ``<sil_cache_dir()>/alertmap``; the last two get a ``<host>`` part
    (from ``url``) so two servers never share files."""
    if cache_dir:
        return Path(cache_dir).expanduser()
    env = os.environ.get(ENV_ALERTMAP_CACHE, "").strip()
    root = Path(env).expanduser() if env else sil_cache_dir() / "alertmap"
    parts = urllib.parse.urlsplit(alertmap_url(url))
    host = re.sub(r"[^A-Za-z0-9._-]+", "_", parts.netloc + parts.path).strip("_")
    return root / (host or "default")


def _fallback_enabled(value: bool | None) -> bool:
    if value is not None:
        return value
    env = os.environ.get(ENV_ALERTMAP_FALLBACK, "").strip().lower()
    if env in ("", "0", "no", "off", "none"):
        return False
    if env == "http":
        return True
    raise ValueError(f"${ENV_ALERTMAP_FALLBACK}={env!r}: expected 'http' or unset")


# --------------------------------------------------------------------------
# The protocol
# --------------------------------------------------------------------------


@runtime_checkable
class AlertSource(Protocol):
    """Read access to an alertMap tree, by day."""

    @property
    def name(self) -> str:
        """Where the files come from (a path list or a URL)."""
        ...

    def list_day(self, day: date) -> tuple[int, ...]:
        """Sorted detection numbers of ``day``; :class:`AlertDayNotFound`
        when the source lacks the day."""
        ...

    def read_event(self, day: date, alert_id: int) -> AlertEvent:
        """Detection ``alert_id`` of ``day`` (parsed, ``strict=False``)."""
        ...

    def local_day_dir(self, day: date) -> Path:
        """A local directory holding the day's ``NNN.event`` files (the
        mount itself, or the cache after a download)."""
        ...


def _parse(text: str, source: str) -> AlertEvent:
    # strict=False: files grow after the detection, so the header count may lag.
    return parse_alert_event(text, source=source, strict=False)


# --------------------------------------------------------------------------
# Directory source
# --------------------------------------------------------------------------


def _denied(path: Path, exc: OSError) -> AlertAccessDenied:
    return AlertAccessDenied(
        f"alertMap directory {path} exists but cannot be read ({exc.strerror or exc}); "
        "this is a permission problem on the mount, not an empty day"
    )


def _is_dir(path: Path) -> bool:
    """``Path.is_dir`` that raises :class:`AlertAccessDenied` instead of
    passing a permission error off as "no directory"."""
    try:
        return path.is_dir()
    except PermissionError as exc:
        raise _denied(path, exc) from exc


class DirectoryAlertSource:
    """alertMap files under one or more local roots (read-only).

    ``roots`` are tried in order per year (``<root>/<yyyy>``). Every root
    must exist when ``check`` (the default): a configured mount that is not
    there raises :class:`AlertSourceUnavailable`. ``fallback`` (normally an
    :class:`HttpAlertSource`) serves the years no root has; without it such a
    year raises :class:`AlertYearUnavailable`.
    """

    def __init__(
        self,
        roots: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        *,
        fallback: AlertSource | None = None,
        check: bool = True,
    ) -> None:
        self.roots = alertmap_dirs(roots)
        if not self.roots:
            raise ValueError("DirectoryAlertSource: no root given")
        self.fallback = fallback
        self.fallback_years: set[int] = set()
        if check:
            missing = [str(r) for r in self.roots if not _is_dir(r)]
            if missing:
                raise AlertSourceUnavailable(
                    f"alertMap root(s) {missing} not found (configured via "
                    f"argument, ${ENV_ALERTMAP_DIR} or postprocess.cfg [PATHS] "
                    f"{CONFIG_KEY_ALERTMAP_DIR}). Mount it read-only, or unset "
                    f"${ENV_ALERTMAP_DIR} / pass source='http' to read "
                    f"{DEFAULT_ALERTMAP_URL} instead."
                )

    @property
    def name(self) -> str:
        return os.pathsep.join(str(r) for r in self.roots)

    def __repr__(self) -> str:
        return f"DirectoryAlertSource({self.name!r}, fallback={self.fallback!r})"

    def year_dir(self, year: int) -> Path | None:
        """The directory of ``year``; ``None`` when the fallback serves it.

        Raises :class:`AlertYearUnavailable` (every root with its reason)
        when no root has it and there is no fallback.
        """
        reasons: list[str] = []
        for root in self.roots:
            p = root / f"{year:04d}"
            try:
                if p.is_dir():
                    return p
            except PermissionError:
                reasons.append(f"{p} (permission denied)")
                continue
            if p.is_symlink():
                try:
                    target = os.readlink(p)
                except OSError as exc:  # pragma: no cover - raced away
                    target = f"? ({exc})"
                reasons.append(f"{p} -> {target} (symlink target unreachable)")
            elif p.exists():
                reasons.append(f"{p} (not a directory)")
            else:
                reasons.append(f"{p} (missing)")
        if self.fallback is not None:
            if year not in self.fallback_years:
                logger.info(
                    "alertMap %d not under %s; reading %s",
                    year,
                    self.name,
                    self.fallback.name,
                )
                self.fallback_years.add(year)
            return None
        raise AlertYearUnavailable(
            f"alertMap year {year} is not readable: {'; '.join(reasons)}. "
            f"Mount it (e.g. frumgogn04:/data/alert/{year}) and add the root "
            f"holding <yyyy>/ to ${ENV_ALERTMAP_DIR} (an {os.pathsep!r}-separated "
            f"list), set ${ENV_ALERTMAP_FALLBACK}=http to read this year from "
            f"{DEFAULT_ALERTMAP_URL}, or unset ${ENV_ALERTMAP_DIR} to use HTTP only."
        )

    def _day_dir(self, day: date) -> Path | None:
        ydir = self.year_dir(day.year)
        if ydir is None:
            return None
        ddir = ydir / MONTHS[day.month - 1] / f"{day.day:02d}"
        if not _is_dir(ddir):
            mdir = ddir.parent
            if _is_dir(mdir) and not os.access(mdir, os.R_OK | os.X_OK):
                raise _denied(mdir, PermissionError(13, "Permission denied"))
            raise AlertDayNotFound(f"no alertMap day directory {ddir}")
        return ddir

    def list_day(self, day: date) -> tuple[int, ...]:
        ddir = self._day_dir(day)
        if ddir is None:
            assert self.fallback is not None
            return self.fallback.list_day(day)
        ids = []
        try:
            names = [p.name for p in ddir.iterdir()]
        except PermissionError as exc:
            raise _denied(ddir, exc) from exc
        for name in names:
            m = _EVENT_FILE.match(name)
            if m:
                ids.append(int(m.group(1)))
        return tuple(sorted(ids))

    def read_event(self, day: date, alert_id: int) -> AlertEvent:
        ddir = self._day_dir(day)
        if ddir is None:
            assert self.fallback is not None
            return self.fallback.read_event(day, alert_id)
        path = ddir / f"{alert_id:03d}.event"
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise AlertEventNotFound(f"no alertMap file {path}") from None
        except PermissionError as exc:
            raise _denied(path, exc) from exc
        return _parse(data.decode("latin-1"), str(path))

    def local_day_dir(self, day: date) -> Path:
        ddir = self._day_dir(day)
        if ddir is None:
            assert self.fallback is not None
            return self.fallback.local_day_dir(day)
        return ddir


# --------------------------------------------------------------------------
# HTTP source
# --------------------------------------------------------------------------


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _sleep(seconds: float) -> None:
    _time.sleep(seconds)


class HttpAlertSource:
    """alertMap files over HTTP with a cache of finished days.

    ``base_url`` per :func:`alertmap_url`; ``cache_dir`` per
    :func:`alertmap_cache_dir` (an explicit one is used as given, with the
    ``yyyy/mon/dd/`` layout); ``use_cache=False`` keeps everything in
    memory. Days less than ``recent_days`` before ``today`` (UTC) are kept in
    memory only and re-read by a new instance. ``session`` is anything with
    ``get(url, *, timeout, headers)`` returning ``status_code`` and
    ``content`` (``urllib`` by default; tests pass a stub). ``pause_s``
    spaces the requests.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        use_cache: bool = True,
        recent_days: int = ALERT_RECENT_DAYS,
        session: Session | None = None,
        timeout_s: float = 30.0,
        pause_s: float = 0.05,
        today: date | None = None,
    ) -> None:
        if int(recent_days) < 1:
            raise ValueError(f"recent_days must be >= 1, got {recent_days}")
        self.base_url = alertmap_url(base_url)
        self.cache_dir = (
            alertmap_cache_dir(cache_dir, url=self.base_url) if use_cache else None
        )
        self.recent_days = int(recent_days)
        self.session: Session = session if session is not None else UrllibSession()
        self.timeout_s = float(timeout_s)
        self.pause_s = float(pause_s)
        self._today = today
        self._memory: dict[tuple[str, str], bytes] = {}
        self._sent = False
        self.requests = 0

    @property
    def name(self) -> str:
        return self.base_url

    def __repr__(self) -> str:
        return f"HttpAlertSource({self.base_url!r}, cache_dir={self.cache_dir!r})"

    def _finished(self, day: date) -> bool:
        today = self._today if self._today is not None else datetime.now(UTC).date()
        return day <= today - timedelta(days=self.recent_days)

    def url(self, day: date, filename: str) -> str:
        """The URL of ``filename`` in the day directory."""
        return f"{self.base_url}{alert_day_path(day)}/{filename}"

    def _mark(self) -> None:
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            marker = self.cache_dir / CACHE_MARKER
            if not marker.exists():
                marker.write_text(f"alertMap HTTP cache of {self.base_url}\n")

    def _cache_path(self, day: date, filename: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / alert_day_path(day) / filename

    def _get(self, day: date, filename: str) -> bytes | None:
        """The file's bytes, ``None`` on 404."""
        key = (alert_day_path(day), filename)
        if key in self._memory:
            return self._memory[key]
        cpath = self._cache_path(day, filename)
        finished = self._finished(day)
        if finished and cpath is not None and cpath.is_file():
            return cpath.read_bytes()
        url = self.url(day, filename)
        if self._sent and self.pause_s > 0:
            _sleep(self.pause_s)
        self._sent = True
        self.requests += 1
        try:
            resp = self.session.get(
                url, timeout=self.timeout_s, headers={"User-Agent": USER_AGENT}
            )
        except OSError as exc:
            raise AlertSourceUnavailable(
                f"alertMap server could not be reached for {url}: {exc!r}. "
                f"{DEFAULT_ALERTMAP_URL} resolves only inside the IMO network; set "
                f"${ENV_ALERTMAP_URL}, or ${ENV_ALERTMAP_DIR} for a mounted tree."
            ) from exc
        status = int(resp.status_code)
        if status == 404:
            return None
        if not 200 <= status < 300:
            raise AlertSourceError(f"HTTP {status} for {url}")
        data = bytes(resp.content)
        if finished and cpath is not None:
            self._mark()
            _atomic_write(cpath, data)
        else:
            self._memory[key] = data
        return data

    def list_day(self, day: date) -> tuple[int, ...]:
        data = self._get(day, "index.html")
        if data is None:
            raise AlertDayNotFound(
                f"no alertMap day index {self.url(day, 'index.html')}"
            )
        idx = parse_alert_day_index(
            data.decode("latin-1"), day, source=self.url(day, "index.html")
        )
        return tuple(sorted(set(int(i) for i in idx["alert_id"])))

    def read_event(self, day: date, alert_id: int) -> AlertEvent:
        name = f"{alert_id:03d}.event"
        data = self._get(day, name)
        if data is None:
            raise AlertEventNotFound(f"no alertMap file {self.url(day, name)}")
        return _parse(data.decode("latin-1"), self.url(day, name))

    def local_day_dir(self, day: date) -> Path:
        """Download the day's ``index.html`` and ``NNN.event`` files.

        A finished day lands in the cache day directory; a recent day (or
        ``use_cache=False``) in ``<cache>/_recent/yyyy/mon/dd`` (or a fresh
        temporary directory), rewritten on each call.
        """
        ids = self.list_day(day)
        if self.cache_dir is not None and self._finished(day):
            ddir = self.cache_dir / alert_day_path(day)
            for i in ids:
                self._get(day, f"{i:03d}.event")
            return ddir
        if self.cache_dir is not None:
            self._mark()
            ddir = self.cache_dir / "_recent" / alert_day_path(day)
        else:
            ddir = Path(tempfile.mkdtemp(prefix="alertmap-"))
        ddir.mkdir(parents=True, exist_ok=True)
        idx = self._get(day, "index.html")
        if idx is not None:
            _atomic_write(ddir / "index.html", idx)
        for i in ids:
            data = self._get(day, f"{i:03d}.event")
            if data is not None:
                _atomic_write(ddir / f"{i:03d}.event", data)
        return ddir


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


def resolve_alert_source(
    source: SourceChoice | AlertSource | None = None,
    *,
    alertmap_dir: (
        str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None
    ) = None,
    url: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    http_fallback: bool | None = None,
    session: Session | None = None,
) -> AlertSource:
    """The alertMap source to use (B2).

    ``source``: an :class:`AlertSource` (returned as is), ``"directory"``
    (roots required), ``"http"`` or ``"auto"``/``None``: the directory
    source when roots are configured (:func:`alertmap_dirs`), else HTTP. A
    configured root that does not exist raises
    :class:`AlertSourceUnavailable`; it never falls back to HTTP by itself.
    ``http_fallback`` (else ``$GPS_ALERTMAP_FALLBACK=http``) lets the
    directory source read the years none of its roots has over HTTP.
    """
    if source is not None and not isinstance(source, str):
        if not isinstance(source, AlertSource):
            raise TypeError(f"not an AlertSource: {source!r}")
        return source
    choice = source or "auto"
    if choice not in ("auto", "directory", "http"):
        raise ValueError(
            f"source={choice!r}: expected 'auto', 'directory', 'http' or an AlertSource"
        )

    def http() -> HttpAlertSource:
        return HttpAlertSource(url, cache_dir=cache_dir, session=session)

    if choice == "http":
        return http()
    roots = alertmap_dirs(alertmap_dir)
    if not roots:
        if choice == "directory":
            raise AlertSourceUnavailable(
                f"source='directory' but no alertMap root is configured: pass "
                f"alertmap_dir=, set ${ENV_ALERTMAP_DIR} or postprocess.cfg [PATHS] "
                f"{CONFIG_KEY_ALERTMAP_DIR}"
            )
        return http()
    fallback = http() if _fallback_enabled(http_fallback) else None
    return DirectoryAlertSource(roots, fallback=fallback)


# --------------------------------------------------------------------------
# Reading days and the event-level PGV magnitude
# --------------------------------------------------------------------------


def read_alert_day(
    source: AlertSource, day: date
) -> tuple[list[tuple[date, AlertEvent]], list[str]]:
    """Every detection of ``day`` as ``(day, event)``, plus problems as text.

    A listed file that is missing or does not parse is reported in the
    second list and skipped; :class:`AlertDayNotFound` and
    :class:`AlertSourceUnavailable` propagate.
    """
    events: list[tuple[date, AlertEvent]] = []
    problems: list[str] = []
    for alert_id in source.list_day(day):
        try:
            events.append((day, source.read_event(day, alert_id)))
        except (AlertEventNotFound, AlertFormatError) as exc:
            problems.append(f"{day.isoformat()} #{alert_id:03d}: {exc}")
    return events, problems


def _days(lo: pd.Timestamp, hi: pd.Timestamp) -> Iterator[date]:
    d = lo.date()
    while d <= hi.date():
        yield d
        d += timedelta(days=1)


@dataclasses.dataclass(frozen=True)
class AlertPgv:
    """PGV magnitude of one SIL event (:func:`read_alert_pgv`).

    ``association`` says which detection was used and why (all candidates
    with their fit), ``result`` is the :class:`PgvResult` (``None`` when no
    detection was associated). :attr:`stations` is the per-station table
    (:data:`~geo_dataread.seismic.sil_alert.PGV_STATION_COLUMNS`).
    """

    event_id: str | None
    method: AlertMethod
    origin: Origin
    evb: EvbFile | None
    association: AlertAssociation
    result: PgvResult | None
    source: str

    @property
    def found(self) -> bool:
        """``True`` when a detection was associated."""
        return self.result is not None

    @property
    def magnitude(self) -> float:
        """The median (``Mpgv_alert``), NaN when nothing was found."""
        return self.result.magnitude if self.result is not None else float("nan")

    @property
    def stations(self) -> pd.DataFrame:
        """Per-station table of the chosen detection (empty when none)."""
        if self.result is None:
            from geo_dataread.seismic.sil_alert import PGV_STATION_COLUMNS

            return pd.DataFrame(columns=list(PGV_STATION_COLUMNS))
        return self.result.stations


def read_alert_pgv(
    event: str | Origin,
    *,
    method: str | AlertMethod = "page",
    source: SourceChoice | AlertSource | None = None,
    sil_path: str | os.PathLike[str] | None = None,
    origin: Origin | None = None,
    evb: EvbFile | None = None,
    window_s: float | None = None,
    settings: PgvSettings | None = None,
    fit: FitSettings | None = None,
    rank: Rank | None = None,
    min_fit_fraction: float | None = None,
) -> AlertPgv:
    """PGV magnitude (``Mpgv_alert``) of a SIL event from the alertMap files.

    ``event``: a SIL event id (``/2026/nov/12/21:50:00/53:43:797``; its evb
    is read from :func:`~geo_dataread.seismic.sil_catalog.sil_dir`
    ``(sil_path)``) or an :class:`Origin` (no evb: the window is origin ±
    ``window_s`` and the ``"legacy"`` method, which needs evb distances,
    is refused unless ``evb`` is passed). ``origin`` overrides the evb
    origin (e.g. the ``cleanaut.mag`` one the Mpgv page used); ``evb`` skips
    reading the file.

    ``method``: ``"page"`` (B3: alert-file coordinates, 15 < r <= 230 km,
    no ``ada``, field 6 ≠ 22 (provisional), exact inverse, median; 150 s
    window, travel-time fit required) or ``"legacy"`` (``find_alert``:
    stations common to evb and ``.event``, evb ``dist``, far-field inverse,
    median; 150 s window, candidates ranked by travel-time fit, at least
    30 % of the stations fitting, the full fit rule not required), ``"find_alert"`` (the legacy one exactly: 30 s window,
    closest header time) or an :class:`AlertMethod`. ``window_s``,
    ``settings``, ``fit``, ``rank`` and ``min_fit_fraction`` override the
    method's.

    ``source``: see :func:`resolve_alert_source`. The days the window
    touches are read (the previous or next one too near midnight); a day the
    source lacks is listed in ``association.missing_days``; a year that a
    directory source cannot read raises :class:`AlertYearUnavailable`.
    """
    m = resolve_method(
        method,
        window_s=window_s,
        settings=settings,
        fit=fit,
        rank=rank,
        min_fit_fraction=min_fit_fraction,
    )
    event_id: str | None = None
    if isinstance(event, str):
        event_id = event.strip()
        if evb is None:
            evb = read_evb(evb_path(event_id, sil_path))
        org = origin or evb.origin
    elif isinstance(event, Origin):
        org = origin or event
    else:
        raise TypeError(f"event must be a SIL event id or an Origin, not {event!r}")
    if m.settings.distance == "evb" and evb is None:
        raise ValueError(
            f"method {m.name!r} takes distances from the evb: pass a SIL event id "
            "or evb="
        )
    src = resolve_alert_source(source)
    window = association_window(org, evb, m.window_s)
    pairs: list[tuple[date, AlertEvent]] = []
    problems: list[str] = []
    missing: list[str] = []
    for day in _days(*window):
        try:
            evs, probs = read_alert_day(src, day)
        except AlertDayNotFound as exc:
            logger.info("%s", exc)
            missing.append(day.isoformat())
            continue
        pairs.extend(evs)
        problems.extend(probs)
    assoc = associate_alert(
        org,
        pairs,
        window=window,
        fit=m.fit,
        require_fit=m.require_fit,
        rank=m.rank,
        min_fit_fraction=m.min_fit_fraction,
        method=m.name,
        ellps=m.settings.ellps,
    )
    assoc = dataclasses.replace(
        assoc, missing_days=tuple(missing), errors=tuple(problems)
    )
    result: PgvResult | None = None
    if assoc.chosen is not None:
        result = estimate_pgv_magnitude(
            assoc.chosen,
            org,
            evb=evb,
            settings=m.settings,
            fit=m.fit,
            event_id=event_id,
        )
    return AlertPgv(
        event_id=event_id,
        method=m,
        origin=org,
        evb=evb,
        association=assoc,
        result=result,
        source=src.name,
    )
