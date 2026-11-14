"""Quakes API reader (SeisComP and SIL) onto the common schema (ticket T10).

:func:`read_quakes` reads the official IMO Quakes API (``PLAN-seismicity.md``
§3.1) for the half-open window ``[start, end)`` and returns an events frame
of :mod:`geo_dataread.seismic.schema`, source ``quakes-seiscomp``
(``system="seiscomp"``) or ``quakes-sil`` (``system="sil"``).

Endpoint (BGÓ Q2: internal by default)
---------------------------------------

=========  ================================================  ==================
Which      List / count                                      Selected by
=========  ================================================  ==================
internal   ``https://vi-api.vedur.is/quakes/internal/events``  default
public     ``https://api.vedur.is/quakes/events``              ``internal=False``
=========  ================================================  ==================

Precedence: the ``internal=`` argument, then ``$GPS_QUAKES_API``
(``internal`` or ``public``), then internal. The internal host resolves only
inside the IMO network (laptop and yang-d01 reach it, T9b). When it cannot be
reached, :class:`QuakesApiUnreachable` says how to switch; the reader never
falls back to the public endpoint by itself, because the public list is a
different catalogue (no ``magnitude_type``, ``phases`` or ``status``, and no
rejected events).

Requests
--------

* GeoJSON only (``format=json``); CSV differs in names and labels (T9).
* All four limits are always sent (``depth_min/max`` default 0–50 km, the
  API's own bounds; ``size_min/max`` default −3…10, the API's bounds; the
  server default ``size_max`` would be 7). Values outside the API bounds
  raise :class:`ValueError` before any request. Events deeper than 50 km or
  above 0 km cannot be fetched through this API at all.
* **Count first, then split.** The API has no paging and no server cap, so a
  window is sized by ``/events/count`` with the same filters; while the count
  is above ``FetchPolicy.max_events`` the window is bisected (midpoints on
  whole milliseconds). A window that is still too large at
  ``min_window`` is fetched anyway and counted (``over_limit_windows``). A
  window with count 0 is not listed. A list whose length differs from its
  count (events revised between the two calls) is counted
  (``count_mismatch``), not an error.
* The server treats the window as ``[start_time, end_time)`` (probed
  2026-11-14 at ms precision); the reader still cuts to the window and
  dedupes on ``event_id`` across sub-windows.
* Polite: ``FetchPolicy.pause_s`` between requests; a timeout per request;
  retries with backoff on HTTP 5xx only. 4xx raise :class:`QuakesApiError`
  with the server's ``detail``. Connection, DNS and timeout errors raise
  :class:`QuakesApiUnreachable` at once (no retry).
* The end of the request window is clamped to the API's upper bound
  (midnight of tomorrow, UTC); the asked window stays in the provenance.
* Transport is :mod:`urllib` (``requests`` is not a dependency of this
  package). Any ``requests.Session``-shaped object can be passed as
  ``session=`` (``get(url, timeout=, headers=)`` returning ``status_code`` and
  ``content``); tests pass a fake one.

Field mapping (GeoJSON feature → schema column)
-----------------------------------------------

=============================  ============================================
Schema column                  From
=============================  ============================================
``time`` (index)               ``properties.time`` (ISO, ``Z``)
``event_id``                   ``properties.event_id`` (``IMO…`` / ``SIL<n>``)
``longitude``, ``latitude``    ``geometry.coordinates[0]``, ``[1]``
``depth_km``                   ``properties.depth`` (km)
``magnitude``                  ``properties.magnitude`` (raw, Q7)
``magnitude_type``             ``properties.magnitude_type``; absent from the
                               public SeisComP list → NA (not guessed: the
                               internal list mixes ``ML_SIL``, ``Mpgv_w``
                               and ``M``)
``mag_mlw``                    ``magnitude`` where the type is ``Mlw``
``mag_ml``                     ``magnitude`` where the type is ``ML``
``mag_auto``                   ``magnitude`` where the type is ``aut.mag``
                               or ``Msil_auto``
``evaluation_mode``            ``properties.evaluation_mode`` (Q8: automatic
                               events are kept and labelled)
``evaluation_status``          ``properties.status`` (internal SeisComP
                               only; NA elsewhere, and NA where unset)
``event_type``                 ``properties.type`` when it is one of the
                               API's types, else ``other``; null → NA
``event_type_code``            ``properties.type``, raw
``n_phases``                   ``properties.phases`` (internal SeisComP)
``updated_time``               ``properties.updated_time`` (SeisComP)
``source``                     ``quakes-seiscomp`` / ``quakes-sil``
``sil_path``/``sil_aut_path``  NA: not in the list
``h_err_km``/``depth_err_km``  NA: detail only (T11; SIL detail is only in
                               the legacy API)
``moment_nm``, ``quality``,    NA: not published
``azimuthal_gap``
=============================  ============================================

Not mapped: ``properties.region`` (SeisComP region name; no schema column).
With ``extras=True`` it is kept as an extra column ``region`` (which
:func:`~geo_dataread.seismic.schema.validate` reports as a warning).

The frame is validated strictly before it is returned, so a magnitude type,
status or evaluation mode the schema does not know fails loudly
(:class:`~geo_dataread.seismic.schema.SchemaError`) instead of being
silently dropped. A ``"features": null`` body (a window before 1991) is an
empty catalogue. The :class:`QuakesReadReport` (requests, splits, retries,
mismatches) rides on ``df.attrs["catalog"]["report"]``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time as _time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import Any, Final, Literal, Protocol, TypeAlias

import numpy as np
import pandas as pd

from geo_dataread.seismic.schema import (
    EVENT_COLUMNS,
    EVENT_TYPES,
    catalog_provenance,
    coerce_catalog,
    validate,
    with_provenance,
)

System = Literal["seiscomp", "sil"]
TimeLike: TypeAlias = date | datetime | str | pd.Timestamp

#: ``system`` → schema source name (T23).
SOURCES_BY_SYSTEM: Final[Mapping[str, str]] = {
    "seiscomp": "quakes-seiscomp",
    "sil": "quakes-sil",
}

#: Base URLs; ``/events`` and ``/events/count`` are appended.
INTERNAL_BASE: Final = "https://vi-api.vedur.is/quakes/internal"
PUBLIC_BASE: Final = "https://api.vedur.is/quakes"

#: Environment variable choosing the endpoint: ``internal`` or ``public``.
ENV_ENDPOINT: Final = "GPS_QUAKES_API"

#: The API's own parameter bounds (OpenAPI 2026-08-06).
DEPTH_BOUNDS: Final = (0.0, 50.0)
SIZE_BOUNDS: Final = (-3.0, 10.0)

#: Generic User-Agent: no personal identifier in any request.
USER_AGENT: Final = "geo_dataread-quakes-api"

#: Magnitude types whose value also fills a per-scale column.
_SCALE_COLUMNS: Final[Mapping[str, str]] = {
    "Mlw": "mag_mlw",
    "ML": "mag_ml",
    "aut.mag": "mag_auto",
    "Msil_auto": "mag_auto",
}

#: Report counters that are always present (0 when nothing happened).
_REPORT_KEYS: Final = (
    "requests",
    "count_requests",
    "list_requests",
    "retries",
    "windows_listed",
    "windows_empty",
    "splits",
    "over_limit_windows",
    "count_mismatch",
    "features",
    "duplicates_dropped",
    "outside_window",
    "dropped_bad_position",
    "magnitude_type_missing",
)

#: Indirection so tests can count pauses without sleeping.
_sleep: Callable[[float], None] = _time.sleep


# --------------------------------------------------------------------------
# Errors, policy, report
# --------------------------------------------------------------------------


class QuakesApiError(RuntimeError):
    """The Quakes API answered with an error (4xx, or 5xx after the retries)."""

    def __init__(self, message: str, *, status: int | None = None, url: str = ""):
        super().__init__(message)
        self.status = status
        self.url = url


class QuakesApiUnreachable(QuakesApiError):
    """The host could not be reached (DNS, connection refused, timeout)."""


@dataclasses.dataclass(frozen=True)
class FetchPolicy:
    """How the reader talks to the API.

    ``max_events``: the largest count one list request may carry before its
    window is bisected. ``min_window``: no window is split below this span.
    ``pause_s``: sleep between consecutive requests. ``timeout_s``: per
    request. ``retries``: extra attempts on HTTP 5xx, with ``backoff_s`` ×
    2^attempt between them.
    """

    max_events: int = 10_000
    min_window: timedelta = timedelta(minutes=1)
    pause_s: float = 1.0
    timeout_s: float = 60.0
    retries: int = 3
    backoff_s: float = 2.0

    def __post_init__(self) -> None:
        if self.max_events < 1:
            raise ValueError("max_events must be >= 1")
        if self.min_window <= timedelta(0):
            raise ValueError("min_window must be positive")
        if self.pause_s < 0 or self.timeout_s <= 0 or self.backoff_s < 0:
            raise ValueError("pause_s/backoff_s must be >= 0, timeout_s > 0")
        if self.retries < 0:
            raise ValueError("retries must be >= 0")


DEFAULT_POLICY: Final = FetchPolicy()


@dataclasses.dataclass(frozen=True)
class QuakesReadReport:
    """What one :func:`read_quakes` call did. ``counts`` keys:

    ``requests`` (HTTP requests sent, retries included), ``count_requests``,
    ``list_requests``, ``retries``, ``windows_listed``, ``windows_empty``
    (count 0, not listed), ``splits``, ``over_limit_windows``,
    ``count_mismatch``, ``features``, ``duplicates_dropped``,
    ``outside_window``, ``out_of_range:<column>``, ``dropped_bad_position``,
    ``magnitude_type_missing``, ``events``, ``events_manual``,
    ``events_automatic``.
    """

    endpoint: str
    counts: Mapping[str, int]
    elapsed_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "counts": dict(sorted(self.counts.items())),
            "elapsed_s": round(self.elapsed_s, 3),
        }


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------


class Response(Protocol):
    """The part of a ``requests.Response`` the reader uses."""

    @property
    def status_code(self) -> int: ...

    @property
    def content(self) -> bytes: ...


class Session(Protocol):
    """The part of a ``requests.Session`` the reader uses."""

    def get(
        self, url: str, *, timeout: float, headers: Mapping[str, str]
    ) -> Response: ...


@dataclasses.dataclass(frozen=True)
class _UrllibResponse:
    status_code: int
    content: bytes


class UrllibSession:
    """Default transport: :mod:`urllib.request`, HTTP errors returned, not raised."""

    def get(
        self, url: str, *, timeout: float, headers: Mapping[str, str]
    ) -> _UrllibResponse:
        req = urllib.request.Request(url, headers=dict(headers), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return _UrllibResponse(int(resp.status), resp.read())
        except urllib.error.HTTPError as exc:
            body = exc.read() if exc.fp is not None else b""
            return _UrllibResponse(int(exc.code), body)


# --------------------------------------------------------------------------
# Endpoint and parameters
# --------------------------------------------------------------------------


def resolve_endpoint(internal: bool | None = None) -> tuple[str, bool]:
    """``(base_url, is_internal)``: the argument, then ``$GPS_QUAKES_API``,
    then internal. An unknown environment value raises :class:`ValueError`."""
    if internal is None:
        env = os.environ.get(ENV_ENDPOINT, "").strip().lower()
        if env in ("", "internal"):
            internal = True
        elif env == "public":
            internal = False
        else:
            raise ValueError(
                f"${ENV_ENDPOINT}={os.environ[ENV_ENDPOINT]!r}: "
                "expected 'internal' or 'public'"
            )
    return (INTERNAL_BASE if internal else PUBLIC_BASE), internal


def _utc(t: TimeLike, what: str) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(t)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} {t!r} is not a time") from exc
    if ts is pd.NaT:
        raise ValueError(f"{what} is NaT")
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _iso_ms(ts: pd.Timestamp) -> str:
    """ISO UTC with millisecond precision and ``Z`` (what the server stores)."""
    head = str(ts.strftime("%Y-%m-%dT%H:%M:%S."))
    return head + f"{ts.microsecond // 1000:03d}Z"


def _floor_ms(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.floor("ms")


def _check_range(name: str, lo: float, hi: float, bounds: tuple[float, float]) -> None:
    for label, v in ((f"{name}_min", lo), (f"{name}_max", hi)):
        if not math.isfinite(v) or not bounds[0] <= v <= bounds[1]:
            raise ValueError(
                f"{label}={v} is outside the API's bounds {bounds[0]}…{bounds[1]}"
            )
    if lo > hi:
        raise ValueError(f"{name}_min {lo} > {name}_max {hi}")


def polygon_wkt(
    polygon: str | Sequence[Sequence[float]] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> str | None:
    """The WKT polygon to send: ``polygon`` as WKT text or ``(lon, lat)``
    vertices (closed here), or ``bbox`` ``(min_lon, min_lat, max_lon, max_lat)``
    (GeoJSON order, as :func:`geo_dataread.seismic.filters.subset`). Not both."""
    if polygon is not None and bbox is not None:
        raise ValueError("give either polygon or bbox, not both")
    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError("bbox is (min_lon, min_lat, max_lon, max_lat)")
        w, s, e, n = (float(v) for v in bbox)
        if not (w < e and s < n):
            raise ValueError(f"bbox {bbox} is empty or inverted")
        polygon = [(w, s), (e, s), (e, n), (w, n)]
    if polygon is None:
        return None
    if isinstance(polygon, str):
        text = polygon.strip()
        if not text.upper().startswith("POLYGON"):
            raise ValueError("polygon text must be WKT: POLYGON((lon lat, …))")
        return text
    pts = [(float(p[0]), float(p[1])) for p in polygon]
    if any(len(p) != 2 for p in polygon):
        raise ValueError("polygon vertices are (lon, lat) pairs")
    if pts and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        raise ValueError("polygon needs at least 3 distinct vertices")
    if not all(math.isfinite(c) for p in pts for c in p):
        raise ValueError("polygon vertices must be finite")
    ring = [*pts, pts[0]]
    return "POLYGON((" + ", ".join(f"{x:.6f} {y:.6f}" for x, y in ring) + "))"


def _tomorrow_midnight(now: pd.Timestamp | None = None) -> pd.Timestamp:
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    return now.normalize() + pd.Timedelta(days=1)


# --------------------------------------------------------------------------
# HTTP with policy
# --------------------------------------------------------------------------


class _Client:
    def __init__(
        self,
        session: Session,
        base: str,
        internal: bool,
        policy: FetchPolicy,
        tally: Counter[str],
    ) -> None:
        self.session = session
        self.base = base
        self.internal = internal
        self.policy = policy
        self.tally = tally
        self._sent = False

    def _unreachable(self, url: str, exc: BaseException) -> QuakesApiUnreachable:
        host = urllib.parse.urlsplit(url).netloc
        msg = f"Quakes API host {host} could not be reached: {exc!r}."
        if self.internal:
            msg += (
                " The internal endpoint resolves only inside the IMO network."
                " To use the public endpoint instead, pass internal=False or set"
                f" {ENV_ENDPOINT}=public (note: the public SeisComP list has no"
                " magnitude_type, phases or status, and drops rejected events)."
                " The reader does not fall back by itself."
            )
        return QuakesApiUnreachable(msg, url=url)

    def get(self, path: str, params: Sequence[tuple[str, str]]) -> bytes:
        url = f"{self.base}{path}?{urllib.parse.urlencode(list(params))}"
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        attempt = 0
        while True:
            if self._sent and self.policy.pause_s > 0:
                _sleep(self.policy.pause_s)
            self._sent = True
            self.tally["requests"] += 1
            try:
                resp = self.session.get(
                    url, timeout=self.policy.timeout_s, headers=headers
                )
            except OSError as exc:  # URLError, socket.timeout, requests errors
                raise self._unreachable(url, exc) from exc
            status = int(resp.status_code)
            if 200 <= status < 300:
                return bytes(resp.content)
            if status >= 500 and attempt < self.policy.retries:
                attempt += 1
                self.tally["retries"] += 1
                if self.policy.backoff_s > 0:
                    _sleep(self.policy.backoff_s * 2 ** (attempt - 1))
                continue
            raise QuakesApiError(
                f"Quakes API {status} for {url}: {_detail(resp.content)}"
                + (f" (after {attempt} retries)" if attempt else ""),
                status=status,
                url=url,
            )

    def count(self, params: Sequence[tuple[str, str]]) -> int:
        self.tally["count_requests"] += 1
        body = self.get("/events/count", params)
        try:
            value = json.loads(body)
        except ValueError as exc:
            raise QuakesApiError(f"count is not JSON: {body[:200]!r}") from exc
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise QuakesApiError(f"count is not a non-negative integer: {value!r}")
        return int(value)

    def features(self, params: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
        self.tally["list_requests"] += 1
        body = self.get("/events", params)
        try:
            doc = json.loads(body)
        except ValueError as exc:
            raise QuakesApiError(f"list is not JSON: {body[:200]!r}") from exc
        if not isinstance(doc, dict) or doc.get("type") != "FeatureCollection":
            raise QuakesApiError(
                f"list is not a GeoJSON FeatureCollection: {str(doc)[:200]}"
            )
        feats = doc.get("features")
        if feats is None:  # a window before 1991 gives "features": null (T9)
            return []
        if not isinstance(feats, list):
            raise QuakesApiError(f"features is not a list: {type(feats).__name__}")
        return feats


def _detail(content: bytes) -> str:
    """The server's error text: FastAPI ``detail`` (list or str) or ``message``."""
    try:
        doc = json.loads(content)
    except ValueError:
        return content[:300].decode("utf-8", "replace")
    if isinstance(doc, dict):
        det = doc.get("detail", doc.get("message"))
        if isinstance(det, list):
            return "; ".join(
                str(d.get("msg", d)) if isinstance(d, dict) else str(d) for d in det
            )
        if det is not None:
            return str(det)
    return str(doc)[:300]


# --------------------------------------------------------------------------
# Window splitting
# --------------------------------------------------------------------------


def _collect(
    client: _Client,
    lo: pd.Timestamp,
    hi: pd.Timestamp,
    filters: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Features of ``[lo, hi)``, bisecting while the count exceeds the limit."""
    params = [("start_time", _iso_ms(lo)), ("end_time", _iso_ms(hi)), *filters]
    n = client.count(params)
    if n == 0:
        client.tally["windows_empty"] += 1
        return []
    policy = client.policy
    if n > policy.max_events:
        mid = _floor_ms(lo + (hi - lo) / 2)
        if hi - lo > policy.min_window and lo < mid < hi:
            client.tally["splits"] += 1
            return _collect(client, lo, mid, filters) + _collect(
                client, mid, hi, filters
            )
        client.tally["over_limit_windows"] += 1
    feats = client.features(params)
    client.tally["windows_listed"] += 1
    if len(feats) != n:
        client.tally["count_mismatch"] += 1
    return feats


# --------------------------------------------------------------------------
# Mapping
# --------------------------------------------------------------------------


def _feature_row(feat: Any) -> dict[str, Any]:
    if not isinstance(feat, dict):
        raise QuakesApiError(f"feature is not an object: {str(feat)[:200]}")
    props = feat.get("properties") or {}
    geom = feat.get("geometry") or {}
    coords = geom.get("coordinates") or [None, None]
    if not isinstance(props, dict) or len(coords) < 2:
        raise QuakesApiError(f"malformed feature: {str(feat)[:200]}")
    return {
        "time": props.get("time"),
        "event_id": props.get("event_id"),
        "longitude": coords[0],
        "latitude": coords[1],
        "depth_km": props.get("depth"),
        "magnitude": props.get("magnitude"),
        "magnitude_type": props.get("magnitude_type"),
        "evaluation_mode": props.get("evaluation_mode"),
        "evaluation_status": props.get("status"),
        "event_type_code": props.get("type"),
        "n_phases": props.get("phases"),
        "updated_time": props.get("updated_time"),
        "region": props.get("region"),
        "_has_magnitude_type": "magnitude_type" in props,
    }


_RANGED: Final = tuple(
    s
    for s in EVENT_COLUMNS
    if s.kind in ("float64", "Int64")
    and (s.minimum is not None or s.maximum is not None)
)


def _range_guard(df: pd.DataFrame, tally: Counter[str]) -> pd.DataFrame:
    """NA out-of-range values (counted); drop rows with an impossible position."""
    drop = np.zeros(len(df), dtype=bool)
    for spec in _RANGED:
        if spec.name not in df.columns:
            continue
        num = pd.to_numeric(df[spec.name], errors="coerce").astype("float64").to_numpy()
        with np.errstate(invalid="ignore"):
            bad = np.zeros(len(df), dtype=bool)
            if spec.minimum is not None:
                bad |= num < spec.minimum
            if spec.maximum is not None:
                bad |= num > spec.maximum
        if spec.name in ("latitude", "longitude"):
            bad |= np.isnan(num)
        if not bad.any():
            continue
        tally[f"out_of_range:{spec.name}"] += int(bad.sum())
        if not spec.nullable:
            drop |= bad
            continue
        df.loc[bad, spec.name] = np.nan
    if drop.any():
        tally["dropped_bad_position"] += int(drop.sum())
        df = df.loc[~drop]
    return df


def features_to_frame(
    features: Sequence[Any],
    *,
    system: System,
    extras: bool = False,
    tally: Counter[str] | None = None,
) -> pd.DataFrame:
    """Map GeoJSON features (one list response, or several concatenated) to a
    coerced (not yet validated) schema frame; see the module docstring."""
    tally = tally if tally is not None else Counter()
    source = SOURCES_BY_SYSTEM[system]
    rows = [_feature_row(f) for f in features]
    if not rows:
        empty = pd.DataFrame({"time": pd.DatetimeIndex([], tz="UTC")}).set_index("time")
        out = coerce_catalog(empty, source=source)
        if extras:
            out["region"] = pd.Series(index=out.index, dtype="str")
        return out
    df = pd.DataFrame(rows)
    missing = ~df.pop("_has_magnitude_type").astype(bool)
    if missing.any():
        tally["magnitude_type_missing"] += int(missing.sum())
    for col in ("longitude", "latitude", "depth_km", "magnitude"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    mtype = df["magnitude_type"].astype(object)
    for label, column in _SCALE_COLUMNS.items():
        hit = (mtype == label).to_numpy()
        if column not in df.columns:
            df[column] = np.nan
        df.loc[hit, column] = df.loc[hit, "magnitude"]
    code = df["event_type_code"].astype(object)
    known = set(EVENT_TYPES)
    df["event_type"] = [
        (
            None
            if v is None or (isinstance(v, float) and math.isnan(v))
            else (v if v in known else "other")
        )
        for v in code
    ]
    df = _range_guard(df, tally)
    if not extras:
        df = df.drop(columns=["region"])
    df["time"] = pd.to_datetime(df["time"], utc=False, format="ISO8601")
    df["updated_time"] = pd.to_datetime(df["updated_time"], utc=False, format="ISO8601")
    return coerce_catalog(df.set_index("time"), source=source)


# --------------------------------------------------------------------------
# Public reader
# --------------------------------------------------------------------------


def _filter_params(
    *,
    system: System,
    depth: tuple[float, float],
    size: tuple[float, float],
    wkt: str | None,
    evaluation_mode: Sequence[str] | None,
    event_type: Sequence[str] | None,
) -> list[tuple[str, str]]:
    params: list[tuple[str, str]] = [
        ("size_min", f"{size[0]:g}"),
        ("size_max", f"{size[1]:g}"),
        ("depth_min", f"{depth[0]:g}"),
        ("depth_max", f"{depth[1]:g}"),
        ("system", system),
        ("format", "json"),
    ]
    if wkt is not None:
        params.append(("polygon", wkt))
    for mode in evaluation_mode or ():
        if mode not in ("manual", "automatic"):
            raise ValueError(f"evaluation_mode {mode!r}: manual or automatic")
        params.append(("evaluation_mode", mode))
    for etype in event_type or ():
        if etype not in EVENT_TYPES:
            raise ValueError(f"event_type {etype!r} is not one of {EVENT_TYPES}")
        params.append(("type", etype))
    return params


def read_quakes(
    start: TimeLike,
    end: TimeLike,
    *,
    system: System = "seiscomp",
    internal: bool | None = None,
    polygon: str | Sequence[Sequence[float]] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    depth_min: float = DEPTH_BOUNDS[0],
    depth_max: float = DEPTH_BOUNDS[1],
    size_min: float = SIZE_BOUNDS[0],
    size_max: float = SIZE_BOUNDS[1],
    evaluation_mode: Sequence[str] | None = None,
    event_type: Sequence[str] | None = None,
    policy: FetchPolicy = DEFAULT_POLICY,
    session: Session | None = None,
    extras: bool = False,
) -> pd.DataFrame:
    """Read the Quakes API for ``[start, end)`` into a schema events frame.

    ``start``/``end``: times without a zone are UTC. ``system``: ``seiscomp``
    (source ``quakes-seiscomp``) or ``sil`` (``quakes-sil``). ``internal``:
    ``None`` → ``$GPS_QUAKES_API`` (``internal``/``public``) → internal.
    Region: ``polygon`` (WKT text, or ``(lon, lat)`` vertices) or ``bbox``
    ``(min_lon, min_lat, max_lon, max_lat)``, sent to the server. Depth and
    size limits are always sent (defaults: the API's full bounds, 0–50 km and
    −3…10). ``evaluation_mode`` / ``event_type``: optional server filters
    (lists; the API applies ``type`` to manual events only). Default: every
    event, automatic ones included and labelled (Q8).

    ``policy`` sets the per-request event limit (windows are bisected by
    count), the pause between requests, the timeout and the 5xx retries.
    ``session``: a ``requests.Session``-shaped transport; default urllib.
    ``extras=True`` keeps the API's ``region`` as an extra column.

    Raises :class:`QuakesApiUnreachable` when the host cannot be reached
    (never falls back to the other endpoint), :class:`QuakesApiError` on an
    HTTP error, :class:`ValueError` on bad arguments, and
    :class:`~geo_dataread.seismic.schema.SchemaError` when the response does
    not fit the schema (e.g. a new magnitude type).
    """
    t0 = _time.perf_counter()
    if system not in SOURCES_BY_SYSTEM:
        raise ValueError(f"system {system!r}: 'seiscomp' or 'sil'")
    lo, hi = _utc(start, "start"), _utc(end, "end")
    if hi < lo:
        raise ValueError(f"end {hi} is before start {lo}")
    _check_range("depth", float(depth_min), float(depth_max), DEPTH_BOUNDS)
    _check_range("size", float(size_min), float(size_max), SIZE_BOUNDS)
    wkt = polygon_wkt(polygon, bbox)
    filters = _filter_params(
        system=system,
        depth=(float(depth_min), float(depth_max)),
        size=(float(size_min), float(size_max)),
        wkt=wkt,
        evaluation_mode=evaluation_mode,
        event_type=event_type,
    )
    base, is_internal = resolve_endpoint(internal)
    tally: Counter[str] = Counter({k: 0 for k in _REPORT_KEYS})
    client = _Client(session or UrllibSession(), base, is_internal, policy, tally)

    req_lo = _floor_ms(lo)
    req_hi = min(hi.ceil("ms"), _tomorrow_midnight())
    feats: list[dict[str, Any]] = []
    if req_lo < req_hi:
        feats = _collect(client, req_lo, req_hi, filters)
    tally["features"] = len(feats)

    df = features_to_frame(feats, system=system, extras=extras, tally=tally)
    dup = df["event_id"].duplicated(keep="first").to_numpy()
    if dup.any():
        tally["duplicates_dropped"] = int(dup.sum())
        df = df.loc[~dup]
    inside = np.asarray((df.index >= lo) & (df.index < hi), dtype=bool)
    if not inside.all():
        tally["outside_window"] = int((~inside).sum())
        df = df.loc[inside]
    is_aut = (df["evaluation_mode"] == "automatic").to_numpy()
    tally["events"] = len(df)
    tally["events_automatic"] = int(is_aut.sum())
    tally["events_manual"] = int((~is_aut).sum())

    report = QuakesReadReport(
        endpoint=f"{base}/events",
        counts=dict(tally),
        elapsed_s=_time.perf_counter() - t0,
    )
    validate(df, strict=True)
    prov = catalog_provenance(
        source=SOURCES_BY_SYSTEM[system],
        location=f"{base}/events",
        start=lo.to_pydatetime(),
        end=hi.to_pydatetime(),
        filters={
            "system": system,
            "endpoint": "internal" if is_internal else "public",
            "depth_km": [float(depth_min), float(depth_max)],
            "size": [float(size_min), float(size_max)],
            "polygon": wkt,
            "evaluation_mode": list(evaluation_mode) if evaluation_mode else None,
            "event_type": list(event_type) if event_type else None,
            "max_events": policy.max_events,
        },
        extra={
            "report": report.as_dict(),
            "notes": (
                []
                if is_internal or system == "sil"
                else [
                    "public SeisComP list: magnitude_type, n_phases and "
                    "evaluation_status are not published (NA); rejected "
                    "events are not listed"
                ]
            ),
        },
    )
    return with_provenance(df, prov)


__all__ = [
    "DEFAULT_POLICY",
    "DEPTH_BOUNDS",
    "ENV_ENDPOINT",
    "INTERNAL_BASE",
    "PUBLIC_BASE",
    "SIZE_BOUNDS",
    "SOURCES_BY_SYSTEM",
    "USER_AGENT",
    "FetchPolicy",
    "QuakesApiError",
    "QuakesApiUnreachable",
    "QuakesReadReport",
    "Session",
    "UrllibSession",
    "features_to_frame",
    "polygon_wkt",
    "read_quakes",
    "resolve_endpoint",
]
