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

Detail on request (T11)
-----------------------

:func:`read_quake_detail` reads ``/events/{event_id}`` (SeisComP only) into a
:class:`QuakeDetail`: every origin (uncertainties normalised to 1σ km and
flagged by method: LOCSAT km at 90 %, SIL-imported degrees at 1σ), every
magnitude, the preferred ones, and ``.to_catalog()`` (one schema row with
``h_err_km``, ``depth_err_km``, ``azimuthal_gap`` and the per-scale
magnitudes). :func:`read_quake_details` fetches a bounded *sample*
(explicit ``max_events``, hard ceiling :data:`MAX_DETAIL_SAMPLE`), never a
catalogue. The unit rules are in the comment block above
:class:`QuakeOrigin`.
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
from statistics import NormalDist
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
from geo_dataread.seismic.sil_catalog import KM_PER_DEG

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


# --------------------------------------------------------------------------
# Detail on request (T11)
# --------------------------------------------------------------------------
#
# ``/events/{event_id}`` answers QuakeML as JSON (SeisComP events only;
# ``SIL…`` ids are 404 there, their detail is in the legacy Skjálftalísa API,
# :func:`geo_dataread.seismic.skjalftalisa_api.read_skjalftalisa_detail`).
#
# Shape: every number is a string; ``origin`` and ``magnitude`` are a dict
# when there is one and a list when there are several (T9). The public
# detail is reduced to the preferred origin and magnitude, with no ids,
# authors, methods or quality.
#
# Uncertainty units (T9, measured on the fixtures)
# ------------------------------------------------
#
# ===========  =====================================  ========================
# Method       Recognised by                          ``latitude/longitude``
#                                                     ``.uncertainty``
# ===========  =====================================  ========================
# ``locsat``   ``methodID`` ends in ``LOCSAT``        km, at ``confidenceLevel``
#                                                     (90 %, per axis)
# ``sil``      author ``SIL``, earth model            degrees, 1σ (the SIL file
#              ``smi:local/imo/SIL`` or publicID      errors)
#              containing ``/imo/SIL``
# ``unknown``  anything else (e.g. the public         not used: NaN
#              detail, which names no method)
# ===========  =====================================  ========================
#
# Normalised values (``*_err_km``, ``time_err_s``) are 1σ in km (s):
#
# * A value with a ``confidenceLevel`` p is divided by the two-sided normal
#   quantile z = Φ⁻¹(0.5 + p/200) (1.645 for 90 %). Evidence that LOCSAT's
#   per-axis numbers are 90 % one-dimensional values: on IMO2026rusibb's
#   manual origin the 2-D 90 % ellipse axes / 2.146 (310.6 → 144.7 m,
#   211.3 → 98.5 m) equal the lat/lon uncertainties / 1.645 (236.8 → 144.0 m,
#   163.7 → 99.5 m).
# * SIL degrees become km as :func:`~geo_dataread.seismic.sil_catalog.
#   horizontal_error_km` does: latitude × 111.19, longitude × 111.19 ×
#   cos(lat). SeisComP's import wrote ``originUncertainty`` without cos(lat),
#   so that block is kept raw and not used.
# * ``h_err_km`` is the larger of the two 1σ axes, the same rule as the SIL
#   files and the legacy API, so the sources compare like with like.
# * A depth with ``depthType: operator assigned`` (no uncertainty) has
#   ``depth_err_km`` NaN and ``depth_fixed=True``.
# * Self-check (``uncertainty_consistent``): for SIL origins the two axes ×
#   111.195 km must equal ``min/maxHorizontalUncertainty``; for LOCSAT
#   hypot(lat, lon) must equal ``horizontalUncertainty``. ``False`` means the
#   server's units may have changed: the values are kept, the flag rides into
#   the provenance. ``None``: nothing to check against.

UncertaintyMethod = Literal["locsat", "sil", "unknown"]

#: Hard ceiling of :func:`read_quake_details`: details are for samples,
#: never for whole catalogues (plan §7 T11).
MAX_DETAIL_SAMPLE: Final = 500

#: Degree → km for the SIL self-check: SeisComP's import uses the mean
#: Earth radius 6371 km (111.195 km/deg), not :data:`KM_PER_DEG` (111.19).
_IMPORT_KM_PER_DEG: Final = 6371.0 * math.pi / 180.0

#: Detail magnitude labels → per-scale schema column. ``Msil_lw`` / ``Msil_l``
#: are SeisComP's names for the SIL Mlw / ML of a back-imported event
#: (IMO2024qnogqh: ``Msil_lw`` 0.25 = SIL1352381's Mlw). They are not in
#: :data:`~geo_dataread.seismic.schema.MAGNITUDE_TYPES` and never reach the
#: ``magnitude_type`` column; only their values fill these columns.
_DETAIL_SCALE_COLUMNS: Final[Mapping[str, str]] = {
    "Mlw": "mag_mlw",
    "Msil_lw": "mag_mlw",
    "ML": "mag_ml",
    "Msil_l": "mag_ml",
    "aut.mag": "mag_auto",
    "Msil_auto": "mag_auto",
}


class QuakeDetailNotFound(QuakesApiError, LookupError):
    """The Quakes API has no detail for this event id (HTTP 404)."""


def _qnum(value: Any) -> float:
    """A QuakeML JSON number (a string, a number or absent) → float/NaN."""
    if value is None or isinstance(value, bool):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _qint(value: Any) -> int | None:
    v = _qnum(value)
    return None if math.isnan(v) else int(v)


def _qstr(value: Any) -> str | None:
    return None if value is None else str(value)


def _quantity(block: Any) -> tuple[float, float, float]:
    """``(value, uncertainty, confidence_level)`` of a QuakeML quantity."""
    if not isinstance(block, Mapping):
        return _qnum(block), math.nan, math.nan
    return (
        _qnum(block.get("value")),
        _qnum(block.get("uncertainty")),
        _qnum(block.get("confidenceLevel")),
    )


def sigma_divisor(confidence_level: float) -> float:
    """The divisor turning a two-sided ``confidence_level`` (percent) value
    into 1σ: Φ⁻¹(0.5 + p/200); 1.0 when no level is given (NaN)."""
    if math.isnan(confidence_level):
        return 1.0
    if not 0.0 < confidence_level < 100.0:
        raise ValueError(f"confidence level {confidence_level} is not in (0, 100)")
    return NormalDist().inv_cdf(0.5 + confidence_level / 200.0)


def _as_list(value: Any, what: str) -> list[Mapping[str, Any]]:
    """Normalise the dict-for-one / list-for-several shape (T9)."""
    if value is None:
        return []
    items = [value] if isinstance(value, Mapping) else value
    if not isinstance(items, list) or not all(isinstance(i, Mapping) for i in items):
        raise QuakesApiError(f"detail {what} is not an object or a list of objects")
    return list(items)


def _ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value) if value is not None else pd.NaT
    if ts is pd.NaT:
        return pd.Timestamp("NaT")
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


@dataclasses.dataclass(frozen=True)
class QuakeOrigin:
    """One origin (solution) of a detail. NaN/None where not given.

    Normalised (1σ, km/s): ``time_err_s``, ``lat_err_km``, ``lon_err_km``,
    ``depth_err_km``, ``h_err_km`` (the larger horizontal axis). Raw, as
    served: ``*_uncertainty_raw`` with ``confidence_level`` (percent, NaN =
    none stated), the ``sigma_factor`` divided out, and ``origin_uncertainty``
    (ellipse/ellipsoid block, metres). ``uncertainty_method`` and
    ``uncertainty_unit`` say how the raw lat/lon numbers were read (module
    notes above). Quality: ``n_phases`` = ``usedPhaseCount`` (what the list's
    ``phases`` gives), distances in degrees.
    """

    origin_id: str | None
    time: pd.Timestamp
    latitude: float
    longitude: float
    depth_km: float
    time_err_s: float
    lat_err_km: float
    lon_err_km: float
    depth_err_km: float
    h_err_km: float
    uncertainty_method: UncertaintyMethod
    uncertainty_unit: Literal["deg", "km"] | None
    confidence_level: float
    sigma_factor: float
    uncertainty_consistent: bool | None
    time_uncertainty_raw: float
    latitude_uncertainty_raw: float
    longitude_uncertainty_raw: float
    depth_uncertainty_raw: float
    depth_fixed: bool
    depth_type: str | None
    evaluation_mode: str | None
    evaluation_status: str | None
    origin_type: str | None
    author: str | None
    method_id: str | None
    earth_model_id: str | None
    creation_time: pd.Timestamp
    n_phases: int | None
    n_phases_associated: int | None
    n_stations: int | None
    azimuthal_gap: float
    standard_error: float
    minimum_distance_deg: float
    median_distance_deg: float
    maximum_distance_deg: float
    origin_uncertainty: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class QuakeMagnitude:
    """One magnitude of a detail: the raw ``magnitude_type`` label (e.g.
    ``ML_SIL``, ``Mpgv_w``, ``Msil_lw``), value, uncertainty (as served; no
    confidence level is given), station count, and the origin it belongs to."""

    magnitude_id: str | None
    magnitude_type: str | None
    value: float
    uncertainty: float
    station_count: int | None
    origin_id: str | None
    method_id: str | None
    author: str | None
    evaluation_status: str | None
    creation_time: pd.Timestamp


def _uncertainty_method(o: Mapping[str, Any]) -> UncertaintyMethod:
    method = str(o.get("methodID") or "")
    if method.rstrip("/").upper().endswith("LOCSAT"):
        return "locsat"
    author = str((o.get("creationInfo") or {}).get("author") or "")
    model = str(o.get("earthModelID") or "")
    pid = str(o.get("publicID") or "")
    if author == "SIL" or model == "smi:local/imo/SIL" or "/imo/SIL" in pid:
        return "sil"
    return "unknown"


def _close(a: float, b: float, rel: float = 2e-3) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-6)


def _parse_origin(o: Mapping[str, Any]) -> QuakeOrigin:
    _, t_unc, t_cl = _quantity(o.get("time"))
    lat, lat_unc, lat_cl = _quantity(o.get("latitude"))
    lon, lon_unc, lon_cl = _quantity(o.get("longitude"))
    dep, dep_unc, dep_cl = _quantity(o.get("depth"))
    tblock = o.get("time")
    tval = tblock.get("value") if isinstance(tblock, Mapping) else tblock
    method = _uncertainty_method(o)
    ou = o.get("originUncertainty")
    ou = dict(ou) if isinstance(ou, Mapping) else {}
    depth_type = _qstr(o.get("depthType"))
    depth_fixed = depth_type == "operator assigned"

    unit: Literal["deg", "km"] | None
    consistent: bool | None = None
    if method == "sil":
        unit = "deg"
        lat_km = lat_unc * KM_PER_DEG
        lon_km = lon_unc * KM_PER_DEG * math.cos(math.radians(lat))
        lo_m = _qnum(ou.get("minHorizontalUncertainty"))
        hi_m = _qnum(ou.get("maxHorizontalUncertainty"))
        axes = sorted((lat_unc, lon_unc))
        if not any(math.isnan(v) for v in (*axes, lo_m, hi_m)):
            consistent = _close(axes[0] * _IMPORT_KM_PER_DEG * 1e3, lo_m) and _close(
                axes[1] * _IMPORT_KM_PER_DEG * 1e3, hi_m
            )
    elif method == "locsat":
        unit = "km"
        lat_km = lat_unc / sigma_divisor(lat_cl)
        lon_km = lon_unc / sigma_divisor(lon_cl)
        h_m = _qnum(ou.get("horizontalUncertainty"))
        if not any(math.isnan(v) for v in (lat_unc, lon_unc, h_m)):
            consistent = _close(math.hypot(lat_unc, lon_unc) * 1e3, h_m)
    else:
        unit = None
        lat_km = lon_km = math.nan
    if method == "unknown":
        dep_err = t_err = math.nan
    else:
        dep_err = math.nan if depth_fixed else dep_unc / sigma_divisor(dep_cl)
        t_err = t_unc / sigma_divisor(t_cl)
    with np.errstate(invalid="ignore"):
        h_err = float(np.fmax(lat_km, lon_km))
    levels = [v for v in (lat_cl, lon_cl) if not math.isnan(v)]
    cl = levels[0] if levels and method == "locsat" else math.nan
    q = o.get("quality")
    q = q if isinstance(q, Mapping) else {}
    info = o.get("creationInfo")
    info = info if isinstance(info, Mapping) else {}
    return QuakeOrigin(
        origin_id=_qstr(o.get("publicID")),
        time=_ts(tval),
        latitude=lat,
        longitude=lon,
        depth_km=dep,
        time_err_s=t_err,
        lat_err_km=lat_km,
        lon_err_km=lon_km,
        depth_err_km=dep_err,
        h_err_km=h_err,
        uncertainty_method=method,
        uncertainty_unit=unit,
        confidence_level=cl,
        sigma_factor=sigma_divisor(cl),
        uncertainty_consistent=consistent,
        time_uncertainty_raw=t_unc,
        latitude_uncertainty_raw=lat_unc,
        longitude_uncertainty_raw=lon_unc,
        depth_uncertainty_raw=dep_unc,
        depth_fixed=depth_fixed,
        depth_type=depth_type,
        evaluation_mode=_qstr(o.get("evaluationMode")),
        evaluation_status=_qstr(o.get("evaluationStatus")),
        origin_type=_qstr(o.get("type")),
        author=_qstr(info.get("author")),
        method_id=_qstr(o.get("methodID")),
        earth_model_id=_qstr(o.get("earthModelID")),
        creation_time=_ts(info.get("creationTime")),
        n_phases=_qint(q.get("usedPhaseCount")),
        n_phases_associated=_qint(q.get("associatedPhaseCount")),
        n_stations=_qint(q.get("usedStationCount")),
        azimuthal_gap=_qnum(q.get("azimuthalGap")),
        standard_error=_qnum(q.get("standardError")),
        minimum_distance_deg=_qnum(q.get("minimumDistance")),
        median_distance_deg=_qnum(q.get("medianDistance")),
        maximum_distance_deg=_qnum(q.get("maximumDistance")),
        origin_uncertainty=ou,
    )


def _parse_magnitude(m: Mapping[str, Any]) -> QuakeMagnitude:
    value, unc, _ = _quantity(m.get("mag"))
    info = m.get("creationInfo")
    info = info if isinstance(info, Mapping) else {}
    return QuakeMagnitude(
        magnitude_id=_qstr(m.get("publicID")),
        magnitude_type=_qstr(m.get("type")),
        value=value,
        uncertainty=unc,
        station_count=_qint(m.get("stationCount")),
        origin_id=_qstr(m.get("originID")),
        method_id=_qstr(m.get("methodID")),
        author=_qstr(info.get("author")),
        evaluation_status=_qstr(m.get("evaluationStatus")),
        creation_time=_ts(info.get("creationTime")),
    )


@dataclasses.dataclass(frozen=True)
class QuakeDetail:
    """The detail of one SeisComP event (``/events/{event_id}``).

    ``origins`` / ``magnitudes``: all of them, in server order.
    ``preferred_origin`` / ``preferred_magnitude``: by the event's
    ``preferredOriginID`` / ``preferredMagnitudeID``; in the public detail
    (no ids, one of each) the single one. ``event_type_code``: the event's
    ``type``, raw. ``region``: ``description.text``. ``raw``: the decoded
    answer.
    """

    event_id: str
    event_type_code: str | None
    region: str | None
    origins: tuple[QuakeOrigin, ...]
    magnitudes: tuple[QuakeMagnitude, ...]
    preferred_origin_id: str | None
    preferred_magnitude_id: str | None
    url: str
    internal: bool
    raw: Mapping[str, Any]

    @property
    def preferred_origin(self) -> QuakeOrigin | None:
        if self.preferred_origin_id is None:
            return self.origins[0] if len(self.origins) == 1 else None
        return next(
            (o for o in self.origins if o.origin_id == self.preferred_origin_id), None
        )

    @property
    def preferred_magnitude(self) -> QuakeMagnitude | None:
        if self.preferred_magnitude_id is None:
            return self.magnitudes[0] if len(self.magnitudes) == 1 else None
        return next(
            (
                m
                for m in self.magnitudes
                if m.magnitude_id == self.preferred_magnitude_id
            ),
            None,
        )

    def magnitudes_of(self, origin_id: str | None) -> tuple[QuakeMagnitude, ...]:
        """The magnitudes computed for one origin."""
        return tuple(m for m in self.magnitudes if m.origin_id == origin_id)

    def scale_magnitudes(self) -> dict[str, float]:
        """``mag_mlw`` / ``mag_ml`` / ``mag_auto`` from the detail's labels
        (``Msil_lw``/``Mlw``, ``Msil_l``/``ML``, ``Msil_auto``/``aut.mag``).

        A value on the preferred origin wins; otherwise the first one on any
        other origin of the same event is taken (``Msil_auto`` sits on the
        SIL automatic origin, which is rarely the preferred one). NaN where
        absent.
        """
        out = {c: math.nan for c in ("mag_mlw", "mag_ml", "mag_auto")}
        pref = self.preferred_origin
        pref_id = pref.origin_id if pref is not None else None
        ordered = sorted(self.magnitudes, key=lambda m: m.origin_id != pref_id)
        for m in ordered:
            col = _DETAIL_SCALE_COLUMNS.get(m.magnitude_type or "")
            if col is not None and math.isnan(out[col]) and not math.isnan(m.value):
                out[col] = m.value
        return out

    def to_catalog(self, *, extras: bool = False) -> pd.DataFrame:
        """A one-row schema events frame (source ``quakes-seiscomp``) of the
        preferred origin and magnitude, with ``h_err_km``, ``depth_err_km``,
        ``azimuthal_gap`` and the per-scale magnitudes filled.

        The list columns are mapped exactly as :func:`read_quakes` maps them
        (same code path), so the row equals that event's list row in the
        columns both have. ``updated_time`` is NaT (the detail does not carry
        the list's value). ``extras=True`` keeps ``region``. Raises
        :class:`ValueError` without a preferred origin, and
        :class:`~geo_dataread.seismic.schema.SchemaError` when the preferred
        magnitude type is not a schema type.
        """
        o = self.preferred_origin
        if o is None:
            raise ValueError(f"event {self.event_id}: no preferred origin")
        m = self.preferred_magnitude
        props: dict[str, Any] = {
            "event_id": self.event_id,
            "time": o.time.isoformat() if o.time is not pd.NaT else None,
            "depth": o.depth_km,
            "magnitude": m.value if m is not None else None,
            "evaluation_mode": o.evaluation_mode,
            "status": o.evaluation_status,
            "type": self.event_type_code,
            "phases": o.n_phases,
            "updated_time": None,
            "region": self.region,
        }
        if m is not None:
            props["magnitude_type"] = m.magnitude_type
        feat = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [o.longitude, o.latitude]},
            "properties": props,
        }
        tally: Counter[str] = Counter()
        df = features_to_frame([feat], system="seiscomp", extras=extras, tally=tally)
        fill = {
            "h_err_km": o.h_err_km,
            "depth_err_km": o.depth_err_km,
            "azimuthal_gap": o.azimuthal_gap,
            **self.scale_magnitudes(),
        }
        for col, val in fill.items():
            spec = next(s for s in EVENT_COLUMNS if s.name == col)
            ok = not math.isnan(val) and (spec.minimum is None or val >= spec.minimum)
            ok = ok and (spec.maximum is None or val <= spec.maximum)
            if not ok and not math.isnan(val):
                tally[f"out_of_range:{col}"] += 1
            if len(df):
                df[col] = np.array([val if ok else np.nan], dtype="float64")
        notes = []
        if o.uncertainty_method == "unknown":
            notes.append(
                "origin method not stated (public detail?): uncertainties not "
                "normalised, h_err_km/depth_err_km NA; use internal=True"
            )
        if o.uncertainty_consistent is False:
            notes.append(
                "lat/lon uncertainties disagree with originUncertainty: the "
                "server's units may have changed"
            )
        prov = catalog_provenance(
            source=SOURCES_BY_SYSTEM["seiscomp"],
            location=self.url,
            extra={
                "detail": True,
                "endpoint": "internal" if self.internal else "public",
                "preferred_origin_id": o.origin_id,
                "preferred_magnitude_id": m.magnitude_id if m is not None else None,
                "uncertainty_method": o.uncertainty_method,
                "confidence_level": (
                    None if math.isnan(o.confidence_level) else o.confidence_level
                ),
                "sigma_factor": o.sigma_factor,
                "uncertainty_consistent": o.uncertainty_consistent,
                "depth_fixed": o.depth_fixed,
                "n_origins": len(self.origins),
                "n_magnitudes": len(self.magnitudes),
                "counts": dict(tally),
                "notes": notes,
            },
        )
        with_provenance(df, prov)
        validate(df, strict=True)
        return df


def _event_id_of(public_id: str | None) -> str | None:
    if not public_id:
        return None
    return public_id.rstrip("/").rsplit("/", 1)[-1]


def parse_quake_detail(
    body: Any, *, event_id: str | None = None, url: str = "", internal: bool = True
) -> QuakeDetail:
    """A decoded detail answer → :class:`QuakeDetail` (no network).

    ``event_id``: the id asked for; it must match the answer's ``publicID``
    when that is present (the public detail has none). Raises
    :class:`QuakesApiError` for a malformed answer, or when a preferred id
    names no origin/magnitude of the answer.
    """
    if not isinstance(body, Mapping) or "origin" not in body:
        raise QuakesApiError(f"{url or 'detail'}: not a QuakeML event answer", url=url)
    origins = tuple(_parse_origin(o) for o in _as_list(body.get("origin"), "origin"))
    mags = tuple(
        _parse_magnitude(m) for m in _as_list(body.get("magnitude"), "magnitude")
    )
    served_id = _event_id_of(_qstr(body.get("publicID")))
    if event_id is not None and served_id is not None and served_id != event_id:
        raise QuakesApiError(
            f"asked for {event_id}, the answer is {served_id}", url=url
        )
    eid = served_id or event_id
    if eid is None:
        raise QuakesApiError("detail answer without an event id", url=url)
    pref_o = _qstr(body.get("preferredOriginID"))
    pref_m = _qstr(body.get("preferredMagnitudeID"))
    if pref_o is not None and pref_o not in {o.origin_id for o in origins}:
        raise QuakesApiError(f"{eid}: preferredOriginID {pref_o} not among origins")
    if pref_m is not None and pref_m not in {m.magnitude_id for m in mags}:
        raise QuakesApiError(
            f"{eid}: preferredMagnitudeID {pref_m} not among magnitudes"
        )
    desc = body.get("description")
    region = _qstr(desc.get("text")) if isinstance(desc, Mapping) else None
    return QuakeDetail(
        event_id=eid,
        event_type_code=_qstr(body.get("type")),
        region=region,
        origins=origins,
        magnitudes=mags,
        preferred_origin_id=pref_o,
        preferred_magnitude_id=pref_m,
        url=url,
        internal=internal,
        raw=dict(body),
    )


def _check_detail_args(event_id: str, system: str) -> str:
    if system not in SOURCES_BY_SYSTEM:
        raise ValueError(f"system {system!r}: 'seiscomp' or 'sil'")
    text = str(event_id).strip()
    if system == "sil" or text.upper().startswith("SIL"):
        raise ValueError(
            f"{event_id!r}: the Quakes API has no detail for SIL events (404); "
            "use geo_dataread.seismic.skjalftalisa_api.read_skjalftalisa_detail"
        )
    if not text or "/" in text or "?" in text or "#" in text:
        raise ValueError(f"event_id {event_id!r} is not a SeisComP event id")
    return text


def _fetch_detail(client: _Client, event_id: str) -> QuakeDetail:
    path = f"/events/{urllib.parse.quote(event_id, safe='')}"
    client.tally["detail_requests"] += 1
    try:
        body = client.get(path, ())
    except QuakesApiError as exc:
        if exc.status == 404 and not isinstance(exc, QuakesApiUnreachable):
            raise QuakeDetailNotFound(
                f"no Quakes API detail for {event_id}: {exc}",
                status=404,
                url=exc.url,
            ) from exc
        raise
    try:
        doc = json.loads(body)
    except ValueError as exc:
        raise QuakesApiError(f"detail is not JSON: {body[:200]!r}") from exc
    return parse_quake_detail(
        doc,
        event_id=event_id,
        url=f"{client.base}{path}",
        internal=client.internal,
    )


def read_quake_detail(
    event_id: str,
    *,
    system: System = "seiscomp",
    internal: bool | None = None,
    session: Session | None = None,
    policy: FetchPolicy = DEFAULT_POLICY,
) -> QuakeDetail:
    """The detail of one SeisComP event: all origins (uncertainties
    normalised to 1σ km and flagged by method), all magnitudes, the
    preferred ones; ``.to_catalog()`` gives the one-row schema frame.

    For samples, never whole catalogues (:func:`read_quake_details` for a
    few). ``system="sil"`` or a ``SIL…`` id raises :class:`ValueError` before
    any request (use the legacy Skjálftalísa detail). Endpoint, transport,
    timeout and 5xx retries as :func:`read_quakes`. Raises
    :class:`QuakeDetailNotFound` (also a :class:`LookupError`) for an
    unknown id, :class:`QuakesApiUnreachable`, :class:`QuakesApiError`.
    """
    eid = _check_detail_args(event_id, system)
    base, is_internal = resolve_endpoint(internal)
    client = _Client(session or UrllibSession(), base, is_internal, policy, Counter())
    return _fetch_detail(client, eid)


@dataclasses.dataclass(frozen=True)
class QuakeDetailSample:
    """What :func:`read_quake_details` fetched: ``details`` in input order
    (ids not found are left out and listed in ``missing``), and the request
    tally (``requests``, ``detail_requests``, ``retries``)."""

    details: tuple[QuakeDetail, ...]
    missing: tuple[str, ...]
    counts: Mapping[str, int]
    elapsed_s: float

    def to_catalog(self, *, extras: bool = False) -> pd.DataFrame:
        """The details' one-row frames concatenated (schema-valid, sorted)."""
        frames = [d.to_catalog(extras=extras) for d in self.details]
        if not frames:
            return features_to_frame([], system="seiscomp", extras=extras)
        df = coerce_catalog(pd.concat(frames), source=SOURCES_BY_SYSTEM["seiscomp"])
        validate(df, strict=True)
        prov = dict(frames[0].attrs.get("catalog", {}))
        for key in (
            "preferred_origin_id",
            "preferred_magnitude_id",
            "uncertainty_method",
            "confidence_level",
            "sigma_factor",
            "uncertainty_consistent",
            "depth_fixed",
            "n_origins",
            "n_magnitudes",
            "counts",
            "notes",
        ):
            prov.pop(key, None)
        prov["location"] = prov.get("location", "").rsplit("/", 1)[0]
        prov["sample"] = {
            "events": len(self.details),
            "missing": list(self.missing),
            "uncertainty_methods": dict(
                Counter(
                    o.uncertainty_method
                    for d in self.details
                    if (o := d.preferred_origin) is not None
                )
            ),
            "counts": dict(self.counts),
        }
        return with_provenance(df, prov)


def read_quake_details(
    event_ids: Sequence[str] | pd.Series,
    *,
    max_events: int,
    system: System = "seiscomp",
    internal: bool | None = None,
    session: Session | None = None,
    policy: FetchPolicy = DEFAULT_POLICY,
    missing: Literal["raise", "skip"] = "raise",
) -> QuakeDetailSample:
    """Details for a *sample* of events, one request each, politely.

    ``max_events`` is required: more ids than that raise :class:`ValueError`
    (never truncated silently), and it may not exceed
    :data:`MAX_DETAIL_SAMPLE`. Draw the sample first (e.g.
    ``df["event_id"].sample(50, random_state=0)``). Duplicate ids are fetched
    once. One client serves the whole loop, so ``policy.pause_s`` separates
    every request (N ids → N−1 pauses). ``missing="skip"`` lists 404 ids in
    :attr:`QuakeDetailSample.missing` instead of raising
    :class:`QuakeDetailNotFound`. Every id is checked before the first
    request.
    """
    t0 = _time.perf_counter()
    if isinstance(max_events, bool) or not isinstance(max_events, int):
        raise TypeError("max_events must be an int")
    if not 1 <= max_events <= MAX_DETAIL_SAMPLE:
        raise ValueError(
            f"max_events={max_events}: must be 1…{MAX_DETAIL_SAMPLE} "
            "(details are for samples, not catalogues)"
        )
    if missing not in ("raise", "skip"):
        raise ValueError("missing must be 'raise' or 'skip'")
    if isinstance(event_ids, str):
        raise TypeError("event_ids is a sequence of ids, not one string")
    ids = list(dict.fromkeys(_check_detail_args(str(e), system) for e in event_ids))
    if len(ids) > max_events:
        raise ValueError(
            f"{len(ids)} event ids > max_events={max_events}: draw a smaller "
            "sample (details are never fetched for whole catalogues)"
        )
    base, is_internal = resolve_endpoint(internal)
    tally: Counter[str] = Counter(
        {"requests": 0, "detail_requests": 0, "retries": 0, "missing": 0}
    )
    client = _Client(session or UrllibSession(), base, is_internal, policy, tally)
    details: list[QuakeDetail] = []
    lost: list[str] = []
    for eid in ids:
        try:
            details.append(_fetch_detail(client, eid))
        except QuakeDetailNotFound:
            if missing == "raise":
                raise
            lost.append(eid)
            tally["missing"] += 1
    return QuakeDetailSample(
        details=tuple(details),
        missing=tuple(lost),
        counts=dict(tally),
        elapsed_s=_time.perf_counter() - t0,
    )


__all__ = [
    "DEFAULT_POLICY",
    "DEPTH_BOUNDS",
    "ENV_ENDPOINT",
    "INTERNAL_BASE",
    "MAX_DETAIL_SAMPLE",
    "PUBLIC_BASE",
    "SIZE_BOUNDS",
    "SOURCES_BY_SYSTEM",
    "USER_AGENT",
    "FetchPolicy",
    "QuakeDetail",
    "QuakeDetailNotFound",
    "QuakeDetailSample",
    "QuakeMagnitude",
    "QuakeOrigin",
    "QuakesApiError",
    "QuakesApiUnreachable",
    "QuakesReadReport",
    "Session",
    "UrllibSession",
    "features_to_frame",
    "parse_quake_detail",
    "polygon_wkt",
    "read_quake_detail",
    "read_quake_details",
    "read_quakes",
    "resolve_endpoint",
    "sigma_divisor",
]
