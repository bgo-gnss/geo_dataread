"""Legacy Skjálftalísa API reader (ticket T12, PLAN-seismicity.md §3.3).

.. warning::
   **Legacy source.** The Skjálftalísa API describes itself as "Legacy API for
   old version of Skjálftalísa … Can be deprecated with no notice". It holds
   reviewed *and* automatic SIL events before 3 February 2026 and **automatic
   only** after. The main SIL reader is the Quakes API (``system=sil``); this
   one is kept (BGÓ Q3) for comparison and for the ``slungapath`` join to the
   SIL files (:func:`read_skjalftalisa_detail`).

Endpoints (OpenAPI ``2026-04-30``): internal (default, Q2)
``https://vi-api.vedur.is/skjalftalisa/internal/quakes``, public
``https://api.vedur.is/skjalftalisa/quakes``; ``/count`` and
``/{event_id}`` beside each. GeoJSON only, no paging, no row cap seen
(2023-11: 56,658 rows / 14 MB in one response). Query times must carry a
zone (a naive one is a 422); response times are naive UTC. Valid times run
from 1991-01-07 to midnight of tomorrow.

**One event, two rows.** The list returns each solution of an event as its
own row sharing one integer ``event_id``: ``originating_system`` ``SIL
picks`` (reviewed) and ``SIL aut.mag`` (automatic). The ad-hoc scripts that
took every row counted reviewed events twice. :func:`resolve_solutions`
keeps one row per ``event_id``:

1. exact repeats of ``(event_id, originating_system)`` (chunk edges; whether
   ``end_time`` is inclusive is not documented) are dropped
   (``duplicate_rows``);
2. the reviewed ``SIL picks`` row wins over ``SIL aut.mag``
   (``twins_resolved`` counts the automatic rows dropped this way); events
   with one row are ``picks_only`` / ``aut_only``;
3. the reviewed row takes ``mag_auto`` from its automatic twin (as
   ``sil-files`` does).

The two rows of an event can be more than a minute apart (68 s in 2023-11)
and kilometres apart (and differ in depth: 1352381 is 0.2 km automatic,
2.1 km reviewed). Therefore:

* the query window is padded by ``pad`` (default 5 min) on both sides, the
  twins are resolved over **all** chunks together, and only then the events
  are cut to ``[start, end)`` on the time of the chosen row
  (``outside_window``);
* the API's depth/size/event-type filters act per *row*, so they are **not**
  sent: depth, magnitude, event type and the exact region are applied after
  resolution with :func:`geo_dataread.seismic.filters.subset`
  (``filtered_client``). Only a region is sent to the server, as the
  bounding box of ``bbox``/``polygon`` grown by ``server_buffer_km``
  (default 30 km), to keep downloads small.

Field mapping (list → schema; ``source`` = ``skjalftalisa``):

========================  ================================================
Schema column             Legacy list field
========================  ================================================
index ``time``            ``time`` of the chosen row (naive → UTC)
``event_id``              ``event_id`` as a string (``"1352381"``; the
                          Quakes API spells it ``SIL1352381``)
``latitude/longitude``    ``geometry.coordinates`` ``[lon, lat]``
``depth_km``              ``depth``
``magnitude``             picks: ``m_mlw`` (``Mlw``), else ``m_ml`` (``ML``);
                          aut: ``m_autmag`` (``aut.mag``, as ``sil-files``
                          and the Quakes API label it)
``mag_mlw`` / ``mag_ml``  picks ``m_mlw`` / ``m_ml`` (−9.99 = none → NA)
``mag_auto``              ``m_autmag`` of the event's automatic row
``moment_nm``             picks ``seismic_moment``
``quality``               aut ``quality`` (NA on reviewed events, as in
                          ``sil-files``: Q belongs to automatic solutions)
``evaluation_mode``       picks → ``manual``, aut → ``automatic``
``evaluation_status``     NA (the source has none; plan §4)
``event_type(_code)``     ``event_type`` (SIL code) via ``sil_event_type``
``sil_path``, errors      NA in the list; :func:`read_skjalftalisa_detail`
others                    NA
========================  ================================================

Extra (non-schema) columns, kept for T12b and checks: ``originating_system``
of the chosen row, ``n_solutions`` (1 or 2), and the reviewed row's raw
``strike`` / ``dip`` / ``slip`` (internal list only: one Aki–Richards plane,
fps plane a, *not normalised* — strikes of −72 and 434 occur). The raw rows
with every field are available from :func:`fetch_skjalftalisa_rows`.

Known limits: the API's ETL truncates some scalars (T3b); the public list
rounds coordinates to 0.001° and ``m_mlw`` to 0.1, drops strike/dip/slip and
drops some whole events (event magnitude < 0?, not verified).
"""

from __future__ import annotations

import dataclasses
import json
import math
import time as _time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import timedelta
from typing import Any, Final, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

from geo_dataread.seismic.filters import TimeLike, subset, to_utc
from geo_dataread.seismic.schema import (
    EVENT_COLUMNS,
    catalog_provenance,
    coerce_catalog,
    sil_event_type,
    validate,
    with_provenance,
)
from geo_dataread.seismic.sil_catalog import KM_PER_DEG, horizontal_error_km

SOURCE: Final = "skjalftalisa"

INTERNAL_URL: Final = "https://vi-api.vedur.is/skjalftalisa/internal/quakes"
PUBLIC_URL: Final = "https://api.vedur.is/skjalftalisa/quakes"

#: ``originating_system`` values.
SYSTEM_PICKS: Final = "SIL picks"
SYSTEM_AUT: Final = "SIL aut.mag"

#: First valid query time (OpenAPI: "between 7th January 1991 and midnight
#: of tomorrow").
API_START: Final = pd.Timestamp("1991-01-07T00:00:00Z")

DEFAULT_CHUNK: Final = timedelta(days=31)
DEFAULT_PAD: Final = timedelta(minutes=5)
DEFAULT_PAUSE_S: Final = 1.2
DEFAULT_TIMEOUT_S: Final = 120.0
DEFAULT_RETRIES: Final = 2
DEFAULT_SERVER_BUFFER_KM: Final = 30.0

USER_AGENT: Final = "geo_dataread-skjalftalisa"

#: "No magnitude" value of the SIL files, passed through by the API.
_MAG_SENTINEL: Final = -9.99

_RETRY_STATUS: Final = frozenset({429, 502, 503, 504})

FloatArray = npt.NDArray[np.float64]


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class SkjalftalisaError(RuntimeError):
    """The API answered with an error, or with something that is not the list."""

    def __init__(self, message: str, *, url: str = "", status: int | None = None):
        super().__init__(message)
        self.url = url
        self.status = status


class SkjalftalisaUnreachable(SkjalftalisaError):
    """No answer (timeout, DNS, refused connection)."""


class SkjalftalisaNotFound(SkjalftalisaError, LookupError):
    """The detail endpoint does not know the event id (404)."""


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class HttpResponse:
    """Status and body of one GET (error statuses too)."""

    status: int
    body: bytes
    url: str


class Transport(Protocol):
    """What the reader needs from an HTTP client (injectable for tests)."""

    def get(self, url: str, *, timeout: float) -> HttpResponse:
        """GET ``url``. Return the response for any HTTP status; raise
        :class:`SkjalftalisaUnreachable` when there is no response."""
        ...


class UrllibTransport:
    """The default :class:`Transport`, on :mod:`urllib` (no extra dependency)."""

    def __init__(self, user_agent: str = USER_AGENT) -> None:
        self.user_agent = user_agent

    def get(self, url: str, *, timeout: float) -> HttpResponse:
        req = urllib.request.Request(
            url, headers={"User-Agent": self.user_agent, "Accept": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return HttpResponse(int(resp.status), resp.read(), url)
        except urllib.error.HTTPError as err:
            return HttpResponse(int(err.code), err.read(), url)
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            reason = getattr(err, "reason", err)
            raise SkjalftalisaUnreachable(
                f"no answer from {url} ({reason})", url=url
            ) from err


class _Client:
    """Polite GETs: a pause between requests, retries, JSON decoding."""

    def __init__(
        self,
        session: Transport | None,
        *,
        internal: bool,
        pause_s: float,
        timeout_s: float,
        retries: int,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.session: Transport = session if session is not None else UrllibTransport()
        self.internal = internal
        self.pause_s = max(0.0, float(pause_s))
        self.timeout_s = float(timeout_s)
        self.retries = max(0, int(retries))
        self._sleep = sleep
        self._last: float | None = None
        self.n_requests = 0
        self.urls: list[str] = []

    def sleep(self, seconds: float) -> None:
        # looked up at call time, so a patched time.sleep is honoured
        (self._sleep or _time.sleep)(seconds)

    def _wait(self) -> None:
        if self._last is not None and self.pause_s > 0:
            left = self.pause_s - (_time.monotonic() - self._last)
            if left > 0:
                self.sleep(left)
        self._last = _time.monotonic()

    def _switch_hint(self) -> str:
        if self.internal:
            return (
                "the internal host needs the IMO network; pass internal=False for "
                "the public endpoint (reduced: rounded values, no fault planes)"
            )
        return "check the network, or try internal=True from inside IMO"

    def get_json(self, url: str) -> Any:
        attempt = 0
        while True:
            self._wait()
            self.n_requests += 1
            self.urls.append(url)
            try:
                resp = self.session.get(url, timeout=self.timeout_s)
            except SkjalftalisaUnreachable as err:
                if attempt < self.retries:
                    attempt += 1
                    self.sleep(self.pause_s * 2**attempt)
                    continue
                raise SkjalftalisaUnreachable(
                    f"{err} after {attempt + 1} tries; {self._switch_hint()}",
                    url=url,
                ) from err
            if resp.status in _RETRY_STATUS and attempt < self.retries:
                attempt += 1
                self.sleep(self.pause_s * 2**attempt)
                continue
            return _decode(resp)


def _error_text(body: Any) -> str:
    """The API's own message: ``detail[].msg`` (422) or ``message`` (404)."""
    if isinstance(body, Mapping):
        detail = body.get("detail")
        if isinstance(detail, list):
            parts = []
            for d in detail:
                if isinstance(d, Mapping):
                    loc = ".".join(str(x) for x in d.get("loc", ()) if x != "query")
                    msg = str(d.get("msg", d))
                    parts.append(f"{loc}: {msg}" if loc else msg)
                else:
                    parts.append(str(d))
            return "; ".join(parts)
        if detail is not None:
            return str(detail)
        if "message" in body:
            return str(body["message"])
    return str(body)[:300]


def _decode(resp: HttpResponse) -> Any:
    try:
        body: Any = json.loads(resp.body.decode("utf-8")) if resp.body else None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        if resp.status >= 400:
            body = resp.body[:300].decode("utf-8", "replace")
        else:
            raise SkjalftalisaError(
                f"{resp.url}: answer is not JSON ({exc})",
                url=resp.url,
                status=resp.status,
            ) from exc
    if resp.status == 404:
        raise SkjalftalisaNotFound(
            f"{resp.url}: 404 {_error_text(body)}", url=resp.url, status=404
        )
    if resp.status >= 400:
        raise SkjalftalisaError(
            f"{resp.url}: HTTP {resp.status}: {_error_text(body)}",
            url=resp.url,
            status=resp.status,
        )
    return body


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SkjalftalisaReport:
    """Counts of what a read did.

    ``counts`` keys (absent = 0): ``requests``, ``chunks``,
    ``rows_fetched``, ``rows:picks``, ``rows:aut``,
    ``dropped_unknown_system``, ``duplicate_rows`` (exact repeats of
    ``(event_id, originating_system)``), ``events`` (after resolution, before
    the window cut), ``twins_resolved`` (automatic rows dropped for their
    reviewed twin), ``picks_only``, ``aut_only``, ``sentinel:<field>``
    (−9.99 magnitudes made NA), ``out_of_range:<column>``,
    ``dropped_bad_position``, ``outside_window``, ``filtered_client``, and
    the totals ``events_reviewed`` / ``events_automatic`` of the frame
    returned. ``urls`` are the requests made, in order.
    """

    counts: Mapping[str, int] = dataclasses.field(default_factory=dict)
    urls: tuple[str, ...] = ()
    seconds: float = 0.0

    def get(self, key: str) -> int:
        """A count (0 when absent)."""
        return int(self.counts.get(key, 0))

    def __add__(self, other: SkjalftalisaReport) -> SkjalftalisaReport:
        c: Counter[str] = Counter(self.counts)
        c.update(other.counts)
        return SkjalftalisaReport(
            dict(c), self.urls + other.urls, self.seconds + other.seconds
        )

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe summary."""
        return {
            "counts": {k: int(v) for k, v in sorted(self.counts.items())},
            "n_requests": len(self.urls),
            "urls": list(self.urls[:20]),
            "seconds": round(self.seconds, 3),
        }

    def summary(self) -> str:
        """One line for a log."""
        return (
            f"skjalftalisa (legacy): {self.get('requests')} requests, "
            f"{self.get('rows_fetched')} rows → {self.get('events')} events "
            f"({self.get('twins_resolved')} twins resolved, "
            f"{self.get('duplicate_rows')} duplicate rows), returned "
            f"{self.get('events_reviewed')} reviewed + "
            f"{self.get('events_automatic')} automatic"
        )


# --------------------------------------------------------------------------
# Query building
# --------------------------------------------------------------------------


def base_url(internal: bool = True) -> str:
    """The list endpoint: internal (default, Q2) or public."""
    return INTERNAL_URL if internal else PUBLIC_URL


def _fmt(ts: pd.Timestamp) -> str:
    return f"{ts.tz_convert('UTC'):%Y-%m-%dT%H:%M:%S}Z"


def _region_ring(
    bbox: Sequence[float] | None,
    polygon: Sequence[Sequence[float]] | npt.ArrayLike | None,
    buffer_km: float,
) -> tuple[list[float], list[float]] | None:
    """The server-side region: the bounding box of ``bbox``/``polygon`` grown
    by ``buffer_km``, as a closed ring of (lat list, lon list)."""
    if bbox is None and polygon is None:
        return None
    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError("bbox is (min_lon, min_lat, max_lon, max_lat)")
        lo_lon, lo_lat, hi_lon, hi_lat = (float(v) for v in bbox)
    else:
        pts = np.asarray(polygon, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            raise ValueError("polygon must be at least three (lon, lat) pairs")
        lo_lon, lo_lat = (float(v) for v in pts.min(axis=0))
        hi_lon, hi_lat = (float(v) for v in pts.max(axis=0))
    if lo_lon > hi_lon or lo_lat > hi_lat:
        raise ValueError("bbox minimum is greater than its maximum")
    dlat = buffer_km / KM_PER_DEG
    mid = math.radians(max(abs(lo_lat), abs(hi_lat)) + dlat)
    dlon = buffer_km / (KM_PER_DEG * max(math.cos(min(mid, math.radians(89.0))), 1e-3))
    a_lat = max(-90.0, lo_lat - dlat)
    b_lat = min(90.0, hi_lat + dlat)
    a_lon = max(-180.0, lo_lon - dlon)
    b_lon = min(180.0, hi_lon + dlon)
    lats = [a_lat, b_lat, b_lat, a_lat, a_lat]
    lons = [a_lon, a_lon, b_lon, b_lon, a_lon]
    return [round(v, 5) for v in lats], [round(v, 5) for v in lons]


def _query_url(
    url: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    ring: tuple[list[float], list[float]] | None,
) -> str:
    q: list[tuple[str, str]] = [("start_time", _fmt(start)), ("end_time", _fmt(end))]
    if ring is not None:
        q += [("lat", repr(v)) for v in ring[0]]
        q += [("lon", repr(v)) for v in ring[1]]
    return f"{url}?{urllib.parse.urlencode(q)}"


def _chunks(
    start: pd.Timestamp, end: pd.Timestamp, chunk: timedelta
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if chunk <= timedelta(0):
        raise ValueError("chunk must be positive")
    step = pd.Timedelta(chunk)
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    a = start
    while a < end:
        b = min(a + step, end)
        out.append((a, b))
        a = b
    return out


def _query_window(
    lo: pd.Timestamp, hi: pd.Timestamp, pad: timedelta
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Padded, whole-second query window clamped to the API's valid span."""
    if hi <= lo:
        raise ValueError(f"end {hi} is not after start {lo}")
    # the API refuses times after "midnight of tomorrow"; no event can be
    # later than now, so the start of tomorrow (UTC) is a safe ceiling
    ceiling = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
    if hi <= API_START:
        raise ValueError(
            f"the window ends before the legacy API's first valid time {API_START}"
        )
    qs = max((lo - pd.Timedelta(pad)).floor("s"), API_START)
    qe = (hi + pd.Timedelta(pad)).ceil("s")
    qe = min(qe, ceiling)
    if qe <= qs:
        raise ValueError(f"the window starts after the API's last valid time {ceiling}")
    return qs, qe


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------

#: Columns of the raw rows frame, in order (any unknown property follows).
ROW_COLUMNS: Final = (
    "event_id",
    "time",
    "originating_system",
    "latitude",
    "longitude",
    "depth",
    "m_mlw",
    "m_ml",
    "m_autmag",
    "seismic_moment",
    "quality",
    "event_type",
    "strike",
    "dip",
    "slip",
)

_NUMERIC_ROW: Final = (
    "latitude",
    "longitude",
    "depth",
    "m_mlw",
    "m_ml",
    "m_autmag",
    "seismic_moment",
    "quality",
    "strike",
    "dip",
    "slip",
)


def _features(body: Any, url: str) -> list[Mapping[str, Any]]:
    if not isinstance(body, Mapping) or "features" not in body:
        raise SkjalftalisaError(
            f"{url}: not a GeoJSON FeatureCollection", url=url, status=200
        )
    feats = body["features"]
    if feats is None:  # an empty window may give null
        return []
    if not isinstance(feats, list):
        raise SkjalftalisaError(f"{url}: `features` is not a list", url=url)
    return feats


def rows_from_features(features: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """The raw rows frame of a list response: one row per feature, every
    property kept, ``event_id`` Int64, ``time`` UTC (the API's naive times
    are UTC), ``latitude``/``longitude`` from the geometry. No resolution."""
    recs: list[dict[str, Any]] = []
    for f in features:
        props = dict(f.get("properties") or {})
        geom = f.get("geometry") or {}
        coords = geom.get("coordinates") if isinstance(geom, Mapping) else None
        lon = coords[0] if coords and len(coords) >= 2 else None
        lat = coords[1] if coords and len(coords) >= 2 else None
        props["longitude"] = lon
        props["latitude"] = lat
        recs.append(props)
    df = pd.DataFrame.from_records(recs)
    for col in ROW_COLUMNS:
        if col not in df.columns:
            df[col] = None
    extra = [c for c in df.columns if c not in ROW_COLUMNS]
    df = df[list(ROW_COLUMNS) + extra].copy()
    df["event_id"] = pd.to_numeric(df["event_id"], errors="raise").astype("Int64")
    if len(df):
        df["time"] = pd.to_datetime(df["time"], format="ISO8601", utc=True).astype(
            "datetime64[ns, UTC]"
        )
    else:
        df["time"] = pd.Series([], dtype="datetime64[ns, UTC]")
    for col in _NUMERIC_ROW:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df["originating_system"] = df["originating_system"].astype(object)
    df["event_type"] = df["event_type"].astype(object)
    return df


def fetch_skjalftalisa_rows(
    start: TimeLike,
    end: TimeLike,
    *,
    internal: bool = True,
    bbox: Sequence[float] | None = None,
    polygon: Sequence[Sequence[float]] | npt.ArrayLike | None = None,
    server_buffer_km: float = DEFAULT_SERVER_BUFFER_KM,
    chunk: timedelta = DEFAULT_CHUNK,
    pause_s: float = DEFAULT_PAUSE_S,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    session: Transport | None = None,
) -> tuple[pd.DataFrame, SkjalftalisaReport]:
    """The raw list rows for ``[start, end]`` as the API gives them.

    Both solutions of an event, every field (the input of T12b's fault
    planes). No padding, no resolution, no cut. The window is split into
    ``chunk``-long requests with ``pause_s`` between them; rows at a chunk
    edge may repeat (see :func:`resolve_solutions`). Naive ``start``/``end``
    are UTC.
    """
    t0 = _time.perf_counter()
    client = _Client(
        session,
        internal=internal,
        pause_s=pause_s,
        timeout_s=timeout_s,
        retries=retries,
    )
    lo, hi = to_utc(start, what="start"), to_utc(end, what="end")
    rows, tally = _fetch(client, lo, hi, bbox, polygon, server_buffer_km, chunk)
    report = SkjalftalisaReport(
        dict(tally), tuple(client.urls), _time.perf_counter() - t0
    )
    return rows, report


def _fetch(
    client: _Client,
    qs: pd.Timestamp,
    qe: pd.Timestamp,
    bbox: Sequence[float] | None,
    polygon: Sequence[Sequence[float]] | npt.ArrayLike | None,
    buffer_km: float,
    chunk: timedelta,
) -> tuple[pd.DataFrame, Counter[str]]:
    if qe <= qs:
        raise ValueError(f"end {qe} is not after start {qs}")
    ring = _region_ring(bbox, polygon, buffer_km)
    url = base_url(client.internal)
    feats: list[Mapping[str, Any]] = []
    tally: Counter[str] = Counter()
    for a, b in _chunks(qs, qe, chunk):
        u = _query_url(url, a, b, ring)
        feats.extend(_features(client.get_json(u), u))
        tally["chunks"] += 1
    rows = rows_from_features(feats)
    tally["requests"] = client.n_requests
    tally["rows_fetched"] = len(rows)
    tally["rows:picks"] = int((rows["originating_system"] == SYSTEM_PICKS).sum())
    tally["rows:aut"] = int((rows["originating_system"] == SYSTEM_AUT).sum())
    return rows, tally


# --------------------------------------------------------------------------
# Resolution: one row per event
# --------------------------------------------------------------------------


def _mag(values: Any, field: str, tally: Counter[str]) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64).copy()
    bad = np.isclose(arr, _MAG_SENTINEL, atol=1e-6)
    if bad.any():
        tally[f"sentinel:{field}"] += int(bad.sum())
        arr[bad] = np.nan
    return arr


def resolve_solutions(rows: pd.DataFrame) -> tuple[pd.DataFrame, Counter[str]]:
    """One row per ``event_id``: reviewed ``SIL picks`` over ``SIL aut.mag``.

    Takes the raw rows (:func:`rows_from_features`, any order, any number of
    chunks concatenated) and returns the raw (not yet coerced) events frame in
    schema column names with the extra columns ``originating_system``,
    ``n_solutions``, ``strike``, ``dip``, ``slip``, plus the counts
    (``duplicate_rows``, ``dropped_unknown_system``, ``twins_resolved``,
    ``picks_only``, ``aut_only``, ``events``, ``sentinel:<field>``).
    """
    tally: Counter[str] = Counter()
    known = rows["originating_system"].isin([SYSTEM_PICKS, SYSTEM_AUT]).to_numpy()
    if (~known).any():
        tally["dropped_unknown_system"] += int((~known).sum())
    r = rows.loc[known]
    dup = r.duplicated(["event_id", "originating_system"], keep="first").to_numpy()
    tally["duplicate_rows"] += int(dup.sum())
    r = r.loc[~dup]

    picks = r.loc[r["originating_system"] == SYSTEM_PICKS].set_index("event_id")
    aut = r.loc[r["originating_system"] == SYSTEM_AUT].set_index("event_id")
    twins = picks.index.intersection(aut.index)
    tally["twins_resolved"] += len(twins)
    tally["picks_only"] += len(picks.index.difference(aut.index))
    aut_alone = aut.loc[~aut.index.isin(picks.index)]
    tally["aut_only"] += len(aut_alone)

    mlw = _mag(picks["m_mlw"], "m_mlw", tally)
    ml = _mag(picks["m_ml"], "m_ml", tally)
    has_mlw = ~np.isnan(mlw)
    has_ml = ~np.isnan(ml)
    rev = pd.DataFrame(
        {
            "event_id": picks.index.astype(str),
            "time": picks["time"].to_numpy(),
            "latitude": picks["latitude"].to_numpy(),
            "longitude": picks["longitude"].to_numpy(),
            "depth_km": picks["depth"].to_numpy(),
            "magnitude": np.where(has_mlw, mlw, ml),
            "magnitude_type": [
                "Mlw" if w else ("ML" if m else None)
                for w, m in zip(has_mlw, has_ml, strict=True)
            ],
            "mag_mlw": mlw,
            "mag_ml": ml,
            "mag_auto": _mag(
                aut["m_autmag"].reindex(picks.index), "m_autmag(twin)", Counter()
            ),
            "moment_nm": picks["seismic_moment"].to_numpy(),
            "quality": np.nan,
            "evaluation_mode": "manual",
            "event_type_code": picks["event_type"].to_numpy(),
            "originating_system": SYSTEM_PICKS,
            "n_solutions": np.where(picks.index.isin(aut.index), 2, 1),
            "strike": picks["strike"].to_numpy(),
            "dip": picks["dip"].to_numpy(),
            "slip": picks["slip"].to_numpy(),
        }
    )
    amag = _mag(aut_alone["m_autmag"], "m_autmag", tally)
    auto = pd.DataFrame(
        {
            "event_id": aut_alone.index.astype(str),
            "time": aut_alone["time"].to_numpy(),
            "latitude": aut_alone["latitude"].to_numpy(),
            "longitude": aut_alone["longitude"].to_numpy(),
            "depth_km": aut_alone["depth"].to_numpy(),
            "magnitude": amag,
            "magnitude_type": [None if math.isnan(v) else "aut.mag" for v in amag],
            "mag_mlw": np.nan,
            "mag_ml": np.nan,
            "mag_auto": amag,
            "moment_nm": np.nan,
            "quality": aut_alone["quality"].to_numpy(),
            "evaluation_mode": "automatic",
            "event_type_code": aut_alone["event_type"].to_numpy(),
            "originating_system": SYSTEM_AUT,
            "n_solutions": 1,
            "strike": np.nan,
            "dip": np.nan,
            "slip": np.nan,
        }
    )
    parts = [p for p in (rev, auto) if len(p)]
    if parts:
        out = pd.concat(parts, ignore_index=True)
    else:
        out = rev.iloc[0:0].copy()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    out["event_type"] = [sil_event_type(c) for c in out["event_type_code"]]
    out["n_solutions"] = out["n_solutions"].astype("Int64")
    tally["events"] += len(out)
    return out, tally


_RANGED: Final = tuple(
    s
    for s in EVENT_COLUMNS
    if s.kind in ("float64", "Int64")
    and (s.minimum is not None or s.maximum is not None)
)


def _range_guard(df: pd.DataFrame, tally: Counter[str]) -> pd.DataFrame:
    """NA out-of-range values (counted); drop rows with impossible positions."""
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
        if spec.name == "magnitude":
            df.loc[bad, "magnitude_type"] = None
    if drop.any():
        tally["dropped_bad_position"] += int(drop.sum())
        df = df.loc[~drop]
    return df


# --------------------------------------------------------------------------
# The reader
# --------------------------------------------------------------------------


def read_skjalftalisa(
    start: TimeLike,
    end: TimeLike,
    *,
    internal: bool = True,
    bbox: Sequence[float] | None = None,
    polygon: Sequence[Sequence[float]] | npt.ArrayLike | None = None,
    min_depth: float | None = None,
    max_depth: float | None = None,
    min_mag: float | None = None,
    max_mag: float | None = None,
    event_types: Iterable[str] | None = None,
    server_buffer_km: float = DEFAULT_SERVER_BUFFER_KM,
    chunk: timedelta = DEFAULT_CHUNK,
    pad: timedelta = DEFAULT_PAD,
    pause_s: float = DEFAULT_PAUSE_S,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    session: Transport | None = None,
) -> tuple[pd.DataFrame, SkjalftalisaReport]:
    """Read the legacy Skjálftalísa list for ``[start, end)`` (LEGACY source).

    Returns ``(events, report)``: a validated events frame of
    :mod:`~geo_dataread.seismic.schema` with one row per event (reviewed
    solution preferred, automatic events included and labelled through
    ``evaluation_mode``, Q8), and the :class:`SkjalftalisaReport`, which also
    rides on ``events.attrs["catalog"]["report"]``.

    Naive ``start``/``end`` are UTC. ``internal=True`` (default) reads the
    internal endpoint; it does not fall back to the public one by itself
    (:class:`SkjalftalisaUnreachable` says how to switch).

    Filters, applied **after** the twins are resolved (see the module
    docstring for why they are not sent to the API): ``bbox``
    ``(min_lon, min_lat, max_lon, max_lat)`` or ``polygon`` ``(lon, lat)``
    vertices (also sent to the server as a box grown by
    ``server_buffer_km``); ``min_depth``/``max_depth`` (km); ``min_mag`` /
    ``max_mag`` on ``magnitude`` (mixed scales: Mlw, ML or aut.mag);
    ``event_types`` in the schema vocabulary (``earthquake``, …). Bounds are
    inclusive (:func:`~geo_dataread.seismic.filters.subset`).

    ``chunk`` is the request length, ``pad`` the margin fetched on each side
    so that a twin just outside the window still resolves, ``pause_s`` the
    minimum gap between requests, ``timeout_s`` per request, ``retries`` on
    no answer / 429 / 502–504. ``session`` replaces the HTTP transport
    (tests).
    """
    t0 = _time.perf_counter()
    lo, hi = to_utc(start, what="start"), to_utc(end, what="end")
    qs, qe = _query_window(lo, hi, pad)
    client = _Client(
        session,
        internal=internal,
        pause_s=pause_s,
        timeout_s=timeout_s,
        retries=retries,
    )
    rows, tally = _fetch(client, qs, qe, bbox, polygon, server_buffer_km, chunk)
    raw, rtally = resolve_solutions(rows)
    tally.update(rtally)
    raw = _range_guard(raw, tally)
    t = pd.DatetimeIndex(raw["time"])
    keep = np.asarray((t >= lo) & (t < hi), dtype=bool)
    tally["outside_window"] += int((~keep).sum())
    raw = raw.loc[keep]

    events = coerce_catalog(raw.set_index("time"), source=SOURCE, assume_utc=True)
    events = events.astype(
        {
            "originating_system": "category",
            "n_solutions": "Int64",
            "strike": "float64",
            "dip": "float64",
            "slip": "float64",
        }
    )
    filters: dict[str, Any] = {}
    sub: dict[str, Any] = {
        "bbox": None if bbox is None else tuple(float(v) for v in bbox),
        "polygon": polygon,
        "min_depth": min_depth,
        "max_depth": max_depth,
        "min_mag": min_mag,
        "max_mag": max_mag,
        "event_types": None if event_types is None else list(event_types),
    }
    sub = {k: v for k, v in sub.items() if v is not None}
    if sub:
        n0 = len(events)
        events = subset(events, **sub)
        tally["filtered_client"] += n0 - len(events)
        for k, v in sub.items():
            filters[k] = np.asarray(v, dtype=float).tolist() if k == "polygon" else v
    is_aut = (events["evaluation_mode"] == "automatic").to_numpy()
    tally["events_reviewed"] = int((~is_aut).sum())
    tally["events_automatic"] = int(is_aut.sum())
    report = SkjalftalisaReport(
        dict(tally), tuple(client.urls), _time.perf_counter() - t0
    )
    subsets = events.attrs.get("catalog", {}).get("subsets")
    prov = catalog_provenance(
        source=SOURCE,
        location=base_url(internal),
        start=lo.to_pydatetime(),
        end=hi.to_pydatetime(),
        filters={
            **filters,
            "server_region_buffer_km": (
                server_buffer_km if bbox is not None or polygon is not None else None
            ),
        },
        extra={
            "legacy": True,
            "endpoint": "internal" if internal else "public",
            "query_window": {"start": _fmt(qs), "end": _fmt(qe)},
            "chunk_s": pd.Timedelta(chunk).total_seconds(),
            "pad_s": pd.Timedelta(pad).total_seconds(),
            "report": report.as_dict(),
            **({"subsets": subsets} if subsets else {}),
        },
    )
    with_provenance(events, prov)
    validate(events, strict=True)
    return events, report


def count_skjalftalisa(
    start: TimeLike,
    end: TimeLike,
    *,
    internal: bool = True,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    session: Transport | None = None,
) -> int:
    """The ``/count`` endpoint: number of **events** (not rows) in the window.

    Unpadded and unfiltered, so it can differ from ``len(read_skjalftalisa)``
    by the twins at the window edges. The endpoint refuses long spans
    ("Start time is out of bounds" for 1991 → 2026).
    """
    lo, hi = to_utc(start, what="start"), to_utc(end, what="end")
    client = _Client(
        session, internal=internal, pause_s=0.0, timeout_s=timeout_s, retries=retries
    )
    url = _query_url(base_url(internal) + "/count", lo, hi, None)
    body = client.get_json(url)
    if not isinstance(body, Mapping) or "quake_count" not in body:
        raise SkjalftalisaError(f"{url}: no `quake_count` in the answer", url=url)
    return int(body["quake_count"])


# --------------------------------------------------------------------------
# Detail
# --------------------------------------------------------------------------


def _num(value: Any) -> float:
    """A float, or NaN for None / blank / unparsable (``"3.9"`` → 3.9)."""
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _int(value: Any) -> int | None:
    v = _num(value)
    return None if math.isnan(v) else int(v)


@dataclasses.dataclass(frozen=True)
class SkjalftalisaOrigin:
    """One origin (solution) of a detail answer. NaN/None where not given."""

    originating_system: str | None
    time: pd.Timestamp
    latitude: float
    longitude: float
    depth_km: float
    latitude_error_deg: float
    longitude_error_deg: float
    depth_err_km: float
    h_err_km: float
    nearest_station_km: float
    n_stations: int | None
    quality: float
    row_num: int | None
    event_type_code: str | None

    @property
    def evaluation_mode(self) -> str | None:
        """``manual`` for ``SIL picks``, ``automatic`` for ``SIL aut.mag``."""
        return {SYSTEM_PICKS: "manual", SYSTEM_AUT: "automatic"}.get(
            self.originating_system or ""
        )


@dataclasses.dataclass(frozen=True)
class SkjalftalisaDetail:
    """The detail of one event (``/quakes/{event_id}``).

    ``sil_path`` is ``slungapath``, the SIL file id (the exact join key to
    the files; for a reviewed event the reviewed id, for an automatic-only
    event the automatic one). ``magnitudes`` are ``(type, size)`` pairs as
    the API types them (``Mlw``, ``Ml``, ``Autmag``; the public detail gives
    one ``M``). ``sp3`` is the fault-plane block decoded from its
    JSON-in-a-string (``sp3_raw`` keeps the string; NA for automatic
    events); T12b takes the planes from its ``fp[ab]_*_AR`` fields.
    ``raw`` is the whole decoded answer.
    """

    event_id: str
    sil_path: str | None
    origins: tuple[SkjalftalisaOrigin, ...]
    magnitudes: tuple[tuple[str, float], ...]
    sp3: Mapping[str, Any] | None
    sp3_raw: str | None
    touched_aut: int | None
    url: str
    internal: bool
    raw: Mapping[str, Any]

    def origin(self, system: str) -> SkjalftalisaOrigin | None:
        """The origin of one ``originating_system``, if present."""
        for o in self.origins:
            if o.originating_system == system:
                return o
        return None

    @property
    def preferred_origin(self) -> SkjalftalisaOrigin | None:
        """The reviewed origin, else the automatic one, else the first."""
        return (
            self.origin(SYSTEM_PICKS)
            or self.origin(SYSTEM_AUT)
            or (self.origins[0] if self.origins else None)
        )

    def magnitude(self, mtype: str) -> float:
        """The size of one magnitude type (NaN if absent)."""
        for t, v in self.magnitudes:
            if t == mtype:
                return v
        return math.nan

    def to_catalog(self) -> pd.DataFrame:
        """A one-row schema events frame of the preferred origin, with
        ``sil_path``, ``h_err_km`` and ``depth_err_km`` filled.

        Raises :class:`ValueError` for an answer without a labelled origin
        (the public detail does not say which solution it shows).
        """
        o = self.preferred_origin
        if o is None or o.evaluation_mode is None:
            raise ValueError(
                f"event {self.event_id}: no origin labelled `SIL picks`/`SIL aut.mag` "
                "(the public detail is reduced; use internal=True)"
            )
        mlw = _clean_mag(self.magnitude("Mlw"))
        ml = _clean_mag(self.magnitude("Ml"))
        aut = _clean_mag(self.magnitude("Autmag"))
        if o.evaluation_mode == "manual":
            mag, mtype = (mlw, "Mlw") if not math.isnan(mlw) else (ml, "ML")
            quality = math.nan
        else:
            mag, mtype = aut, "aut.mag"
            quality = o.quality
        moment = math.nan
        if self.sp3 is not None:
            moment = _num(self.sp3.get("seismic_moment"))
        rec = {
            "time": o.time,
            "event_id": self.event_id,
            "sil_path": self.sil_path,
            "latitude": o.latitude,
            "longitude": o.longitude,
            "depth_km": o.depth_km,
            "h_err_km": o.h_err_km,
            "depth_err_km": o.depth_err_km,
            "magnitude": mag,
            "magnitude_type": None if math.isnan(mag) else mtype,
            "mag_mlw": mlw,
            "mag_ml": ml,
            "mag_auto": aut,
            "moment_nm": moment,
            "quality": quality,
            "evaluation_mode": o.evaluation_mode,
            "event_type": sil_event_type(o.event_type_code),
            "event_type_code": o.event_type_code,
        }
        df = coerce_catalog(pd.DataFrame([rec]).set_index("time"), source=SOURCE)
        prov = catalog_provenance(
            source=SOURCE,
            location=self.url,
            extra={"legacy": True, "detail": True},
        )
        with_provenance(df, prov)
        validate(df, strict=True)
        return df


def _clean_mag(v: float) -> float:
    return math.nan if math.isclose(v, _MAG_SENTINEL, abs_tol=1e-6) else v


def _origin(o: Mapping[str, Any]) -> SkjalftalisaOrigin:
    lat, lon = _num(o.get("latitude")), _num(o.get("longitude"))
    dlat, dlon = _num(o.get("latitude_error")), _num(o.get("longitude_error"))
    h = float(horizontal_error_km(dlat, dlon, lat))
    t = pd.Timestamp(o.get("origin_time"))
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    code = o.get("event_type")
    return SkjalftalisaOrigin(
        originating_system=o.get("originating_system"),
        time=t,
        latitude=lat,
        longitude=lon,
        depth_km=_num(o.get("depth")),
        latitude_error_deg=dlat,
        longitude_error_deg=dlon,
        depth_err_km=_num(o.get("depth_error")),
        h_err_km=h,
        nearest_station_km=_num(o.get("nearest_station_km")),
        n_stations=_int(o.get("number_of_stations_in_location")),
        quality=_num(o.get("quality")),
        row_num=_int(o.get("row_num")),
        event_type_code=None if code is None else str(code),
    )


def parse_detail(
    body: Mapping[str, Any], *, url: str = "", internal: bool = True
) -> SkjalftalisaDetail:
    """A decoded detail answer → :class:`SkjalftalisaDetail` (no network)."""
    if not isinstance(body, Mapping) or "id" not in body:
        raise SkjalftalisaError(f"{url}: not a detail answer", url=url)
    sp3_raw = body.get("sp3")
    sp3: Mapping[str, Any] | None = None
    if isinstance(sp3_raw, str) and sp3_raw.strip():
        try:
            decoded = json.loads(sp3_raw)
        except json.JSONDecodeError:
            decoded = None
        sp3 = decoded if isinstance(decoded, Mapping) else None
    elif isinstance(sp3_raw, Mapping):
        sp3, sp3_raw = sp3_raw, json.dumps(sp3_raw)
    mags = tuple(
        (str(m.get("magnitude_type")), _num(m.get("size")))
        for m in body.get("magnitudes") or ()
        if isinstance(m, Mapping)
    )
    origins = tuple(
        _origin(o) for o in body.get("origins") or () if isinstance(o, Mapping)
    )
    path = body.get("slungapath")
    return SkjalftalisaDetail(
        event_id=str(body["id"]),
        sil_path=str(path) if path else None,
        origins=origins,
        magnitudes=mags,
        sp3=sp3,
        sp3_raw=sp3_raw if isinstance(sp3_raw, str) else None,
        touched_aut=_int(body.get("touched_aut")),
        url=url,
        internal=internal,
        raw=dict(body),
    )


def read_skjalftalisa_detail(
    event_id: int | str,
    *,
    internal: bool = True,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    session: Transport | None = None,
) -> SkjalftalisaDetail:
    """The detail of one event: ``sil_path``, every origin with its errors,
    every magnitude, the decoded ``sp3`` block (LEGACY source).

    ``event_id`` is the legacy integer id (``1352381``; ``"SIL1352381"`` as
    the Quakes API spells it is accepted too). For samples, not catalogues.
    Raises :class:`SkjalftalisaNotFound` (a :class:`LookupError`) for an
    unknown id.
    """
    text = str(event_id).strip()
    if text.upper().startswith("SIL"):
        text = text[3:]
    if not text.isdigit():
        raise ValueError(f"event_id {event_id!r} is not a legacy integer id")
    url = f"{base_url(internal)}/{int(text)}"
    client = _Client(
        session, internal=internal, pause_s=0.0, timeout_s=timeout_s, retries=retries
    )
    body = client.get_json(url)
    return parse_detail(body, url=url, internal=internal)


__all__ = [
    "API_START",
    "DEFAULT_CHUNK",
    "DEFAULT_PAD",
    "INTERNAL_URL",
    "PUBLIC_URL",
    "ROW_COLUMNS",
    "SOURCE",
    "SYSTEM_AUT",
    "SYSTEM_PICKS",
    "HttpResponse",
    "SkjalftalisaDetail",
    "SkjalftalisaError",
    "SkjalftalisaNotFound",
    "SkjalftalisaOrigin",
    "SkjalftalisaReport",
    "SkjalftalisaUnreachable",
    "Transport",
    "UrllibTransport",
    "base_url",
    "count_skjalftalisa",
    "fetch_skjalftalisa_rows",
    "parse_detail",
    "read_skjalftalisa",
    "read_skjalftalisa_detail",
    "resolve_solutions",
    "rows_from_features",
]
