"""SeisComP FDSN event service reader onto the common schema (ticket T13).

:func:`read_fdsn` reads the internal fdsnws-event service of SeisComP
(``PLAN-seismicity.md`` §3.2) for the half-open window ``[start, end)`` and
returns an events frame of :mod:`geo_dataread.seismic.schema`, source
``fdsn-seiscomp``. :func:`read_fdsn_catalog` (QuakeML only) returns the same
events plus every origin, magnitude, (optionally) arrival and focal mechanism
in a :class:`FdsnCatalog`; :func:`read_fdsn_event` does that for one event id.
:func:`cross_check_quakes` compares a window with
:func:`geo_dataread.seismic.quakes_api.read_quakes` (``system="seiscomp"``) by
``event_id``: the reason T13 exists (R2), an independent door to SeisComP and
a check on the Quakes API layer.

Endpoint
--------

``http://seiscomp-p01.vedur.is:8080/fdsnws/event/1`` (plain HTTP; resolves
only inside the IMO network; laptop and yang-d01 reach it, T9b). Precedence:
the ``base_url=`` argument, then ``$GPS_FDSN_EVENT_URL``, then the default.
When the host cannot be reached, :class:`FdsnUnreachable` says so; the reader
never falls back to another service (the Quakes API is a different layer
with different fields, so switching is the caller's decision).

Requests and paging (probed 2026-11-13/14)
-------------------------------------------

* The server has **no** ``updatedafter`` (400) and no count endpoint.
* ``starttime`` and ``endtime`` are both **inclusive** on the server and take
  microseconds; the reader sends ``[start, end]`` and cuts to ``[start, end)``
  on the client.
* The default order is newest first, so ``orderby=time-asc`` is always sent.
* Paging is **keyset with an offset tie-break**: each page asks
  ``limit=FdsnPolicy.page_size``; after a full page the next request starts
  at the last origin time seen, with ``offset = 1 + (events already seen at
  or after that time)`` (``offset`` is 1-based: ``offset=0`` is a 400). A page
  shorter than ``limit``, or HTTP 204 (empty body: no events), ends the scan.
  Re-anchoring on time (rather than a growing offset) means an event that
  was already returned and then leaves the result set during the scan (a
  deletion, or a revision that moves its origin time) does not make the next
  page skip an unseen event, as a pure offset scan would; events seen twice
  are dropped by ``event_id`` (first version kept, counted). Not covered: an
  unseen event revised to before the anchor during the scan, and the
  tie-break assumes the server orders same-time events alike on each request
  (re-fetching a recent window, T14, covers both). More than
  ``FdsnPolicy.max_pages`` pages raises :class:`FdsnError` instead of
  returning a truncated catalogue.
* Errors are plain text (``Error 400: Bad Request`` / blank / detail line).
  4xx raise :class:`FdsnError` with the detail line; 5xx are retried with
  backoff; a 404 "no matching events" (only with ``nodata=404``, never sent
  by the reader) is an empty page. Connection, DNS and timeout errors raise
  :class:`FdsnUnreachable` at once.
* Polite: ``FdsnPolicy.pause_s`` between requests. Transport is
  :mod:`urllib` (via :class:`~geo_dataread.seismic.quakes_api.UrllibSession`);
  any ``requests.Session``-shaped object can be passed as ``session=``.

Formats
-------

``format="xml"`` (QuakeML 1.2, default) is the full door. ``format="text"``
is the compact list (``#EventID|Time|Latitude|Longitude|Depth/km|Author|…|
MagType|Magnitude|MagAuthor|EventLocationName|EventType``), but it has **no
evaluation mode and no status**, and ``evaluation_mode`` is non-nullable in
the schema. The text reader therefore refuses unless the caller opts in to
``text_evaluation_mode="author"``: a machine author (``…@host``) →
``automatic``, anything else → ``manual``. That rule is wrong for 5 of the 48
events of the 2026-09-10 fixture (analysts reject or retype an automatic
origin without relocating it, and the text ``Author`` is then the analyst),
so it is labelled in the provenance and the report.

Field mapping (QuakeML → schema column)
---------------------------------------

All location fields come from the event's **preferred origin**
(``preferredOriginID``), the magnitude from the **preferred magnitude**.

=====================  ======================================================
Schema column          From
=====================  ======================================================
``time`` (index)       ``origin/time/value`` (``Z``)
``event_id``           last path segment of ``event/@publicID`` (``IMO…``)
``latitude`` …         ``origin/latitude|longitude/value``
``depth_km``           ``origin/depth/value`` / 1000 (QuakeML is in metres)
``h_err_km``           SIL origins (``earthModelID`` ``…/imo/SIL``): lat/lon
                       uncertainty is in **degrees** (SIL file errors) →
                       :func:`~geo_dataread.seismic.sil_catalog.horizontal_error_km`
                       (with cos(lat), as the file reader). LOCSAT and other
                       origins: **NA** — SeisComP writes km at a 90 %
                       confidence level there, not 1σ; the raw values are in
                       the origins frame (decision for BGÓ, see the report).
``depth_err_km``       SIL origins: ``depth/uncertainty`` / 1000 (= SIL
                       ``Ddepth``); other origins NA (same reason)
``magnitude``          preferred ``magnitude/mag/value`` (raw, Q7). NA
                       when the ``preferredMagnitudeID`` is absent or names a
                       magnitude the service does not export (2 of 64 events
                       on 2026-11-13, automatic SILCLEAN origins; the Quakes
                       API list still has a value for them); never replaced
                       by another magnitude
``magnitude_type``     preferred ``magnitude/type`` (``ML_SIL``, ``Mpgv_w``,
                       ``Msil_auto``; an unknown type fails validation)
``mag_mlw``            ``Msil_lw`` (= SIL Mlw, 0.25 for IMO2024qnogqh)
``mag_ml``             ``Msil_l`` (= SIL ML, −0.5 for IMO2024qnogqh)
``mag_auto``           ``Msil_auto``. These three need
                       ``include_all_magnitudes=True`` (the default list
                       carries the preferred magnitude only); one on the
                       preferred origin wins, else the newest.
``evaluation_mode``    preferred origin ``evaluationMode``
``evaluation_status``  preferred origin ``evaluationStatus``
``event_type``         ``event/type`` if in the schema vocabulary; QuakeML
                       ``other event`` (and anything else) → ``other``;
                       absent (back-imported events) → NA
``event_type_code``    ``event/type``, raw
``n_phases``           preferred origin ``quality/usedPhaseCount`` (= the
                       Quakes API ``phases``; 50 for IMO2026rusibb)
``azimuthal_gap``      preferred origin ``quality/azimuthalGap``
``updated_time``       ``event/creationInfo/modificationTime`` if present,
                       else NA (SeisComP does not write it; no creationTime
                       equals the Quakes API ``updated_time``)
``sil_path`` …         NA (``sil_path``, ``sil_aut_path``, ``moment_nm``,
                       ``quality``)
=====================  ======================================================

Text format: the same columns from the text fields (depth already km);
``evaluation_status``, ``n_phases``, ``azimuthal_gap``, errors and the
per-scale magnitudes are NA. ``extras=True`` keeps ``region`` (the event
description / ``EventLocationName``) and, for QuakeML, ``n_origins``,
``n_magnitudes`` and ``earth_model``.

SeisComP maps its own types ``not locatable`` (and other non-QuakeML types)
to QuakeML ``other event`` on export, while the Quakes API publishes
``not locatable``: FDSN loses that distinction (found in T13).

Detail frames (:class:`FdsnCatalog`)
------------------------------------

Plain DataFrames (not schema-validated: they carry SeisComP's own magnitude
types such as ``Msil_lw``): ``origins`` (one row per origin, raw
uncertainties with their unit), ``magnitudes`` (one row per magnitude),
``arrivals`` (``include_arrivals=True``: one row per arrival of every origin,
joined to its pick for station and pick time; ``includepicks`` is sent too,
because the station code lives in the pick) and ``mechanisms`` (schema
mechanisms frame from ``focalMechanism``; SeisComP has none today (T9), but
they are parsed and counted if they appear; ``include_focal_mechanisms=True``
asks for them). Arrivals cost about 0.8 KB per arrival (45 KB for a 53-pick
event), so use them for windows, not whole catalogues.
"""

from __future__ import annotations

import dataclasses
import math
import os
import time as _time
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date, datetime
from typing import Any, Final, Literal, TypeAlias, TypeVar

import numpy as np
import pandas as pd

from geo_dataread.seismic.quakes_api import (
    DEPTH_BOUNDS,
    SIZE_BOUNDS,
    Session,
    UrllibSession,
    _range_guard,
    _utc,
    read_quakes,
)
from geo_dataread.seismic.schema import (
    EVENT_TYPES,
    catalog_provenance,
    coerce_catalog,
    coerce_mechanisms,
    empty_mechanisms,
    validate,
    validate_mechanisms,
    with_provenance,
)
from geo_dataread.seismic.sil_catalog import horizontal_error_km

TimeLike: TypeAlias = date | datetime | str | pd.Timestamp
Format = Literal["xml", "text"]

#: Schema source name (T23).
SOURCE: Final = "fdsn-seiscomp"

#: fdsnws-event base; ``/query`` is appended.
DEFAULT_BASE: Final = "http://seiscomp-p01.vedur.is:8080/fdsnws/event/1"

#: Environment variable overriding the base URL.
ENV_BASE: Final = "GPS_FDSN_EVENT_URL"

#: Generic User-Agent: no personal identifier in any request.
USER_AGENT: Final = "geo_dataread-fdsn"

#: QuakeML 1.2 BED namespace.
QML_NS: Final = "http://quakeml.org/xmlns/bed/1.2"
_NS: Final = "{" + QML_NS + "}"

#: Text-format header (fdsnws-event 1.2 text).
TEXT_COLUMNS: Final = (
    "EventID",
    "Time",
    "Latitude",
    "Longitude",
    "Depth/km",
    "Author",
    "Catalog",
    "Contributor",
    "ContributorID",
    "MagType",
    "Magnitude",
    "MagAuthor",
    "EventLocationName",
    "EventType",
)

#: SeisComP magnitude type → per-scale schema column (QuakeML with all
#: magnitudes; checked on IMO2024qnogqh against the SIL files).
SCALE_COLUMNS: Final[Mapping[str, str]] = {
    "Msil_lw": "mag_mlw",
    "Msil_l": "mag_ml",
    "Msil_auto": "mag_auto",
}

#: Earth-model id suffix of SeisComP's SIL back-imported origins, whose lat/lon
#: uncertainty is SIL's error in degrees (1σ) and depth uncertainty SIL Ddepth.
SIL_EARTH_MODEL_SUFFIX: Final = "/imo/SIL"

#: Default tolerances of :func:`compare_with_quakes`. The Quakes API rounds
#: magnitude and depth to about 6 decimals, and times to the microsecond.
DEFAULT_TOLERANCES: Final[Mapping[str, float]] = {
    "time_s": 1e-3,
    "latitude": 1e-5,
    "longitude": 1e-5,
    "depth_km": 1e-4,
    "magnitude": 1e-5,
}

#: Fields compared by :func:`compare_with_quakes` without a tolerance.
EXACT_FIELDS: Final = (
    "magnitude_type",
    "evaluation_mode",
    "evaluation_status",
    "event_type",
    "n_phases",
)

_REPORT_KEYS: Final = (
    "requests",
    "pages",
    "retries",
    "empty_pages",
    "events_parsed",
    "duplicates_dropped",
    "outside_window",
    "dropped_no_origin",
    "preferred_origin_implicit",
    "magnitude_missing",
    "preferred_magnitude_dangling",
    "dropped_bad_position",
    "h_err_not_sil_model",
    "focal_mechanisms",
    "mode_inferred",
)

#: Indirection so tests can count pauses without sleeping.
_sleep: Callable[[float], None] = _time.sleep

T = TypeVar("T")


# --------------------------------------------------------------------------
# Errors, policy, report
# --------------------------------------------------------------------------


class FdsnError(RuntimeError):
    """The FDSN service answered with an error, or a response cannot be read."""

    def __init__(self, message: str, *, status: int | None = None, url: str = ""):
        super().__init__(message)
        self.status = status
        self.url = url


class FdsnUnreachable(FdsnError):
    """The host could not be reached (DNS, connection refused, timeout)."""


@dataclasses.dataclass(frozen=True)
class FdsnPolicy:
    """How the reader talks to the service.

    ``page_size``: ``limit`` of one request. ``max_pages``: more pages than
    this raise :class:`FdsnError` (never a silent truncation). ``pause_s``:
    sleep between consecutive requests. ``timeout_s``: per request.
    ``retries``: extra attempts on HTTP 5xx, ``backoff_s`` × 2^attempt apart.
    """

    page_size: int = 1000
    max_pages: int = 500
    pause_s: float = 1.0
    timeout_s: float = 120.0
    retries: int = 3
    backoff_s: float = 2.0

    def __post_init__(self) -> None:
        if self.page_size < 1 or self.max_pages < 1:
            raise ValueError("page_size and max_pages must be >= 1")
        if self.pause_s < 0 or self.timeout_s <= 0 or self.backoff_s < 0:
            raise ValueError("pause_s/backoff_s must be >= 0, timeout_s > 0")
        if self.retries < 0:
            raise ValueError("retries must be >= 0")


DEFAULT_POLICY: Final = FdsnPolicy()


@dataclasses.dataclass(frozen=True)
class FdsnReadReport:
    """What one read did. ``counts`` keys: ``requests`` (retries included),
    ``pages``, ``retries``, ``empty_pages`` (204), ``events_parsed``,
    ``duplicates_dropped``, ``outside_window``, ``dropped_no_origin``,
    ``preferred_origin_implicit`` (no ``preferredOriginID``, one origin),
    ``magnitude_missing`` (of which ``preferred_magnitude_dangling``: a
    ``preferredMagnitudeID`` that is not in the response), ``dropped_bad_position``, ``out_of_range:<col>``,
    ``h_err_not_sil_model`` (errors left NA), ``focal_mechanisms``,
    ``mode_inferred`` (text), ``events``, ``events_manual``,
    ``events_automatic``."""

    endpoint: str
    counts: Mapping[str, int]
    elapsed_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "counts": dict(sorted(self.counts.items())),
            "elapsed_s": round(self.elapsed_s, 3),
        }


@dataclasses.dataclass(frozen=True)
class FdsnCatalog:
    """Events (schema) plus the QuakeML detail of the same read."""

    events: pd.DataFrame
    origins: pd.DataFrame
    magnitudes: pd.DataFrame
    arrivals: pd.DataFrame
    mechanisms: pd.DataFrame
    report: FdsnReadReport


# --------------------------------------------------------------------------
# Endpoint and parameters
# --------------------------------------------------------------------------


def resolve_base(base_url: str | None = None) -> str:
    """The base URL: the argument, then ``$GPS_FDSN_EVENT_URL``, then default."""
    base = base_url or os.environ.get(ENV_BASE, "").strip() or DEFAULT_BASE
    base = base.rstrip("/")
    if base.endswith("/query"):
        base = base[: -len("/query")]
    if urllib.parse.urlsplit(base).scheme not in ("http", "https"):
        raise ValueError(f"FDSN base URL {base!r} is not http(s)")
    return base


def _fmt_time(ts: pd.Timestamp) -> str:
    """ISO UTC with microseconds and no zone (what the service accepts)."""
    return str(ts.strftime("%Y-%m-%dT%H:%M:%S.%f"))


def _region_params(
    bbox: tuple[float, float, float, float] | None,
    radius: tuple[float, ...] | None,
) -> list[tuple[str, str]]:
    if bbox is not None and radius is not None:
        raise ValueError("give either bbox or radius, not both")
    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError("bbox is (min_lon, min_lat, max_lon, max_lat)")
        w, s, e, n = (float(v) for v in bbox)
        if not (w < e and s < n) or not all(map(math.isfinite, (w, s, e, n))):
            raise ValueError(f"bbox {bbox} is empty, inverted or not finite")
        return [
            ("minlatitude", f"{s:g}"),
            ("maxlatitude", f"{n:g}"),
            ("minlongitude", f"{w:g}"),
            ("maxlongitude", f"{e:g}"),
        ]
    if radius is not None:
        if len(radius) not in (3, 4):
            raise ValueError("radius is (lat, lon, maxradius_deg[, minradius_deg])")
        lat, lon, rmax = (float(v) for v in radius[:3])
        rmin = float(radius[3]) if len(radius) == 4 else 0.0
        if not (-90 <= lat <= 90 and -180 <= lon <= 180 and 0 <= rmin < rmax <= 180):
            raise ValueError(f"radius {radius} is out of range")
        out = [
            ("latitude", f"{lat:g}"),
            ("longitude", f"{lon:g}"),
            ("maxradius", f"{rmax:g}"),
        ]
        if rmin > 0:
            out.append(("minradius", f"{rmin:g}"))
        return out
    return []


def _limit_params(
    name: str, lo: float | None, hi: float | None
) -> list[tuple[str, str]]:
    for label, v in (("min", lo), ("max", hi)):
        if v is not None and not math.isfinite(v):
            raise ValueError(f"{label}_{name} must be finite")
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"min_{name} {lo} > max_{name} {hi}")
    out = []
    if lo is not None:
        out.append((f"min{name}", f"{lo:g}"))
    if hi is not None:
        out.append((f"max{name}", f"{hi:g}"))
    return out


# --------------------------------------------------------------------------
# HTTP with policy
# --------------------------------------------------------------------------


def _detail(content: bytes) -> str:
    """The detail line of a plain-text fdsnws error body."""
    lines = [ln.strip() for ln in content.decode("utf-8", "replace").splitlines()]
    lines = [ln for ln in lines if ln]
    if len(lines) >= 2 and lines[0].startswith("Error"):
        return lines[1]
    return " ".join(lines)[:300]


class _Client:
    def __init__(
        self, session: Session, base: str, policy: FdsnPolicy, tally: Counter[str]
    ) -> None:
        self.session = session
        self.base = base
        self.policy = policy
        self.tally = tally
        self._sent = False

    def get(self, params: Sequence[tuple[str, str]], accept: str) -> bytes:
        """Body of one query; ``b""`` for 204 (and a 404 "no matching events")."""
        url = f"{self.base}/query?{urllib.parse.urlencode(list(params))}"
        headers = {"User-Agent": USER_AGENT, "Accept": accept}
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
                host = urllib.parse.urlsplit(url).netloc
                raise FdsnUnreachable(
                    f"FDSN event service {host} could not be reached: {exc!r}. "
                    "The default host resolves only inside the IMO network; "
                    f"another fdsnws-event service can be set with base_url= or "
                    f"${ENV_BASE}. The reader does not fall back by itself.",
                    url=url,
                ) from exc
            status = int(resp.status_code)
            if status == 204:
                self.tally["empty_pages"] += 1
                return b""
            if 200 <= status < 300:
                return bytes(resp.content)
            detail = _detail(resp.content)
            if status == 404 and detail.startswith("no matching events"):
                self.tally["empty_pages"] += 1
                return b""
            if status >= 500 and attempt < self.policy.retries:
                attempt += 1
                self.tally["retries"] += 1
                if self.policy.backoff_s > 0:
                    _sleep(self.policy.backoff_s * 2 ** (attempt - 1))
                continue
            raise FdsnError(
                f"FDSN {status} for {url}: {detail}"
                + (f" (after {attempt} retries)" if attempt else ""),
                status=status,
                url=url,
            )


def _page_through(
    client: _Client,
    lo: pd.Timestamp,
    hi: pd.Timestamp,
    filters: Sequence[tuple[str, str]],
    accept: str,
    parse: Callable[[bytes], tuple[list[T], int]],
    key: Callable[[T], tuple[str, pd.Timestamp | None]],
) -> list[T]:
    """Every item of ``[lo, hi]`` (server-inclusive), deduped by event id.

    ``parse(body)`` → (items, number of events in the body, parsed or not).
    """
    policy = client.policy
    seen: dict[str, T] = {}
    times: dict[str, pd.Timestamp] = {}
    anchor = lo
    skip = 0
    for _ in range(policy.max_pages):
        params = [
            ("starttime", _fmt_time(anchor)),
            ("endtime", _fmt_time(hi)),
            *filters,
            ("orderby", "time-asc"),
            ("limit", str(policy.page_size)),
        ]
        if skip:
            params.append(("offset", str(skip + 1)))
        body = client.get(params, accept)
        client.tally["pages"] += 1
        items, n_raw = parse(body) if body.strip() else ([], 0)
        for item in items:
            eid, t = key(item)
            if eid in seen:
                client.tally["duplicates_dropped"] += 1
                continue
            seen[eid] = item
            if t is not None:
                times[eid] = t
        if n_raw < policy.page_size:
            return list(seen.values())
        page_times = [t for t in (key(i)[1] for i in items) if t is not None]
        if not page_times:
            raise FdsnError("a full page without any origin time: cannot page on")
        anchor = max(page_times)
        skip = sum(1 for t in times.values() if t >= anchor)
    raise FdsnError(
        f"more than {policy.max_pages} pages of {policy.page_size} events for "
        f"[{lo}, {hi}]; raise FdsnPolicy.max_pages/page_size or split the window"
    )


# --------------------------------------------------------------------------
# Text format
# --------------------------------------------------------------------------


def _event_type(code: Any) -> str | None:
    if code is None or (isinstance(code, float) and math.isnan(code)) or code == "":
        return None
    return str(code) if code in EVENT_TYPES else "other"


def _is_machine(author: Any) -> bool:
    return isinstance(author, str) and "@" in author


def parse_text(body: bytes) -> pd.DataFrame:
    """Rows of one fdsnws text response as raw strings (columns
    :data:`TEXT_COLUMNS`). An empty body is an empty frame."""
    text = body.decode("utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return pd.DataFrame(columns=list(TEXT_COLUMNS))
    head = lines[0].lstrip("#").strip().split("|")
    if tuple(c.strip() for c in head) != TEXT_COLUMNS:
        raise FdsnError(f"unexpected text header: {lines[0][:200]!r}")
    rows = []
    for ln in lines[1:]:
        cells = ln.split("|")
        if len(cells) != len(TEXT_COLUMNS):
            raise FdsnError(f"text row with {len(cells)} fields: {ln[:200]!r}")
        rows.append(cells)
    return pd.DataFrame(rows, columns=list(TEXT_COLUMNS))


def text_to_frame(
    raw: pd.DataFrame,
    *,
    evaluation_mode: Literal["author"],
    extras: bool = False,
    tally: Counter[str] | None = None,
) -> pd.DataFrame:
    """Map :func:`parse_text` rows to a coerced (not validated) schema frame.

    ``evaluation_mode="author"`` is the only rule (see the module docstring:
    machine author → automatic, else manual; not exact)."""
    if evaluation_mode != "author":
        raise ValueError("text evaluation_mode rule: only 'author'")
    tally = tally if tally is not None else Counter()
    if raw.empty:
        out = coerce_catalog(
            pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="time")),
            source=SOURCE,
        )
        if extras:
            out["region"] = pd.Series(index=out.index, dtype="str")
        return out

    def num(col: str) -> Any:
        return pd.to_numeric(raw[col].replace("", np.nan), errors="coerce").astype(
            "float64"
        )

    t = pd.to_datetime(raw["Time"], format="ISO8601")
    t = t.dt.tz_localize("UTC") if t.dt.tz is None else t.dt.tz_convert("UTC")
    mtype = raw["MagType"].replace("", None)
    df = pd.DataFrame(
        {
            "time": t,
            "event_id": raw["EventID"].to_numpy(),
            "latitude": num("Latitude").to_numpy(),
            "longitude": num("Longitude").to_numpy(),
            "depth_km": num("Depth/km").to_numpy(),
            "magnitude": num("Magnitude").to_numpy(),
            "magnitude_type": mtype.to_numpy(),
            "evaluation_mode": [
                "automatic" if _is_machine(a) else "manual" for a in raw["Author"]
            ],
            "event_type": [_event_type(c) for c in raw["EventType"]],
            "event_type_code": raw["EventType"].replace("", None).to_numpy(),
            "region": raw["EventLocationName"].replace("", None).to_numpy(),
        }
    )
    tally["mode_inferred"] += len(df)
    tally["magnitude_missing"] += int(df["magnitude"].isna().sum())
    for label, column in SCALE_COLUMNS.items():
        hit = (df["magnitude_type"] == label).to_numpy()
        df[column] = np.where(hit, df["magnitude"], np.nan)
    df = _range_guard(df, tally)
    if not extras:
        df = df.drop(columns=["region"])
    return coerce_catalog(df.set_index("time"), source=SOURCE)


# --------------------------------------------------------------------------
# QuakeML
# --------------------------------------------------------------------------


def _q(path: str) -> str:
    return "/".join(_NS + p for p in path.split("/"))


def _txt(el: ET.Element | None, path: str) -> str | None:
    if el is None:
        return None
    v = el.findtext(_q(path))
    if v is None:
        return None
    v = v.strip()
    return v or None


def _flt(el: ET.Element | None, path: str) -> float:
    v = _txt(el, path)
    if v is None:
        return math.nan
    try:
        return float(v)
    except ValueError as exc:
        raise FdsnError(f"QuakeML {path} is not a number: {v!r}") from exc


def _int(el: ET.Element | None, path: str) -> int | None:
    v = _flt(el, path)
    return None if math.isnan(v) else int(v)


def _ts(v: str | None) -> pd.Timestamp | None:
    if v is None:
        return None
    ts = pd.Timestamp(v)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _short_id(public_id: str | None) -> str | None:
    if not public_id:
        return None
    return public_id.rstrip("/").rsplit("/", 1)[-1]


@dataclasses.dataclass
class _QmlEvent:
    """One parsed ``<event>``: schema row + detail rows."""

    event_id: str
    time: pd.Timestamp | None
    row: dict[str, Any] | None
    origins: list[dict[str, Any]]
    magnitudes: list[dict[str, Any]]
    arrivals: list[dict[str, Any]]
    mechanisms: list[dict[str, Any]]


def _origin_row(
    event_id: str, o: ET.Element, preferred_id: str | None
) -> dict[str, Any]:
    pid = o.get("publicID")
    model = _txt(o, "earthModelID")
    method = _txt(o, "methodID")
    is_sil = bool(model and model.endswith(SIL_EARTH_MODEL_SUFFIX))
    unit = "deg" if is_sil else ("km" if method and method.endswith("LOCSAT") else None)
    depth_m = _flt(o, "depth/value")
    depth_unc_m = _flt(o, "depth/uncertainty")
    return {
        "event_id": event_id,
        "origin_id": pid,
        "preferred": pid is not None and pid == preferred_id,
        "time": _ts(_txt(o, "time/value")),
        "latitude": _flt(o, "latitude/value"),
        "longitude": _flt(o, "longitude/value"),
        "depth_km": depth_m / 1000.0,
        "time_unc_s": _flt(o, "time/uncertainty"),
        "lat_unc": _flt(o, "latitude/uncertainty"),
        "lon_unc": _flt(o, "longitude/uncertainty"),
        "latlon_unc_unit": unit,
        "confidence_level": _flt(o, "latitude/confidenceLevel"),
        "depth_unc_km": depth_unc_m / 1000.0,
        "depth_type": _txt(o, "depthType"),
        "h_unc_km": _flt(o, "originUncertainty/horizontalUncertainty") / 1000.0,
        "min_h_unc_km": _flt(o, "originUncertainty/minHorizontalUncertainty") / 1000.0,
        "max_h_unc_km": _flt(o, "originUncertainty/maxHorizontalUncertainty") / 1000.0,
        "azimuth_max_h_unc": _flt(
            o, "originUncertainty/azimuthMaxHorizontalUncertainty"
        ),
        "evaluation_mode": _txt(o, "evaluationMode"),
        "evaluation_status": _txt(o, "evaluationStatus"),
        "origin_type": _txt(o, "type"),
        "author": _txt(o, "creationInfo/author"),
        "creation_time": _ts(_txt(o, "creationInfo/creationTime")),
        "method": method,
        "earth_model": model,
        "used_phase_count": _int(o, "quality/usedPhaseCount"),
        "associated_phase_count": _int(o, "quality/associatedPhaseCount"),
        "used_station_count": _int(o, "quality/usedStationCount"),
        "associated_station_count": _int(o, "quality/associatedStationCount"),
        "standard_error_s": _flt(o, "quality/standardError"),
        "azimuthal_gap": _flt(o, "quality/azimuthalGap"),
        "min_distance_deg": _flt(o, "quality/minimumDistance"),
        "median_distance_deg": _flt(o, "quality/medianDistance"),
        "max_distance_deg": _flt(o, "quality/maximumDistance"),
    }


def _magnitude_row(
    event_id: str, m: ET.Element, preferred_id: str | None
) -> dict[str, Any]:
    pid = m.get("publicID")
    return {
        "event_id": event_id,
        "magnitude_id": pid,
        "preferred": pid is not None and pid == preferred_id,
        "origin_id": _txt(m, "originID"),
        "magnitude_type": _txt(m, "type"),
        "magnitude": _flt(m, "mag/value"),
        "uncertainty": _flt(m, "mag/uncertainty"),
        "station_count": _int(m, "stationCount"),
        "method": _txt(m, "methodID"),
        "evaluation_mode": _txt(m, "evaluationMode"),
        "evaluation_status": _txt(m, "evaluationStatus"),
        "author": _txt(m, "creationInfo/author"),
        "creation_time": _ts(_txt(m, "creationInfo/creationTime")),
    }


def _arrival_rows(
    event_id: str, o: ET.Element, picks: Mapping[str, ET.Element]
) -> list[dict[str, Any]]:
    rows = []
    for a in o.findall(_q("arrival")):
        pick_id = _txt(a, "pickID")
        p = picks.get(pick_id or "")
        wf = p.find(_q("waveformID")) if p is not None else None
        rows.append(
            {
                "event_id": event_id,
                "origin_id": o.get("publicID"),
                "pick_id": pick_id,
                "phase": _txt(a, "phase"),
                "time_residual_s": _flt(a, "timeResidual"),
                "time_weight": _flt(a, "timeWeight"),
                "distance_deg": _flt(a, "distance"),
                "azimuth_deg": _flt(a, "azimuth"),
                "network": wf.get("networkCode") if wf is not None else None,
                "station": wf.get("stationCode") if wf is not None else None,
                "location": wf.get("locationCode") if wf is not None else None,
                "channel": wf.get("channelCode") if wf is not None else None,
                "pick_time": _ts(_txt(p, "time/value")),
                "pick_evaluation_mode": _txt(p, "evaluationMode"),
                "polarity": _txt(p, "polarity"),
                "pick_author": _txt(p, "creationInfo/author"),
            }
        )
    return rows


def _mechanism_row(event_id: str, fm: ET.Element) -> dict[str, Any] | None:
    np1 = fm.find(_q("nodalPlanes/nodalPlane1"))
    if np1 is None:
        return None
    np2 = fm.find(_q("nodalPlanes/nodalPlane2"))

    def ang(el: ET.Element | None, what: str) -> float:
        return _flt(el, f"{what}/value") if el is not None else math.nan

    return {
        "event_id": event_id,
        "method": (
            "moment_tensor" if fm.find(_q("momentTensor")) is not None else "other"
        ),
        "strike1": ang(np1, "strike"),
        "dip1": ang(np1, "dip"),
        "rake1": ang(np1, "rake"),
        "strike2": ang(np2, "strike"),
        "dip2": ang(np2, "dip"),
        "rake2": ang(np2, "rake"),
        "misfit": _flt(fm, "misfit"),
        "n_polarities": _int(fm, "stationPolarityCount"),
        "focal_mechanism_id": fm.get("publicID"),
        "method_id": _txt(fm, "methodID"),
    }


def _pick_scale(
    mags: Sequence[dict[str, Any]], mtype: str, preferred_origin: str | None
) -> float:
    cands = [m for m in mags if m["magnitude_type"] == mtype]
    if not cands:
        return math.nan
    on_pref = [m for m in cands if m["origin_id"] == preferred_origin]
    pool = on_pref or cands
    newest = max(
        pool,
        key=lambda m: (m["creation_time"] or pd.Timestamp.min.tz_localize("UTC"),),
    )
    return float(newest["magnitude"])


def _parse_event(ev: ET.Element, tally: Counter[str], *, extras: bool) -> _QmlEvent:
    event_id = _short_id(ev.get("publicID"))
    if event_id is None:
        raise FdsnError("QuakeML event without publicID")
    pref_o = _txt(ev, "preferredOriginID")
    pref_m = _txt(ev, "preferredMagnitudeID")
    origins_el = ev.findall(_q("origin"))
    picks = {p.get("publicID") or "": p for p in ev.findall(_q("pick"))}
    origins = [_origin_row(event_id, o, pref_o) for o in origins_el]
    mags = [_magnitude_row(event_id, m, pref_m) for m in ev.findall(_q("magnitude"))]
    arrivals = [r for o in origins_el for r in _arrival_rows(event_id, o, picks)]
    mechs = []
    for fm in ev.findall(_q("focalMechanism")):
        tally["focal_mechanisms"] += 1
        mrow = _mechanism_row(event_id, fm)
        if mrow is not None:
            mechs.append(mrow)

    po = next((o for o in origins if o["preferred"]), None)
    if po is None and pref_o is None and len(origins) == 1:
        po = origins[0]
        po["preferred"] = True
        tally["preferred_origin_implicit"] += 1
    if po is None or po["time"] is None:
        tally["dropped_no_origin"] += 1
        return _QmlEvent(event_id, None, None, origins, mags, arrivals, mechs)

    pm = next((m for m in mags if m["preferred"]), None)
    if pm is None:
        tally["magnitude_missing"] += 1
        if pref_m is not None:
            # SeisComP names a preferred magnitude it does not export (seen
            # 2026-11-13 on SILCLEAN origins); never replaced by another one.
            tally["preferred_magnitude_dangling"] += 1
    if po["latlon_unc_unit"] == "deg":
        h_err = float(horizontal_error_km(po["lat_unc"], po["lon_unc"], po["latitude"]))
        d_err = po["depth_unc_km"]
    else:
        h_err = d_err = math.nan
        tally["h_err_not_sil_model"] += 1
    code = _txt(ev, "type")
    row: dict[str, Any] = {
        "time": po["time"],
        "event_id": event_id,
        "latitude": po["latitude"],
        "longitude": po["longitude"],
        "depth_km": po["depth_km"],
        "h_err_km": h_err,
        "depth_err_km": d_err,
        "magnitude": pm["magnitude"] if pm else math.nan,
        "magnitude_type": pm["magnitude_type"] if pm else None,
        "evaluation_mode": po["evaluation_mode"],
        "evaluation_status": po["evaluation_status"],
        "event_type": _event_type(code),
        "event_type_code": code,
        "n_phases": po["used_phase_count"],
        "azimuthal_gap": po["azimuthal_gap"],
        "updated_time": _ts(_txt(ev, "creationInfo/modificationTime")),
    }
    for mtype, column in SCALE_COLUMNS.items():
        row[column] = _pick_scale(mags, mtype, po["origin_id"])
    if extras:
        region = None
        for d in ev.findall(_q("description")):
            if _txt(d, "type") in (None, "region name"):
                region = _txt(d, "text")
                break
        row["region"] = region
        row["n_origins"] = len(origins)
        row["n_magnitudes"] = len(mags)
        row["earth_model"] = po["earth_model"]
    for m in mechs:
        m["time"] = po["time"]
    return _QmlEvent(event_id, po["time"], row, origins, mags, arrivals, mechs)


def parse_quakeml(
    body: bytes, *, extras: bool = False, tally: Counter[str] | None = None
) -> tuple[list[_QmlEvent], int]:
    """Parse one QuakeML 1.2 response: (events, number of ``<event>`` elements)."""
    tally = tally if tally is not None else Counter()
    if not body.strip():
        return [], 0
    try:
        root = ET.fromstring(body)  # noqa: S314 - trusted internal service
    except ET.ParseError as exc:
        raise FdsnError(f"response is not XML: {exc}; {body[:200]!r}") from exc
    params = root.find(_NS + "eventParameters")
    if params is None:
        raise FdsnError(
            f"no QuakeML 1.2 eventParameters (root {root.tag!r}); expected "
            f"namespace {QML_NS}"
        )
    elements = params.findall(_NS + "event")
    events = [_parse_event(ev, tally, extras=extras) for ev in elements]
    tally["events_parsed"] += len(elements)
    return events, len(elements)


_ORIGIN_COLUMNS: Final = tuple(_origin_row("", ET.Element(_NS + "origin"), None).keys())
_MAGNITUDE_COLUMNS: Final = tuple(
    _magnitude_row("", ET.Element(_NS + "magnitude"), None).keys()
)
_ARRIVAL_COLUMNS: Final = (
    "event_id",
    "origin_id",
    "pick_id",
    "phase",
    "time_residual_s",
    "time_weight",
    "distance_deg",
    "azimuth_deg",
    "network",
    "station",
    "location",
    "channel",
    "pick_time",
    "pick_evaluation_mode",
    "polarity",
    "pick_author",
)
_INT_DETAIL: Final = (
    "used_phase_count",
    "associated_phase_count",
    "used_station_count",
    "associated_station_count",
    "station_count",
)


def _detail_frame(
    rows: Iterable[dict[str, Any]], columns: Sequence[str]
) -> pd.DataFrame:
    df = pd.DataFrame(list(rows), columns=list(columns))
    for col in df.columns:
        if col in _INT_DETAIL:
            df[col] = df[col].astype("Int64")
        elif col in ("time", "creation_time", "pick_time"):
            df[col] = pd.to_datetime(df[col], utc=True)
        elif col == "preferred":
            df[col] = df[col].astype(bool)
    return df


def quakeml_to_catalog(
    events: Sequence[_QmlEvent],
    *,
    extras: bool = False,
    tally: Counter[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(events, origins, magnitudes, arrivals, mechanisms) frames. Events and
    mechanisms are coerced (not yet validated) schema frames."""
    tally = tally if tally is not None else Counter()
    rows = [e.row for e in events if e.row is not None]
    if rows:
        df = pd.DataFrame(rows)
        for col in ("latitude", "longitude", "depth_km", "magnitude"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        df = _range_guard(df, tally)
        df["time"] = pd.DatetimeIndex(df["time"])
        df["updated_time"] = pd.to_datetime(df["updated_time"], utc=True)
        ev_frame = coerce_catalog(df.set_index("time"), source=SOURCE)
    else:
        ev_frame = coerce_catalog(
            pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="time")),
            source=SOURCE,
        )
        if extras:
            for col, dt in (
                ("region", "str"),
                ("n_origins", "Int64"),
                ("n_magnitudes", "Int64"),
                ("earth_model", "str"),
            ):
                ev_frame[col] = pd.Series(index=ev_frame.index, dtype=dt)
    origins = _detail_frame((o for e in events for o in e.origins), _ORIGIN_COLUMNS)
    mags = _detail_frame((m for e in events for m in e.magnitudes), _MAGNITUDE_COLUMNS)
    arrivals = _detail_frame((a for e in events for a in e.arrivals), _ARRIVAL_COLUMNS)
    mech_rows = [m for e in events if e.row is not None for m in e.mechanisms]
    if mech_rows:
        mech = coerce_mechanisms(
            pd.DataFrame(mech_rows).set_index("time"), source=SOURCE
        )
    else:
        mech = empty_mechanisms()
    return ev_frame, origins, mags, arrivals, mech


# --------------------------------------------------------------------------
# Public readers
# --------------------------------------------------------------------------


def _finish_events(
    df: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp, tally: Counter[str]
) -> pd.DataFrame:
    dup = df["event_id"].duplicated(keep="first").to_numpy()
    if dup.any():
        tally["duplicates_dropped"] += int(dup.sum())
        df = df.loc[~dup]
    inside = np.asarray((df.index >= lo) & (df.index < hi), dtype=bool)
    if not inside.all():
        tally["outside_window"] += int((~inside).sum())
        df = df.loc[inside]
    is_aut = (df["evaluation_mode"] == "automatic").to_numpy()
    tally["events"] = len(df)
    tally["events_automatic"] = int(is_aut.sum())
    tally["events_manual"] = int((~is_aut).sum())
    return df


def _read(
    start: TimeLike | None,
    end: TimeLike | None,
    *,
    event_id: str | None,
    format: Format,
    base_url: str | None,
    bbox: tuple[float, float, float, float] | None,
    radius: tuple[float, ...] | None,
    min_depth_km: float | None,
    max_depth_km: float | None,
    min_magnitude: float | None,
    max_magnitude: float | None,
    magnitude_type: str | None,
    include_all_origins: bool,
    include_all_magnitudes: bool,
    include_arrivals: bool,
    include_focal_mechanisms: bool,
    text_evaluation_mode: Literal["author"] | None,
    policy: FdsnPolicy,
    session: Session | None,
    extras: bool,
) -> FdsnCatalog:
    t0 = _time.perf_counter()
    if format not in ("xml", "text"):
        raise ValueError(f"format {format!r}: 'xml' or 'text'")
    xml_only = (
        include_all_origins
        or include_all_magnitudes
        or include_arrivals
        or include_focal_mechanisms
    )
    if format == "text":
        if xml_only:
            raise ValueError("include_* options need format='xml'")
        if text_evaluation_mode is None:
            raise ValueError(
                "FDSN text has no evaluation mode or status, but the schema "
                "needs evaluation_mode. Use format='xml' (exact), or pass "
                "text_evaluation_mode='author' to infer it from Author "
                "(machine author → automatic, else manual; wrong for 5 of 48 "
                "events on 2026-09-10)."
            )
    filters: list[tuple[str, str]] = [
        *_region_params(bbox, radius),
        *_limit_params("depth", min_depth_km, max_depth_km),
        *_limit_params("magnitude", min_magnitude, max_magnitude),
    ]
    if magnitude_type is not None:
        filters.append(("magnitudetype", magnitude_type))
    if format == "text":
        filters.append(("format", "text"))
    for flag, name in (
        (include_all_origins, "includeallorigins"),
        (include_all_magnitudes, "includeallmagnitudes"),
        (include_arrivals, "includearrivals"),
        (include_arrivals, "includepicks"),
        (include_focal_mechanisms, "includefocalmechanism"),
    ):
        if flag:
            filters.append((name, "true"))
    accept = "text/plain" if format == "text" else "application/xml"

    base = resolve_base(base_url)
    tally: Counter[str] = Counter({k: 0 for k in _REPORT_KEYS})
    client = _Client(session or UrllibSession(), base, policy, tally)

    if event_id is not None:
        lo = hi = None
        body = client.get([("eventid", event_id), *filters], accept)
        tally["pages"] += 1
        if format == "text":
            raw = parse_text(body)
            tally["events_parsed"] += len(raw)
            qml: list[_QmlEvent] = []
        else:
            qml, _n = parse_quakeml(body, extras=extras, tally=tally)
        if (len(raw) if format == "text" else _n) == 0:
            raise FdsnError(
                f"event {event_id!r} not found (HTTP 204 / no <event>)",
                status=204,
            )
    else:
        assert start is not None and end is not None
        lo, hi = _utc(start, "start"), _utc(end, "end")
        if hi < lo:
            raise ValueError(f"end {hi} is before start {lo}")
        req_lo, req_hi = lo.floor("us"), hi.ceil("us")
        if format == "text":

            def parse_t(b: bytes) -> tuple[list[list[str]], int]:
                r = parse_text(b)
                tally["events_parsed"] += len(r)
                return r.to_numpy().tolist(), len(r)

            def key_t(r: list[str]) -> tuple[str, pd.Timestamp | None]:
                return r[0], _ts(r[1])

            rows = (
                _page_through(client, req_lo, req_hi, filters, accept, parse_t, key_t)
                if req_lo < req_hi
                else []
            )
            raw = pd.DataFrame(rows, columns=list(TEXT_COLUMNS))
            qml = []
        else:

            def parse_x(b: bytes) -> tuple[list[_QmlEvent], int]:
                return parse_quakeml(b, extras=extras, tally=tally)

            def key_x(e: _QmlEvent) -> tuple[str, pd.Timestamp | None]:
                return e.event_id, e.time

            qml = (
                _page_through(client, req_lo, req_hi, filters, accept, parse_x, key_x)
                if req_lo < req_hi
                else []
            )

    if format == "text":
        assert text_evaluation_mode is not None
        events = text_to_frame(
            raw, evaluation_mode=text_evaluation_mode, extras=extras, tally=tally
        )
        empty = pd.DataFrame()
        origins = mags = arrivals = empty
        mech = empty_mechanisms()
    else:
        events, origins, mags, arrivals, mech = quakeml_to_catalog(
            qml, extras=extras, tally=tally
        )
    if lo is not None and hi is not None:
        events = _finish_events(events, lo, hi, tally)
        keep = set(events["event_id"])
        if format == "xml":
            origins = origins[origins["event_id"].isin(keep)]
            mags = mags[mags["event_id"].isin(keep)]
            arrivals = arrivals[arrivals["event_id"].isin(keep)]
            mech = mech[mech["event_id"].isin(keep)]
    else:
        far = pd.Timestamp.max.tz_localize("UTC")
        events = _finish_events(events, pd.Timestamp.min.tz_localize("UTC"), far, tally)

    report = FdsnReadReport(
        endpoint=f"{base}/query",
        counts=dict(tally),
        elapsed_s=_time.perf_counter() - t0,
    )
    validate(events, strict=True)
    if len(mech):
        validate_mechanisms(mech, strict=True)
    notes = []
    if format == "text":
        notes.append(
            "evaluation_mode inferred from Author (machine author → automatic, "
            "else manual); not exact; evaluation_status is not in the text format"
        )
    else:
        notes.append(
            "h_err_km/depth_err_km only for SIL-model origins (1σ, degrees "
            "converted with cos(lat)); LOCSAT errors (km, 90 %) are NA here and "
            "raw in the origins frame"
        )
    prov = catalog_provenance(
        source=SOURCE,
        location=f"{base}/query",
        start=lo.to_pydatetime() if lo is not None else None,
        end=hi.to_pydatetime() if hi is not None else None,
        filters={
            "format": format,
            "eventid": event_id,
            "bbox": list(bbox) if bbox is not None else None,
            "radius": list(radius) if radius is not None else None,
            "depth_km": [min_depth_km, max_depth_km],
            "magnitude": [min_magnitude, max_magnitude],
            "magnitude_type": magnitude_type,
            "include_all_origins": include_all_origins,
            "include_all_magnitudes": include_all_magnitudes,
            "include_arrivals": include_arrivals,
            "include_focal_mechanisms": include_focal_mechanisms,
            "text_evaluation_mode": text_evaluation_mode,
            "page_size": policy.page_size,
        },
        extra={"report": report.as_dict(), "notes": notes},
    )
    with_provenance(events, prov)
    if len(mech):
        with_provenance(mech, {**prov, "frame": "mechanisms"})
    return FdsnCatalog(events, origins, mags, arrivals, mech, report)


def read_fdsn(
    start: TimeLike,
    end: TimeLike,
    *,
    format: Format = "xml",
    base_url: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    radius: tuple[float, ...] | None = None,
    min_depth_km: float | None = None,
    max_depth_km: float | None = None,
    min_magnitude: float | None = None,
    max_magnitude: float | None = None,
    magnitude_type: str | None = None,
    include_all_origins: bool = False,
    include_all_magnitudes: bool = False,
    text_evaluation_mode: Literal["author"] | None = None,
    policy: FdsnPolicy = DEFAULT_POLICY,
    session: Session | None = None,
    extras: bool = False,
) -> pd.DataFrame:
    """Read the FDSN event service for ``[start, end)`` into a schema events frame.

    ``start``/``end``: times without a zone are UTC. ``format``: ``xml``
    (QuakeML, default) or ``text`` (needs ``text_evaluation_mode="author"``,
    see the module docstring). Region: ``bbox`` ``(min_lon, min_lat, max_lon,
    max_lat)`` or ``radius`` ``(lat, lon, maxradius_deg[, minradius_deg])``.
    Depth (km) and magnitude limits and ``magnitude_type`` are sent only when
    given (the service has no default bounds, unlike the Quakes API).
    ``include_all_origins`` / ``include_all_magnitudes`` (xml): ask for every
    solution; the frame still holds the preferred one, plus ``mag_mlw`` /
    ``mag_ml`` / ``mag_auto`` from all magnitudes. Use
    :func:`read_fdsn_catalog` for the detail frames.

    Raises :class:`FdsnUnreachable`, :class:`FdsnError`, :class:`ValueError`
    on bad arguments, and :class:`~geo_dataread.seismic.schema.SchemaError`
    when the response does not fit the schema.
    """
    return _read(
        start,
        end,
        event_id=None,
        format=format,
        base_url=base_url,
        bbox=bbox,
        radius=radius,
        min_depth_km=min_depth_km,
        max_depth_km=max_depth_km,
        min_magnitude=min_magnitude,
        max_magnitude=max_magnitude,
        magnitude_type=magnitude_type,
        include_all_origins=include_all_origins,
        include_all_magnitudes=include_all_magnitudes,
        include_arrivals=False,
        include_focal_mechanisms=False,
        text_evaluation_mode=text_evaluation_mode,
        policy=policy,
        session=session,
        extras=extras,
    ).events


def read_fdsn_catalog(
    start: TimeLike,
    end: TimeLike,
    *,
    base_url: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    radius: tuple[float, ...] | None = None,
    min_depth_km: float | None = None,
    max_depth_km: float | None = None,
    min_magnitude: float | None = None,
    max_magnitude: float | None = None,
    magnitude_type: str | None = None,
    include_all_origins: bool = True,
    include_all_magnitudes: bool = True,
    include_arrivals: bool = False,
    include_focal_mechanisms: bool = False,
    policy: FdsnPolicy = DEFAULT_POLICY,
    session: Session | None = None,
    extras: bool = False,
) -> FdsnCatalog:
    """QuakeML read of ``[start, end)`` with the detail frames
    (:class:`FdsnCatalog`); every origin and magnitude by default."""
    return _read(
        start,
        end,
        event_id=None,
        format="xml",
        base_url=base_url,
        bbox=bbox,
        radius=radius,
        min_depth_km=min_depth_km,
        max_depth_km=max_depth_km,
        min_magnitude=min_magnitude,
        max_magnitude=max_magnitude,
        magnitude_type=magnitude_type,
        include_all_origins=include_all_origins,
        include_all_magnitudes=include_all_magnitudes,
        include_arrivals=include_arrivals,
        include_focal_mechanisms=include_focal_mechanisms,
        text_evaluation_mode=None,
        policy=policy,
        session=session,
        extras=extras,
    )


def read_fdsn_event(
    event_id: str,
    *,
    base_url: str | None = None,
    include_all_origins: bool = True,
    include_all_magnitudes: bool = True,
    include_arrivals: bool = False,
    include_focal_mechanisms: bool = False,
    policy: FdsnPolicy = DEFAULT_POLICY,
    session: Session | None = None,
    extras: bool = False,
) -> FdsnCatalog:
    """One event by id (``IMO…``), QuakeML, with the detail frames."""
    if not event_id or not event_id.strip():
        raise ValueError("event_id is empty")
    return _read(
        None,
        None,
        event_id=event_id.strip(),
        format="xml",
        base_url=base_url,
        bbox=None,
        radius=None,
        min_depth_km=None,
        max_depth_km=None,
        min_magnitude=None,
        max_magnitude=None,
        magnitude_type=None,
        include_all_origins=include_all_origins,
        include_all_magnitudes=include_all_magnitudes,
        include_arrivals=include_arrivals,
        include_focal_mechanisms=include_focal_mechanisms,
        text_evaluation_mode=None,
        policy=policy,
        session=session,
        extras=extras,
    )


# --------------------------------------------------------------------------
# Cross-check with the Quakes API
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class QuakesCrossCheck:
    """Result of :func:`compare_with_quakes`.

    ``only_fdsn`` / ``only_quakes``: the unmatched events, with a ``reason``
    column on ``only_fdsn`` (``depth outside Quakes bounds``, ``no
    magnitude``, ``magnitude outside Quakes bounds``, ``unexplained``).
    ``diffs``: one row per (event, field) that differs beyond its tolerance
    (columns ``event_id, field, fdsn, quakes, delta``). ``fields``: per field
    ``compared``, ``differ``, ``max_abs`` (numeric) and ``pairs`` (exact
    fields: ``"fdsn → quakes": n``)."""

    n_fdsn: int
    n_quakes: int
    n_both: int
    only_fdsn: pd.DataFrame
    only_quakes: pd.DataFrame
    diffs: pd.DataFrame
    fields: Mapping[str, Mapping[str, Any]]
    tolerances: Mapping[str, float]

    @property
    def ok(self) -> bool:
        """No unmatched event and no field difference."""
        return not (len(self.only_fdsn) or len(self.only_quakes) or len(self.diffs))

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_fdsn": self.n_fdsn,
            "n_quakes": self.n_quakes,
            "n_both": self.n_both,
            "only_fdsn": {
                "n": len(self.only_fdsn),
                "by_reason": {
                    str(k): int(v)
                    for k, v in Counter(self.only_fdsn.get("reason", [])).items()
                },
                "event_ids": [str(v) for v in self.only_fdsn["event_id"]],
            },
            "only_quakes": {
                "n": len(self.only_quakes),
                "event_ids": [str(v) for v in self.only_quakes["event_id"]],
            },
            "fields": {k: dict(v) for k, v in self.fields.items()},
            "tolerances": dict(self.tolerances),
            "ok": self.ok,
        }


def _unmatched_reason(row: Mapping[str, Any]) -> str:
    d, m = row.get("depth_km"), row.get("magnitude")
    if (
        d is not None
        and not pd.isna(d)
        and not (DEPTH_BOUNDS[0] <= d <= DEPTH_BOUNDS[1])
    ):
        return "depth outside Quakes bounds"
    if m is None or pd.isna(m):
        return "no magnitude"
    if not (SIZE_BOUNDS[0] <= m <= SIZE_BOUNDS[1]):
        return "magnitude outside Quakes bounds"
    return "unexplained"


def _same(a: Any, b: Any) -> bool:
    na_a, na_b = pd.isna(a), pd.isna(b)
    if na_a or na_b:
        return bool(na_a and na_b)
    return bool(str(a) == str(b))


def compare_with_quakes(
    fdsn: pd.DataFrame,
    quakes: pd.DataFrame,
    *,
    tolerances: Mapping[str, float] | None = None,
) -> QuakesCrossCheck:
    """Compare two events frames of the same SeisComP window by ``event_id``.

    Numeric fields (``time`` in seconds, ``latitude``, ``longitude``,
    ``depth_km``, ``magnitude``) differ when ``|Δ|`` exceeds the tolerance
    (:data:`DEFAULT_TOLERANCES`, overridable per key); NaN on one side only is
    a difference. :data:`EXACT_FIELDS` are compared as values (NA = NA).
    ``updated_time`` is not compared (FDSN does not carry it)."""
    tol = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    f = fdsn.reset_index()
    q = quakes.reset_index()
    for name, df in (("fdsn", f), ("quakes", q)):
        if df["event_id"].duplicated().any():
            raise ValueError(f"{name} frame has duplicate event_id values")
    fi, qi = f.set_index("event_id"), q.set_index("event_id")
    both = fi.index.intersection(qi.index)
    only_f = f[~f["event_id"].isin(both)].copy()
    only_f["reason"] = [_unmatched_reason(r) for r in only_f.to_dict("records")]
    only_q = q[~q["event_id"].isin(both)].copy()

    diffs: list[dict[str, Any]] = []
    fields: dict[str, dict[str, Any]] = {}
    fb, qb = fi.loc[both], qi.loc[both]
    numeric = {
        "time": ("time_s", None),
        "latitude": ("latitude", None),
        "longitude": ("longitude", None),
        "depth_km": ("depth_km", None),
        "magnitude": ("magnitude", None),
    }
    for col, (tkey, _) in numeric.items():
        if col == "time":
            a = (fb[col] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
            b = (qb[col] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
        else:
            a = fb[col].astype("float64")
            b = qb[col].astype("float64")
        a_np, b_np = a.to_numpy(dtype="float64"), b.to_numpy(dtype="float64")
        delta = a_np - b_np
        one_nan = np.isnan(a_np) ^ np.isnan(b_np)
        with np.errstate(invalid="ignore"):
            bad = one_nan | (np.abs(delta) > tol[tkey])
        finite = np.abs(delta[np.isfinite(delta)])
        fields[col] = {
            "compared": len(both),
            "differ": int(bad.sum()),
            "max_abs": float(finite.max()) if finite.size else None,
            "tolerance": tol[tkey],
        }
        for i in np.flatnonzero(bad):
            diffs.append(
                {
                    "event_id": str(both[i]),
                    "field": col,
                    "fdsn": fb[col].iloc[i],
                    "quakes": qb[col].iloc[i],
                    "delta": float(delta[i]),
                }
            )
    for col in EXACT_FIELDS:
        pairs: Counter[str] = Counter()
        n_bad = 0
        for eid, a_v, b_v in zip(both, fb[col], qb[col], strict=True):
            if _same(a_v, b_v):
                continue
            n_bad += 1
            pairs[
                f"{'NA' if pd.isna(a_v) else a_v} → {'NA' if pd.isna(b_v) else b_v}"
            ] += 1
            diffs.append(
                {
                    "event_id": str(eid),
                    "field": col,
                    "fdsn": a_v,
                    "quakes": b_v,
                    "delta": math.nan,
                }
            )
        fields[col] = {"compared": len(both), "differ": n_bad, "pairs": dict(pairs)}
    diff_df = pd.DataFrame(
        diffs, columns=["event_id", "field", "fdsn", "quakes", "delta"]
    )
    return QuakesCrossCheck(
        n_fdsn=len(f),
        n_quakes=len(q),
        n_both=len(both),
        only_fdsn=only_f.reset_index(drop=True),
        only_quakes=only_q.reset_index(drop=True),
        diffs=diff_df,
        fields=fields,
        tolerances=tol,
    )


def cross_check_quakes(
    start: TimeLike,
    end: TimeLike,
    *,
    fdsn_session: Session | None = None,
    quakes_session: Session | None = None,
    base_url: str | None = None,
    policy: FdsnPolicy = DEFAULT_POLICY,
    tolerances: Mapping[str, float] | None = None,
) -> QuakesCrossCheck:
    """Read ``[start, end)`` from FDSN (QuakeML, no limits) and from the
    internal Quakes API (``system="seiscomp"``, its full depth and size
    bounds), back to back, and :func:`compare_with_quakes`. FDSN-only events
    outside the Quakes API bounds are labelled as such, so a short count on
    the Quakes side is split into "cannot be fetched there" and "unexplained".
    """
    fd = read_fdsn(start, end, base_url=base_url, policy=policy, session=fdsn_session)
    qk = read_quakes(
        start,
        end,
        system="seiscomp",
        internal=True,
        depth_min=DEPTH_BOUNDS[0],
        depth_max=DEPTH_BOUNDS[1],
        size_min=SIZE_BOUNDS[0],
        size_max=SIZE_BOUNDS[1],
        session=quakes_session,
    )
    return compare_with_quakes(fd, qk, tolerances=tolerances)


__all__ = [
    "DEFAULT_BASE",
    "DEFAULT_POLICY",
    "DEFAULT_TOLERANCES",
    "ENV_BASE",
    "EXACT_FIELDS",
    "QML_NS",
    "SCALE_COLUMNS",
    "SOURCE",
    "TEXT_COLUMNS",
    "USER_AGENT",
    "FdsnCatalog",
    "FdsnError",
    "FdsnPolicy",
    "FdsnReadReport",
    "FdsnUnreachable",
    "QuakesCrossCheck",
    "compare_with_quakes",
    "cross_check_quakes",
    "parse_quakeml",
    "parse_text",
    "quakeml_to_catalog",
    "read_fdsn",
    "read_fdsn_catalog",
    "read_fdsn_event",
    "resolve_base",
    "text_to_frame",
]
