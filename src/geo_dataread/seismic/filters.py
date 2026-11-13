"""Catalogue filters and derived values for schema events frames (T5).

Replaces the legacy ``sil_read.seism_subset`` (``PLAN-seismicity.md`` §2.4):

* every bound is applied on its own (the legacy ``max_lon`` filter was guarded
  by ``min_lon``);
* every bound is **inclusive** (``>=`` / ``<=``), on time too; the legacy
  bounds were exclusive;
* no blanket ``dropna``: a NaN in a filtered column fails that filter only,
  and a NaN in a column nobody filters on never removes a row;
* times are UTC: a naive ``start``/``end`` is taken as UTC, never local time;
* an empty frame in gives an empty frame out, never an exception.

Every function takes and returns a frame in the common schema
(:mod:`geo_dataread.seismic.schema`). Filters append what they applied to
``df.attrs["catalog"]["subsets"]`` (one record per call, in call order), so
the provenance of a cut catalogue says how it was cut.

Derived values are functions, not stored columns (§4): :func:`daily_counts`,
:func:`moment_from_magnitude` (the inverse of
:func:`~geo_dataread.seismic.schema.moment_magnitude`) and
:func:`with_derived_moment`, whose result feeds
:func:`~geo_dataread.seismic.schema.cumulative_moment` unchanged.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from typing import Any, Final, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

from geo_dataread.seismic.schema import (
    ATTRS_KEY,
    HK79_CONSTANT_NM,
    INDEX_NAME,
    FloatArray,
    column_spec,
    validate,
)

__all__ = [
    "POLYGON_EDGE_TOL_DEG",
    "DERIVED_MOMENT_FLAG",
    "FilterError",
    "TimeLike",
    "daily_counts",
    "distance_km",
    "moment_from_magnitude",
    "points_in_polygon",
    "to_utc",
    "subset",
    "with_derived_moment",
    "within_distance",
]

#: Anything :func:`subset` accepts as a time bound.
TimeLike: TypeAlias = str | datetime | date | pd.Timestamp | np.datetime64

#: Absolute tolerance (degrees) within which a point counts as *on* a polygon
#: edge, and so inside (1e-9° is about 0.1 mm).
POLYGON_EDGE_TOL_DEG: Final = 1e-9

#: Key under ``df.attrs["catalog"]`` holding the list of applied filters.
SUBSETS_KEY: Final = "subsets"


class FilterError(ValueError):
    """A filter cannot be applied: missing column, bad index or bad argument."""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def to_utc(value: TimeLike, *, what: str = "time") -> pd.Timestamp:
    """Parse ``value`` to a UTC :class:`pandas.Timestamp`.

    A naive value (no time zone, e.g. ``"2024-08-22 12:00"``) is taken as
    **UTC**, never local time; an aware value (``"…+01:00"``, a ``datetime``
    with ``tzinfo``) is converted to UTC. A ``date`` is its midnight UTC.
    """
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise FilterError(f"{what}: cannot parse {value!r} as a time") from exc
    if ts is pd.NaT:
        raise FilterError(f"{what}: {value!r} is not a time (NaT)")
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _iso(ts: pd.Timestamp) -> str:
    return str(ts.isoformat().replace("+00:00", "Z"))


def _require(
    df: pd.DataFrame, columns: Iterable[str], *, index: bool, what: str
) -> None:
    """Raise :class:`FilterError` if a column (or the time index) is unusable.

    Uses :func:`~geo_dataread.seismic.schema.validate` (non-strict) and only
    acts on the issues that touch what this call needs: an unrelated missing
    column or a range warning elsewhere does not stop a filter.
    """
    needed = set(columns)
    report = validate(df)
    missing = sorted(
        i.column
        for i in report.issues
        if i.kind == "missing_column" and i.column in needed
    )
    # A column outside the schema (e.g. a by= column) is not reported by
    # validate; check it directly.
    missing += sorted(c for c in needed if c not in df.columns and c not in missing)
    if missing:
        raise FilterError(
            f"{what}: column(s) {missing} absent; build the frame with "
            "schema.coerce_catalog"
        )
    if index:
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise FilterError(
                f"{what}: needs a DatetimeIndex named {INDEX_NAME!r}, got "
                f"{type(idx).__name__}"
            )
        if idx.tz is None:
            raise FilterError(
                f"{what}: the frame's time index is tz-naive; coerce it to UTC "
                "with schema.coerce_catalog (assume_utc only for UTC sources)"
            )


def _check_allowed(values: Iterable[str], column: str) -> list[str]:
    """Return ``values`` as a list; raise on a value outside the schema set."""
    vals = [values] if isinstance(values, str) else list(values)
    allowed = column_spec(column).allowed
    bad = sorted(set(vals) - set(allowed))
    if bad:
        raise FilterError(f"{column}: unknown value(s) {bad}; allowed {allowed}")
    return vals


def _num(df: pd.DataFrame, column: str) -> FloatArray:
    out: FloatArray = (
        pd.to_numeric(df[column], errors="coerce").astype("float64").to_numpy()
    )
    return out


def _bounds(
    mask: npt.NDArray[np.bool_],
    values: FloatArray,
    lo: float | None,
    hi: float | None,
    name: str,
) -> None:
    """``mask &= lo <= values <= hi`` in place, each bound independently."""
    if lo is not None and hi is not None and lo > hi:
        raise FilterError(f"{name}: minimum {lo} is greater than maximum {hi}")
    with np.errstate(invalid="ignore"):
        if lo is not None:
            mask &= values >= lo  # NaN compares False: fails the filter
        if hi is not None:
            mask &= values <= hi


def _append_subset(
    src: pd.DataFrame, out: pd.DataFrame, record: Mapping[str, Any]
) -> pd.DataFrame:
    """Give ``out`` a deep copy of ``src.attrs`` with ``record`` appended."""
    attrs = copy.deepcopy(dict(src.attrs))
    catalog = dict(attrs.get(ATTRS_KEY) or {})
    subsets = list(catalog.get(SUBSETS_KEY) or [])
    subsets.append({**record, "n_in": len(src), "n_out": len(out)})
    catalog[SUBSETS_KEY] = subsets
    attrs[ATTRS_KEY] = catalog
    out.attrs = attrs
    return out


# --------------------------------------------------------------------------
# Polygon and distance
# --------------------------------------------------------------------------


def _polygon_array(polygon: Sequence[Sequence[float]] | npt.ArrayLike) -> FloatArray:
    poly = np.asarray(polygon, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise FilterError(f"polygon must be (lon, lat) pairs, got shape {poly.shape}")
    if len(poly) > 1 and np.array_equal(poly[0], poly[-1]):
        poly = poly[:-1]  # closed ring: drop the repeated vertex
    if len(poly) < 3:
        raise FilterError("polygon needs at least 3 distinct vertices")
    if not np.isfinite(poly).all():
        raise FilterError("polygon vertices must be finite")
    return poly


def points_in_polygon(
    lon: npt.ArrayLike,
    lat: npt.ArrayLike,
    polygon: Sequence[Sequence[float]] | npt.ArrayLike,
    *,
    edge_tol_deg: float = POLYGON_EDGE_TOL_DEG,
) -> npt.NDArray[np.bool_]:
    """Ray-casting point-in-polygon test on plain lon/lat degrees.

    ``polygon`` is a sequence of ``(lon, lat)`` vertices (GeoJSON order), open
    or closed (a repeated first vertex is dropped); one outer ring, no holes;
    concave and self-touching rings are fine. Returns a boolean array, one
    entry per point.

    Rules:

    * A point on an edge or a vertex (within ``edge_tol_deg`` degrees) is
      **inside**, matching the inclusive bounds of :func:`subset`.
    * Elsewhere the even-odd rule with half-open edge crossings, so a ray
      through a vertex is counted once.
    * A NaN coordinate is outside.

    Limits: edges are straight lines in the lon/lat plane, not great circles.
    That is harmless for Iceland-sized polygons with vertices a few tens of km
    apart, but a long edge at high latitude differs from the geodesic one, a
    polygon around a pole cannot be expressed, and one crossing the
    antimeridian (±180°) must be given in a continuous longitude range (e.g.
    170…190) with the points shifted to match.
    """
    x = np.asarray(lon, dtype=np.float64).ravel()
    y = np.asarray(lat, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise FilterError(f"lon and lat differ in length: {x.size} vs {y.size}")
    poly = _polygon_array(polygon)
    inside = np.zeros(x.size, dtype=bool)
    on_edge = np.zeros(x.size, dtype=bool)
    if x.size == 0:
        return inside
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        for ax, ay, bx, by in zip(x1, y1, x2, y2, strict=True):
            dx, dy = bx - ax, by - ay
            # On the segment: collinear within tolerance and between its ends.
            length = float(np.hypot(dx, dy))
            cross = (x - ax) * dy - (y - ay) * dx
            dot = (x - ax) * dx + (y - ay) * dy
            near = np.abs(cross) <= edge_tol_deg * length
            between = (dot >= -edge_tol_deg * length) & (
                dot <= length * length + edge_tol_deg * length
            )
            on_edge |= near & between
            # Half-open crossing rule: an edge counts when exactly one end is
            # strictly above the point's latitude.
            straddle = (ay > y) != (by > y)
            if dy != 0.0:
                x_cross = ax + (y - ay) * dx / dy
                inside ^= straddle & (x < x_cross)
    result: npt.NDArray[np.bool_] = (inside | on_edge) & np.isfinite(x) & np.isfinite(y)
    return result


def distance_km(
    df: pd.DataFrame, lat: float, lon: float, *, ellps: str = "GRS80"
) -> pd.Series:
    """Geodesic epicentral distance (km) from ``(lat, lon)`` to every event.

    Uses :func:`geofunc.frames.geodesic` (``pyproj.Geod`` on ``ellps``,
    Karney's algorithm, accurate to well under a millimetre), not a spherical
    haversine. Returns a float Series on the frame's index named
    ``distance_km``; NaN where an epicentre is NaN.
    """
    _require(df, ("latitude", "longitude"), index=False, what="distance_km")
    if not (np.isfinite(lat) and np.isfinite(lon)) or abs(lat) > 90.0:
        raise FilterError(f"distance_km: bad reference point ({lat}, {lon})")
    out = np.full(len(df), np.nan, dtype=np.float64)
    elat, elon = _num(df, "latitude"), _num(df, "longitude")
    ok = np.isfinite(elat) & np.isfinite(elon)
    if ok.any():
        from geofunc.frames import geodesic

        n = int(ok.sum())
        _, _, dist_m = geodesic(ellps).inv(
            np.full(n, float(lon)), np.full(n, float(lat)), elon[ok], elat[ok]
        )
        out[ok] = np.asarray(dist_m, dtype=np.float64) / 1000.0
    return pd.Series(out, index=df.index, name="distance_km")


def within_distance(
    df: pd.DataFrame,
    lat: float,
    lon: float,
    max_km: float,
    *,
    min_km: float | None = None,
    ellps: str = "GRS80",
) -> pd.DataFrame:
    """Events whose epicentre lies within ``max_km`` of ``(lat, lon)``.

    Note the argument order: **lat, lon** (as the schema columns read), while
    polygons are ``(lon, lat)`` pairs. Inclusive: an event at exactly
    ``max_km`` (or ``min_km``, for a ring) is kept. Epicentral (surface)
    distance on the ellipsoid via :func:`distance_km`; depth is ignored. A NaN
    epicentre fails the filter. The call is appended to
    ``attrs["catalog"]["subsets"]``.
    """
    if not np.isfinite(max_km) or max_km < 0:
        raise FilterError(f"within_distance: max_km must be ≥ 0, got {max_km}")
    record: dict[str, Any] = {
        "filter": "within_distance",
        "lat": float(lat),
        "lon": float(lon),
        "max_km": float(max_km),
        "min_km": None if min_km is None else float(min_km),
        "ellps": ellps,
    }
    if len(df) == 0:
        _require(df, ("latitude", "longitude"), index=False, what="within_distance")
        return _append_subset(df, df.copy(), record)
    dist = distance_km(df, lat, lon, ellps=ellps).to_numpy()
    mask = np.ones(len(df), dtype=bool)
    _bounds(mask, dist, min_km, float(max_km), "distance_km")
    return _append_subset(df, df[mask].copy(), record)


# --------------------------------------------------------------------------
# subset
# --------------------------------------------------------------------------


def subset(
    df: pd.DataFrame,
    *,
    start: TimeLike | None = None,
    end: TimeLike | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    min_lat: float | None = None,
    max_lat: float | None = None,
    min_lon: float | None = None,
    max_lon: float | None = None,
    polygon: Sequence[Sequence[float]] | npt.ArrayLike | None = None,
    min_depth: float | None = None,
    max_depth: float | None = None,
    min_mag: float | None = None,
    max_mag: float | None = None,
    magnitude_types: Iterable[str] | None = None,
    evaluation_modes: Iterable[str] | None = None,
    event_types: Iterable[str] | None = None,
    min_quality: float | None = None,
    max_quality: float | None = None,
    reviewed_only: bool = False,
) -> pd.DataFrame:
    """Cut an events frame by time, region, depth, magnitude, type and quality.

    All filters are combined with AND; each is applied on its own, so any
    single bound works without its partner. A filter left at ``None`` (or
    ``False``) is not applied. Returns a new frame (the input and its
    ``attrs`` are not changed) with the rows in their original order.

    **Bounds are inclusive**: ``start <= time <= end``,
    ``min_lat <= latitude <= max_lat`` and so on for longitude, ``depth_km``,
    ``magnitude`` and ``quality``. A minimum greater than its maximum raises
    :class:`FilterError` (a box across the antimeridian is not supported).

    **Time**: ``start``/``end`` accept a string, ``datetime``, ``date`` or
    ``Timestamp``. Naive values are UTC, never local time; aware values are
    converted to UTC (:func:`to_utc`). ``end`` is inclusive *at that
    instant*: ``end="2024-08-22"`` means 2024-08-22T00:00Z and so excludes
    most of that day; pass ``"2024-08-22 23:59:59.999"`` or the next day to
    include it. The frame's index must be tz-aware (the schema's UTC index).

    **Region**: ``bbox`` is ``(min_lon, min_lat, max_lon, max_lat)`` (GeoJSON
    order); it cannot be combined with the separate ``min_/max_ lat/lon``
    arguments. ``polygon`` is ``(lon, lat)`` vertices, tested with
    :func:`points_in_polygon` (a point on the edge is inside). For a radius
    use :func:`within_distance`.

    **Magnitude**: ``min_mag``/``max_mag`` act on the ``magnitude`` column,
    which mixes scales (Q7); combine with ``magnitude_types`` (e.g.
    ``["Mlw"]``) to stay on one scale. ``magnitude_types``,
    ``evaluation_modes`` and ``event_types`` keep rows whose value is in the
    given set; a value outside the schema's allowed set raises (a typo must
    not silently return nothing).

    **reviewed_only** keeps ``evaluation_mode == "manual"`` rows whose
    ``evaluation_status`` is not ``"rejected"`` (an NA status counts as not
    rejected; SIL rows have none).

    **NaN** never removes a row through a column that is not filtered (no
    blanket ``dropna``). A NaN (or NA category) in a filtered column fails
    that filter: an event without a depth is dropped by ``min_depth`` but not
    by a magnitude filter. In particular ``quality`` is NaN on reviewed SIL
    events, so ``min_quality``/``max_quality`` drop every reviewed event;
    apply them to automatic rows only if that is not wanted.

    The columns a requested filter needs are checked with
    :func:`~geo_dataread.seismic.schema.validate`; an absent one raises
    :class:`FilterError`. The applied filters (times as ISO UTC strings) are
    appended to ``attrs["catalog"]["subsets"]``.
    """
    if bbox is not None:
        if any(v is not None for v in (min_lat, max_lat, min_lon, max_lon)):
            raise FilterError("give either bbox or min_/max_ lat/lon, not both")
        if len(bbox) != 4:
            raise FilterError("bbox is (min_lon, min_lat, max_lon, max_lat)")
        min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox)

    t0 = None if start is None else to_utc(start, what="start")
    t1 = None if end is None else to_utc(end, what="end")
    if t0 is not None and t1 is not None and t0 > t1:
        raise FilterError(f"start {_iso(t0)} is after end {_iso(t1)}")
    mag_types = (
        None
        if magnitude_types is None
        else _check_allowed(magnitude_types, "magnitude_type")
    )
    modes = (
        None
        if evaluation_modes is None
        else _check_allowed(evaluation_modes, "evaluation_mode")
    )
    etypes = None if event_types is None else _check_allowed(event_types, "event_type")
    poly = None if polygon is None else _polygon_array(polygon)

    numeric: list[tuple[str, float | None, float | None]] = [
        ("latitude", min_lat, max_lat),
        ("longitude", min_lon, max_lon),
        ("depth_km", min_depth, max_depth),
        ("magnitude", min_mag, max_mag),
        ("quality", min_quality, max_quality),
    ]
    needed = [c for c, lo, hi in numeric if lo is not None or hi is not None]
    if poly is not None:
        needed += ["latitude", "longitude"]
    if mag_types is not None:
        needed.append("magnitude_type")
    if modes is not None or reviewed_only:
        needed.append("evaluation_mode")
    if reviewed_only:
        needed.append("evaluation_status")
    if etypes is not None:
        needed.append("event_type")
    timed = t0 is not None or t1 is not None
    _require(df, needed, index=timed, what="subset")

    record: dict[str, Any] = {"filter": "subset"}
    if t0 is not None:
        record["start"] = _iso(t0)
    if t1 is not None:
        record["end"] = _iso(t1)
    for col, lo, hi in numeric:
        if lo is not None:
            record[f"min_{col}"] = float(lo)
        if hi is not None:
            record[f"max_{col}"] = float(hi)
    if poly is not None:
        record["polygon"] = [[float(a), float(b)] for a, b in poly]
    if mag_types is not None:
        record["magnitude_types"] = mag_types
    if modes is not None:
        record["evaluation_modes"] = modes
    if etypes is not None:
        record["event_types"] = etypes
    if reviewed_only:
        record["reviewed_only"] = True

    for name, lo, hi in numeric:
        if lo is not None and hi is not None and lo > hi:
            raise FilterError(f"{name}: minimum {lo} is greater than maximum {hi}")
    if len(df) == 0:
        return _append_subset(df, df.copy(), record)

    mask = np.ones(len(df), dtype=bool)
    if timed:
        i8 = df.index.tz_convert("UTC")
        if t0 is not None:
            mask &= np.asarray(i8 >= t0, dtype=bool)
        if t1 is not None:
            mask &= np.asarray(i8 <= t1, dtype=bool)
    for col, lo, hi in numeric:
        if lo is not None or hi is not None:
            _bounds(mask, _num(df, col), lo, hi, col)
    if poly is not None:
        mask &= points_in_polygon(_num(df, "longitude"), _num(df, "latitude"), poly)

    def _isin(column: str, values: list[str]) -> npt.NDArray[np.bool_]:
        col = df[column].astype(object)
        return np.asarray(col.isin(values) & col.notna(), dtype=bool)

    if mag_types is not None:
        mask &= _isin("magnitude_type", mag_types)
    if modes is not None:
        mask &= _isin("evaluation_mode", modes)
    if etypes is not None:
        mask &= _isin("event_type", etypes)
    if reviewed_only:
        mask &= _isin("evaluation_mode", ["manual"])
        mask &= ~_isin("evaluation_status", ["rejected"])
    return _append_subset(df, df[mask].copy(), record)


# --------------------------------------------------------------------------
# Derived values
# --------------------------------------------------------------------------


def daily_counts(
    df: pd.DataFrame,
    *,
    by: str | None = None,
    start: TimeLike | None = None,
    end: TimeLike | None = None,
) -> pd.Series | pd.DataFrame:
    """Number of events per UTC day.

    Returns a Series named ``count`` on a daily ``DatetimeIndex`` (UTC
    midnights, named ``day``) with every day from the first to the last event,
    zero-filled. ``start``/``end`` (taken as UTC, see :func:`to_utc`) fix
    either end of that span to the day containing them, inclusive; events
    outside it are not counted. An empty frame without
    ``start``/``end`` gives an empty Series.

    With ``by`` (a column name, e.g. ``"evaluation_mode"`` or
    ``"magnitude_type"``) it returns a DataFrame with one integer column per
    value. Rows whose ``by`` value is NA are counted in a column ``"<NA>"``,
    so the columns always add up to the total. For a category column every
    allowed value gets a column, even if it never occurs.
    """
    _require(df, [] if by is None else [by], index=True, what="daily_counts")
    days = df.index.tz_convert("UTC").floor("D")
    lo = to_utc(start, what="start").floor("D") if start is not None else None
    hi = to_utc(end, what="end").floor("D") if end is not None else None
    if lo is None and len(days):
        lo = days.min()
    if hi is None and len(days):
        hi = days.max()
    if lo is not None and hi is None:
        hi = max(lo, days.max()) if len(days) else lo
    if hi is not None and lo is None:
        lo = min(hi, days.min()) if len(days) else hi
    if lo is not None and hi is not None and lo > hi:
        raise FilterError(f"daily_counts: start {_iso(lo)} is after end {_iso(hi)}")
    span = (
        pd.DatetimeIndex([], tz="UTC")
        if lo is None
        else pd.date_range(lo, hi, freq="D", tz="UTC")
    )
    span = span.rename("day")
    if start is not None or end is not None:
        keep = np.asarray((days >= lo) & (days <= hi), dtype=bool)
        frame = df[keep]
        days = frame.index.tz_convert("UTC").floor("D")
    else:
        frame = df
    if by is None:
        counts = pd.Series(1, index=days).groupby(level=0).sum()
        out = counts.reindex(span, fill_value=0).astype("int64")
        return out.rename("count").rename_axis("day")
    key = frame[by]
    labels = key.astype(object).where(key.notna(), "<NA>").astype(str)
    table = pd.crosstab(days, labels.to_numpy()) if len(frame) else pd.DataFrame()
    columns: list[str] = []
    if isinstance(key.dtype, pd.CategoricalDtype):
        columns = [str(c) for c in key.dtype.categories]
    columns += [str(c) for c in table.columns if str(c) not in columns]
    table = table.reindex(index=span, columns=columns, fill_value=0).astype("int64")
    table.index.name = "day"
    table.columns.name = by
    return table


def moment_from_magnitude(mw: npt.ArrayLike) -> FloatArray:
    """Seismic moment (N·m) from moment magnitude: the inverse of ``moment_magnitude``.

    ``M0 = 10**(1.5·(Mw + 6.0333…))`` with the same Hanks & Kanamori (1979)
    constant as :func:`~geo_dataread.seismic.schema.moment_magnitude`
    (``HK79_CONSTANT_NM``), so the two round-trip exactly. Note that the
    legacy ``sil_read.MW`` uses 6.03. Valid only for a magnitude that *is*
    Hanks–Kanamori Mw; SIL's Mlw is not (see :func:`with_derived_moment`). NaN gives NaN. Scalar in, 0-d array out.
    """
    m = np.asarray(mw, dtype=np.float64)
    out: FloatArray = np.power(10.0, 1.5 * (m + HK79_CONSTANT_NM))
    return out


#: Name of the flag column :func:`with_derived_moment` adds.
DERIVED_MOMENT_FLAG: Final = "moment_derived"


def with_derived_moment(
    df: pd.DataFrame, *, magnitude_types: Iterable[str]
) -> pd.DataFrame:
    """Fill a missing ``moment_nm`` from ``magnitude`` as if it were Mw, flagged.

    For rows where ``moment_nm`` is NA, ``magnitude`` is finite and
    ``magnitude_type`` is in ``magnitude_types``, sets
    ``moment_nm = moment_from_magnitude(magnitude)`` (Hanks & Kanamori). An
    existing moment is never overwritten. Adds the boolean column
    ``moment_derived`` (True where filled); it is not a schema column, so
    ``validate`` reports it as an ``extra_column`` warning. The choice of
    types is recorded in ``attrs["catalog"]["subsets"]``.

    ``magnitude_types`` is required, with no default, because no magnitude in
    these catalogues is Hanks–Kanamori Mw: treating one as Mw is a magnitude
    conversion, which the schema leaves to the caller (Q7). In particular SIL
    **Mlw is not HK Mw**: on the T0 ``events.fps`` fixtures (22 events, Mlw
    −0.5…3.1) ``Mlw ≈ log10(M0[N·m]) − 10`` (median residual 0.00 below
    Mlw 2, −0.04 at 2.2, −0.12 at 3.1), so the HK inverse under-derives the
    moment of an Mlw 1 event about 3× and of an Mlw 0.5 event about 5×. A
    SIL-specific relation is an open question (PLAN-seismicity, for BGÓ).
    The result feeds :func:`~geo_dataread.seismic.schema.cumulative_moment`
    unchanged; label any figure that uses derived moments.
    """
    types = _check_allowed(magnitude_types, "magnitude_type")
    _require(
        df,
        ("moment_nm", "magnitude", "magnitude_type"),
        index=False,
        what="with_derived_moment",
    )
    out = df.copy()
    m0 = _num(df, "moment_nm")
    mag = _num(df, "magnitude")
    mtype = df["magnitude_type"].astype(object)
    fill = (
        np.isnan(m0)
        & np.isfinite(mag)
        & np.asarray(mtype.isin(types) & mtype.notna(), dtype=bool)
    )
    m0 = m0.copy()
    m0[fill] = moment_from_magnitude(mag[fill])
    out["moment_nm"] = m0
    out[DERIVED_MOMENT_FLAG] = fill
    record = {
        "filter": "with_derived_moment",
        "magnitude_types": types,
        "n_derived": int(fill.sum()),
    }
    return _append_subset(df, out, record)
