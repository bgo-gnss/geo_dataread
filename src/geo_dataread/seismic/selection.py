"""Quake selection and the catalogue across the SIL → SeisComP seam (Q-T1).

PLAN-ts-quakes Q7: one selector type, many sources of the region. A
:class:`QuakeSelection` is a region (a named region of ``quake_regions.yaml``,
a box, a polygon or a radius) plus depth, magnitude and event-type bounds.
The caller decides where the region comes from (the station area's named
region for a ts plot, the map's extent for a map, a radius around a .pos
position); the readers never know which plot asked.

:func:`catalogue_for` reads a selection for ``[start, end)``:

* ``source="auto"`` (default): SIL files before :data:`SEAM`, SeisComP from
  it. The cut is hard and half-open on both sides — never a union: SeisComP
  before 2026 is a back-import of SIL's reviewed events, and SIL still has
  scattered reviewed events after the seam, so a union counts events twice.
  With ``mode="automatic"`` it reads SIL for the whole window: SIL automatic
  is the one series of the same kind of row on both sides (PLAN-seismicity
  §9e, point 5).
* ``"sil"``: SIL files only. ``"seiscomp"``: the Quakes API (internal
  endpoint, server-side box) only. ``"fdsn"``: SeisComP through the FDSN
  event service only.

``mode`` (``catalog`` | ``automatic`` | ``reviewed``) means for SIL what
:func:`~geo_dataread.seismic.sil_catalog.read_sil_catalog` says. For SeisComP:
``catalog`` = every event, ``reviewed`` = manual only, ``automatic`` =
events not yet reviewed (not the automatic solutions of reviewed events,
which only the FDSN origins hold). Rejected events (``evaluation_status``
``rejected`` or type ``not existing``) are always dropped. ``min_quality``
and ``duplicate_tol`` act on SIL automatic rows only (SeisComP rows have no Q).
"""

from __future__ import annotations

import dataclasses
import math
import time as _time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

from geo_dataread.seismic.filters import (
    FilterError,
    TimeLike,
    subset,
    to_utc,
    within_distance,
)
from geo_dataread.seismic.schema import (
    ATTRS_KEY,
    EVENT_TYPES,
    SCHEMA_VERSION,
    coerce_catalog,
)
from geo_dataread.seismic.sil_catalog import (
    DEFAULT_DUPLICATE_TOL,
    DEFAULT_MIN_QUALITY,
    KM_PER_DEG,
    MODES,
    CatalogAutomaticFile,
    MatchTolerance,
    Mode,
)

#: SeisComP took over review on 2026-02-01 (SIL's Mlw share: 0.92 in January,
#: 0.27 in February). ``source="auto"`` switches here.
SEAM: Final = pd.Timestamp("2026-02-01", tz="UTC")

#: :func:`catalogue_for` sources.
Source = Literal["auto", "sil", "seiscomp", "fdsn"]
SOURCES: Final[tuple[Source, ...]] = ("auto", "sil", "seiscomp", "fdsn")

#: Event types kept by default (report recommendation b: other, explosions
#: and not-existing events stay out of quake products). A missing type is kept.
DEFAULT_EVENT_TYPES: Final[tuple[str, ...]] = ("earthquake",)

#: ``quake_regions.yaml``: ``[FILES] quake_regions`` in postprocess.cfg, else
#: this name in the gpsconfig dir (as events.yaml).
REGIONS_FILES_KEY: Final = "quake_regions"
REGIONS_FILENAME: Final = "quake_regions.yaml"

Bbox = tuple[float, float, float, float]


class SelectionError(ValueError):
    """A selection or region is malformed or unknown."""


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def _bbox(value: Sequence[float]) -> Bbox:
    if len(value) != 4:
        raise SelectionError("bbox is (min_lon, min_lat, max_lon, max_lat)")
    w, s, e, n = (float(v) for v in value)
    if not (w < e and s < n):
        raise SelectionError(f"bbox {tuple(value)} is empty or inverted")
    if not (-180 <= w and e <= 180 and -90 <= s and n <= 90):
        raise SelectionError(f"bbox {tuple(value)} is outside lon/lat range")
    return (w, s, e, n)


@dataclasses.dataclass(frozen=True)
class QuakeSelection:
    """Which quakes: a region and bounds. Every bound is inclusive.

    Region: at most one of ``bbox`` ``(min_lon, min_lat, max_lon, max_lat)``,
    ``polygon`` (``(lon, lat)`` vertices) or ``radius`` ``(lat, lon, km)``
    (epicentral, on the ellipsoid); none = no region cut. ``name`` labels it
    (the named region, or the caller's own label). ``event_types=None``
    keeps every type; otherwise rows whose type is listed or missing.
    ``min_mag``/``max_mag`` act on ``magnitude``, which mixes scales.
    """

    bbox: Bbox | None = None
    polygon: tuple[tuple[float, float], ...] | None = None
    radius: tuple[float, float, float] | None = None
    name: str | None = None
    min_mag: float | None = None
    max_mag: float | None = None
    min_depth: float | None = None
    max_depth: float | None = None
    event_types: tuple[str, ...] | None = DEFAULT_EVENT_TYPES

    def __post_init__(self) -> None:
        given = [k for k in ("bbox", "polygon", "radius") if getattr(self, k)]
        if len(given) > 1:
            raise SelectionError(f"give one region kind, not {given}")
        if self.bbox is not None:
            object.__setattr__(self, "bbox", _bbox(self.bbox))
        if self.polygon is not None:
            pts = tuple((float(p[0]), float(p[1])) for p in self.polygon)
            if len(pts) < 3:
                raise SelectionError("a polygon needs at least 3 (lon, lat) vertices")
            object.__setattr__(self, "polygon", pts)
        if self.radius is not None:
            lat, lon, km = (float(v) for v in self.radius)
            if abs(lat) > 90 or abs(lon) > 180 or not km > 0:
                raise SelectionError(f"radius {self.radius}: (lat, lon, km > 0)")
            object.__setattr__(self, "radius", (lat, lon, km))
        if self.event_types is not None:
            bad = sorted(set(self.event_types) - set(EVENT_TYPES))
            if bad:
                raise SelectionError(f"unknown event types {bad}; known {EVENT_TYPES}")
            object.__setattr__(self, "event_types", tuple(self.event_types))

    def replace(self, **changes: Any) -> QuakeSelection:
        """A copy with ``changes`` (dataclasses.replace)."""
        return dataclasses.replace(self, **changes)

    def server_bbox(self) -> Bbox | None:
        """A box enclosing the region, for a server-side cut (then refined)."""
        if self.bbox is not None:
            return self.bbox
        if self.polygon is not None:
            lon = [p[0] for p in self.polygon]
            lat = [p[1] for p in self.polygon]
            return (min(lon), min(lat), max(lon), max(lat))
        if self.radius is not None:
            lat0, lon0, km = self.radius
            dlat = km / KM_PER_DEG * 1.01
            dlon = dlat / max(math.cos(math.radians(min(abs(lat0) + dlat, 89.9))), 1e-6)
            return (
                max(lon0 - dlon, -180.0),
                max(lat0 - dlat, -90.0),
                min(lon0 + dlon, 180.0),
                min(lat0 + dlat, 90.0),
            )
        return None

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """The rows of a schema events frame inside this selection."""
        out = subset(
            df,
            bbox=self.bbox,
            polygon=self.polygon,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            min_mag=self.min_mag,
            max_mag=self.max_mag,
        )
        if self.radius is not None:
            out = within_distance(out, self.radius[0], self.radius[1], self.radius[2])
        if self.event_types is not None and len(out):
            et = out["event_type"].astype(object)
            keep = et.isna() | et.isin(self.event_types)
            out = out.loc[np.asarray(keep, dtype=bool)]
        return out

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


# --------------------------------------------------------------------------
# Named regions (quake_regions.yaml)
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class QuakeRegion:
    """One named region of ``quake_regions.yaml``."""

    name: str
    selection: QuakeSelection
    title: str | None = None
    area: str | None = None
    default: bool = False
    note: str | None = None


def parse_quake_regions(doc: Mapping[str, Any]) -> dict[str, QuakeRegion]:
    """The regions of a loaded ``quake_regions.yaml``.

    Layout::

        regions:
          bardarbunga-caldera:
            title: Bárðarbunga, askja
            area: bardarbunga        # station area (station_areas.yaml)
            default: true            # the area's region for ts plots
            bbox: [-17.65, 64.59, -17.25, 64.72]   # or polygon: / radius:
            min_depth: 0             # optional bounds, as QuakeSelection
            note: free text

    ``polygon`` is a list of ``[lon, lat]``; ``radius`` is
    ``[lat, lon, km]``. At most one region per area may be ``default``.
    """
    rows = doc.get("regions") if isinstance(doc, Mapping) else None
    if not isinstance(rows, Mapping):
        raise SelectionError("quake_regions.yaml: top-level 'regions:' mapping missing")
    keys = {"bbox", "polygon", "radius", "min_mag", "max_mag", "min_depth", "max_depth"}
    meta = {"title", "area", "default", "note", "event_types"}
    out: dict[str, QuakeRegion] = {}
    defaults: dict[str, str] = {}
    for name, row in rows.items():
        if not isinstance(row, Mapping):
            raise SelectionError(f"region {name!r}: expected a mapping")
        unknown = sorted(set(row) - keys - meta)
        if unknown:
            raise SelectionError(f"region {name!r}: unknown keys {unknown}")
        kw: dict[str, Any] = {k: row[k] for k in keys if row.get(k) is not None}
        if "bbox" in kw:
            kw["bbox"] = tuple(kw["bbox"])
        if "polygon" in kw:
            kw["polygon"] = tuple(tuple(p) for p in kw["polygon"])
        if "radius" in kw:
            kw["radius"] = tuple(kw["radius"])
        if "event_types" in row:
            et = row["event_types"]
            kw["event_types"] = None if et in (None, "all") else tuple(et)
        try:
            sel = QuakeSelection(name=str(name), **kw)
        except (SelectionError, TypeError, ValueError) as exc:
            raise SelectionError(f"region {name!r}: {exc}") from exc
        if sel.bbox is None and sel.polygon is None and sel.radius is None:
            raise SelectionError(f"region {name!r}: needs bbox, polygon or radius")
        area = row.get("area")
        is_default = bool(row.get("default", False))
        if is_default:
            if area is None:
                raise SelectionError(f"region {name!r}: default needs an area")
            if area in defaults:
                raise SelectionError(
                    f"area {area!r} has two default regions: "
                    f"{defaults[area]!r} and {name!r}"
                )
            defaults[str(area)] = str(name)
        out[str(name)] = QuakeRegion(
            name=str(name),
            selection=sel,
            title=row.get("title"),
            area=None if area is None else str(area),
            default=is_default,
            note=row.get("note"),
        )
    return out


def default_quake_regions_path() -> Path | None:
    """The deployed ``quake_regions.yaml`` (``[FILES] quake_regions``, else
    ``<gpsconfig dir>/quake_regions.yaml``); None without a gpsconfig."""
    try:
        from gps_parser import outlier_catalogs as _oc
    except ImportError:  # pragma: no cover - gps_parser is a dependency
        return None
    try:
        path = _oc.catalog_path(REGIONS_FILES_KEY, REGIONS_FILENAME)
    except Exception:  # noqa: BLE001 - no gpsconfig deployed
        return None
    return None if path is None else Path(path)


def read_quake_regions(path: str | Path | None = None) -> dict[str, QuakeRegion]:
    """Read ``quake_regions.yaml`` (None: the deployed one).

    Raises :class:`FileNotFoundError` when there is no file, naming where it
    was looked for.
    """
    import yaml

    p = Path(path) if path is not None else default_quake_regions_path()
    if p is None or not p.is_file():
        where = p if p is not None else f"<gpsconfig>/{REGIONS_FILENAME}"
        raise FileNotFoundError(
            f"no quake regions file at {where}; pass a path (example: "
            "geo_dataread docs/seismic/quake_regions.example.yaml)"
        )
    with p.open(encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    return parse_quake_regions(doc)


def region_for_area(
    area: str, regions: Mapping[str, QuakeRegion]
) -> QuakeRegion | None:
    """The ``default`` region of a station area, if any."""
    for r in regions.values():
        if r.default and r.area == area:
            return r
    return None


def parse_region_spec(
    spec: str,
    regions: Mapping[str, QuakeRegion] | Callable[[], Mapping[str, QuakeRegion]],
) -> QuakeSelection:
    """A selection from a CLI region spec.

    ``W,S,E,N`` (four numbers) → bbox; ``LAT,LON,KM`` (three) → radius;
    anything else → a named region (``regions`` may be a callable, so the
    file is read only for a name).
    """
    parts = [p.strip() for p in spec.split(",")]
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        nums = []
    if len(nums) == 4:
        return QuakeSelection(bbox=(nums[0], nums[1], nums[2], nums[3]), name=spec)
    if len(nums) == 3:
        return QuakeSelection(radius=(nums[0], nums[1], nums[2]), name=spec)
    table = regions() if callable(regions) else regions
    if spec not in table:
        known = ", ".join(sorted(table)) or "(none)"
        raise SelectionError(f"unknown quake region {spec!r}; known: {known}")
    return table[spec].selection


# --------------------------------------------------------------------------
# The catalogue
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CataloguePart:
    """What one source contributed."""

    source: str
    start: pd.Timestamp
    end: pd.Timestamp
    read: int
    kept: int
    counts: Mapping[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "window": {"start": self.start.isoformat(), "end": self.end.isoformat()},
            "read": self.read,
            "kept": self.kept,
            "counts": dict(self.counts),
        }


@dataclasses.dataclass(frozen=True)
class CatalogueReport:
    parts: tuple[CataloguePart, ...]
    seconds: float

    def summary(self) -> str:
        bits = [
            f"{p.source} {p.start:%Y-%m-%d}→{p.end:%Y-%m-%d}: {p.kept} of {p.read}"
            for p in self.parts
        ]
        return "; ".join(bits) or "no source read"


def _empty() -> pd.DataFrame:
    df = pd.DataFrame({"time": pd.DatetimeIndex([], tz="UTC")}).set_index("time")
    return coerce_catalog(df, source="sil-files")


def _windows(
    source: Source, mode: Mode, lo: pd.Timestamp, hi: pd.Timestamp, seam: pd.Timestamp
) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    if source == "sil" or (source == "auto" and mode == "automatic"):
        return [("sil", lo, hi)]
    if source in ("seiscomp", "fdsn"):
        return [(source, lo, hi)]
    out: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    if lo < seam:
        out.append(("sil", lo, min(hi, seam)))
    if hi > seam:
        out.append(("seiscomp", max(lo, seam), hi))
    return out


def _seiscomp_mode(df: pd.DataFrame, mode: Mode, counts: dict[str, int]) -> Any:
    status = df["evaluation_status"].astype(object)
    etype = df["event_type"].astype(object)
    rejected = np.asarray(
        (status == "rejected") | (etype == "not existing"), dtype=bool
    )
    counts["rejected_dropped"] = int(rejected.sum())
    keep = ~rejected
    em = df["evaluation_mode"].astype(object).to_numpy()
    if mode == "reviewed":
        keep &= em == "manual"
    elif mode == "automatic":
        keep &= em == "automatic"
    counts[f"mode_{mode}_dropped"] = int((~rejected & ~keep).sum())
    return df.loc[keep]


def catalogue_for(
    selection: QuakeSelection,
    start: TimeLike,
    end: TimeLike,
    *,
    source: Source = "auto",
    mode: Mode = "catalog",
    min_quality: float | None = DEFAULT_MIN_QUALITY,
    duplicate_tol: MatchTolerance | None = DEFAULT_DUPLICATE_TOL,
    drop_duplicates: bool = False,
    automatic_file: CatalogAutomaticFile = "aut.mag",
    seam: TimeLike = SEAM,
    sil_base: str | Path | None = None,
    use_cache: bool = True,
    session: Any = None,
    readers: Mapping[str, Callable[..., pd.DataFrame]] | None = None,
) -> tuple[pd.DataFrame, CatalogueReport]:
    """The quakes of ``selection`` in ``[start, end)``, across the seam.

    See the module docstring for ``source`` and ``mode``. ``min_quality``,
    ``duplicate_tol`` and ``automatic_file`` go to the SIL reader (cached:
    :func:`~geo_dataread.seismic.cache.read_sil_catalog_cached`, archive
    ``sil_base``); ``drop_duplicates`` removes rows flagged
    ``possible_duplicate``. ``session`` is the HTTP transport of the
    SeisComP readers (tests). ``readers`` overrides the per-source read
    functions (``"sil"``, ``"seiscomp"``, ``"fdsn"``; tests).

    The result is one schema frame, sorted by time; its ``source`` column
    says where each row came from and ``magnitude_type`` keeps each row's
    scale (no conversion here; :func:`~geo_dataread.seismic.moment.moment_estimate`
    adds a moment). ``attrs["catalog"]`` records the selection, the seam and
    each part. Times without a zone are UTC.
    """
    t0 = _time.perf_counter()
    if source not in SOURCES:
        raise SelectionError(f"source {source!r}: one of {SOURCES}")
    if mode not in MODES:
        raise SelectionError(f"mode {mode!r}: one of {MODES}")
    lo, hi = to_utc(start, what="start"), to_utc(end, what="end")
    if hi < lo:
        raise FilterError(f"end {hi} is before start {lo}")
    seam_ts = to_utc(seam, what="seam")
    rd = dict(_default_readers())
    rd.update(readers or {})
    frames: list[pd.DataFrame] = []
    parts: list[CataloguePart] = []
    for kind, a, b in _windows(source, mode, lo, hi, seam_ts):
        if not a < b:
            continue
        counts: dict[str, int] = {}
        if kind == "sil":
            raw = rd["sil"](
                a,
                b,
                base=sil_base,
                mode=mode,
                automatic_file=automatic_file,
                min_quality=min_quality,
                duplicate_tol=duplicate_tol,
                use_cache=use_cache,
            )
            got = raw
        else:
            raw = rd[kind](a, b, bbox=selection.server_bbox(), session=session)
            got = _seiscomp_mode(raw, mode, counts)
        got = selection.apply(got)
        if drop_duplicates and len(got):
            dup = np.asarray(got["possible_duplicate"].fillna(False), dtype=bool)
            counts["duplicates_dropped"] = int(dup.sum())
            got = got.loc[~dup]
        parts.append(
            CataloguePart(
                source=str(raw.attrs.get(ATTRS_KEY, {}).get("source", kind)),
                start=a,
                end=b,
                read=len(raw),
                kept=len(got),
                counts={k: v for k, v in counts.items() if v},
            )
        )
        frames.append(got)
    nonempty = [f for f in frames if len(f)]
    if nonempty:
        df = pd.concat(nonempty)
        df.attrs = {}
        df = coerce_catalog(df)
    else:
        df = _empty()
    report = CatalogueReport(tuple(parts), _time.perf_counter() - t0)
    df.attrs = {
        ATTRS_KEY: {
            "schema_version": SCHEMA_VERSION,
            "frame": "events",
            "source": "catalogue_for",
            "window": {"start": lo.isoformat(), "end": hi.isoformat()},
            "selection": selection.as_dict(),
            "filters": {
                "source": source,
                "mode": mode,
                "min_quality": min_quality,
                "duplicate_tol": (
                    None if duplicate_tol is None else dataclasses.asdict(duplicate_tol)
                ),
                "drop_duplicates": drop_duplicates,
                "automatic_file": automatic_file,
            },
            "seam": seam_ts.isoformat(),
            "parts": [p.as_dict() for p in parts],
            "fetched_at": datetime.now(UTC).isoformat(),
        }
    }
    return df, report


def _read_sil(a: pd.Timestamp, b: pd.Timestamp, **kw: Any) -> pd.DataFrame:
    from geo_dataread.seismic.cache import read_sil_catalog_cached
    from geo_dataread.seismic.sil_catalog import read_sil_catalog

    use_cache = kw.pop("use_cache", True)
    if use_cache:
        df, _ = read_sil_catalog_cached(a, b, **kw)
    else:
        df, _ = read_sil_catalog(a, b, **kw)
    return df


def _read_quakes(a: pd.Timestamp, b: pd.Timestamp, **kw: Any) -> pd.DataFrame:
    from geo_dataread.seismic.quakes_api import read_quakes

    return read_quakes(a, b, system="seiscomp", **kw)


def _read_fdsn(a: pd.Timestamp, b: pd.Timestamp, **kw: Any) -> pd.DataFrame:
    from geo_dataread.seismic.fdsn import read_fdsn

    return read_fdsn(a, b, include_all_magnitudes=True, **kw)


def _default_readers() -> dict[str, Callable[..., pd.DataFrame]]:
    return {"sil": _read_sil, "seiscomp": _read_quakes, "fdsn": _read_fdsn}


__all__ = [
    "DEFAULT_EVENT_TYPES",
    "REGIONS_FILENAME",
    "REGIONS_FILES_KEY",
    "SEAM",
    "SOURCES",
    "CataloguePart",
    "CatalogueReport",
    "QuakeRegion",
    "QuakeSelection",
    "SelectionError",
    "Source",
    "catalogue_for",
    "default_quake_regions_path",
    "parse_quake_regions",
    "parse_region_spec",
    "read_quake_regions",
    "region_for_area",
]
