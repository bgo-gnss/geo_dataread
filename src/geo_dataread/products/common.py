"""Shared pieces of the gpsprod products: records, record sets, provenance.

Every record carries what it is — frame, series (pos product), session,
reference, view, method, period, σ method, units — so a number can never be
taken for another kind. Velocity keeps its own record
(:class:`geo_dataread.velocity.VelocityRecord`, schema 3); the products here
start at :data:`SCHEMA_VERSION` 1.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterator, Mapping, Sequence
from importlib import metadata
from typing import Any, Generic, TypeVar

import pandas as pd

SCHEMA_VERSION = 1
COMPONENTS = ("north", "east", "up")
#: the references a product series can be read in: the station's plate
#: removed and its stored seasonal removed (plate-periodic), its whole
#: background removed (background), plate only (plate), or the raw frame
REFS = ("plate-periodic", "background", "plate", "itrf")
#: the record terms each detrended ref removes (gps_views terms)
TERMS = {"plate-periodic": "periodic", "background": "all"}
VIEWS = ("declared", "cleaned", "raw")
SESSIONS = {"24h": 1.0, "8h": 1.0 / 3.0}  # session length [days]

Vec3 = tuple[float, float, float]
NAN3: Vec3 = (math.nan, math.nan, math.nan)


def software() -> dict[str, str]:
    """Package versions behind a record."""
    out = {}
    for pkg in ("geo_dataread", "gps_analysis"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "unknown"
    return out


def jsonable(v: Any) -> Any:
    """NaN → None, tuples → lists, recursively (JSON-safe)."""
    if isinstance(v, float):
        return None if not math.isfinite(v) else v
    if isinstance(v, Mapping):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return v


def _vec3(v: Any) -> Vec3:
    a, b, c = (math.nan if x is None else float(x) for x in v)
    return a, b, c


@dataclasses.dataclass(frozen=True)
class DisplacementRecord:
    """One station's displacement across a period.

    ``displacement`` and ``sigma`` are N, E, U in ``units`` (mm unless
    written with ``--units m``). ``period`` is the requested half-open
    window in fractional years; ``ends`` holds what each end window used.
    A station that gives no displacement is ``refused`` with the reason.
    """

    station: str
    lon: float
    lat: float
    period: tuple[float, float]
    period_dates: tuple[str, str]
    frame: str
    series: str
    session: str
    ref: str
    ref_requested: str
    view: str
    method: str
    sigma_method: str
    displacement: Vec3 = NAN3
    sigma: Vec3 = NAN3
    units: str = "mm"
    uncert: float = math.nan
    relative_to: str | None = None
    period_names: tuple[str, ...] = ()
    ends: dict[str, Any] | None = None
    #: method step: the event, the two lines, the windows, the k used
    step: dict[str, Any] | None = None
    #: method rate: the velocity record the displacement was projected from
    rate: dict[str, Any] | None = None
    flags: tuple[str, ...] = ()
    refused: str | None = None
    input_fingerprint: str = ""
    software: dict[str, str] = dataclasses.field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    kind: str = "displacement"

    @property
    def accepted(self) -> bool:
        return self.refused is None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = jsonable(dataclasses.asdict(self))
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> DisplacementRecord:
        kw = dict(d)
        for key in ("displacement", "sigma"):
            kw[key] = _vec3(kw[key])
        kw["period"] = tuple(kw["period"])
        kw["period_dates"] = tuple(kw["period_dates"])
        kw["period_names"] = tuple(kw.get("period_names") or ())
        kw["flags"] = tuple(kw.get("flags") or ())
        kw["lon"] = math.nan if kw["lon"] is None else kw["lon"]
        kw["lat"] = math.nan if kw["lat"] is None else kw["lat"]
        kw["uncert"] = math.nan if kw.get("uncert") is None else kw["uncert"]
        return cls(**kw)

    def scaled(self, units: str) -> DisplacementRecord:
        """The record in ``units`` ("mm" or "m")."""
        if units == self.units:
            return self
        factor = {("mm", "m"): 1e-3, ("m", "mm"): 1e3}[(self.units, units)]
        # the provenance dicts (ends, step, rate) keep the product's own
        # units, mm and mm/yr; they say so
        return dataclasses.replace(
            self,
            displacement=_vec3(x * factor for x in self.displacement),
            sigma=_vec3(x * factor for x in self.sigma),
            units=units,
            ends=None if self.ends is None else {**self.ends, "units": "mm"},
            step=None if self.step is None else {**self.step, "units": "mm, mm/yr"},
        )


R = TypeVar("R")


@dataclasses.dataclass
class RecordSet(Generic[R]):
    """The records of one product run, in station order."""

    records: list[R]
    kind: str
    settings: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __iter__(self) -> Iterator[R]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def accepted(self) -> list[R]:
        return [r for r in self.records if getattr(r, "accepted", False)]

    def to_frame(self) -> pd.DataFrame:
        """One row per record: the CSV columns (:mod:`.files`)."""
        from geo_dataread.products.files import rows

        cols, data = rows(self.records)
        return pd.DataFrame(data, columns=list(cols))

    def write(self, path: str, *, units: str = "mm") -> str:
        """Write to ``path``; the format comes from its extension."""
        from geo_dataread.products.files import write

        return write(self, path, units=units)


def provenance_header(rs: RecordSet[Any]) -> list[str]:
    """``key value`` lines describing a set (the CSV ``#`` header)."""
    lines = [f"gpsprod {rs.kind} schema {SCHEMA_VERSION}"]
    for key, value in rs.settings.items():
        if isinstance(value, Sequence) and not isinstance(value, str):
            value = ",".join(str(v) for v in value)
        lines.append(f"{key} {value}")
    return lines
