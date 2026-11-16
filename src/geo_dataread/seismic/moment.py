"""Seismic moment estimate across the SIL → SeisComP seam (Q-T1, PLAN-ts-quakes Q4).

:func:`moment_estimate` gives every event it can a seismic moment, choosing
the relation from the row's ``magnitude_type`` (never from a magnitude
threshold alone):

==============  ============================================================
type            moment
==============  ============================================================
any, measured   ``moment_nm`` as read (SIL ``events.fps``), relation ``measured``
``Mlw``         the SIL archive relation :func:`~geo_dataread.seismic.filters.moment_from_mlw`
``ML_SIL``      :data:`SEISCOMP_ML_SIL` (log10 M0 = 10.41 + 1.045 M)
``Mpgv_w``      :data:`SEISCOMP_MPGV_W` (log10 M0 = 8.95 + 1.347 M) for M ≥ 3;
                below 3 it is not estimated (Mpgv_w runs 0.7–1.2 above
                ML_SIL there, so neither relation fits) and is counted
other types     not estimated, counted (``aut.mag``, ``events.ref``, ``ML``,
                ``Msil_auto``, ``M``, ``Mpgv_alert``: no relation yet)
==============  ============================================================

BGÓ's Q4 (2026-11-15) allows the two SeisComP relations for plotting now,
labelled as estimates; it overrides R1 for this use only.
:func:`~geo_dataread.seismic.filters.with_derived_moment` keeps its R1 guard.
The relations, their fit sets and their limits are in
``docs/seismic/seiscomp_moment_relation.md``; T18 replaces them.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from geo_dataread.seismic.filters import MLW_M0_FIT, moment_from_mlw
from geo_dataread.seismic.schema import ATTRS_KEY

FloatArray = npt.NDArray[np.float64]

#: Columns :func:`moment_estimate` adds (not schema columns).
ESTIMATED_FLAG: Final = "moment_estimated"
RELATION_COLUMN: Final = "moment_relation"

MEASURED: Final = "measured"
SIL_MLW: Final = "sil_mlw_m0"


@dataclasses.dataclass(frozen=True)
class MomentRelation:
    """``log10 M0 [N·m] = intercept + slope · M`` for one magnitude type."""

    name: str
    magnitude_type: str
    intercept: float
    slope: float
    min_mag: float | None
    n: int
    rms_log10: float | None
    fit: str
    reference: str = "docs/seismic/seiscomp_moment_relation.md"

    def log10_moment(self, mag: npt.ArrayLike) -> FloatArray:
        m = np.asarray(mag, dtype=np.float64)
        out: FloatArray = self.intercept + self.slope * m
        return out

    def applies(self, mag: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        m = np.asarray(mag, dtype=np.float64)
        ok = np.isfinite(m)
        if self.min_mag is not None:
            ok &= m >= self.min_mag
        return ok

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


#: SeisComP ``ML_SIL`` → M0. Fit 2026-11-15: SeisComP back-import ``ML_SIL``
#: of SIL events with a measured SIL moment, matched 2023-01 – 2026-01;
#: n 2622, rms 0.29 in log10 M0. ML_SIL saturates near 4 (4.03 for
#: Mpgv_w 5.24), so large events are underestimated.
SEISCOMP_ML_SIL: Final = MomentRelation(
    name="seiscomp_ml_sil_m0",
    magnitude_type="ML_SIL",
    intercept=10.41,
    slope=1.045,
    min_mag=None,
    n=2622,
    rms_log10=0.29,
    fit="SeisComP ML_SIL vs SIL events.fps moment, matched events 2023-01..2026-01",
)

#: SeisComP ``Mpgv_w`` (M ≥ 3) → M0, treating Mpgv_w as Mlw (5.38 vs 5.27 for
#: 2026-01-31): the SIL Mlw ≥ 4 events with a measured moment, n 24, rms not
#: recorded. Our Mw from it is ~0.75 below GEOFON Mw, consistently (BGÓ:
#: keep our scale).
SEISCOMP_MPGV_W: Final = MomentRelation(
    name="seiscomp_mpgv_w_m0",
    magnitude_type="Mpgv_w",
    intercept=8.95,
    slope=1.347,
    min_mag=3.0,
    n=24,
    rms_log10=None,
    fit="SIL Mlw >= 4 vs events.fps moment; Mpgv_w taken as Mlw",
)

#: The relations :func:`moment_estimate` uses, by magnitude type.
DEFAULT_RELATIONS: Final[Mapping[str, MomentRelation]] = {
    r.magnitude_type: r for r in (SEISCOMP_ML_SIL, SEISCOMP_MPGV_W)
}


def moment_estimate(
    df: pd.DataFrame,
    *,
    relations: Mapping[str, MomentRelation] = DEFAULT_RELATIONS,
    mlw: bool = True,
) -> pd.DataFrame:
    """A copy of ``df`` with ``moment_nm`` filled where a relation exists.

    A measured moment is never replaced. ``Mlw`` rows use the SIL relation
    (``mlw=False`` leaves them), the types in ``relations`` their own
    relation within its range; every other row keeps NA. Adds
    ``moment_estimated`` (True where filled here) and ``moment_relation``
    (``measured``, the relation name, or NA). The counts per relation and
    the unestimated rows by type are appended to
    ``attrs["catalog"]["subsets"]``; label any figure that uses them.
    """
    for col in ("moment_nm", "magnitude", "magnitude_type"):
        if col not in df.columns:
            raise ValueError(f"moment_estimate: column {col!r} missing")
    out = df.copy()
    n = len(df)
    m0 = np.array(
        pd.to_numeric(df["moment_nm"]).to_numpy(dtype="float64", na_value=np.nan)
    )
    mag = np.array(
        pd.to_numeric(df["magnitude"]).to_numpy(dtype="float64", na_value=np.nan)
    )
    mtype = np.array(df["magnitude_type"].astype(object).to_numpy(), dtype=object)
    measured = np.isfinite(m0)
    rel = np.full(n, None, dtype=object)
    rel[measured] = MEASURED
    est = np.zeros(n, dtype=bool)
    if mlw:
        sel = ~measured & (mtype == "Mlw") & np.isfinite(mag)
        m0[sel] = moment_from_mlw(mag[sel])
        rel[sel] = SIL_MLW
        est |= sel
    for mt, r in relations.items():
        if mt != r.magnitude_type:
            raise ValueError(f"relation {r.name} is for {r.magnitude_type}, not {mt}")
        sel = ~measured & ~est & (mtype == mt) & r.applies(mag)
        m0[sel] = np.power(10.0, r.log10_moment(mag[sel]))
        rel[sel] = r.name
        est |= sel
    out["moment_nm"] = m0
    out[ESTIMATED_FLAG] = est
    out[RELATION_COLUMN] = pd.array(rel, dtype="string")
    left = ~measured & ~est & np.isfinite(mag)
    unestimated = {
        str(t): int(c)
        for t, c in zip(*np.unique(mtype[left].astype(str), return_counts=True))
    }
    named = rel[np.array([v is not None for v in rel], dtype=bool)].astype(str)
    used = {str(t): int(c) for t, c in zip(*np.unique(named, return_counts=True))}
    record: dict[str, Any] = {
        "filter": "moment_estimate",
        "relations": {k: v.as_dict() for k, v in relations.items()},
        "mlw_relation": MLW_M0_FIT.as_dict() if mlw else None,
        "rows_by_relation": used,
        "unestimated_by_type": unestimated,
        "no_magnitude": int((~measured & ~np.isfinite(mag)).sum()),
        "n_in": n,
        "n_out": n,
    }
    attrs = copy.deepcopy(dict(df.attrs))
    cat = dict(attrs.get(ATTRS_KEY) or {})
    cat["subsets"] = [*list(cat.get("subsets") or []), record]
    attrs[ATTRS_KEY] = cat
    out.attrs = attrs
    return out


__all__ = [
    "DEFAULT_RELATIONS",
    "ESTIMATED_FLAG",
    "MEASURED",
    "RELATION_COLUMN",
    "SEISCOMP_ML_SIL",
    "SEISCOMP_MPGV_W",
    "SIL_MLW",
    "MomentRelation",
    "moment_estimate",
]
