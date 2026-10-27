"""The secular rate of a station's stored detrend record (plate removed).

What a velocity map draws when no window is given: the rate the station's
curated record holds (``detrend_params.json``), with the record's formal σ.
Moved from gps_plot's ``sources/velocity.record_velocity`` (gpsprod step 6) so
the rate comes from the products like every other number.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclasses.dataclass(frozen=True)
class SecularRate:
    """N/E/U rate [mm/yr] and its formal 1-σ; ``n`` epochs the record fitted."""

    station: str
    rate: FloatArray
    sigma: FloatArray
    n: int
    window: tuple[float, float] | None = None
    sigma_method: str = "formal (record)"


def secular_rate(sta: str, record: Mapping[str, Any] | None) -> SecularRate:
    """The secular rate of ``record`` (the station's stored detrend record).

    Raises:
        ValueError: no record, a borrowed secular (the donor's rate, not this
            station's), no rate parameter, or components other than N/E/U.
    """
    from gps_analysis import trajectory_from_record

    if record is None:
        raise ValueError("no stored detrend record")
    if record.get("borrowed"):
        # which groups were borrowed is not checked: refuse rather than
        # give a donor's rate as this station's
        raise ValueError(f"record borrowed from {record['borrowed']}")
    names = list(record.get("param_names") or ())
    if "rate" not in names:
        raise ValueError(f"record has no secular rate (params {names})")
    k = names.index("rate")
    _model, fits = trajectory_from_record(record)
    by_name = {f.component: f for f in fits}
    if set(by_name) != {"north", "east", "up"}:
        raise ValueError(f"record components {sorted(map(str, by_name))}, not N/E/U")
    rows = [by_name[c] for c in ("north", "east", "up")]
    window = record.get("window")
    return SecularRate(
        station=sta.upper(),
        rate=np.array([float(f.params[k]) for f in rows]),
        sigma=np.array([float(f.uncertainties[k]) for f in rows]),
        n=int(record.get("n_epochs") or 0),
        window=(float(window[0]), float(window[1])) if window else None,
    )
