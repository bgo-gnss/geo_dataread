"""gpsprod — one place, one estimator for station motion (PLAN-gpsprod.md).

    from geo_dataread import products as gprod

    src = gprod.Source(frame="itrf14", view="declared")
    period = gprod.resolve_period(since="20231110")
    d = gprod.estimate_displacements(["SENG", "THOB"], period.bounds,
                                     gprod.DisplacementSettings(source=src))
    d.to_frame(); d.write("d.csv")

The maths is gps_analysis's (:mod:`gps_analysis.displacement`); this package
reads the series (:func:`load`), runs it, and writes self-describing records.
"""

from geo_dataread.products.common import (
    SCHEMA_VERSION,
    DisplacementRecord,
    RecordSet,
)
from geo_dataread.products.displacement import (
    DisplacementSettings,
    displacement_from_view,
    estimate_displacements,
    estimate_station_displacement,
)
from geo_dataread.products.loader import Source, StationView, load
from geo_dataread.products.period import Period, resolve_period
from geo_dataread.products.secular import SecularRate, secular_rate

__all__ = [
    "SCHEMA_VERSION",
    "DisplacementRecord",
    "DisplacementSettings",
    "Period",
    "RecordSet",
    "SecularRate",
    "Source",
    "StationView",
    "displacement_from_view",
    "estimate_displacements",
    "estimate_station_displacement",
    "load",
    "resolve_period",
    "secular_rate",
]
