"""The series product: a station's time series as the products see it, to file.

``source="pos"`` (the default): the ``.pos`` series of a frame and session
through :func:`geo_dataread.products.loader.load` — the reference (plate and
the stored seasonal removed by default), the view (cleaned by default: the
running-window outliers out), the σ screen — the same epochs every other
gpsprod product uses. Kept epochs only, or ``with_dropped`` all of them with
the raw values and why each was dropped or marked (outlier, declared, icing,
snow, network). Files: CSV (``#`` provenance header, one row per epoch),
JSONL (one epoch per line), or ``.NEU`` in exactly the layout
``gps-savetimes`` writes (``gps_read.gtoNEU`` + ``gamittoFile``), with a
``.prov.json`` beside it.

``source="tot"``: the legacy ITRF2008 TOT series through
``gps_read.gamittooneuf`` — byte for byte what ``gps-savetimes`` writes
(``ref`` plate | detrend | itrf2008 there).
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from geo_dataread.products.common import COMPONENTS, SCHEMA_VERSION, jsonable, software
from geo_dataread.products.loader import (
    DAY_YEARS,
    REASON_NAMES,
    Source,
    load,
)

FloatArray = npt.NDArray[np.float64]
SOURCES = ("pos", "tot")
FORMATS = ("csv", "jsonl", "neu", "xlsx")
#: --source tot: the gps-savetimes references
TOT_REFS = {"plate": "plate", "background": "detrend", "itrf": "itrf2008"}
#: the default period of a whole series
ALL_TIME = (1990.0, 2100.0)


@dataclasses.dataclass(frozen=True)
class SeriesSettings:
    """How a set of series is exported."""

    source: Source = Source(view="cleaned", uncert=10.0)
    with_dropped: bool = False

    def __post_init__(self) -> None:
        if self.source.relative_to and self.with_dropped:
            raise ValueError("with_dropped is not kept for a baseline (--relative-to)")


@dataclasses.dataclass(frozen=True)
class SeriesRecord:
    """One station's series [mm]. ``t`` decimal years; ``y``/``sigma``
    (3, N) N/E/U; ``kept`` (N,) whether the view kept the epoch (all three
    components); ``reasons`` (3, N) the drop/mark bits
    (:data:`~geo_dataread.products.loader.REASON_NAMES`)."""

    station: str
    frame: str
    series: str
    session: str
    ref: str
    ref_requested: str
    view: str
    period: tuple[float, float]
    period_dates: tuple[str, str]
    t: FloatArray
    y: FloatArray
    sigma: FloatArray
    kept: npt.NDArray[np.bool_]
    reasons: npt.NDArray[np.uint8] | None = None
    uncert: float = math.nan
    relative_to: str | None = None
    flags: tuple[str, ...] = ()
    refused: str | None = None
    software: dict[str, str] = dataclasses.field(default_factory=dict)
    units: str = "mm"
    schema_version: int = SCHEMA_VERSION
    kind: str = "series"

    @property
    def accepted(self) -> bool:
        return self.refused is None

    @property
    def n_kept(self) -> int:
        return int(self.kept.sum())

    def provenance(self) -> dict[str, Any]:
        """Everything but the arrays (the header / sidecar)."""
        d = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in ("t", "y", "sigma", "kept", "reasons")
        }
        d["n_epochs"] = int(self.t.size)
        d["n_kept"] = self.n_kept
        out: dict[str, Any] = jsonable(d)
        return out

    def scaled(self, units: str) -> SeriesRecord:
        if units == self.units:
            return self
        f = {("mm", "m"): 1e-3, ("m", "mm"): 1e3}[(self.units, units)]
        return dataclasses.replace(
            self, y=self.y * f, sigma=self.sigma * f, units=units
        )


def reason_names(bits: int) -> str:
    return "|".join(name for bit, name in REASON_NAMES.items() if bits & bit)


def _iso(t: float) -> str:
    from gtimes.timefunc import TimefromYearf

    return str(TimefromYearf(t + 1e-6, "%Y-%m-%d"))


def station_series(
    sta: str,
    period: tuple[float, float] | None,
    settings: SeriesSettings,
    *,
    doc: Any = None,
    period_dates: tuple[str, str] | None = None,
) -> SeriesRecord:
    """One station's series (``source="pos"``). Never raises for a
    station-level reason: the record comes back ``refused``."""
    src = settings.source
    sta = sta.upper()
    p0, p1 = period or ALL_TIME
    empty = np.zeros(0)
    base: dict[str, Any] = {
        "station": sta, "frame": src.frame_name, "series": src.pos_id,
        "session": src.session, "ref_requested": src.ref, "view": src.view,
        "period": (p0, p1),
        "period_dates": period_dates or (("all", "all") if period is None
                                         else (_iso(p0), _iso(p1 - 0.5 * DAY_YEARS))),
        "uncert": src.uncert,
        "relative_to": src.relative_to.upper() if src.relative_to else None,
        "software": software(),
    }  # fmt: skip
    if doc is None:
        doc = src.params_doc()
    try:
        view = load(sta, p0, p1, src, doc=doc)
    except (OSError, ValueError, KeyError) as err:
        return SeriesRecord(
            **base, ref=src.ref, t=empty, y=np.zeros((3, 0)), sigma=np.zeros((3, 0)),
            kept=np.zeros(0, bool), refused=f"read: {err}",
        )  # fmt: skip
    view = view.sliced(p0, p1)
    kept = np.isfinite(view.y).all(axis=0)
    if settings.with_dropped and view.y_raw is not None:
        y = view.y_raw
        t, sig, keep_rows = view.t, view.sigma, np.ones(view.t.size, bool)
    else:
        y, t, sig, keep_rows = view.y, view.t, view.sigma, kept
    reasons = None if view.reasons is None else view.reasons[:, keep_rows]
    return SeriesRecord(
        **base, ref=view.ref, t=t[keep_rows], y=y[:, keep_rows],
        sigma=sig[:, keep_rows], kept=kept[keep_rows], reasons=reasons,
        flags=view.flags,
    )  # fmt: skip


def estimate_series(
    stations: Sequence[str],
    period: tuple[float, float] | None,
    settings: SeriesSettings | None = None,
    *,
    period_dates: tuple[str, str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[SeriesRecord]:
    """Every station's series, in station order."""
    settings = settings or SeriesSettings()
    doc = settings.source.params_doc()
    out = []
    for sta in sorted({s.upper() for s in stations}):
        if progress is not None:
            progress(sta)
        out.append(
            station_series(sta, period, settings, doc=doc, period_dates=period_dates)
        )
    return out


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------

SERIES_COLUMNS = ("date", "yearf", "dN", "dE", "dU", "sN", "sE", "sU", "kept", "flags")


def _fmt(v: float, decimals: int) -> str:
    return "" if not math.isfinite(v) else f"{v:.{decimals}f}"


def series_rows(rec: SeriesRecord) -> list[list[str]]:
    """One text row per epoch (:data:`SERIES_COLUMNS`)."""
    from gtimes.timefunc import TimefromYearf

    dec = 2 if rec.units == "mm" else 5
    out = []
    for j in range(rec.t.size):
        bits = (
            0 if rec.reasons is None else int(np.bitwise_or.reduce(rec.reasons[:, j]))
        )
        out.append(
            [
                str(TimefromYearf(float(rec.t[j]), "%Y-%m-%d %H:%M:%S")),
                f"{rec.t[j]:.6f}",
                *(_fmt(float(v), dec) for v in rec.y[:, j]),
                *(_fmt(float(v), dec) for v in rec.sigma[:, j]),
                "1" if rec.kept[j] else "0",
                reason_names(bits),
            ]
        )
    return out


def header(rec: SeriesRecord) -> list[str]:
    lines = [f"gpsprod series schema {SCHEMA_VERSION}"]
    for key in ("station", "frame", "series", "session", "ref", "view", "uncert",
                "relative_to", "units"):  # fmt: skip
        lines.append(
            f"{key} {getattr(rec, key) if getattr(rec, key) is not None else '-'}"
        )
    lines.append("period {}..{}".format(*rec.period_dates))
    lines.append(f"epochs {rec.t.size} kept {rec.n_kept}")
    if rec.flags:
        lines.append("flags " + ";".join(rec.flags))
    return lines


def write_series(
    rec: SeriesRecord, path: str | Path, fmt: str, *, time_format: str | None = None
) -> list[Path]:
    """Write one station's series; returns the files written (``.neu``
    also writes ``<path>.prov.json``)."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        lines = [f"# {x}" for x in header(rec)] + [",".join(SERIES_COLUMNS)]
        lines += [",".join(r) for r in series_rows(rec)]
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return [target]
    if fmt == "jsonl":
        prov = rec.provenance()
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"provenance": prov}, sort_keys=True, ensure_ascii=False)
                + "\n"
            )
            for row in series_rows(rec):
                fh.write(
                    json.dumps(
                        dict(zip(SERIES_COLUMNS, row, strict=True)), sort_keys=True
                    )
                    + "\n"
                )
        return [target]
    if fmt == "neu":
        write_neu(rec, target, time_format=time_format)
        side = target.with_name(target.name + ".prov.json")
        side.write_text(
            json.dumps(rec.provenance(), indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return [target, side]
    if fmt == "xlsx":
        from geo_dataread.products.xlsx import write_xlsx

        meta = [x.split(" ", 1) if " " in x else [x, ""] for x in header(rec)]
        write_xlsx(
            target,
            [("data", [list(SERIES_COLUMNS), *series_rows(rec)]),
             ("meta", [["key", "value"], *meta])],
        )  # fmt: skip
        return [target]
    raise ValueError(f"format {fmt!r}: one of {FORMATS}")


def write_neu(
    rec: SeriesRecord, path: str | Path, *, time_format: str | None = None
) -> None:
    """The kept epochs in gps-savetimes' ``.NEU`` layout (``gtoNEU`` +
    ``gamittoFile``, the time of day at noon as ``rhour``)."""
    from geo_dataread import gps_read

    keep = rec.kept
    neu = gps_read.gtoNEU(  # type: ignore[no-untyped-call]
        rec.t[keep], rec.y[:, keep], rec.sigma[:, keep], dstring=time_format, rhour=True
    )
    gps_read.gamittoFile(  # type: ignore[no-untyped-call]
        neu, str(path), mm=rec.units == "mm", ref=rec.ref, dstring=time_format
    )


def write_tot(
    sta: str,
    path: str | Path | None,
    *,
    ref: str,
    mm: bool = True,
    time_format: str | None = None,
) -> None:
    """``source="tot"``: exactly gps-savetimes' file (``gamittooneuf``);
    ``path`` None prints to stdout."""
    import sys

    from geo_dataread import gps_read

    out: Any = sys.stdout if path is None else str(Path(path).expanduser())
    gps_read.gamittooneuf(  # type: ignore[no-untyped-call]
        sta, out, mm=mm, ref=ref, dstring=time_format, rhour=True
    )


__all__ = [
    "COMPONENTS", "FORMATS", "SERIES_COLUMNS", "SOURCES", "TOT_REFS",
    "SeriesRecord", "SeriesSettings", "estimate_series", "header", "series_rows",
    "station_series", "write_neu", "write_series", "write_tot",
]  # fmt: skip
