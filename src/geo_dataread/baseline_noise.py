"""Baseline noise record: the fitted inputs for empirical baseline σ.

Formal GAMIT/GLOBK σ count the daily error two neighbouring stations SHARE
twice, so a short baseline's quadrature σ is several times its real
scatter. :mod:`gps_analysis.empirical_sigma` holds the math; this module is
its data side:

- :class:`BaselineNoiseRecord` — ONE JSON record: the distance law ρ(d)
  (``CommonModeModel``), each station's σ scale factor k, the network-median
  k fallback, the fit window and provenance, and the validation numbers.
- :func:`fit_baseline_noise` — build it from station series + positions.
- :func:`pair_inputs` — ``(rho, (k_a, k_b), how)`` for one baseline, ready
  for :func:`geo_dataread.baseline.baseline` ``rho=`` / ``sigma_scale=``.
  A pair with ≥ ``MIN_OWN_DAYS`` consecutive common days uses its OWN
  shared fraction (a distance law cannot know a pair's unshared error —
  VFLN–VFLS Up, 2026-10-09); otherwise ρ(d).
- :func:`read_baseline_noise` / :func:`write_baseline_noise`, and the
  ``gps-baseline-noise`` console script that refits from TOT.

Home: the precompute store, ``$GPS_API_STORE`` (default
``~/.cache/gps_analysis``) ``/noise/baseline_noise.json`` — NOT the config
deploy tree, which a deploy may overwrite (the 2026-08-27 secular-store
wipe). Refit periodically over the most recent ``window_years``: the
station scale drifts (Up 0.63 → 0.57 between 2023–24 and 2025–26).
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import tempfile
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from gps_analysis import empirical_sigma as es

__all__ = [
    "MIN_OWN_DAYS",
    "RECORD_VERSION",
    "BaselineNoiseRecord",
    "default_path",
    "fit_baseline_noise",
    "pair_inputs",
    "read_baseline_noise",
    "write_baseline_noise",
]

RECORD_VERSION = 1

#: A pair needs this many consecutive common days for its own ρ.
MIN_OWN_DAYS = es.MIN_PAIRS_DEFAULT

FloatArray = npt.NDArray[np.float64]
Series = tuple[
    npt.ArrayLike, npt.ArrayLike, npt.ArrayLike
]  # (yearf, data(C,N), sigma(C,N))


@dataclasses.dataclass(frozen=True)
class BaselineNoiseRecord:
    """Everything needed to give a baseline a realistic daily σ.

    Attributes:
        model: the fitted distance law (distance unit km).
        station_scale: per station, k per component (formal σ × k ≈ scatter).
        median_scale: network-median k, for stations without their own.
        fit_window: ``(start, end)`` fractional years the fit used.
        fitted_at: ISO-8601 UTC time of the fit.
        source: where the series came from (e.g. the TOT directory).
        n_stations / n_pairs: how much went into the fit.
        validation: free-form numbers (e.g. out-of-sample NRMS).
        notes: caveats the reader should see.
    """

    model: es.CommonModeModel
    station_scale: Mapping[str, tuple[float, ...]]
    median_scale: tuple[float, ...]
    fit_window: tuple[float, float]
    fitted_at: str
    source: str
    n_stations: int
    n_pairs: int
    validation: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def scale(self, station: str) -> tuple[FloatArray, bool]:
        """``(k, own)`` — the station's k, or the network median (own=False)."""
        k = self.station_scale.get(station.upper())
        if k is None:
            return np.asarray(self.median_scale, dtype=np.float64), False
        return np.asarray(k, dtype=np.float64), True

    def to_record(self) -> dict[str, Any]:
        return {
            "record_version": RECORD_VERSION,
            "model": self.model.to_record(),
            "station_scale": {
                s: list(k) for s, k in sorted(self.station_scale.items())
            },
            "median_scale": list(self.median_scale),
            "fit_window": list(self.fit_window),
            "fitted_at": self.fitted_at,
            "source": self.source,
            "n_stations": self.n_stations,
            "n_pairs": self.n_pairs,
            "validation": dict(self.validation),
            "notes": list(self.notes),
        }

    @classmethod
    def from_record(cls, rec: Mapping[str, Any]) -> BaselineNoiseRecord:
        if rec.get("record_version") != RECORD_VERSION:
            raise ValueError(
                f"baseline noise record version {rec.get('record_version')!r}, "
                f"expected {RECORD_VERSION}"
            )
        return cls(
            model=es.CommonModeModel.from_record(rec["model"]),
            station_scale={s: tuple(k) for s, k in rec["station_scale"].items()},
            median_scale=tuple(rec["median_scale"]),
            fit_window=(float(rec["fit_window"][0]), float(rec["fit_window"][1])),
            fitted_at=str(rec["fitted_at"]),
            source=str(rec["source"]),
            n_stations=int(rec["n_stations"]),
            n_pairs=int(rec["n_pairs"]),
            validation=dict(rec.get("validation", {})),
            notes=tuple(rec.get("notes", ())),
        )


def default_path() -> Path:
    """``$GPS_API_STORE/noise/baseline_noise.json`` (store default ``~/.cache/gps_analysis``)."""
    root = os.environ.get("GPS_API_STORE") or str(
        Path.home() / ".cache" / "gps_analysis"
    )
    return Path(root) / "noise" / "baseline_noise.json"


def write_baseline_noise(rec: BaselineNoiseRecord, path: Path | None = None) -> Path:
    """Write atomically (temp file + rename): a reader never sees half a record."""
    out = Path(path) if path is not None else default_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=out.parent, prefix=".baseline_noise.", suffix=".json"
    )
    with os.fdopen(fd, "w") as fh:
        json.dump(rec.to_record(), fh, indent=1)
        fh.write("\n")
    os.replace(tmp, out)
    return out


def read_baseline_noise(path: Path | None = None) -> BaselineNoiseRecord:
    """Read the record; ``FileNotFoundError`` when there is none yet."""
    p = Path(path) if path is not None else default_path()
    return BaselineNoiseRecord.from_record(json.loads(p.read_text()))


def _unique_days(series: Series) -> Series:
    """Sorted, one epoch per day (TOT can repeat a day; the first wins)."""
    t = np.asarray(series[0], dtype=np.float64)
    day = np.round(t * 365.25).astype(np.int64)
    order = np.argsort(t, kind="stable")
    _, first = np.unique(day[order], return_index=True)
    keep = order[first]
    return t[keep], np.asarray(series[1])[:, keep], np.asarray(series[2])[:, keep]


def _window(
    series: Series, window: tuple[float, float]
) -> tuple[FloatArray, FloatArray, FloatArray]:
    t = np.asarray(series[0], dtype=np.float64)
    y = np.asarray(series[1], dtype=np.float64)
    s = np.asarray(series[2], dtype=np.float64)
    m = (t >= window[0]) & (t < window[1])
    return t[m], y[:, m], s[:, m]


def _common(a: Series, b: Series) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Epoch-matched (t, y_a, y_b) by day (``baseline.match_epochs``)."""
    from .baseline import match_epochs

    ta = np.asarray(a[0], dtype=np.float64)
    ia, ib = match_epochs(ta, b[0])
    return (
        ta[ia],
        np.asarray(a[1], dtype=np.float64)[:, ia],
        np.asarray(b[1], dtype=np.float64)[:, ib],
    )


def own_shared_fraction(
    a: Series, b: Series, window: tuple[float, float] | None = None
) -> FloatArray:
    """The pair's own ρ per component from its common epochs (NaN if too few days)."""
    t, ya, yb = _common(a, b)
    if window is not None:
        m = (t >= window[0]) & (t < window[1])
        t, ya, yb = t[m], ya[:, m], yb[:, m]
    if t.size < 2:
        return np.full(np.asarray(a[1]).shape[0], np.nan)
    return es.shared_fraction(
        es.daily_scatter(t, ya), es.daily_scatter(t, yb), es.daily_scatter(t, ya - yb)
    )


def fit_baseline_noise(
    series: Mapping[str, Series],
    positions_km: Mapping[str, npt.ArrayLike],
    *,
    window: tuple[float, float],
    max_km: float = 50.0,
    source: str = "",
    notes: tuple[str, ...] = (),
) -> BaselineNoiseRecord:
    """Fit station scales and the ρ(d) law on ``window``.

    ``series[code] = (yearf, data(C,N), sigma(C,N))`` in one consistent
    unit; ``positions_km[code]`` = ECEF position in km (distances are
    Euclidean — chord, identical to geodesic at these ranges).
    """
    series = {c: _unique_days(s) for c, s in series.items()}
    k: dict[str, tuple[float, ...]] = {}
    for code, s in series.items():
        kk = es.sigma_scale_factor(*_window(s, window))
        if np.isfinite(kk).all():
            k[code.upper()] = tuple(float(v) for v in kk)
    if not k:
        raise ValueError("fit_baseline_noise: no station has enough data in the window")
    codes = sorted(c for c in series if c.upper() in k and c in positions_km)
    dist, rho = [], []
    for p, q in combinations(codes, 2):
        d = float(np.linalg.norm(np.subtract(positions_km[p], positions_km[q])))
        if d > max_km:
            continue
        r = own_shared_fraction(series[p], series[q], window)
        if np.isfinite(r).all():
            dist.append(d)
            rho.append(r)
    if len(dist) < 5:
        raise ValueError(
            f"fit_baseline_noise: only {len(dist)} usable pairs within {max_km} km"
        )
    model = es.fit_common_mode(np.array(dist), np.array(rho).T, distance_unit="km")
    return BaselineNoiseRecord(
        model=model,
        station_scale=k,
        median_scale=tuple(
            float(v) for v in np.median(np.array(list(k.values())), axis=0)
        ),
        fit_window=(float(window[0]), float(window[1])),
        fitted_at=datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        source=source,
        n_stations=len(k),
        n_pairs=len(dist),
        notes=notes,
    )


def pair_inputs(
    rec: BaselineNoiseRecord,
    a: str,
    b: str,
    *,
    distance_km: float,
    own_rho: npt.ArrayLike | None = None,
) -> tuple[FloatArray, tuple[FloatArray, FloatArray], str]:
    """``(rho, (k_a, k_b), how)`` for baseline ``a − b``.

    ``own_rho`` (from :func:`own_shared_fraction`, finite only with
    ≥ ``MIN_OWN_DAYS`` consecutive common days) wins over the distance
    law; ``how`` says which was used, for the plot/API provenance.
    """
    ka, own_a = rec.scale(a)
    kb, own_b = rec.scale(b)
    own = None if own_rho is None else np.asarray(own_rho, dtype=np.float64)
    if own is not None and np.isfinite(own).all():
        rho = np.clip(own, -1.0, 1.0)
        how = "own ρ"
    else:
        rho = es.shared_fraction_model(distance_km, rec.model)
        how = f"ρ(d = {distance_km:.3g} km)"
    if not (own_a and own_b):
        how += ", network-median k for " + ", ".join(
            s for s, o in ((a, own_a), (b, own_b)) if not o
        )
    return rho, (ka, kb), how


def main(argv: list[str] | None = None) -> int:
    """``gps-baseline-noise``: refit the record from TOT and write it."""
    import argparse

    ap = argparse.ArgumentParser(prog="gps-baseline-noise", description=main.__doc__)
    ap.add_argument("--dir", help="TOT directory (default: the configured one)")
    ap.add_argument("--window-years", type=float, default=2.0)
    ap.add_argument(
        "--end", type=float, help="window end, fractional year (default: now)"
    )
    ap.add_argument("--max-km", type=float, default=50.0)
    ap.add_argument(
        "--stations", nargs="*", help="default: every station in stations.cfg"
    )
    ap.add_argument("--out", type=Path, default=None, help=f"default {default_path()}")
    a = ap.parse_args(argv)

    import warnings

    import gps_parser as cp
    from geofunc import local

    from .gps_read import getData

    cfg = cp.ConfigParser()
    stations = [s.upper() for s in (a.stations or cfg.getStationInfo())]
    now = datetime.datetime.now(datetime.UTC)
    end = (
        a.end
        if a.end is not None
        else now.year + (now.timetuple().tm_yday - 0.5) / 365.25
    )
    window = (end - a.window_years, end)
    series: dict[str, Series] = {}
    pos: dict[str, FloatArray] = {}
    for s in stations:
        try:
            info = cfg.getStationInfo(s)["station"]  # inline comments stripped
            lat, lon = float(info["latitude"]), float(info["longitude"])
            h = float(info.get("height") or 0.0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t, y, sy, _ = getData(s, Dir=a.dir, tType="TOT")  # type: ignore[no-untyped-call]
        except Exception as exc:  # noqa: BLE001 - one bad station must not stop the fit
            print(f"  skip {s}: {exc}")
            continue
        series[s] = (np.asarray(t), np.asarray(y), np.asarray(sy))
        pos[s] = local.geodetic_to_ecef(np.radians(lat), np.radians(lon), h) / 1e3
    rec = fit_baseline_noise(
        series,
        pos,
        window=window,
        max_km=a.max_km,
        source=f"TOT via geo_dataread.getData (Dir={a.dir or 'configured'})",
        notes=(
            "Daily (white) sigma only; rate sigma needs the colored-noise model.",
            "Validated 2026-10-09 out of sample: NRMS 0.91/0.91/0.94 (N/E/U), "
            "slightly optimistic < 1 km (U 1.2).",
        ),
    )
    out = write_baseline_noise(rec, a.out)
    print(
        f"{rec.n_stations} stations, {rec.n_pairs} pairs, window "
        f"{window[0]:.2f}-{window[1]:.2f}; rho0 {np.round(rec.model.rho0, 2)}, "
        f"rho_inf {np.round(rec.model.rho_inf, 2)}, L {np.round(rec.model.length, 1)} km; "
        f"median k {np.round(rec.median_scale, 2)} -> {out}"
    )
    return 0
