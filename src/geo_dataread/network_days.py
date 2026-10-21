"""Network days: whole-solution bad days, scanned over every station's series.

A bad network solution day (frame realisation, orbits, a failed adjustment)
moves every station at once; the leaf (:mod:`gps_analysis.network`) decides a
day from every station's two-sided running verdicts. This module is the I/O
around it:

- :func:`scan` reads each station's full 24 h ``.pos`` series of
  :data:`NETWORK_FRAME`, leaves out the station's icing and snow days (as the
  cleaned view does before detection), runs the default running detector and
  aligns the votes on one calendar-day axis;
- :func:`write_network_days` writes ``network_days.<frame>.csv`` into the
  ``.pos`` directory — regenerated WHOLE each run, never appended: every day
  with a dominant vote, its counts per component and its status
  (``decided`` / ``partial`` / empty), so the cut can be redrawn without a
  rescan;
- :func:`network_epochs` gives the decided days as noon fractional years for
  ``gps_views.resolve_network_epochs`` and every cleaned-view consumer.

The scan is defined on detection WITHOUT the network mask (masking a day
changes its neighbours' side lines, not their votes on that day).

CLI: ``gps-network-days`` (run before the plots, after the ``.pos`` update).
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import tempfile
import warnings
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from gps_analysis import OutlierParams, detect_outliers_running
from gps_analysis.network import (
    NETWORK_DOMINANT_SIGMA,
    NETWORK_MIN_STATIONS,
    NETWORK_PARTIAL_SHARE,
    NETWORK_SHARE,
    day_scores,
    network_days,
)

#: the ``.pos`` product the list is scanned from and read for, whatever frame
#: the caller holds: a bad day belongs to the processing chain, and okada's
#: itrf08/itrf14 share their solutions (gamit-p01's IGb20 will need its own)
NETWORK_FRAME = "itrf14"

COLUMNS = (
    "date",
    "status",
    "share",
    "n_bad",
    "n_voting",
    "n_bad_north",
    "n_bad_east",
    "n_bad_up",
)


def network_days_path(directory: str | Path, frame: str = NETWORK_FRAME) -> Path:
    """``<directory>/network_days.<frame>.csv``."""
    return Path(directory) / f"network_days.{frame}.csv"


def _default_pos_dir() -> Path:
    from gps_parser import ConfigParser

    return Path(str(ConfigParser().getPostProcessDir("pos_dir")))


def _days(yearf: np.ndarray) -> np.ndarray:
    """Calendar days (datetime64[D]) of fractional-year epochs (doy/365|366)."""
    year = np.floor(yearf).astype(int)
    length = np.where(
        (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0)), 366, 365
    )
    doy = np.floor((yearf - year) * length).astype(int)  # 0-based
    base = (year - 1970).astype("datetime64[Y]").astype("datetime64[D]")
    return np.asarray(base + doy.astype("timedelta64[D]"), dtype="datetime64[D]")


def noon_yearf(days: Sequence[dt.date] | pd.DatetimeIndex) -> np.ndarray:
    """Noon fractional years of calendar days (the icing-day convention)."""
    out = []
    for d in pd.DatetimeIndex(days):
        length = 366 if d.is_leap_year else 365
        out.append(d.year + (d.dayofyear - 0.5) / length)
    return np.asarray(out, dtype=np.float64)


def station_votes(
    sta: str, directory: str | Path, frame: str = NETWORK_FRAME
) -> tuple[np.ndarray, np.ndarray] | None:
    """One station's ``(days, scores)``: datetime64[D] (N,), scores (3, N).

    None when the station has no usable series (too short, unreadable).
    Icing and snow days are left out before detection, as in the cleaned
    view, so a buried antenna never votes.
    """
    from geo_dataread import gps_read
    from geo_dataread.gps_views import resolve_icing_epochs, resolve_snow_epochs
    from geo_dataread.pos import pos_product
    from gps_parser import outlier_catalogs as _oc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            yearf, data, _dd, _off = gps_read.getData(  # type: ignore[no-untyped-call]
                sta,
                ref="plate",
                Dir=str(directory),
                tType="POS",
                uncert=15,
                pos_id=pos_product(frame),
            )
        except Exception:  # noqa: BLE001 - a station that cannot be read: no votes
            return None
        t = np.asarray(yearf, dtype=np.float64)
        y = np.asarray(data, dtype=np.float64)
        keep = np.isfinite(t) & np.all(np.isfinite(y), axis=0)
        for resolve in (resolve_icing_epochs, resolve_snow_epochs):
            epochs, _src = resolve(sta, directory)
            if epochs:
                idx = _oc.match_declared_epochs(t, np.asarray(epochs))
                keep[list(idx)] = False
        t, y = t[keep], y[:, keep]
        if t.size < 30:
            return None
        try:
            det = detect_outliers_running(t, y, params=OutlierParams())
        except Exception:  # noqa: BLE001 - detection failure: no votes
            return None
    return _days(t), day_scores(det)


def _votes(args: tuple[str, str, str]) -> tuple[str, object]:
    sta, directory, frame = args
    return sta, station_votes(sta, directory, frame)


def scan(
    stations: Sequence[str],
    directory: str | Path,
    *,
    frame: str = NETWORK_FRAME,
    workers: int | None = None,
) -> tuple[pd.DataFrame, int]:
    """Every station's votes → the per-day table (all days with a vote).

    Returns ``(table, n_stations_used)``; ``table`` has :data:`COLUMNS` with
    ``date`` as ``datetime64``, one row per day with at least one dominant
    vote or a status.
    """
    jobs = [(s, str(directory), frame) for s in stations]
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for sta, res in pool.map(_votes, jobs):
            if res is not None:
                results[sta] = res  # type: ignore[assignment]
    if not results:
        return pd.DataFrame(columns=list(COLUMNS)), 0
    axis = np.unique(np.concatenate([d for d, _ in results.values()]))
    scores = np.full((len(results), 3, axis.size), np.nan, dtype=np.float64)
    for i, (days, sc) in enumerate(results.values()):
        # one epoch per day in a 24 h product; keep the last if duplicated
        scores[i][:, np.searchsorted(axis, days)] = sc
    nd = network_days(scores)
    status = np.where(nd.decided, "decided", np.where(nd.partial, "partial", ""))
    rows = (nd.n_bad > 0) | (status != "")
    table = pd.DataFrame(
        {
            "date": axis[rows],
            "status": status[rows],
            "share": np.round(nd.share[rows], 3),
            "n_bad": nd.n_bad[rows],
            "n_voting": nd.n_voting[rows],
            "n_bad_north": nd.n_bad_component[0][rows],
            "n_bad_east": nd.n_bad_component[1][rows],
            "n_bad_up": nd.n_bad_component[2][rows],
        }
    )
    return table, len(results)


def write_network_days(
    table: pd.DataFrame,
    path: str | Path,
    *,
    frame: str = NETWORK_FRAME,
    n_stations: int = 0,
) -> Path:
    """Write the table WHOLE (temp file + atomic rename), with a provenance
    header: frame, creation time, levels and station count."""
    path = Path(path)
    header = (
        f"# gps-network-days frame={frame} "
        f"created={dt.datetime.now(dt.UTC):%Y-%m-%dT%H:%M:%SZ} "
        f"stations={n_stations} dominant={NETWORK_DOMINANT_SIGMA} "
        f"share={NETWORK_SHARE} partial_share={NETWORK_PARTIAL_SHARE} "
        f"min_stations={NETWORK_MIN_STATIONS}\n"
    )
    out = table.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(header)
            out.to_csv(fh, index=False, columns=list(COLUMNS))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def read_network_days(path: str | Path) -> pd.DataFrame:
    """The table written by :func:`write_network_days` (``date`` parsed,
    ``status`` "" for unlabelled days)."""
    df = pd.read_csv(path, comment="#", dtype={"status": str})
    df["status"] = df["status"].fillna("")
    df["date"] = pd.to_datetime(df["date"])
    return df


def network_epochs(
    path: str | Path, statuses: Sequence[str] = ("decided",)
) -> tuple[float, ...]:
    """Noon fractional years of the days with one of ``statuses``."""
    df = read_network_days(path)
    days = df.loc[df["status"].isin(list(statuses)), "date"]
    return tuple(float(v) for v in noon_yearf(pd.DatetimeIndex(days)))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="gps-network-days",
        description="Scan every station's 24 h .pos series for whole-network bad "
        "solution days and write network_days.<frame>.csv into the .pos "
        "directory (regenerated whole). Run after the .pos update, before plots.",
    )
    ap.add_argument("--pos-dir", default=None, help="default: postprocess pos_dir")
    ap.add_argument("--frame", default=NETWORK_FRAME)
    ap.add_argument("--stations", default=None, help="comma list (default: all)")
    ap.add_argument(
        "--out", default=None, help="default: <pos-dir>/network_days.<frame>.csv"
    )
    ap.add_argument("-j", "--workers", type=int, default=None)
    ap.add_argument(
        "--dry-run", action="store_true", help="print the days, write nothing"
    )
    a = ap.parse_args(argv)
    from geo_dataread.pos import pos_product, pos_stations

    directory = Path(a.pos_dir) if a.pos_dir else _default_pos_dir()
    stations = (
        [s.strip().upper() for s in a.stations.split(",") if s.strip()]
        if a.stations
        else [s for s, _lon, _lat in pos_stations(pos_product(a.frame), directory)]
    )
    table, used = scan(stations, directory, frame=a.frame, workers=a.workers)
    decided = table[table["status"] == "decided"]
    partial = table[table["status"] == "partial"]
    for _, r in pd.concat([decided, partial]).sort_values("date").iterrows():
        print(
            f"{r['date']:%Y-%m-%d} {r['status']:8s} {r['share']:5.0%} "
            f"({r['n_bad']}/{r['n_voting']}; N{r['n_bad_north']} "
            f"E{r['n_bad_east']} U{r['n_bad_up']})"
        )
    print(
        f"network days ({a.frame}): {len(decided)} decided, {len(partial)} partial, "
        f"{used} of {len(stations)} station(s) voting"
    )
    if a.dry_run:
        return 0
    out = Path(a.out) if a.out else network_days_path(directory, a.frame)
    write_network_days(table, out, frame=a.frame, n_stations=used)
    print(f"wrote {out}")
    return 0
