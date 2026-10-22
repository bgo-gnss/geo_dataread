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
    COMMON_MODE_MIN_STATIONS,
    NETWORK_DOMINANT_SIGMA,
    NETWORK_MIN_STATIONS,
    NETWORK_PARTIAL_SHARE,
    NETWORK_SHARE,
    common_mode,
    day_scores,
    midpoint_residuals,
    network_days,
)

#: the ``.pos`` product the list is scanned from and read for, whatever frame
#: the caller holds: a bad day belongs to the processing chain, and okada's
#: itrf08/itrf14 share their solutions (gamit-p01's IGb20 will need its own)
NETWORK_FRAME = "itrf14"

#: the plotting box (gps_plot ICELAND_REGION): W, E, S, N
ICELAND_REGION = (-25.5, -12.5, 62.5, 67.5)

CM_COLUMNS = (
    "date",
    "cm_north",
    "cm_east",
    "cm_up",
    "n_north",
    "n_east",
    "n_up",
    "provisional",
)

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


def _read_series(
    sta: str,
    directory: str | Path,
    frame: str,
    drop_epochs: Sequence[float] = (),
) -> tuple[np.ndarray, np.ndarray] | None:
    """The station's full 24 h series (mm, plate), its icing and snow days
    and ``drop_epochs`` left out; None when unreadable or under 30 epochs."""
    from geo_dataread import gps_read
    from geo_dataread.gps_views import resolve_icing_epochs, resolve_snow_epochs
    from geo_dataread.pos import pos_product
    from gps_parser import outlier_catalogs as _oc

    try:
        yearf, data, _dd, _off = gps_read.getData(  # type: ignore[no-untyped-call]
            sta,
            ref="plate",
            Dir=str(directory),
            tType="POS",
            uncert=15,
            pos_id=pos_product(frame),
        )
    except Exception:  # noqa: BLE001 - a station that cannot be read: nothing
        return None
    t = np.asarray(yearf, dtype=np.float64)
    y = np.asarray(data, dtype=np.float64)
    keep = np.isfinite(t) & np.all(np.isfinite(y), axis=0)
    drops: list[tuple[float, ...]] = [tuple(drop_epochs)]
    for resolve in (resolve_icing_epochs, resolve_snow_epochs):
        drops.append(resolve(sta, directory)[0])
    for epochs in drops:
        if epochs:
            idx = _oc.match_declared_epochs(t, np.asarray(epochs))
            keep[list(idx)] = False
    t, y = t[keep], y[:, keep]
    return (t, y) if t.size >= 30 else None


def station_votes(
    sta: str, directory: str | Path, frame: str = NETWORK_FRAME
) -> tuple[np.ndarray, np.ndarray] | None:
    """One station's ``(days, scores)``: datetime64[D] (N,), scores (3, N).

    None when the station has no usable series (too short, unreadable).
    Icing and snow days are left out before detection, as in the cleaned
    view, so a buried antenna never votes.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        series = _read_series(sta, directory, frame)
        if series is None:
            return None
        t, y = series
        try:
            det = detect_outliers_running(t, y, params=OutlierParams())
        except Exception:  # noqa: BLE001 - detection failure: no votes
            return None
    return _days(t), day_scores(det)


def station_residuals(
    sta: str,
    directory: str | Path,
    frame: str = NETWORK_FRAME,
    network_epochs: Sequence[float] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """One station's ``(days, residuals, one_sided)`` for the common mode:
    midpoint residuals (3, N) [mm], the network days left out like the
    icing and snow days."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        series = _read_series(sta, directory, frame, network_epochs)
        if series is None:
            return None
        t, y = series
        r, one = midpoint_residuals(t, y)
    return _days(t), r, one


def _votes(args: tuple[str, str, str]) -> tuple[str, object]:
    sta, directory, frame = args
    return sta, station_votes(sta, directory, frame)


def _residuals(args: tuple[str, str, str, tuple[float, ...]]) -> tuple[str, object]:
    sta, directory, frame, epochs = args
    return sta, station_residuals(sta, directory, frame, epochs)


def active_area_stations(path: str | Path | None = None) -> frozenset[str]:
    """Every station of a ``volcanic_areas`` area in station_areas.yaml --
    the active deformation kept out of the common mode (none on a host
    without the file)."""
    import yaml

    from geo_dataread.events import (
        STATION_AREAS_FILENAME,
        STATION_AREAS_KEY,
        _gpsconfig_file,
    )

    resolved = (
        _gpsconfig_file(STATION_AREAS_KEY, STATION_AREAS_FILENAME)
        if path is None
        else Path(path)
    )
    if resolved is None or not resolved.is_file():
        return frozenset()
    doc = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    out: set[str] = set()
    for area in (doc.get("volcanic_areas") or {}).values():
        out |= {str(x).upper() for x in (area or {}).get("stations") or ()}
    return frozenset(out)


def common_mode_stations(
    directory: str | Path, frame: str = NETWORK_FRAME
) -> list[str]:
    """Stations the common mode is stacked from: in the Iceland box and in
    no volcanic area -- so a regional signal cannot leak into everyone."""
    from geo_dataread.pos import pos_product, pos_stations

    active = active_area_stations()
    return [
        s
        for s, _lon, _lat in pos_stations(pos_product(frame), directory, ICELAND_REGION)
        if s not in active
    ]


def scan_common_mode(
    stations: Sequence[str],
    directory: str | Path,
    *,
    network_epochs: Sequence[float] = (),
    frame: str = NETWORK_FRAME,
    workers: int | None = None,
) -> pd.DataFrame:
    """The daily network common mode (:data:`CM_COLUMNS`), one row per day
    with any residual; ``cm_*`` NaN where fewer than 15 stations voted."""
    jobs = [(s, str(directory), frame, tuple(network_epochs)) for s in stations]
    got: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _sta, res in pool.map(_residuals, jobs):
            if res is not None:
                got.append(res)  # type: ignore[arg-type]
    if not got:
        return pd.DataFrame(columns=list(CM_COLUMNS))
    axis = np.unique(np.concatenate([d for d, _r, _o in got]))
    r = np.full((len(got), 3, axis.size), np.nan)
    one = np.zeros(r.shape, dtype=bool)
    for i, (days, ri, oi) in enumerate(got):
        j = np.searchsorted(axis, days)
        r[i][:, j] = ri
        one[i][:, j] = oi
    cm = common_mode(r, one, min_stations=COMMON_MODE_MIN_STATIONS)
    return pd.DataFrame(
        {
            "date": axis,
            "cm_north": np.round(cm.value[0], 2),
            "cm_east": np.round(cm.value[1], 2),
            "cm_up": np.round(cm.value[2], 2),
            "n_north": cm.n[0],
            "n_east": cm.n[1],
            "n_up": cm.n[2],
            "provisional": cm.provisional.astype(int),
        }
    )


def common_mode_path(directory: str | Path, frame: str = NETWORK_FRAME) -> Path:
    """``<directory>/network_cm.<frame>.csv``."""
    return Path(directory) / f"network_cm.{frame}.csv"


def write_common_mode(
    table: pd.DataFrame,
    path: str | Path,
    *,
    frame: str = NETWORK_FRAME,
    n_stations: int = 0,
) -> Path:
    """Write the daily common mode WHOLE (atomic), with a provenance header."""
    header = (
        f"# gps-network-days common mode frame={frame} "
        f"created={dt.datetime.now(dt.UTC):%Y-%m-%dT%H:%M:%SZ} "
        f"stations={n_stations} min_stations={COMMON_MODE_MIN_STATIONS} "
        "residual=midpoint-of-one-sided-lines stack=median unit=mm\n"
    )
    out = table.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return _atomic_csv(out, Path(path), header, CM_COLUMNS)


def read_common_mode(path: str | Path) -> pd.DataFrame:
    """The table written by :func:`write_common_mode` (``date`` parsed)."""
    df = pd.read_csv(path, comment="#")
    df["date"] = pd.to_datetime(df["date"])
    return df


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
    return _atomic_csv(out, path, header, COLUMNS)


def _atomic_csv(
    df: pd.DataFrame, path: Path, header: str, columns: Sequence[str]
) -> Path:
    """Header + CSV through a temp file and an atomic rename."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(header)
            df.to_csv(fh, index=False, columns=list(columns))
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
    ap.add_argument(
        "--no-common-mode",
        action="store_true",
        help="skip the daily network common mode (network_cm.<frame>.csv)",
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
    if a.no_common_mode:
        return 0
    # second pass: the common mode, on the series WITHOUT the decided days
    epochs = tuple(float(v) for v in noon_yearf(pd.DatetimeIndex(decided["date"])))
    cm_stations = common_mode_stations(directory, a.frame)
    cm = scan_common_mode(
        cm_stations, directory, network_epochs=epochs, frame=a.frame, workers=a.workers
    )
    cm_out = common_mode_path(out.parent, a.frame)
    write_common_mode(cm, cm_out, frame=a.frame, n_stations=len(cm_stations))
    have = cm[["cm_north", "cm_east", "cm_up"]].notna().all(axis=1)
    print(
        f"common mode ({a.frame}): {int(have.sum())} days from "
        f"{len(cm_stations)} stations outside the volcanic areas; wrote {cm_out}"
    )
    return 0
