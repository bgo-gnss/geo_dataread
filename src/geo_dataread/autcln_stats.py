"""GAMIT autcln per-station daily residual statistics, and the icing index.

GAMIT's ``autcln.post.sum`` (one per network and day) summarises each
station's post-fit one-way LC phase residuals: the RMS, the number of
observations, the elevation-dependent noise model ``RMS² = A² + B²/sin²(el)``
(the ``ATELV`` line — ×1.7 it is the q-file "elev" row that
``sh_sigelv`` feeds SOLVE), and the mean residual per elevation and per
azimuth bin. Snow and rime on a radome raise all of them at that station
and nowhere else; the azimuth mean reacts most (snow lies unevenly). Measured
2026-10-19 against HOFN (flat all year): AUST winter/summer azimuth mean
3.6/1.9 mm, GFUM episodes 5-24 mm against 1-2 (vault note
``check-station-for-suspect-icing-signal``; Larson 2013 JGR 118:4503 for the
SNR route; Kocharski 2015 MIT for multipath at rimed Alaska sites).

Two halves:

- **harvest** (``gps-autcln-harvest``): walks okada's GAMIT day directories
  (``WEEKS/<gpsweek>/<net>/<doy>/autcln.post.sum.gz``, read-only) and keeps one
  CSV per station, ``<SITE>.autcln.24hr.csv``, in the ``.pos`` directory —
  the INDICATORS, never flags, so a threshold can change without
  re-harvesting. Incremental: ``.autcln.harvested`` lists every file taken
  (path, size, mtime); a re-made day replaces its row.
- **icing index** (:func:`icing_days`): per day, the azimuth mean divided by
  the network's median that day (``network.autcln.24hr.csv``; network-wide
  ionosphere days cancel), against the station's own rolling low quantile of
  that ratio (no calendar baseline — GFUM ices in May and September too),
  flagged above 2.0×; or the ``sh_sigelv`` clip (A + B > 50 → SOLVE gets
  85/0).

The statistics come from okada's 24 h GAMIT networks (ice1-ice6); they are a
proxy for any frame's 24 h product and say nothing per 8 h session.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import math
import os
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

AUTCLN_FILE = "autcln.post.sum.gz"
STATS_SUFFIX = ".autcln.24hr.csv"
STATE_FILE = ".autcln.harvested"
#: the network's daily median azimuth mean (written by every harvest)
NETWORK_FILE = "network" + STATS_SUFFIX
DEFAULT_WEEKS_DIR = "/mnt_data/okada/WEEKS"
DEFAULT_NETWORKS = ("ice1", "ice2", "ice3", "ice4", "ice5", "ice6")
#: the ``sh_sigelv`` clip: A + B above this and SOLVE gets 85/0 (worst days)
SIGELV_CLIP_SUM = 50.0
#: an ELMEAN/AZMEAN/ATELV bin with no data
_NO_DATA = 99.0
_GPS_EPOCH = dt.date(1980, 1, 6)


@dataclass(frozen=True)
class SiteStats:
    """One station's autcln post-fit statistics for one network-day [mm]."""

    rms: float = math.nan  # RMS of all one-way LC residuals
    num: float = math.nan  # number of residuals
    ams: float = math.nan  # RMS of 25-point averages
    ams_ratio: float = math.nan  # autcln: ~5 white; near 1 = long-period
    atelv_a: float = math.nan  # elevation model RMS² = A² + B²/sin²(el)
    atelv_b: float = math.nan
    atelv_low: float = math.nan  # RMS in the lowest populated elevation bin
    az_absmean: float = math.nan  # mean |mean residual| over azimuth bins
    el_absmean: float = math.nan  # mean |mean residual| over elevation bins


STAT_FIELDS = tuple(f.name for f in fields(SiteStats))


def _num(token: str) -> float:
    try:
        return float(token)
    except ValueError:
        return math.nan


def _absmean(tokens: Sequence[str]) -> float:
    vals = [abs(v) for v in map(_num, tokens) if math.isfinite(v) and v < _NO_DATA]
    return statistics.fmean(vals) if vals else math.nan


def _unglue(p: list[str]) -> list[str]:
    """Split a value the fixed-width format ran into the 4-char site code.

    autcln prints ``RMS  20 SAFH1001.0`` when the RMS reaches 1000 mm (and
    ``ATELV SAFH120.5 …`` for a wide A): the site is always 4 characters.
    """
    site_at = 2 if p and p[0] in ("RMS", "NUM", "AMS") else 1
    if len(p) > site_at and len(p[site_at]) > 4:
        tok = p[site_at]
        head, tail = tok[:4], tok[4:]
        if head.isalnum() and (tail[0].isdigit() or tail[0] in "-."):
            return [*p[:site_at], head, tail, *p[site_at + 1 :]]
    return p


def parse_autcln_post_sum(lines: Iterable[str]) -> dict[str, SiteStats]:
    """Per-station statistics from the text of one ``autcln.post.sum``.

    Lines read (``~/gg/help/autcln.hlp``; ``kf/ctogobs/proc_phsfin.f``):
    ``RMS IT Site All …``, ``NUM IT Site All …``, ``AMS IT Site rms ratio …``,
    ``ATELV Site A B <5° bins>``, ``AZMEAN Site <bins>``, ``ELMEAN Site
    <bins>``; 99.9 marks an empty bin. A later iteration's line replaces an
    earlier one.
    """
    acc: dict[str, dict[str, float]] = {}
    for line in lines:
        p = _unglue(line.split())
        if len(p) < 3:
            continue
        tag = p[0]
        if tag in ("RMS", "NUM", "AMS") and len(p) >= 4 and p[1].isdigit():
            site = p[2]
            row = acc.setdefault(site, {})
            if tag == "RMS":
                row["rms"] = _num(p[3])
            elif tag == "NUM":
                row["num"] = _num(p[3])
            else:
                row["ams"] = _num(p[3])
                row["ams_ratio"] = _num(p[4]) if len(p) > 4 else math.nan
        elif tag == "ATELV" and len(p) >= 4 and p[1] != "Site":
            row = acc.setdefault(p[1], {})
            row["atelv_a"], row["atelv_b"] = _num(p[2]), _num(p[3])
            bins = [v for v in map(_num, p[4:]) if 0.0 < v < _NO_DATA]
            row["atelv_low"] = bins[0] if bins else math.nan
        elif tag in ("AZMEAN", "ELMEAN") and p[1] != "Site":
            key = "az_absmean" if tag == "AZMEAN" else "el_absmean"
            acc.setdefault(p[1], {})[key] = _absmean(p[2:])
    # "ALL" is autcln's every-site summary row, not a station
    return {site: SiteStats(**vals) for site, vals in acc.items() if site != "ALL"}


def date_of(gps_week: int, doy: int) -> dt.date:
    """The calendar day of day-of-year ``doy`` in the year GPS week ``gps_week``
    falls in (the week dir may straddle a new year)."""
    start = _GPS_EPOCH + dt.timedelta(weeks=gps_week)
    for year in (start.year, start.year + 1, start.year - 1):
        day = dt.date(year, 1, 1) + dt.timedelta(days=doy - 1)
        if -1 <= (day - start).days <= 7:
            return day
    raise ValueError(f"doy {doy} is not in GPS week {gps_week}")


# ---------------------------------------------------------------------------
# harvest
# ---------------------------------------------------------------------------


def stats_path(directory: str | Path, sta: str) -> Path:
    return Path(directory) / f"{sta.upper()}{STATS_SUFFIX}"


def _read_state(path: Path) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 3:
                out[parts[0]] = (int(parts[1]), int(parts[2]))
    return out


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _merge_station(path: Path, rows: Mapping[tuple[str, str], SiteStats]) -> int:
    """Merge (date, net) → stats rows into a station CSV; return its row count."""
    table: dict[tuple[str, str], dict[str, str]] = {}
    if path.is_file():
        with path.open(newline="") as fh:
            for r in csv.DictReader(fh):
                table[(r["date"], r["net"])] = r
    for (day, net), st in rows.items():
        table[(day, net)] = {
            "date": day,
            "net": net,
            **{k: ("" if math.isnan(v) else f"{v:g}") for k, v in asdict(st).items()},
        }
    header = ["date", "net", *STAT_FIELDS]
    lines = [",".join(header)]
    for key in sorted(table):
        lines.append(",".join(table[key].get(h, "") for h in header))
    _write_atomic(path, "\n".join(lines) + "\n")
    return len(table)


def harvest(
    out_dir: str | Path,
    *,
    weeks_dir: str | Path = DEFAULT_WEEKS_DIR,
    networks: Sequence[str] = DEFAULT_NETWORKS,
    first_week: int,
    last_week: int,
    stations: Sequence[str] | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Harvest new/changed ``autcln.post.sum`` files into the station CSVs.

    Returns ``{"files": n_read, "skipped": n_unchanged, "stations": n_written}``.
    """
    out = Path(out_dir)
    weeks = Path(weeks_dir)
    state_path = out / STATE_FILE
    state = _read_state(state_path)
    want = None if stations is None else {s.upper() for s in stations}
    new_rows: dict[str, dict[tuple[str, str], SiteStats]] = {}
    taken: dict[str, tuple[int, int]] = {}
    n_read = n_skip = 0
    for week in range(first_week, last_week + 1):
        for net in networks:
            net_dir = weeks / str(week) / net
            if not net_dir.is_dir():
                continue
            for day_dir in sorted(net_dir.glob("[0-9][0-9][0-9]")):
                f = day_dir / AUTCLN_FILE
                try:
                    st = f.stat()
                except OSError:
                    continue
                sig = (st.st_size, int(st.st_mtime))
                if state.get(str(f)) == sig:
                    n_skip += 1
                    continue
                try:
                    with gzip.open(f, "rt", errors="replace") as fh:
                        sites = parse_autcln_post_sum(fh)
                except (OSError, EOFError):
                    continue  # a file being written; next run takes it
                day = date_of(week, int(day_dir.name)).isoformat()
                for site, stats in sites.items():
                    if want is None or site in want:
                        new_rows.setdefault(site, {})[(day, net)] = stats
                taken[str(f)] = sig
                n_read += 1
    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)
        for site, rows in new_rows.items():
            _merge_station(stats_path(out, site), rows)
        state.update(taken)
        _write_atomic(
            state_path,
            "".join(f"{p} {s} {m}\n" for p, (s, m) in sorted(state.items())),
        )
        if new_rows:
            write_network_median(out)
    return {"files": n_read, "skipped": n_skip, "stations": len(new_rows)}


def write_network_median(directory: str | Path) -> Path:
    """Write the network's daily median azimuth mean over every station file.

    The denominator of the normalised icing rule: on days the whole network's
    residuals rise (ionosphere / space weather; 2026-04-30 flagged 66 % of
    stations un-normalised) no single station stands out.
    """
    d = Path(directory)
    series = {}
    for f in sorted(d.glob("*" + STATS_SUFFIX)):
        if f.name == NETWORK_FILE:
            continue
        sta = f.name[: -len(STATS_SUFFIX)]
        series[sta] = read_autcln_stats(sta, d)["az_absmean"]
    net = pd.DataFrame(series).median(axis=1).dropna()
    path = d / NETWORK_FILE
    lines = ["date,az_absmean_median,n_stations"]
    counts = pd.DataFrame(series).notna().sum(axis=1)
    for day, v in net.items():
        lines.append(f"{pd.Timestamp(day).date()},{v:g},{int(counts[day])}")
    _write_atomic(path, "\n".join(lines) + "\n")
    return path


def read_network_median(directory: str | Path) -> pd.Series | None:
    """The network's daily median azimuth mean, or None without one."""
    path = Path(directory) / NETWORK_FILE
    if not path.is_file():
        return None
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return df["az_absmean_median"].astype(float)


# ---------------------------------------------------------------------------
# read + icing index
# ---------------------------------------------------------------------------


def read_autcln_stats(sta: str, directory: str | Path) -> pd.DataFrame:
    """A station's daily statistics, indexed by date (one row per day: the
    median over the networks a tie station is processed in).

    Raises:
        FileNotFoundError: No harvest for this station there.
    """
    path = stats_path(directory, sta)
    if not path.is_file():
        raise FileNotFoundError(f"no autcln statistics at {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    # a station autcln kept no data for (NUM 0, RMS 0: GFUM 2025-12-16) is
    # no statistic, not a perfect day
    df = df[~(df["num"] <= 0)]
    return df.groupby("date")[list(STAT_FIELDS)].median().sort_index()


@dataclass(frozen=True)
class IcingParams:
    """Thresholds of the icing index (BGÓ 2026-10-19: network-normalised, 2.0).

    Measured over 181 stations, two years: normalised at 2.0 flags a median
    1.3 % of days per station (2.1 % un-normalised at 2.5, 28 % of those on
    network-wide days); AUST Nov-Dec 2025 46/49 days, REYK 0 %.
    """

    ratio: float = 2.0  # (normalised) azimuth mean over the station's own low level
    quantile: float = 0.25  # the low level: this quantile …
    window_days: int = 365  # … over a centred window of this many days
    min_days: int = 60  # fewer days in the window: whole-record quantile
    clip_sum: float = SIGELV_CLIP_SUM  # A + B above this: flagged regardless
    exit_ratio: float = 1.5  # an episode lasts while above this (hysteresis)
    max_gap_days: int = 5  # bridge gaps up to this long between flagged days


def icing_days(
    stats: pd.DataFrame,
    params: IcingParams | None = None,
    network: pd.Series | None = None,
) -> pd.Series:
    """Days with icing suspected (bool, one per CALENDAR day of ``stats``).

    With ``network`` (the network's daily median azimuth mean,
    :func:`read_network_median`) the station's azimuth mean is first divided
    by it, so network-wide days cancel. ``r`` = that over the baseline, the
    rolling ``quantile`` of the station's own value (centred ``window_days``;
    the whole-record quantile where the window is thin). Then:

    - an episode STARTS on a day with ``r > ratio`` or the ``sh_sigelv`` clip
      (``atelv_a + atelv_b > clip_sum``);
    - it LASTS while ``r > exit_ratio`` (hysteresis: GFUM 2025-12-06..11 sat
      at 1.5-2.0 inside an obvious episode, the network itself disturbed);
      a day without statistics neither ends nor extends it;
    - a gap of at most ``max_gap_days`` (unflagged or no statistics) between
      two flagged days is bridged — the ice does not leave for a day (5 since
      2026-10-19: GFUM 2026-10-11..14 had no okada statistics; median flagged
      share 1.5 → 1.8 %).

    The index is every calendar day from the first to the last day of
    ``stats``, so a bridged day without statistics is still a flagged day.
    """
    p = IcingParams() if params is None else params
    if stats.empty:
        return pd.Series([], dtype=bool, index=stats.index)
    az = stats["az_absmean"].astype(float)
    if network is not None:
        az = az / network.reindex(az.index)
    az = az.where(np.isfinite(az))
    days = pd.date_range(az.index.min(), az.index.max(), freq="D")
    daily = az.reindex(days)
    base = daily.rolling(p.window_days, center=True, min_periods=p.min_days).quantile(
        p.quantile
    )
    base = base.fillna(az.quantile(p.quantile))
    r = (daily / base).to_numpy()
    clip = ((stats["atelv_a"] + stats["atelv_b"]) > p.clip_sum).reindex(days)
    start = (r > p.ratio) | clip.fillna(False).to_numpy(dtype=bool)
    stay = r > p.exit_ratio
    known = np.isfinite(r)
    flags = np.zeros(len(days), dtype=bool)
    inside = False
    for i in range(len(days)):
        if start[i]:
            inside = True
        elif known[i] and not stay[i]:
            inside = False
        flags[i] = inside and known[i]
    # bridge short gaps between flagged days
    idx = np.flatnonzero(flags)
    for a, b in zip(idx[:-1], idx[1:], strict=True):
        if 1 < b - a <= p.max_gap_days + 1:
            flags[a:b] = True
    return pd.Series(flags, index=days)


def icing_epochs(
    sta: str, directory: str | Path, params: IcingParams | None = None
) -> np.ndarray:
    """Noon fractional-year epochs of a station's icing days (empty without
    a harvest; un-normalised when the network file is missing)."""
    try:
        stats = read_autcln_stats(sta, directory)
    except FileNotFoundError:
        return np.zeros(0)
    network = read_network_median(directory)
    flagged = icing_days(stats, params, network)
    days = flagged.index[flagged.to_numpy()]
    out = []
    for d in days:
        year_len = 366 if pd.Timestamp(d).is_leap_year else 365
        out.append(d.year + (d.dayofyear - 0.5) / year_len)
    return np.asarray(out, dtype=float)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_out() -> Path:
    from gps_parser import ConfigParser

    return Path(str(ConfigParser().getPostProcessDir("pos_dir"))).expanduser()


def _gps_week(day: dt.date) -> int:
    return (day - _GPS_EPOCH).days // 7


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="gps-autcln-harvest",
        description="Harvest GAMIT autcln per-station daily residual statistics "
        "(the icing indicators) from okada's WEEKS tree into "
        "<SITE>.autcln.24hr.csv in the .pos directory. Incremental.",
    )
    ap.add_argument("--weeks-dir", default=DEFAULT_WEEKS_DIR)
    ap.add_argument("--out", default=None, help="default: postprocess pos_dir")
    ap.add_argument("--networks", default=",".join(DEFAULT_NETWORKS))
    ap.add_argument("--first-week", type=int, default=None,
                    help="default: 104 weeks back from today")  # fmt: skip
    ap.add_argument("--last-week", type=int, default=None, help="default: this week")
    ap.add_argument("--stations", default=None, help="comma list (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    this_week = _gps_week(dt.date.today())
    res = harvest(
        a.out or _default_out(),
        weeks_dir=a.weeks_dir,
        networks=[n.strip() for n in a.networks.split(",") if n.strip()],
        first_week=a.first_week if a.first_week is not None else this_week - 104,
        last_week=a.last_week if a.last_week is not None else this_week,
        stations=None if a.stations is None else a.stations.split(","),
        dry_run=a.dry_run,
    )
    print(
        f"autcln: read {res['files']} file(s), {res['skipped']} unchanged, "
        f"{res['stations']} station(s) updated"
    )
    return 0
