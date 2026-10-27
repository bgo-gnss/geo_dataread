"""``gpsprod`` — station motion products from one place (PLAN-gpsprod).

Subcommands arrive step by step: ``displacement`` (steps 1–2), ``velocity``
(step 3), ``coords`` (step 4), ``series`` (step 5).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any
from pathlib import Path

from geo_dataread.products.common import REFS, SESSIONS, VIEWS
from geo_dataread.products.displacement import (
    METHODS,
    DisplacementSettings,
    estimate_displacements,
    to_velocity,
)
from geo_dataread.products.files import table
from geo_dataread.products.loader import Source
from geo_dataread.products.period import resolve_period

TOP_EPILOG = """\
subcommands:
  displacement   how far stations moved across a period (ends | points | step | rate)
  velocity       the velocity product: a rate per station over a period
  coords         where stations are at an epoch (ECEF and lat/lon/h)
  series         a station's time series to file (csv, jsonl, .NEU)
  k-table        build the k(T) table the velocity σ falls back on
  show           print a saved velocity set (.jsonl)

  gpsprod displacement --help    every option, with examples
  gpsprod velocity --help
  gpsprod coords --help
  gpsprod series --help

The library behind it: from geo_dataread import products  (docs/gpsprod.md)
"""

DISPLACEMENT_EPILOG = """\
methods:
  ends   level at each end of the period (mean of --ends N days inside it),
         end minus start. Any motion. The default.
  points the last epoch of the period minus the first: no averaging. For fast
         motion (a dike, the first days after it), where a mean over days
         moves the effective date. Missing end days: the nearest epochs inside
         the period are used and named (start_at / end_at).
  step   the jump across one event: a line through --step-window days on each
         side, both evaluated at the event. Dikes, eruptions, earthquakes.
  rate   the velocity product's rate over the period × its length. Steady
         motion only (velocity -> displacement).
  --as-velocity turns any of them into an average velocity (displacement -> velocity).

periods (--period):
  20231112:20240110          dates (YYYYMMDD or YYYY-MM-DD), both days included
  20240901:                  from a date to today (or --end)
  DIKE:ERUPTION              between two events: the day after the first ends
                             to the day before the second starts
  after:EVENT  or  EVENT:    from the day after an event to today (or --end)
  20231001:EVENT             from a date to the day before an event
  EVENT                      the event's own days, onset to last day
  EVENT is an events.yaml id, or any part of one that names one event
  (2023-11-10-dike, 2024-03-eruption).

  or with options:
  --start 20231109 --end 20231112     both days included
  --start 20231109 --days 4           4 days from the start (9-12 Nov)
  --days 30 [--end 20241031]          the 30 days ending --end (default today)
  (--since and --last are the old names of --start and --days)

examples:
  gpsprod displacement SENG THOB --start 20231112
  gpsprod displacement --area reykjanes --period after:2023-11-10-dike -o d.csv
  gpsprod displacement all --days 7 --ends 1            # one day per end, checked
  gpsprod displacement SENG THOB SKSH --method points --start 20231109 --days 4
  gpsprod displacement SENG --relative-to SKSH --period 20240901:20241031
  gpsprod displacement --area reykjanes --method step --event 2024-03-02-dike
  gpsprod displacement THOB --method step --at 20231110 --step-window 20
  gpsprod displacement THOB --method rate --period 20250101:20250601
  gpsprod displacement THOB --period 2023-11-10-dike:2023-12-eruption --as-velocity
  gpsprod displacement HOFN --start 20200101 --keep-seasonal --units m -o h.jsonl

flags (a note on a row; the number is still given):
  end_disputed:<start|end>:<N|E|U>
         that end day does not agree with the days right next to it (a line
         through the 5 days outside the period, > 4 sigma off). Around a dike
         or eruption this is the rapid movement itself; on a quiet station it
         points at a bad solution. Only with one or two days per end.
  end_unchecked:...   too few neighbour days to check that end
  steps_in_period     a declared step lies inside the period (expected for events)
  start_at/end_at:D   points: the end day had no solution; D was used
  end_early:Nd        the data stop N days before the period ends
  step_nonlinear:<before|after>:<c>   step: that side curves (relaxation)
  fallback:plate      no detrend record: only the plate removed

output: a table on screen, or -o FILE (.csv with a # provenance header, or
.jsonl); every row says frame, ref, view, method, period and its sigma method.
With --as-velocity, -o DIR/NAME writes NAME.gmt, NAME.csv and NAME.jsonl
(the velocity product's files). Refused stations are listed with the reason.
"""


def _box(value: str) -> tuple[float, float, float, float]:
    try:
        w, e, s, n = (float(v) for v in value.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r}: use W/E/S/N") from None
    return w, e, s, n


def select(
    names: Sequence[str],
    *,
    area: str | None,
    region: tuple[float, float, float, float] | None,
    source: Source,
) -> list[str]:
    """Stations by name (``all``: every station with a series of the
    source's product), plus an area's members; a W/E/S/N box picks every
    station in it, or narrows the named ones."""
    from geo_dataread.pos import pos_stations

    picked = {n.upper() for n in names if n.lower() != "all"}
    if area is not None:
        from geo_dataread.events import read_station_areas

        areas = read_station_areas()
        if area not in areas:
            raise SystemExit(f"gpsprod: no area {area!r} ({', '.join(sorted(areas))})")
        picked |= set(areas[area])
    want_all = any(n.lower() == "all" for n in names)
    if want_all or region is not None:
        found = {
            s
            for s, _lon, _lat in pos_stations(
                source.pos_id, source.directory(), region=region
            )
        }
        if want_all or not picked:
            # the network's own stations (stations.cfg), not the global
            # IGS sites the .pos products also carry
            from geo_dataread.products.velocity_cli import select_stations

            picked |= found & set(select_stations([], None))
        else:  # a box narrows the named stations
            picked &= found
    return sorted(picked)


REF_HELP = (
    "plate-periodic (default: plate and the stored seasonal removed), "
    "background (the whole stored trajectory removed), plate (plate only), "
    "itrf (the raw frame)"
)
VELOCITY_REF_HELP = (
    "plate-periodic (default: plate removed, the stored seasonal removed), "
    "background (the whole stored trajectory removed: the transient), plate "
    "(plate only), pre-onset (plate-periodic less the station's own rate over "
    "--pre-window)"
)


def _common(
    p: argparse.ArgumentParser,
    *,
    refs: Sequence[str] = REFS,
    ref_help: str = REF_HELP,
    uncert_help: str = "",
    out_help: str = "write FILE.csv or FILE.jsonl (default: a table on screen)",
    units: bool = True,
) -> None:
    g = p.add_argument_group("stations")
    g.add_argument(
        "stations", nargs="*",
        help="station markers (SENG THOB …), or 'all' for every stations.cfg station with a series",
    )  # fmt: skip
    g.add_argument(
        "--area", help="the members of a station_areas.yaml area (reykjanes …)"
    )
    g.add_argument(
        "--region", type=_box, metavar="W/E/S/N",
        help="every station in the box, or narrows the named ones (-22.8/-21.8/63.7/64.1)",
    )  # fmt: skip
    g = p.add_argument_group(
        "period (--start/--end/--days, or --period with dates or event names)"
    )
    g.add_argument(
        "--start", "--since", dest="start", metavar="DATE",
        help="first day (YYYYMMDD or YYYY-MM-DD); to --end, or for --days days",
    )  # fmt: skip
    g.add_argument(
        "-e", "--end", metavar="DATE",
        help="last day (default today); with --days alone, the days end here",
    )  # fmt: skip
    g.add_argument(
        "--days", "--last", dest="days", type=int, metavar="N",
        help="the period's length in days: from --start, or ending --end/today",
    )  # fmt: skip
    g.add_argument(
        "--period", metavar="A:B",
        help="dates and/or event names in one, see 'periods' below",
    )  # fmt: skip
    g = p.add_argument_group("series")
    g.add_argument(
        "--frame", default="itrf14",
        help="the .pos product: itrf14 (default, IGb14), itrf08 (okada), itrf20 (gamit-p01)",
    )  # fmt: skip
    g.add_argument(
        "--session", default="24h", choices=tuple(SESSIONS),
        help="solution session (default 24h; 8h only where 8 h series exist)",
    )  # fmt: skip
    g.add_argument(
        "--ref", choices=refs, help=ref_help,
    )  # fmt: skip
    g.add_argument(
        "--keep-seasonal", action="store_true",
        help="keep the seasonal signal: the same as --ref plate",
    )  # fmt: skip
    g.add_argument(
        "--relative-to", metavar="STA",
        help="difference every station against STA (a baseline STATION − STA)",
    )  # fmt: skip
    g.add_argument(
        "-u", "--uncert", type=float, metavar="MM",
        help=uncert_help or "drop solutions with a formal σ above MM. Default 100 "
        "for displacements (an event day is noisy but wanted; its σ is carried), "
        "10 for --method rate (the velocity product's)",
    )  # fmt: skip
    g.add_argument(
        "--pos-dir",
        metavar="DIR",
        help="the .pos directory (default postprocess.cfg pos_dir)",
    )
    g.add_argument(
        "--params",
        metavar="FILE",
        help="detrend parameter document (default the frame's deployed one)",
    )
    g.add_argument(
        "--steps", metavar="FILE", help="steps catalog (default the deployed steps.csv)"
    )
    g = p.add_argument_group("output")
    g.add_argument("-o", "--out", metavar="FILE", help=out_help)
    if units:
        g.add_argument(
            "--units", choices=("mm", "m"), default="mm", help="mm (default) or m"
        )
    g.add_argument("-q", "--quiet", action="store_true", help="no progress on stderr")


def build_parser() -> argparse.ArgumentParser:
    raw = argparse.RawDescriptionHelpFormatter
    p = argparse.ArgumentParser(
        prog="gpsprod",
        description="GNSS station motion products from one place: one estimator "
        "for displacements (velocities, coordinates and series follow).",
        epilog=TOP_EPILOG,
        formatter_class=raw,
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")
    d = sub.add_parser(
        "displacement",
        help="how far stations moved across a period",
        formatter_class=raw,
        description=(
            "How far stations moved across a period, N/E/U with σ.\n"
            "Defaults: itrf14 .pos, plate and seasonal removed, declared view "
            "(known blunders out), method ends with 5 days per end."
        ),
        epilog=DISPLACEMENT_EPILOG,
    )
    _common(d)
    g = d.add_argument_group("method")
    g.add_argument(
        "--method", choices=METHODS, default="ends",
        help="ends (default), points, step or rate — see 'methods' below",
    )  # fmt: skip
    g.add_argument(
        "--view", choices=VIEWS, default="declared",
        help="declared (default: known blunders out), cleaned (outliers detected "
        "and out — beware: it can take real motion in unrest), raw",
    )  # fmt: skip
    g.add_argument(
        "--ends", type=int, default=5, metavar="N",
        help="ends: days averaged at each end (default 5; 1 = single days)",
    )  # fmt: skip
    g.add_argument(
        "--neighbour-days", type=int, default=5, metavar="M",
        help="ends with N ≤ 2 and points: each end is checked against a line "
        "through the M days outside the period next to it (inside if too few) "
        "and flagged end_disputed when it disagrees (default 5)",
    )  # fmt: skip
    g.add_argument(
        "--event", metavar="EVENT",
        help="step: the event (events.yaml id or part of one); default the "
        "station's one dike/eruption inside the period",
    )  # fmt: skip
    g.add_argument(
        "--at", metavar="DATE", help="step: a one-day event on DATE instead of --event"
    )
    g.add_argument(
        "--step-window", type=int, default=10, metavar="DAYS",
        help="step: days of line on each side of the event (default 10: fast motion after an event bends a longer line)",
    )  # fmt: skip
    g.add_argument(
        "--as-velocity", action="store_true",
        help="report the average velocity (displacement / its span) in mm/yr",
    )  # fmt: skip
    _velocity_parser(sub, raw)
    _coords_parser(sub, raw)
    _series_parser(sub, raw)
    k = sub.add_parser(
        "k-table", help="build the k(T) table (gps-velocities k-table)",
        add_help=False, description="Runs gps-velocities k-table; see its --help.",
    )  # fmt: skip
    k.add_argument("args", nargs=argparse.REMAINDER)
    sh = sub.add_parser(
        "show", help="print a saved velocity set (gps-velocities show)",
        add_help=False, description="Runs gps-velocities show; see its --help.",
    )  # fmt: skip
    sh.add_argument("args", nargs=argparse.REMAINDER)
    return p


VELOCITY_EPILOG = """\
what it is:
  the velocity product (one estimator for every velocity): a weighted line
  through each component over the period, gated (>= 5 epochs, >= 80 % of the
  period covered, no declared step inside), sigma = formal x k(T), the
  station's own empirical scale for that window length, else the k(T) table,
  else 5. Short windows (a few days in unrest) stand on |v|/sigma instead.
  Every row says its ref, k and sigma method; refused stations say why.

periods: as gpsprod displacement (--start/--end/--days, or --period with
  dates and events.yaml names: DIKE:ERUPTION, after:EVENT, EVENT ...)

examples:
  gpsprod velocity --area reykjanes --start 20200101 --end 20221231
  gpsprod velocity SENG THOB --period 2023-11-10-dike:2023-12-eruption
  gpsprod velocity all --days 365 --save                # into velocity_dir
  gpsprod velocity THOB --start 20231112 --days 3       # short tier, unrest
  gpsprod velocity --area reykjanes --ref pre-onset --pre-window 20230806:20231024 \\
                   --period after:2023-11-10-dike
  gpsprod velocity SENG --relative-to SKSH --start 20240901 --end 20241031
  gpsprod velocity THOB --start 20240101 --days 90 -o ~/tmp/thob   # thob.gmt/.csv/.jsonl

flags: fallback:plate (no detrend record), background:none (ref plate),
  short:n=N (short tier), nonlinear:<c> (curved: a pulse, not a rate),
  steps_in_window, truncated@T (the data stop early), k:baseline_unmeasured /
  k:session_unmeasured (k from the table or 5), k_table_settings_differ.

output: a table on screen; -o DIR/NAME writes NAME.gmt (psvelo), NAME.csv and
NAME.jsonl; --save writes the set under velocity_dir (as gps-velocities).
"""


def _velocity_parser(sub: Any, raw: type[argparse.HelpFormatter]) -> None:
    from geo_dataread.products.velocity_core import (
        DEFAULT_K,
        NO_RECORD_POLICIES,
        VelocitySettings,
    )
    from geo_dataread.products.velocity_core import REFS as VREFS

    v = sub.add_parser(
        "velocity",
        help="the velocity product: a rate per station over a period",
        formatter_class=raw,
        description=(
            "Velocities N/E/U [mm/yr] with an honest sigma, one per station over "
            "a period.\nDefaults: itrf14 .pos, plate and seasonal removed, "
            "declared view, sigma screen 10 mm."
        ),
        epilog=VELOCITY_EPILOG,
    )
    _common(
        v, refs=VREFS, ref_help=VELOCITY_REF_HELP,
        uncert_help="drop solutions with a formal σ above MM (default 10)",
        out_help="write DIR/NAME.gmt, .csv and .jsonl (default: a table on screen)",
        units=False,
    )  # fmt: skip
    g = v.add_argument_group("velocity")
    g.add_argument(
        "--view", choices=VIEWS, default="declared",
        help="declared (default: known blunders out), cleaned, raw",
    )  # fmt: skip
    g.add_argument(
        "--pre-window", metavar="A:B",
        help="ref pre-onset: the station's reference days before the onset, "
        "YYYYMMDD:YYYYMMDD (e.g. 20230806:20231024)",
    )  # fmt: skip
    g.add_argument(
        "--no-record", choices=NO_RECORD_POLICIES, default="plate",
        help="a station without a detrend record: fall back to ref plate "
        "(default, flagged) or refuse",
    )  # fmt: skip
    g.add_argument(
        "--allow-steps", action="store_true",
        help="keep a window with a declared step inside (flagged)",
    )  # fmt: skip
    g.add_argument(
        "--min-obs", type=int, default=VelocitySettings.min_obs, metavar="N",
        help=f"fewest epochs (default {VelocitySettings.min_obs})",
    )  # fmt: skip
    g.add_argument(
        "--min-cover", type=float, default=VelocitySettings.min_cover, metavar="F",
        help=f"least share of the period covered (default {VelocitySettings.min_cover})",
    )  # fmt: skip
    g.add_argument(
        "--no-short-tier", dest="short_tier", action="store_false",
        help="refuse a window failing the epoch/cover gates outright",
    )  # fmt: skip
    g.add_argument(
        "--short-snr", type=float, default=VelocitySettings.short_snr, metavar="X",
        help=f"short tier: |v|/σ a part needs (default {VelocitySettings.short_snr})",
    )  # fmt: skip
    g.add_argument(
        "--short-snr-pair", type=float, default=VelocitySettings.short_snr_pair,
        metavar="X",
        help=f"short tier on 2 epochs (default {VelocitySettings.short_snr_pair})",
    )  # fmt: skip
    g.add_argument(
        "--default-k", type=float, default=DEFAULT_K, metavar="K",
        help=f"k when neither the station nor the table has one (default {DEFAULT_K})",
    )  # fmt: skip
    g.add_argument(
        "--k-table", metavar="FILE",
        help="the k(T) table (default velocity_dir/k_table_<frame>.json; 'none': no table)",
    )  # fmt: skip
    g.add_argument(
        "--save", action="store_true",
        help="write the set under velocity_dir as gps-velocities does (name from the settings)",
    )  # fmt: skip


def _velocity(args: argparse.Namespace) -> int:
    from geo_dataread.pos import frame_name
    from geo_dataread.products.period import parse_date
    from geo_dataread.products.velocity_cli import _show
    from geo_dataread.products.velocity_core import (
        VelocitySettings,
        estimate_velocities,
        velocity_dir,
        window_bounds,
    )
    from geo_dataread.products.velocity_files import set_name, write_velocity_set

    if args.ref and args.keep_seasonal and args.ref != "plate":
        raise SystemExit("gpsprod: --keep-seasonal is --ref plate; not both")
    ref = args.ref or ("plate" if args.keep_seasonal else "plate-periodic")
    try:
        pre = None
        if args.pre_window:
            a, b = args.pre_window.split(":", 1)
            pre = window_bounds(parse_date(a), parse_date(b))
        settings = VelocitySettings(
            frame=frame_name(args.frame), ref=ref, view=args.view,
            uncert=args.uncert if args.uncert is not None else UNCERT_RATE,
            no_record=args.no_record, allow_steps=args.allow_steps,
            min_obs=args.min_obs, min_cover=args.min_cover,
            short_tier=args.short_tier, short_snr=args.short_snr,
            short_snr_pair=args.short_snr_pair, default_k=args.default_k,
            pos_dir=args.pos_dir, params=args.params, steps=args.steps,
            k_table=args.k_table, pre_window=pre, session=args.session,
            relative_to=args.relative_to,
        )  # fmt: skip
        period = resolve_period(
            args.period, start=args.start, end=args.end, days=args.days
        )
        src = Source(frame=args.frame, session=args.session, pos_dir=args.pos_dir)
    except ValueError as err:
        raise SystemExit(f"gpsprod: {err}") from None
    stations = select(args.stations, area=args.area, region=args.region, source=src)
    if not stations:
        raise SystemExit("gpsprod: no stations (give markers, --area, --region or all)")

    def progress(rec: Any) -> None:
        if not args.quiet:
            print(f"  {rec.station}", file=sys.stderr, end="\r")

    records = estimate_velocities(
        stations, period.first_day, period.last_day, settings, progress=progress
    )
    if not args.quiet:
        print(" " * 12, file=sys.stderr, end="\r")
    if args.out or args.save:
        if args.save:
            out_dir = velocity_dir()
            name = set_name(settings, period.first_day, period.last_day)
        else:
            out = Path(args.out).expanduser()
            out_dir, name = out.parent, (out.stem if out.suffix else out.name)
        paths = write_velocity_set(records, out_dir, name)
        n_ok = sum(r.accepted for r in records)
        print(f"gpsprod: {n_ok}/{len(records)} stations → "
              f"{paths['csv'].with_suffix('')}.{{gmt,csv,jsonl}}", file=sys.stderr)  # fmt: skip
    else:
        names = f"  ({', '.join(period.names)})" if period.names else ""
        print(f"# {period.dates[0]} .. {period.dates[1]}{names}  {settings.frame} "
              f"{settings.session} ref {ref} view {settings.view} [mm/yr]")  # fmt: skip
        _show(records, show_all=True)
    return 0


#: formal-σ screen [mm]: a displacement keeps the noisy solution of an event
#: day (σ says how noisy); a rate takes the velocity product's screen
UNCERT_DISPLACEMENT = 100.0
UNCERT_RATE = 10.0


def default_uncert(args: argparse.Namespace) -> float:
    return (
        UNCERT_RATE if getattr(args, "method", None) == "rate" else UNCERT_DISPLACEMENT
    )


COORDS_EPILOG = """\
methods:
  observed  the weighted (GLS, full 3x3 covariance) mean of the .pos X/Y/Z
            solutions within --window-days either side of the epoch, after the
            junction offsets and the view. sigma = max(k_day, sqrt(chi2/nu)) x
            the mean's formal sigma. The default. No solution that close: the
            coordinate is extrapolated from the nearest window by the station's
            detrend record + plate motion (flag extrapolated:Nd).
  model     always that extrapolation from the nearest window: a coordinate
            without the day-to-day noise (sigma: the window's only).
  apr       the GLOBK apriori (GAMIT tables igb14_comb.apr/.eq), for comparing.

examples:
  gpsprod coords SENG THOB --epoch 20240101
  gpsprod coords --area reykjanes --epoch 20231109 -o coords.csv
  gpsprod coords all --epoch 20250601 --window-days 0          # single days
  gpsprod coords THOB --relative-to SKSH --epoch 20240601      # baseline vector
  gpsprod coords SENG --epoch 20240110 --method model          # in SENG's Jan-2024 gap
  gpsprod coords HOFN REYK --epoch 20240101 --method apr

columns: X Y Z [m] ECEF, sX sY sZ; lat lon [deg] h [m, ellipsoidal]; sN sE sU
[m] the sigma in the local frame; with --relative-to dN dE dU [m] the vector at
the station. n_used solutions; t_obs the epoch the value rests on.

flags:
  fast_motion:chi2nu=N  the station moved within the window (an event, fast
                        unrest); the mean blurs it: use --window-days 0 or 1
  window_cut_at_step:D  a declared step in the window: only the solutions on
                        the epoch's side of it were used
  extrapolated:Nd       no solution near the epoch: carried N days from the
                        nearest window by the detrend record + plate motion
  steps_between         ... across a declared step the model does not know
  model:plate_only      no detrend record: plate motion only
  sigma:observed_only   the model's own sigma is not included
  apriori               the GLOBK apriori, no sigma
  view:none             the view could not be read: all solutions used
"""


def _coords_parser(sub: Any, raw: type[argparse.HelpFormatter]) -> None:
    from geo_dataread.products.coords import METHODS as CMETHODS

    c = sub.add_parser(
        "coords",
        help="where stations are at an epoch (ECEF and lat/lon/h)",
        formatter_class=raw,
        description=(
            "Station coordinates at an epoch [m]: X/Y/Z, lat/lon/h and their "
            "sigma.\nDefaults: itrf14 .pos, cleaned view, mean of the solutions "
            "within 5 days of the epoch."
        ),
        epilog=COORDS_EPILOG,
    )
    g = c.add_argument_group("stations")
    g.add_argument(
        "stations", nargs="*",
        help="station markers, or 'all' for every stations.cfg station with a series",
    )  # fmt: skip
    g.add_argument("--area", help="the members of a station_areas.yaml area")
    g.add_argument(
        "--region", type=_box, metavar="W/E/S/N",
        help="every station in the box, or narrows the named ones",
    )  # fmt: skip
    g = c.add_argument_group("epoch and method")
    g.add_argument(
        "--epoch", metavar="DATE",
        help="the day (YYYYMMDD or YYYY-MM-DD; default today: the newest solutions)",
    )  # fmt: skip
    g.add_argument(
        "--method", choices=CMETHODS, default="observed",
        help="observed (default), model or apr — see 'methods' below",
    )  # fmt: skip
    g.add_argument(
        "--window-days", type=int, default=5, metavar="N",
        help="solutions within N days either side of the epoch (default 5; 0 = that day)",
    )  # fmt: skip
    g.add_argument(
        "--view", choices=VIEWS, default="cleaned",
        help="cleaned (default: outliers out), declared (known blunders out), raw",
    )  # fmt: skip
    g.add_argument(
        "--apr",
        nargs="+",
        metavar="FILE",
        help="apr: GLOBK .apr file(s) (default the GAMIT tables')",
    )
    g.add_argument(
        "--eq", nargs="+", metavar="FILE", help="apr: GLOBK .eq rename file(s)"
    )
    g = c.add_argument_group("series")
    g.add_argument("--frame", default="itrf14", help="itrf14 (default), itrf08, itrf20")
    g.add_argument(
        "--session", default="24h", choices=tuple(SESSIONS), help="24h (default) or 8h"
    )
    g.add_argument(
        "--relative-to", metavar="STA",
        help="the baseline vector STATION − STA at the epoch (dN dE dU at the station)",
    )  # fmt: skip
    g.add_argument(
        "-u", "--uncert", type=float, default=100.0, metavar="MM",
        help="drop solutions with a formal σ above MM (default 100; the mean weights them anyway)",
    )  # fmt: skip
    g.add_argument(
        "--pos-dir",
        metavar="DIR",
        help="the .pos directory (default postprocess.cfg pos_dir)",
    )
    g.add_argument(
        "--params",
        metavar="FILE",
        help="detrend parameter document (model, extrapolation)",
    )
    g.add_argument("--steps", metavar="FILE", help="steps catalog (the cleaned view)")
    g = c.add_argument_group("output")
    g.add_argument(
        "-o",
        "--out",
        metavar="FILE",
        help="write FILE.csv or FILE.jsonl (default: a table on screen)",
    )
    g.add_argument("-q", "--quiet", action="store_true", help="no progress on stderr")


def _coords(args: argparse.Namespace) -> int:
    import datetime as dt

    from geo_dataread.products.coords import CoordSettings, estimate_coords
    from geo_dataread.products.period import parse_date
    from geo_dataread.products.velocity_core import yearf

    try:
        day = parse_date(args.epoch) if args.epoch else dt.date.today()
        src = Source(
            frame=args.frame, session=args.session, view=args.view, ref="plate",
            uncert=args.uncert, pos_dir=args.pos_dir, params=args.params,
            steps=args.steps, relative_to=args.relative_to,
        )  # fmt: skip
        settings = CoordSettings(
            source=src, method=args.method, window_days=args.window_days,
            apr_files=tuple(args.apr or ()), eq_files=tuple(args.eq or ()),
        )  # fmt: skip
    except ValueError as err:
        raise SystemExit(f"gpsprod: {err}") from None
    stations = select(args.stations, area=args.area, region=args.region, source=src)
    if not stations:
        raise SystemExit("gpsprod: no stations (give markers, --area, --region or all)")
    # the epoch of a day's solution is its noon
    epoch = yearf(dt.datetime(day.year, day.month, day.day, 12))

    def progress(sta: str) -> None:
        if not args.quiet:
            print(f"  {sta}", file=sys.stderr, end="\r")

    rs = estimate_coords(
        stations, epoch, settings, epoch_date=day.isoformat(), progress=progress
    )
    if not args.quiet:
        print(" " * 12, file=sys.stderr, end="\r")
    if args.out:
        path = rs.write(args.out)
        print(
            f"gpsprod: {len(rs.accepted())}/{len(rs)} stations → {path}",
            file=sys.stderr,
        )
    else:
        print(f"# {day.isoformat()}  {src.frame_name} {src.session} view {src.view} "
              f"method {args.method} ±{args.window_days} d [m]")  # fmt: skip
        print(table(rs.records))
    return 0


SERIES_EPILOG = """\
what it writes:
  the series every gpsprod product reads: the .pos series of the frame, the
  reference (plate and the stored seasonal removed by default), the view
  (cleaned by default: outliers out), the sigma screen. Kept epochs only, or
  --with-dropped every epoch with its raw value, kept 0/1 and why (outlier,
  declared, icing, snow, network).

formats (--format, or from the -o extension):
  csv    # provenance header, one row per epoch:
         date, yearf, dN dE dU, sN sE sU [mm], kept, flags
  jsonl  a provenance line, then one epoch per line
  neu    gps-savetimes' .NEU layout (kept epochs) + NAME.prov.json beside it

--source tot: the legacy ITRF2008 TOT series exactly as gps-savetimes writes
  it (gamittooneuf; --ref plate | background (= detrend) | itrf (= itrf2008);
  whole series, .NEU only). gps-savetimes itself is unchanged.

examples:
  gpsprod series OLAC                                    # csv to the screen
  gpsprod series OLAC --start 20240101 -o olac.csv
  gpsprod series --area reykjanes --period after:2023-11-10-dike -o out/ --format neu
  gpsprod series THOB --with-dropped --days 90 -o thob_all.csv
  gpsprod series THOB --relative-to SKSH --start 20240901 -o thob-sksh.csv
  gpsprod series HOFN --ref itrf --view raw --units m -o hofn.jsonl
  gpsprod series SENG --source tot --ref plate -o tot/          # = gps-savetimes --file

output: one station without -o: csv on screen. -o FILE.csv|.jsonl|.neu for one
station; -o DIR/ for several: DIR/STA-<ref>.<format>.
"""


def _series_parser(sub: Any, raw: type[argparse.HelpFormatter]) -> None:
    from geo_dataread.products.series import FORMATS as SFORMATS
    from geo_dataread.products.series import SOURCES

    sp = sub.add_parser(
        "series",
        help="a station's time series to file (csv, jsonl, .NEU)",
        formatter_class=raw,
        description=(
            "Station time series N/E/U [mm] as the gpsprod products see them.\n"
            "Defaults: itrf14 .pos, plate and seasonal removed, cleaned view, "
            "the whole series."
        ),
        epilog=SERIES_EPILOG,
    )
    _common(
        sp,
        uncert_help="drop solutions with a formal σ above MM (default 10)",
        out_help="FILE.csv|.jsonl|.neu (one station) or DIR/ (several); default: the screen",
    )  # fmt: skip
    g = sp.add_argument_group("series")
    g.add_argument(
        "--view", choices=VIEWS, default="cleaned",
        help="cleaned (default: outliers out), declared (known blunders out), raw",
    )  # fmt: skip
    g.add_argument(
        "--source", choices=SOURCES, default="pos",
        help="pos (default: the .pos product) or tot (legacy TOT, as gps-savetimes)",
    )  # fmt: skip
    g.add_argument(
        "--format", choices=SFORMATS,
        help="csv, jsonl or neu (default: from the -o extension, else csv)",
    )  # fmt: skip
    g.add_argument(
        "--with-dropped", action="store_true",
        help="every epoch with its raw value, kept 0/1 and why it was dropped",
    )  # fmt: skip
    g.add_argument(
        "--time-format", metavar="FMT",
        help="neu: the time column, 'yearf' or a strftime format (default date and time)",
    )  # fmt: skip


def _series(args: argparse.Namespace) -> int:
    from geo_dataread.products.series import (
        TOT_REFS,
        SeriesSettings,
        estimate_series,
        series_rows,
        write_series,
        write_tot,
    )

    if args.ref and args.keep_seasonal and args.ref != "plate":
        raise SystemExit("gpsprod: --keep-seasonal is --ref plate; not both")
    ref = args.ref or ("plate" if args.keep_seasonal else "plate-periodic")
    out = Path(args.out).expanduser() if args.out else None
    is_dir = out is not None and (
        out.is_dir() or not out.suffix or args.out.endswith("/")
    )
    fmt = args.format or (
        out.suffix.lstrip(".").lower() if out is not None and not is_dir else "csv"
    )
    if fmt not in ("csv", "jsonl", "neu"):
        raise SystemExit(f"gpsprod: format {fmt!r}: csv, jsonl or neu")
    src = Source(
        frame=args.frame, session=args.session, view=args.view, ref=ref,
        uncert=args.uncert if args.uncert is not None else UNCERT_RATE,
        pos_dir=args.pos_dir, params=args.params, steps=args.steps,
        relative_to=args.relative_to,
    )  # fmt: skip
    stations = select(args.stations, area=args.area, region=args.region, source=src)
    if not stations:
        raise SystemExit("gpsprod: no stations (give markers, --area, --region or all)")
    if len(stations) > 1 and (out is None or not is_dir):
        raise SystemExit("gpsprod: several stations: give -o DIR/")
    mm = args.units == "mm"
    if args.source == "tot":
        if ref not in TOT_REFS:
            raise SystemExit(f"gpsprod: --source tot takes --ref {', '.join(TOT_REFS)}")
        if args.period or args.start or args.days or fmt != "neu" and args.format:
            raise SystemExit("gpsprod: --source tot writes the whole series, as .NEU")
        tref = TOT_REFS[ref]
        for sta in stations:
            path = (
                None if out is None else (out / f"{sta}-{tref}.NEU" if is_dir else out)
            )
            write_tot(sta, path, ref=tref, mm=mm, time_format=args.time_format)
        return 0
    period = None
    if args.period or args.start or args.days:
        try:
            p = resolve_period(
                args.period, start=args.start, end=args.end, days=args.days
            )
        except ValueError as err:
            raise SystemExit(f"gpsprod: {err}") from None
        period, dates = p.bounds, p.dates
    else:
        dates = None
    try:
        settings = SeriesSettings(source=src, with_dropped=args.with_dropped)
    except ValueError as err:
        raise SystemExit(f"gpsprod: {err}") from None

    def progress(sta: str) -> None:
        if not args.quiet:
            print(f"  {sta}", file=sys.stderr, end="\r")

    recs = estimate_series(
        stations, period, settings, period_dates=dates, progress=progress
    )
    if not args.quiet:
        print(" " * 12, file=sys.stderr, end="\r")
    for rec in recs:
        rec = rec.scaled(args.units)
        if not rec.accepted:
            print(f"gpsprod: {rec.station}: {rec.refused}", file=sys.stderr)
            continue
        if out is None:
            from geo_dataread.products.series import SERIES_COLUMNS, header

            print("\n".join(f"# {x}" for x in header(rec)))
            print(",".join(SERIES_COLUMNS))
            for row in series_rows(rec):
                print(",".join(row))
            continue
        path = out / f"{rec.station}-{ref}.{fmt}" if is_dir else out
        written = write_series(rec, path, fmt, time_format=args.time_format)
        print(f"gpsprod: {rec.station}: {rec.n_kept}/{rec.t.size} epochs → "
              f"{', '.join(str(w) for w in written)}", file=sys.stderr)  # fmt: skip
    return 0 if all(r.accepted for r in recs) else 1


def _source(args: argparse.Namespace) -> Source:
    if args.ref and args.keep_seasonal and args.ref != "plate":
        raise SystemExit("gpsprod: --keep-seasonal is --ref plate; not both")
    ref = args.ref or ("plate" if args.keep_seasonal else "plate-periodic")
    return Source(
        frame=args.frame, session=args.session, view=args.view, ref=ref,
        uncert=args.uncert if args.uncert is not None else default_uncert(args),
        pos_dir=args.pos_dir, params=args.params,
        steps=args.steps, relative_to=args.relative_to,
    )  # fmt: skip


def _step_period(args: argparse.Namespace) -> str:
    """The default period of a step given only its event: the event ± the
    step window."""
    import datetime as dt

    from geo_dataread.events import read_events
    from geo_dataread.products.period import find_event, parse_date

    if args.at:
        first = last = parse_date(args.at)
    else:
        e = find_event(args.event, read_events())
        first, last = e.onset, e.last_day
    pad = dt.timedelta(days=args.step_window)
    return f"{(first - pad):%Y%m%d}:{(last + pad):%Y%m%d}"


def _displacement(args: argparse.Namespace) -> int:
    try:
        src = _source(args)
        settings = DisplacementSettings(
            source=src, method=args.method, ends_days=args.ends,
            neighbour_days=args.neighbour_days, step_window_days=args.step_window,
            event=args.event, at=args.at,
        )  # fmt: skip
        spec = args.period
        no_period = spec is None and args.start is None and args.days is None
        if args.method == "step" and no_period and (args.event or args.at):
            spec = _step_period(args)
        period = resolve_period(spec, start=args.start, end=args.end, days=args.days)
    except ValueError as err:
        raise SystemExit(f"gpsprod: {err}") from None
    stations = select(args.stations, area=args.area, region=args.region, source=src)
    if not stations:
        raise SystemExit("gpsprod: no stations (give markers, --area, --region or all)")

    def progress(sta: str) -> None:
        if not args.quiet:
            print(f"  {sta}", file=sys.stderr, end="\r")

    rs = estimate_displacements(
        stations, period.bounds, settings, period_dates=period.dates,
        period_names=period.names, progress=progress,
    )  # fmt: skip
    if not args.quiet:
        print(" " * 12, file=sys.stderr, end="\r")
    names = f"  ({', '.join(period.names)})" if period.names else ""
    head = (
        f"# {period.dates[0]} .. {period.dates[1]}{names}  {src.frame_name} "
        f"{src.session} ref {src.ref} view {src.view} method {args.method}"
    )
    if args.as_velocity:
        from geo_dataread.products.velocity_cli import _show
        from geo_dataread.products.velocity_files import write_velocity_set

        vel = [to_velocity(r) for r in rs]
        if args.out:
            out = Path(args.out).expanduser()
            paths = write_velocity_set(
                vel, out.parent, out.stem if out.suffix else out.name
            )
            print(
                f"gpsprod: {', '.join(str(p) for p in paths.values())}", file=sys.stderr
            )
        else:
            print(head + "  as velocity [mm/yr]")
            _show(vel, show_all=True)
        return 0
    if args.out:
        path = rs.write(args.out, units=args.units)
        print(
            f"gpsprod: {len(rs.accepted())}/{len(rs)} stations → {path}",
            file=sys.stderr,
        )
    else:
        print(f"{head} [{args.units}]")
        print(table([r.scaled(args.units) for r in rs]))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("k-table", "show"):
        # gps-velocities' own subcommands, options and all
        from geo_dataread.products.velocity_cli import main as velocity_main

        return int(velocity_main(argv))
    args = build_parser().parse_args(argv)
    if args.command == "displacement":
        return _displacement(args)
    if args.command == "velocity":
        return _velocity(args)
    if args.command == "coords":
        return _coords(args)
    if args.command == "series":
        return _series(args)
    return 2  # pragma: no cover


def displacemnts_alias(argv: Sequence[str] | None = None) -> int:
    """``gps-displacemnts STA… -s START [-e END] [--file]``, now gpsprod.

    The old tool (TOT, ITRF2008, end day minus start day) is retired; this
    translates its arguments to ``gpsprod displacement`` (itrf14, 5-day
    ends) and says so.
    """
    p = argparse.ArgumentParser(
        prog="gps-displacemnts",
        description="Retired: translates to 'gpsprod displacement' (see gpsprod displacement --help).",
    )
    p.add_argument("stations", nargs="+", help="station markers")
    p.add_argument("-s", "--start", required=True, help="first day, YYYYMMDD")
    p.add_argument("-e", "--end", help="last day, YYYYMMDD (default today)")
    p.add_argument("--file", action="store_true", help="write START-END.csv")
    p.add_argument("-f", default="%Y%m%d", help="date format (only %%Y%%m%%d)")
    a = p.parse_args(argv)
    if a.f != "%Y%m%d":
        raise SystemExit("gps-displacemnts: only -f %Y%m%d; use gpsprod displacement")
    new = ["displacement", *a.stations, "--period", f"{a.start}:{a.end or ''}"]
    if a.file:
        new += ["--out", f"{a.start}-{a.end or 'today'}.csv"]
    print("gps-displacemnts is now: gpsprod " + " ".join(new), file=sys.stderr)
    return main(new)
