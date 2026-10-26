"""``gpsprod`` — station motion products from one place (PLAN-gpsprod).

Subcommands arrive step by step: ``displacement`` (steps 1–2), ``velocity``
(step 3), ``coords`` (step 4), ``series`` (step 5).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
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
  displacement   how far stations moved across a period (ends | step | rate)

  gpsprod displacement --help    every option, with examples

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
            from geo_dataread.velocity import select_stations

            picked |= found & set(select_stations([], None))
        else:  # a box narrows the named stations
            picked &= found
    return sorted(picked)


def _common(p: argparse.ArgumentParser) -> None:
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
        "--ref", choices=REFS,
        help="plate-periodic (default: plate and the stored seasonal removed), "
        "background (the whole stored trajectory removed), plate (plate only), "
        "itrf (the raw frame)",
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
        help="drop solutions with a formal σ above MM. Default 100 for "
        "displacements (an event day is noisy but wanted; its σ is carried), "
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
    g.add_argument(
        "-o", "--out", metavar="FILE",
        help="write FILE.csv or FILE.jsonl (default: a table on screen)",
    )  # fmt: skip
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
        "--step-window", type=int, default=30, metavar="DAYS",
        help="step: days of line on each side of the event (default 30)",
    )  # fmt: skip
    g.add_argument(
        "--as-velocity", action="store_true",
        help="report the average velocity (displacement / its span) in mm/yr",
    )  # fmt: skip
    return p


#: formal-σ screen [mm]: a displacement keeps the noisy solution of an event
#: day (σ says how noisy); a rate takes the velocity product's screen
UNCERT_DISPLACEMENT = 100.0
UNCERT_RATE = 10.0


def default_uncert(args: argparse.Namespace) -> float:
    return (
        UNCERT_RATE if getattr(args, "method", None) == "rate" else UNCERT_DISPLACEMENT
    )


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
        from geo_dataread.velocity import _show, write_velocity_set

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
    args = build_parser().parse_args(argv)
    if args.command == "displacement":
        return _displacement(args)
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
