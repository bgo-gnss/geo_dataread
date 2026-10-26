"""``gpsprod`` — station motion products from one place (PLAN-gpsprod).

    gpsprod displacement SENG THOB --since 20231110
    gpsprod displacement --area reykjanes --period 20231110:20240110 --out d.csv
    gpsprod displacement all --last 30 --ends 1

Subcommands arrive step by step: ``displacement`` (step 1), ``velocity``
(step 3), ``coords`` (step 4), ``series`` (step 5).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from geo_dataread.products.common import REFS, SESSIONS, VIEWS
from geo_dataread.products.displacement import (
    DisplacementSettings,
    estimate_displacements,
)
from geo_dataread.products.files import table
from geo_dataread.products.loader import Source
from geo_dataread.products.period import resolve_period


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
    source's product), plus an area's members, inside a W/E/S/N box."""
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
            picked |= found
        else:  # a box narrows the named stations
            picked &= found
    return sorted(picked)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("stations", nargs="*", help="markers, or 'all'")
    p.add_argument("--area", help="a station_areas.yaml area (its members)")
    p.add_argument("--region", type=_box, help="W/E/S/N box")
    g = p.add_argument_group("period (one of)")
    g.add_argument("--period", help="A:B dates (YYYYMMDD), A: to today")
    g.add_argument("--since", help="from this date to --end (default today)")
    g.add_argument("--last", type=int, help="the N whole days ending --end")
    g.add_argument("-e", "--end", help="last day for --since/--last (default today)")
    p.add_argument("--frame", default="itrf14", help="itrf14 (default), itrf08, itrf20")
    p.add_argument("--session", default="24h", choices=tuple(SESSIONS))
    p.add_argument("--ref", choices=REFS, help="default plate-periodic")
    p.add_argument(
        "--keep-seasonal", action="store_true", help="ref plate (seasonal kept)"
    )
    p.add_argument("--relative-to", metavar="STA", help="difference against STA")
    p.add_argument("-u", "--uncert", type=float, default=10.0, help="max formal σ [mm]")
    p.add_argument("--pos-dir", help="the .pos directory (default postprocess.cfg)")
    p.add_argument("--params", help="detrend parameter document")
    p.add_argument("--steps", help="steps catalog")
    p.add_argument("-o", "--out", help="write here: .csv or .jsonl (default screen)")
    p.add_argument("--units", choices=("mm", "m"), default="mm")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gpsprod",
        description="GNSS station motion products: displacements (more to come).",
    )
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser(
        "displacement",
        help="displacement across a period",
        description=(
            "Displacement across a period: the level at each end is the weighted "
            "mean of --ends N whole days inside the period. Defaults: itrf14, "
            "plate and seasonal removed, declared view."
        ),
    )
    _common(d)
    d.add_argument("--view", choices=VIEWS, default="declared")
    d.add_argument("--method", choices=("ends",), default="ends")
    d.add_argument("--ends", type=int, default=5, metavar="N", help="days per end (5)")
    d.add_argument(
        "--neighbour-days", type=int, default=5, metavar="M",
        help="days the N ≤ 2 end check fits its line through (5)",
    )  # fmt: skip
    return p


def _source(args: argparse.Namespace) -> Source:
    ref = args.ref or ("plate" if args.keep_seasonal else "plate-periodic")
    if args.ref and args.keep_seasonal and args.ref != "plate":
        raise SystemExit("gpsprod: --keep-seasonal is --ref plate; not both")
    return Source(
        frame=args.frame, session=args.session, view=args.view, ref=ref,
        uncert=args.uncert, pos_dir=args.pos_dir, params=args.params,
        steps=args.steps, relative_to=args.relative_to,
    )  # fmt: skip


def _displacement(args: argparse.Namespace) -> int:
    try:
        src = _source(args)
        settings = DisplacementSettings(
            source=src, method=args.method, ends_days=args.ends,
            neighbour_days=args.neighbour_days,
        )  # fmt: skip
        period = resolve_period(
            args.period, since=args.since, end=args.end, last=args.last
        )
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
    if args.out:
        path = rs.write(args.out, units=args.units)
        n_ok = len(rs.accepted())
        print(f"gpsprod: {n_ok}/{len(rs)} stations → {path}", file=sys.stderr)
    else:
        print(f"# {period.dates[0]} .. {period.dates[1]}  {src.frame_name} "
              f"{src.session} ref {src.ref} view {src.view} ends {args.ends} d "
              f"[{args.units}]")  # fmt: skip
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
    p = argparse.ArgumentParser(prog="gps-displacemnts")
    p.add_argument("stations", nargs="+")
    p.add_argument("-s", "--start", required=True)
    p.add_argument("-e", "--end")
    p.add_argument("--file", action="store_true")
    p.add_argument("-f", default="%Y%m%d")
    a = p.parse_args(argv)
    if a.f != "%Y%m%d":
        raise SystemExit("gps-displacemnts: only -f %Y%m%d; use gpsprod displacement")
    new = ["displacement", *a.stations, "--period", f"{a.start}:{a.end or ''}"]
    if a.file:
        new += ["--out", f"{a.start}-{a.end or 'today'}.csv"]
    print("gps-displacemnts is now: gpsprod " + " ".join(new), file=sys.stderr)
    return main(new)
