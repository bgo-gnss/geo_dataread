"""``gpsprod transient``: the velocity on a date from a fitted recharge curve."""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

TRANSIENT_EPILOG = """\
what it is:
  after an event (an eruption that drained a reservoir) the stations move on
  a curve: fast at first, then slowing to a steady rate. Every station and
  component is fitted from the start of the period with

      x(t) = offset + v*t + phi*(1 - exp(-(t - t0)/tau))

  and the rate on a date is the curve's slope there:

      dx/dt = v + (phi/tau)*exp(-(t - t0)/tau)

  v is the steady rate the curve approaches. tau is ONE decay time for all
  the stations of the run (one source): estimated jointly (the default),
  given (--half-life), or read from an earlier run (--decay). The half-life
  is ln2*tau. Its sigma is carried into every rate's sigma. Pick stations
  that see the one source (a box or area around it); stations with another
  source are flagged tau_differs when their own curve is well measured.

  t0 is the first day of the period: start it the day after the eruption
  ended (--period after:EVENT does that: an events.yaml eruption's last day
  is its end).

which numbers to use:
  the rate on a date (default: the newest day with data) is the station's
  velocity now. Months after an event it is close to v; right after it, much
  faster. A straight line over the same months (gpsprod velocity) averages
  the curve and lags it.

periods: as gpsprod displacement (--start/--end/--days, or --period with
  dates and events.yaml names: after:EVENT, DIKE:ERUPTION ...)

examples:
  gpsprod transient --area reykjanes --period after:sundhnukur-2025-07-eruption
  gpsprod transient NBIO ELDC SUDV --start 20250806
  gpsprod transient --region=-22.72/-22.17/63.76/63.945 --start 20250806 \\
                    --at 20260101 --at 20261025 -o ~/tmp/svartsengi
      # svartsengi-at20260101.{gmt,csv,jsonl}, -at20261025..., and
      # svartsengi.decay.json (tau, its sigma, the stations it came from)
  gpsprod transient SENG THOB --start 20250806 --decay ~/tmp/svartsengi.decay.json
      # the same tau again: SENG's rate no longer depends on who else ran
  gpsprod transient SENG --start 20250806 --half-life 60

  draw it: gpsplot map velocity --velocities ~/tmp/svartsengi-at20261025.jsonl ...

flags:
  sigma:formal          sigma = formal x sqrt(chi2/nu), with tau's sigma
                        added: no coloured-noise scale for curves yet, so
                        smaller than gpsprod velocity's sigma
  tau_differs:<N>d      the station's own decay is well measured and differs
                        from the shared one (> 5 sigma): its own half-life N
  rejected:N            N epochs off the curve by > 4 robust sigma, dropped
  extrapolated:Nd       the date is N days past the station's last data
  steps_in_window       --allow-steps: a declared step inside, fitted as an
                        offset (that station sits out of the shared tau)
  fallback:plate, seasonal:regional, truncated@T   as gpsprod velocity

refused: too few epochs (< 30) or too short (< 90 days) after t0, a declared
  step inside (without --allow-steps), no detrend record (--no-record refuse).

output: a table on screen; -o DIR/NAME writes NAME.gmt, NAME.csv, NAME.jsonl
  as NAME-atYYYYMMDD.{gmt,csv,jsonl} (the velocity product's files, one set
  per date) and NAME.decay.json. The .jsonl rows carry 'at' (the date of the rate) and
  'transient' (t0, tau, half-life, per component v, phi and A = -phi).
"""


def add_parser(sub: Any, raw: type[argparse.HelpFormatter]) -> None:
    from geo_dataread.products.cli import _common
    from geo_dataread.products.velocity_core import NO_RECORD_POLICIES

    p = sub.add_parser(
        "transient",
        help="the velocity on a date from a fitted recharge curve (after an event)",
        formatter_class=raw,
        description=(
            "Velocities N/E/U [mm/yr] on a date, from one curve per station: a "
            "steady rate plus\nan exponential with one decay time shared by the "
            "stations.\nDefaults: itrf14 .pos, plate and seasonal removed, "
            "declared view, sigma screen 10 mm,\ntau estimated jointly, the rate "
            "on the newest day with data."
        ),
        epilog=TRANSIENT_EPILOG,
    )
    _common(
        p, refs=("plate-periodic", "background", "plate"),
        ref_help="plate-periodic (default: plate and the stored seasonal removed), "
        "background (the whole stored trajectory removed), plate (plate only)",
        uncert_help="drop solutions with a formal σ above MM (default 10)",
        out_help="write DIR/NAME.gmt, .csv, .jsonl and NAME.decay.json "
        "(default: a table on screen)",
        units=False,
    )  # fmt: skip
    g = p.add_argument_group("transient")
    g.add_argument(
        "--at", action="append", metavar="DATE",
        help="the date to give the rate on (repeat, or DATE,DATE); default the "
        "newest day with data",
    )  # fmt: skip
    tau = g.add_mutually_exclusive_group()
    tau.add_argument(
        "--half-life", type=float, metavar="DAYS",
        help="a fixed half-life instead of estimating one (tau = DAYS/ln2)",
    )  # fmt: skip
    tau.add_argument(
        "--decay", metavar="FILE",
        help="the tau of an earlier run's NAME.decay.json (its t0 must match)",
    )  # fmt: skip
    g.add_argument(
        "--view", choices=("declared", "cleaned", "raw"), default="declared",
        help="declared (default: known blunders out), cleaned, raw",
    )  # fmt: skip
    g.add_argument(
        "--no-record", choices=NO_RECORD_POLICIES, default="plate",
        help="a station without a detrend record: ref plate (default, flagged) or refuse",
    )  # fmt: skip
    g.add_argument(
        "--allow-steps", action="store_true",
        help="fit a declared step inside the period as an offset (flagged)",
    )  # fmt: skip


def _dates(values: Sequence[str] | None) -> list[dt.date]:
    from geo_dataread.products.period import parse_date

    out: list[dt.date] = []
    for v in values or ():
        out += [parse_date(x) for x in v.split(",") if x.strip()]
    return out


def _noon(day: dt.date) -> float:
    from geo_dataread.products.velocity_core import window_bounds

    return window_bounds(day, day)[0] + 0.5 / 365.25


def _day(yearf: float) -> dt.date:
    from gtimes.timefunc import TimefromYearf

    return dt.date.fromisoformat(str(TimefromYearf(yearf, "%Y-%m-%d")))


def _show(records: Sequence[Any]) -> None:
    print(f"{'sta':4} {'vn':>8} {'ve':>8} {'vu':>8} {'sn':>6} {'se':>6} {'su':>6}"
          f" {'v∞ n':>8} {'v∞ e':>8} {'v∞ u':>8}  flags")  # fmt: skip
    for r in records:
        if r.accepted:
            v = r.transient["v"]
            nums = (f"{r.rate[0]:8.1f} {r.rate[1]:8.1f} {r.rate[2]:8.1f} "
                    f"{r.sigma[0]:6.1f} {r.sigma[1]:6.1f} {r.sigma[2]:6.1f} "
                    f"{v[0]:8.1f} {v[1]:8.1f} {v[2]:8.1f}")  # fmt: skip
            note = "|".join((r.ref, *r.flags))
        else:
            nums = f"{'refused':>8}" + " " * 73
            note = r.refused or ""
        print(f"{r.station:4} {nums}  {note}")


def run(args: argparse.Namespace) -> int:
    from geo_dataread.pos import frame_name
    from geo_dataread.products.cli import UNCERT_RATE, select
    from geo_dataread.products.loader import Source
    from geo_dataread.products.period import resolve_period
    from geo_dataread.products.transient import (
        Decay,
        estimate_transients,
        read_decay,
        tau_from_half_life,
        write_decay,
    )
    from geo_dataread.products.velocity_core import VelocitySettings, window_bounds
    from geo_dataread.products.velocity_files import write_velocity_set

    try:
        if args.ref and args.keep_seasonal and args.ref != "plate":
            raise ValueError("--keep-seasonal is --ref plate; not both")
        ref = args.ref or ("plate" if args.keep_seasonal else "plate-periodic")
        settings = VelocitySettings(
            frame=frame_name(args.frame), ref=ref, view=args.view,
            uncert=args.uncert if args.uncert is not None else UNCERT_RATE,
            no_record=args.no_record, allow_steps=args.allow_steps,
            pos_dir=args.pos_dir, params=args.params, steps=args.steps,
            session=args.session, relative_to=args.relative_to,
        )  # fmt: skip
        period = resolve_period(
            args.period, start=args.start, end=args.end, days=args.days
        )
        t0, t1 = window_bounds(period.first_day, period.last_day)
        at = [_noon(d) for d in _dates(args.at)] or None
        decay: Decay | None = None
        if args.decay:
            decay = read_decay(args.decay)
        elif args.half_life is not None:
            if not args.half_life > 0:
                raise ValueError("--half-life must be > 0 days")
            decay = Decay(t0=t0, tau=tau_from_half_life(args.half_life),
                          tau_sigma=0.0, source=f"half-life {args.half_life:g} d")  # fmt: skip
        src = Source(frame=args.frame, session=args.session, pos_dir=args.pos_dir)
    except (ValueError, OSError) as err:
        raise SystemExit(f"gpsprod: {err}") from None
    stations = select(args.stations, area=args.area, region=args.region, source=src)
    if not stations:
        raise SystemExit("gpsprod: no stations (give markers, --area, --region or all)")

    def progress(sta: str) -> None:
        if not args.quiet:
            print(f"  {sta}", file=sys.stderr, end="\r")

    try:
        records, decay = estimate_transients(
            stations, t0, t1, at, settings, decay=decay, progress=progress
        )
    except ValueError as err:
        raise SystemExit(f"gpsprod: {err}") from None
    if not args.quiet:
        print(" " * 12, file=sys.stderr, end="\r")
    epochs = sorted({r.at for r in records if r.at is not None})
    lo, hi = decay.tau_interval or (math.nan, math.nan)
    head = (f"# t0 {period.first_day} .. {period.last_day}  {settings.frame} ref "
            f"{ref} view {settings.view}  half-life {decay.half_life_days:.1f} d "
            f"(tau {decay.tau:.4f} ± {decay.tau_sigma:.4f} yr"
            + (f", {lo:.4f}..{hi:.4f}" if decay.tau_interval else "")
            + f"; {decay.source}"
            + (f", {len(decay.stations)} stations" if decay.stations else "")
            + ")")  # fmt: skip
    if args.out:
        import dataclasses

        out = Path(args.out).expanduser()
        out_dir, name = out.parent, (out.stem if out.suffix else out.name)
        dpath = out_dir / f"{name}.decay.json"
        if decay.source == "shared":  # the rows name the file they can be rerun with
            decay = dataclasses.replace(decay, source=str(dpath))
            records = [
                dataclasses.replace(
                    r,
                    k_source={**r.k_source, "decay": str(dpath)},
                    transient=(
                        {**r.transient, "decay": str(dpath)} if r.transient else None
                    ),
                )
                for r in records
            ]
        written = []
        for a in epochs:
            rows = [r for r in records if r.at == a]
            # always dated: the .csv/.gmt have no 'at' column, and their
            # window is the fit span, not an averaging window
            set_name = f"{name}-at{_day(a):%Y%m%d}"
            paths = write_velocity_set(rows, out_dir, set_name)
            written.append(paths["csv"].with_suffix(""))
        dpath = write_decay(decay, dpath)
        n_ok = len({r.station for r in records if r.accepted})
        print(head, file=sys.stderr)
        print(f"gpsprod: {n_ok}/{len(stations)} stations → "
              + ", ".join(f"{p}.{{gmt,csv,jsonl}}" for p in written)
              + f", {dpath}", file=sys.stderr)  # fmt: skip
        return 0
    print(head)
    for a in epochs:
        print(f"# rate on {_day(a)} [mm/yr]; v∞ = the steady rate")
        _show([r for r in records if r.at == a])
    return 0
