"""Batch join of GLOBK ``pre``/``rap`` segments into a local TOT directory.

Console script ``gps-globk-tot`` — the local replacement for the okada
``tododaily`` → ``compGLOBK -f TOT -o`` step: for each station it discovers
the ``mb_<STA>_*.dat{1,2,3}`` segments across the Pre and Rap directories,
joins them datum-consistently (:mod:`geo_dataread.globk_join`: 10 m de-wrap
+ min-σ dedup) and writes ``mb_<STA>_TOT.dat{1,2,3}`` files that
``gps_read.openGlobkTimes`` reads directly (3-line header contract — see
:func:`geo_dataread.globk_join.format_tot_file`).

Per-station segment exclusions
------------------------------

Some stations need *explicit, named* segment drops before the join. These
are NOT a gap heuristic — each is a reviewed human decision (the general
name-clash/monument-reuse policy is future work). The reviewed seed rules:

* ``SEY1`` — a 1996 segment from an old reference site that reused the
  station code (name clash): drop segments whose last epoch is before 2020.
* ``SUND`` — the Rap solution is a strict subset of Pre (0 unique epochs)
  and unreliable during the Nov-2023 dike intrusion; dropping the ``rap``
  directory's segments is lossless.

Rules are a CSV with columns ``station,drop_dir,drop_before_year,reason``
— one rule per row, ``reason`` mandatory (every exclusion must be
justified); ``#`` comment lines allowed. The DEPLOYED catalog is
``segment_exclusions.csv``, resolved through the shared
:mod:`gps_parser.outlier_catalogs` mechanism exactly like ``steps.csv``
(``postprocess.cfg`` ``[FILES] segment_exclusions``, else
``<gpsconfig dir>/segment_exclusions.csv``; source of the deployed copy:
``gps-config-data/analysis-lane/segment_exclusions.csv`` — the reviewed
seed rules above live there). ``--exclusions`` is the dev override; with
no deployed catalog and no flag the join proceeds WITHOUT exclusions,
loudly.

Usage::

    gps-globk-tot SENG DYNG REYK \\
        --pre /mnt_data/gps_gmt_data/pre --rap /mnt_data/gps_gmt_data/rap \\
        --out ~/gps-data/TOT [--exclusions segment_exclusions.csv]

Exit status: 0 when every requested station/component was written or was an
intentional exclusion skip; 1 when any component failed (missing segments,
guard trip, …) — failures are reported per component, the batch continues.
"""

from __future__ import annotations

import argparse
import csv
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from gps_parser import outlier_catalogs as _oc

from geo_dataread.globk_join import (
    GlobkJoinError,
    MbSegment,
    discover_segments,
    join_segments,
    read_mb_segment,
    write_joined_series,
)

__all__ = [
    "AXES",
    "EXCLUSIONS_FILENAME",
    "AxisResult",
    "SegmentExclusion",
    "default_exclusions_path",
    "load_exclusions",
    "resolve_exclusions",
    "exclusion_reason",
    "join_station",
    "main",
]

#: mb ``.dat`` file suffix ↔ component letter.
AXES: dict[int, str] = {1: "N", 2: "E", 3: "U"}

EXCLUSIONS_FILENAME = "segment_exclusions.csv"
"""Deployed segment-exclusion catalog filename (gpsconfig-owned)."""

_EXCLUSION_COLUMNS = ("station", "drop_dir", "drop_before_year", "reason")


def default_exclusions_path() -> Path | None:
    """Resolve the deployed ``segment_exclusions.csv`` path via gps_parser.

    Resolution order (the shared
    :func:`gps_parser.outlier_catalogs.catalog_path` mechanism — identical
    to ``steps.csv`` / ``protect_windows.csv``):

    1. ``postprocess.cfg`` ``[FILES] segment_exclusions``;
    2. ``<gpsconfig dir>/segment_exclusions.csv`` (the deploy-target
       default).

    Returns:
        The resolved path (which may not exist yet — the catalog is an
        optional reviewed-policy enhancement), or None when no gpsconfig
        is reachable.
    """
    return cast(
        "Path | None", _oc.catalog_path("segment_exclusions", EXCLUSIONS_FILENAME)
    )


@dataclass(frozen=True)
class SegmentExclusion:
    """One reviewed per-station segment-drop rule.

    A segment is dropped before the join when it matches ``drop_dir`` (its
    parent directory name, e.g. ``"rap"``) or when its **last** epoch is
    before ``drop_before_year`` (fractional year — catches name-clash
    orphans that ended long before the real station started). At least one
    criterion is set; ``reason`` records the human justification verbatim.
    """

    station: str
    reason: str
    drop_dir: str | None = None
    drop_before_year: float | None = None


def load_exclusions(path: Path) -> dict[str, tuple[SegmentExclusion, ...]]:
    """Load per-station segment-exclusion rules from a CSV file.

    Expected columns (exactly): ``station,drop_dir,drop_before_year,reason``
    — one rule per row; several rows per station are allowed; ``#`` comment
    lines and blank lines are skipped (the deployed-catalog convention).
    ``reason`` is mandatory (an unjustified exclusion is a config error,
    not a default) and every row must set ``drop_dir`` and/or
    ``drop_before_year``. Station codes are upper-cased. Raises
    :class:`GlobkJoinError` with the offending row number on any malformed
    row.
    """
    rules: dict[str, list[SegmentExclusion]] = {}
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    reader = csv.DictReader(lines)
    if reader.fieldnames is None or tuple(reader.fieldnames) != _EXCLUSION_COLUMNS:
        raise GlobkJoinError(
            f"{path}: exclusion CSV must have exactly the columns "
            f"{','.join(_EXCLUSION_COLUMNS)!r}, got {reader.fieldnames!r}"
        )
    for lineno, row in enumerate(reader, start=2):
        station = (row["station"] or "").strip().upper()
        reason = (row["reason"] or "").strip()
        drop_dir = (row["drop_dir"] or "").strip() or None
        raw_year = (row["drop_before_year"] or "").strip()
        if not station:
            raise GlobkJoinError(f"{path}:{lineno}: missing station")
        if not reason:
            raise GlobkJoinError(
                f"{path}:{lineno}: missing reason — every exclusion is a "
                "reviewed decision and must be justified"
            )
        drop_before_year: float | None = None
        if raw_year:
            try:
                drop_before_year = float(raw_year)
            except ValueError:
                raise GlobkJoinError(
                    f"{path}:{lineno}: drop_before_year must be a "
                    f"fractional year, got {raw_year!r}"
                ) from None
        if drop_dir is None and drop_before_year is None:
            raise GlobkJoinError(
                f"{path}:{lineno}: rule must set drop_dir and/or drop_before_year"
            )
        rules.setdefault(station, []).append(
            SegmentExclusion(
                station=station,
                reason=reason,
                drop_dir=drop_dir,
                drop_before_year=drop_before_year,
            )
        )
    return {station: tuple(station_rules) for station, station_rules in rules.items()}


def resolve_exclusions(
    explicit: Path | None,
) -> tuple[dict[str, tuple[SegmentExclusion, ...]], str | None]:
    """Resolve + load the exclusion rules (explicit path > deployed catalog).

    An EXPLICIT ``--exclusions`` path must exist and parse (hard error —
    the operator asked for it). With no explicit path the DEPLOYED catalog
    (:func:`default_exclusions_path`) is used when present; an absent
    catalog warns loudly (``UserWarning``) and joins WITHOUT exclusions —
    a name-clash station like SEY1 will then trip the residual guard
    instead of silently joining wrong. A corrupt deployed catalog is a
    hard error (reviewed policy must never be half-applied).

    Returns:
        ``(rules, source)`` — the per-station rules (possibly empty) and
        the resolved catalog path (None when no catalog was available).
    """
    if explicit is not None:
        return load_exclusions(explicit), str(explicit)
    resolved = default_exclusions_path()
    if resolved is None or not resolved.is_file():
        warnings.warn(
            f"no deployed segment-exclusion catalog ({EXCLUSIONS_FILENAME}); "
            "joining WITHOUT per-station exclusions — reviewed drops (e.g. "
            "SEY1 name-clash, SUND rap subset) will NOT be applied",
            UserWarning,
            stacklevel=2,
        )
        return {}, None
    return load_exclusions(resolved), str(resolved)


def exclusion_reason(
    rules: Sequence[SegmentExclusion], path: Path, segment: MbSegment
) -> str | None:
    """Return why this segment is excluded, or None if it is kept.

    Checks the station's rules in order: ``drop_dir`` matches the segment
    file's parent directory name; ``drop_before_year`` matches when the
    segment's last epoch is strictly before the given fractional year.
    """
    for rule in rules:
        if rule.drop_dir is not None and path.parent.name == rule.drop_dir:
            return f"drop_dir={rule.drop_dir}: {rule.reason}"
        if (
            rule.drop_before_year is not None
            and float(segment.epochs[-1]) < rule.drop_before_year
        ):
            return (
                f"last epoch {segment.epochs[-1]:.3f} < "
                f"{rule.drop_before_year}: {rule.reason}"
            )
    return None


@dataclass(frozen=True)
class AxisResult:
    """Outcome of joining one station component (axis 1/2/3 ↔ N/E/U).

    ``status`` is ``"written"`` (file produced), ``"skipped"`` (every
    segment excluded by rule — intentional, not a failure) or ``"error"``
    (parse/join failure, reported loudly but without aborting the batch).
    """

    axis: int
    status: str
    detail: str


def join_station(
    station: str,
    segment_dirs: Sequence[Path],
    out_dir: Path,
    exclusions: Mapping[str, Sequence[SegmentExclusion]] | None = None,
) -> list[AxisResult]:
    """Join all three components of one station and write its TOT files.

    For each axis: discover segments across ``segment_dirs`` (typically
    ``[pre, rap]``), drop any matching the station's exclusion rules, join
    the rest (:func:`geo_dataread.globk_join.join_segments`) and write
    ``mb_<station>_TOT.dat<axis>`` into ``out_dir``. A
    :class:`~geo_dataread.globk_join.GlobkJoinError` on one axis (missing
    segments, residual-guard trip, …) is reported as an ``"error"`` result
    and the remaining axes still run; unexpected exceptions propagate.
    """
    rules: Sequence[SegmentExclusion] = (exclusions or {}).get(station, ())
    results: list[AxisResult] = []
    for axis in sorted(AXES):
        try:
            segments: list[MbSegment] = []
            n_excluded = 0
            for segment_path in discover_segments(station, axis, segment_dirs):
                segment = read_mb_segment(segment_path)
                if exclusion_reason(rules, segment_path, segment) is not None:
                    n_excluded += 1
                    continue
                segments.append(segment)
            if not segments:
                results.append(
                    AxisResult(axis, "skipped", "all segments excluded by rule")
                )
                continue
            joined = join_segments(segments)
        except GlobkJoinError as exc:
            results.append(AxisResult(axis, "error", str(exc)))
            continue
        out_path = write_joined_series(joined, out_dir / f"mb_{station}_TOT.dat{axis}")
        detail = f"{joined.epochs.size} epochs -> {out_path.name}"
        if n_excluded:
            detail += f" ({n_excluded} segment(s) excluded by rule)"
        results.append(AxisResult(axis, "written", detail))
    return results


#: ``postprocess.cfg`` key behind each directory flag, and whether the
#: resolved directory must already exist. ``pre``/``rap`` are INPUTS, so a
#: missing one means the join silently produces nothing; ``out`` is created.
DIR_OPTIONS: dict[str, tuple[str, bool]] = {
    "pre": ("pre_path", True),
    "rap": ("rap_path", True),
    "out": ("tot_dir", False),
}


def resolve_dir(explicit: Path | None, flag: str) -> tuple[Path, str]:
    """Resolve one directory: explicit flag > ``postprocess.cfg``.

    Same precedence as :func:`resolve_exclusions` — what the operator typed
    wins, otherwise the deployed config decides — and the source is reported,
    so a run is reproducible from its own output.

    A missing INPUT directory is a hard error naming both the flag and the
    config key, because the failure it prevents is silent: the legacy
    ``[Configs] prePath``/``rapPath`` still point at a 2013
    ``aqaba-reference_test`` path that does not exist, and joining from a
    missing directory writes nothing while exiting 0.

    Returns:
        ``(path, source)`` — the resolved path and where it came from.
    """
    option, must_exist = DIR_OPTIONS[flag]
    if explicit is not None:
        path, source = explicit.expanduser(), f"--{flag}"
    else:
        from gps_parser import ConfigParser

        try:
            path = Path(ConfigParser().getPostProcessDir(option)).expanduser()
        except Exception as exc:
            raise SystemExit(
                f"error: no --{flag} given and postprocess.cfg has no '{option}': {exc}"
            ) from None
        source = f"postprocess.cfg {option}"

    if must_exist and not path.is_dir():
        raise SystemExit(
            f"error: {flag} directory does not exist: {path}  (from {source})\n"
            f"       Joining from a missing input writes nothing, so this is "
            f"refused rather than reported as success. Pass --{flag}, or set "
            f"'{option}' in [PATHS] of postprocess.cfg."
        )
    return path, source


def resolve_stations(explicit: Sequence[str], out_dir: Path) -> tuple[list[str], str]:
    """The stations to join: those named, else those already joined.

    Defaulting to the CONTENTS of the output directory makes a bare
    ``gps-globk-tot`` a pure refresh — same station set as last time, newer
    data — which is what the documented ``xargs`` procedure did. A new
    station is therefore opt-in (name it once). That is deliberate:
    ``pre``/``rap`` carry hundreds of global IGS reference sites that are not
    plotted here, so defaulting to "everything upstream" would silently grow
    the dataset by an order of magnitude.

    Returns:
        ``(stations, source)``. Exits rather than returning empty, so a first
        run against an empty directory cannot report success having done
        nothing.
    """
    if explicit:
        return [s.upper() for s in explicit], "command line"

    found = sorted(
        {
            m.group(1)
            for p in out_dir.glob("mb_*_TOT.dat*")
            if (m := re.fullmatch(r"mb_(\w{4})_TOT\.dat\d", p.name))
        }
    )
    if not found:
        raise SystemExit(
            f"error: no stations given and none already joined in {out_dir}. "
            f"Name the stations explicitly for the first run."
        )
    return found, f"already joined in {out_dir}"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: batch stations' pre/rap segments into a TOT dir."""
    parser = argparse.ArgumentParser(
        prog="gps-globk-tot",
        description=(
            "Join GLOBK pre/rap multibase segments into local "
            "mb_<STA>_TOT.dat{1,2,3} files (10 m de-wrap + min-sigma dedup)."
        ),
    )
    parser.add_argument(
        "stations",
        nargs="*",
        help=(
            "4-char station codes. Default: every station already joined in "
            "the output directory, i.e. a refresh of the existing dataset"
        ),
    )
    parser.add_argument(
        "--pre",
        type=Path,
        default=None,
        help="Pre (prior years) segment directory (default: config pre_path)",
    )
    parser.add_argument(
        "--rap",
        type=Path,
        default=None,
        help="Rap (current year) segment directory (default: config rap_path)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output TOT directory, created (default: config tot_dir)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "resolve and print the directories and station list, then stop "
            "without reading or writing any data"
        ),
    )
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=None,
        help=(
            "per-station segment-exclusion CSV dev override "
            "(columns: station,drop_dir,drop_before_year,reason); "
            "default: the deployed segment_exclusions.csv via gps_parser"
        ),
    )
    args = parser.parse_args(argv)

    # Directories first: every one is reported with its source, because with
    # config defaults in play "which directories did this run actually use"
    # is no longer visible in the command that started it.
    pre_dir, pre_src = resolve_dir(args.pre, "pre")
    rap_dir, rap_src = resolve_dir(args.rap, "rap")
    out_dir, out_src = resolve_dir(args.out, "out")
    stations, sta_src = resolve_stations(args.stations, out_dir)

    exclusions, exclusions_source = resolve_exclusions(
        args.exclusions.expanduser() if args.exclusions else None
    )
    print(f"pre:        {pre_dir}  ({pre_src})")
    print(f"rap:        {rap_dir}  ({rap_src})")
    print(f"out:        {out_dir}  ({out_src})")
    print(f"stations:   {len(stations)}  ({sta_src})")
    if exclusions_source is not None:
        print(f"exclusions: {exclusions_source}")
    if args.dry_run:
        print("\ndry run — nothing read, nothing written")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    segment_dirs = [pre_dir, rap_dir]

    n_errors = 0
    for station in stations:
        print(f"== {station} ==")
        for result in join_station(station, segment_dirs, out_dir, exclusions):
            print(
                f"   {AXES[result.axis]} (dat{result.axis}): "
                f"[{result.status}] {result.detail}"
            )
            if result.status == "error":
                n_errors += 1
    return 1 if n_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
