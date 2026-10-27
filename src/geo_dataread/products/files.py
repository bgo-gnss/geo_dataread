"""Writers of the gpsprod record sets; the format comes from the file name.

- ``.csv`` — ``#`` provenance header, then one row per record (refused ones
  too, with the reason); fixed decimals, station order, no timestamps, so
  a re-run on the same data writes the same bytes.
- ``.jsonl`` — one record per line, every field (``to_dict``).
- ``-`` — the screen table.

- ``.gmt`` — displacements as GMT psvelo rows (``lon lat dE dN sE sN corr
  station``, mm; refused stations left out) for ``gmt velo -Se``/pygmt.
- ``.gbis.txt`` — displacements for GBIS (Bagnardi & Hooper 2018) in metres:
  ``lon lat dE dN dU sE sN sU station``. GBIS itself reads GPS from a .mat
  (``ll``, ``displacements``, ``covariance``): this is the text those are
  made from.
- ``.xlsx`` — a ``data`` sheet (the CSV columns) and a ``meta`` sheet (the
  provenance), standard library only (:mod:`.xlsx`).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from geo_dataread.products.common import (
    DisplacementRecord,
    RecordSet,
    provenance_header,
)

DISPLACEMENT_COLUMNS = (
    "station", "lon", "lat", "start", "end",
    "dN", "dE", "dU", "sN", "sE", "sU", "units",
    "n_start", "n_end", "method", "ref", "view", "frame", "session",
    "relative_to", "flags", "refused",
)  # fmt: skip
FORMATS = (".csv", ".jsonl", ".gmt", ".gbis.txt", ".xlsx")


def suffix_of(path: str | Path) -> str:
    """The format suffix of a file name (``.gbis.txt`` is one)."""
    name = Path(path).name.lower()
    if name.endswith(".gbis.txt"):
        return ".gbis.txt"
    return Path(name).suffix


def _fmt(v: float, decimals: int) -> str:
    return "" if not math.isfinite(v) else f"{v:.{decimals}f}"


def _ends_n(rec: DisplacementRecord, key: str) -> int | str:
    """Epochs behind the start/end: the end windows (ends), the lines
    before/after the event (step)."""
    if rec.ends:
        return min(int(c["n"]) for c in rec.ends[key].values())
    if rec.step:
        side = rec.step["lines"]["before" if key == "start" else "after"]
        return min(int(c["n"]) for c in side.values())
    return ""


COORD_COLUMNS = (
    "station", "epoch", "X", "Y", "Z", "sX", "sY", "sZ",
    "lat", "lon", "h", "sN", "sE", "sU", "dN", "dE", "dU",
    "n_used", "t_obs", "method", "view", "frame", "session",
    "relative_to", "flags", "refused",
)  # fmt: skip


def rows(records: Sequence[Any]) -> tuple[tuple[str, ...], list[list[Any]]]:
    """(columns, rows) of a set, values formatted for text."""
    if records and getattr(records[0], "kind", "") == "coords":
        return _coord_rows(records)
    return _displacement_rows(records)


def _coord_rows(records: Sequence[Any]) -> tuple[tuple[str, ...], list[list[Any]]]:
    out = []
    for r in records:
        dneu = r.dneu if r.dneu is not None else (math.nan,) * 3
        out.append(
            [
                r.station,
                r.epoch_date,
                *(_fmt(v, 5) for v in r.xyz),
                *(_fmt(v, 5) for v in r.sigma_xyz),
                _fmt(r.llh[0], 9),
                _fmt(r.llh[1], 9),
                _fmt(r.llh[2], 5),
                *(_fmt(v, 5) for v in r.sigma_neu),
                *(_fmt(v, 5) for v in dneu),
                r.n_used,
                _fmt(r.t_obs, 5),
                r.method,
                r.view,
                r.frame,
                r.session,
                r.relative_to or "",
                ";".join(r.flags),
                r.refused or "",
            ]  # fmt: skip
        )
    return COORD_COLUMNS, out


def _displacement_rows(
    records: Sequence[DisplacementRecord],
) -> tuple[tuple[str, ...], list[list[Any]]]:
    """(columns, rows) of a displacement set, values formatted for text."""
    out = []
    for r in records:
        dec = 3 if r.units == "mm" else 6
        out.append(
            [
                r.station,
                _fmt(r.lon, 6),
                _fmt(r.lat, 6),
                r.period_dates[0],
                r.period_dates[1],
                *(_fmt(v, dec) for v in r.displacement),
                *(_fmt(v, dec) for v in r.sigma),
                r.units,
                _ends_n(r, "start"),
                _ends_n(r, "end"),
                r.method,
                r.ref,
                r.view,
                r.frame,
                r.session,
                r.relative_to or "",
                ";".join(r.flags),
                r.refused or "",
            ]  # fmt: skip
        )
    return DISPLACEMENT_COLUMNS, out


def _csv_cell(v: Any) -> str:
    text = str(v)
    return f'"{text}"' if any(c in text for c in ',"\n') else text


def write_csv(rs: RecordSet[Any], path: Path, records: Sequence[Any]) -> None:
    cols, data = rows(records)
    lines = [f"# {line}" for line in provenance_header(rs)]
    lines.append(",".join(cols))
    lines += [",".join(_csv_cell(v) for v in row) for row in data]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Sequence[Any]) -> None:
    text = "".join(
        json.dumps(r.to_dict(), sort_keys=True, ensure_ascii=False) + "\n"
        for r in records
    )
    path.write_text(text, encoding="utf-8")


def table(records: Sequence[Any]) -> str:
    """The screen table: one aligned line per station."""
    cols, data = rows(records)
    coords = cols is COORD_COLUMNS
    keep: tuple[str, ...]
    if coords:
        relative = any(getattr(r, "dneu", None) is not None for r in records)
        keep = (
            ("station", "dN", "dE", "dU", "sN", "sE", "sU", "n_used", "flags", "refused")
            if relative
            else ("station", "X", "Y", "Z", "lat", "lon", "h", "sN", "sE", "sU",
                  "n_used", "flags", "refused")
        )  # fmt: skip
    else:
        keep = (
            "station", "dN", "dE", "dU", "sN", "sE", "sU",
            "n_start", "n_end", "flags", "refused",
        )  # fmt: skip
    idx = [cols.index(c) for c in keep]
    grid = [[str(c) for c in keep]] + [[str(row[i]) for i in idx] for row in data]
    widths = [max(len(r[j]) for r in grid) for j in range(len(keep))]
    right = {"dN", "dE", "dU", "sN", "sE", "sU", "n_start", "n_end", "X", "Y", "Z",
             "lat", "lon", "h", "n_used"}  # fmt: skip
    lines = []
    for r in grid:
        cells = [
            v.rjust(w) if keep[j] in right else v.ljust(w)
            for j, (v, w) in enumerate(zip(r, widths, strict=True))
        ]
        lines.append("  ".join(cells).rstrip())
    return "\n".join(lines)


def write_gmt(rs: RecordSet[Any], path: Path) -> None:
    """Accepted displacements as psvelo rows [mm]."""
    if rs.kind != "displacement":
        raise ValueError(f".gmt: displacements only (this set is {rs.kind})")
    lines = [f"# {x}" for x in provenance_header(rs)]
    lines.append("# lon lat dE dN sE sN corr_en station  [mm]")
    for r in rs.records:
        if not r.accepted:
            continue
        r = r.scaled("mm")
        dn, de, _du = r.displacement
        sn, se, _su = r.sigma
        lines.append(
            f"{r.lon:.6f} {r.lat:.6f} {de:.3f} {dn:.3f} {se:.3f} {sn:.3f} 0.000 {r.station}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gbis(rs: RecordSet[Any], path: Path) -> None:
    """Accepted displacements for GBIS, in metres."""
    if rs.kind != "displacement":
        raise ValueError(f".gbis.txt: displacements only (this set is {rs.kind})")
    lines = [f"% {x}" for x in provenance_header(rs)]
    lines.append("% lon lat dE dN dU sE sN sU station  [m]")
    for r in rs.records:
        if not r.accepted:
            continue
        m = r.scaled("m")
        dn, de, du = m.displacement
        sn, se, su = m.sigma
        lines.append(
            f"{r.lon:.6f} {r.lat:.6f} {de:.6f} {dn:.6f} {du:.6f} "
            f"{se:.6f} {sn:.6f} {su:.6f} {r.station}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_xlsx_set(rs: RecordSet[Any], path: Path, records: Sequence[Any]) -> None:
    """A ``data`` sheet (the CSV columns, numbers as numbers) and a ``meta``
    sheet (the provenance header)."""
    from geo_dataread.products.xlsx import write_xlsx

    cols, data = rows(records)
    meta = [
        line.split(" ", 1) if " " in line else [line, ""]
        for line in provenance_header(rs)
    ]
    write_xlsx(
        path, [("data", [list(cols), *data]), ("meta", [["key", "value"], *meta])]
    )


def write(rs: RecordSet[Any], path: str, *, units: str = "mm") -> str:
    """Write ``rs`` to ``path`` (format by extension); returns the path."""
    records = [r.scaled(units) for r in rs.records]
    target = Path(path).expanduser()
    suffix = suffix_of(target)
    if suffix not in FORMATS:
        raise ValueError(f"{path}: unknown format {suffix!r}; one of {FORMATS}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        write_csv(rs, target, records)
    elif suffix == ".jsonl":
        write_jsonl(target, records)
    elif suffix == ".gmt":
        write_gmt(rs, target)
    elif suffix == ".gbis.txt":
        write_gbis(rs, target)
    else:
        write_xlsx_set(rs, target, records)
    return str(target)
