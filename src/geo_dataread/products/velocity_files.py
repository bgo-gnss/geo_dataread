"""The velocity product's files: .gmt/.csv/.jsonl sets, byte-stable (moved
from geo_dataread.velocity in gpsprod step 3; bodies unchanged)."""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from geo_dataread.products.velocity_core import (
    _COORD_DECIMALS,
    _RATE_DECIMALS,
    SCHEMA_VERSION,
    VelocityRecord,
    VelocitySettings,
)

# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------


def _fmt(v: float, decimals: int) -> str:
    return "" if not math.isfinite(v) else f"{v:.{decimals}f}"


CSV_COLUMNS = (
    "station", "lon", "lat", "ve", "vn", "vu", "sve", "svn", "svu", "corr_en",
    "n_epochs", "window_start", "window_end", "ref", "frame", "plate",
    "sigma_method", "k_e", "k_n", "k_u", "flags", "refused",
    "tier", "snr_h", "snr_u", "parts_refused",
)  # fmt: skip


def write_velocity_set(
    records: Sequence[VelocityRecord], directory: str | Path, name: str
) -> dict[str, Path]:
    """Write ``<name>.gmt``, ``.csv`` and ``.jsonl`` under ``directory``.

    Byte-stable: rows sorted by station, fixed decimals, no timestamps.

    Raises:
        ValueError: records in more than one frame, or none.
    """
    if not records:
        raise ValueError("no velocity records to write")
    frames = {r.frame for r in records}
    if len(frames) > 1:
        raise ValueError(f"a velocity set holds one frame, got {sorted(frames)}")
    rows = sorted(records, key=lambda r: (r.station, r.window))
    out_dir = Path(directory).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    (frame,) = frames
    head = (
        f"# gps-velocities schema {SCHEMA_VERSION}, set {name}, frame {frame}, "
        f"{sum(r.accepted for r in rows)} of {len(rows)} stations accepted\n"
    )
    n_h = sum(r.accepted and bool(np.isfinite(r.rate[:2]).all()) for r in rows)
    paths = {
        "gmt": out_dir / f"{name}.gmt",
        "csv": out_dir / f"{name}.csv",
        "jsonl": out_dir / f"{name}.jsonl",
    }
    d, c = _RATE_DECIMALS, _COORD_DECIMALS
    with open(paths["gmt"], "w", encoding="utf-8") as fh:
        fh.write(head.replace("accepted", f"accepted, {n_h} with a horizontal rate")
                 + "# lon lat ve vn sve svn corr_en station  [mm/yr]\n")  # fmt: skip
        for r in rows:
            if not r.accepted or not np.isfinite(r.rate[:2]).all():
                continue  # refused, or a short-tier record with up alone
            vn, ve, _vu = r.rate
            sn, se, _su = r.sigma
            fh.write(
                f"{r.lon:.{c}f} {r.lat:.{c}f} {ve:.{d}f} {vn:.{d}f} {se:.{d}f} "
                f"{sn:.{d}f} {r.corr_en:.{d}f} {r.station}\n"
            )
    with open(paths["csv"], "w", encoding="utf-8") as fh:
        fh.write(head + ",".join(CSV_COLUMNS) + "\n")
        for r in rows:
            vn, ve, vu = r.rate
            sn, se, su = r.sigma
            kn, ke, ku = r.k
            cells = [
                r.station, _fmt(r.lon, c), _fmt(r.lat, c),
                _fmt(ve, d), _fmt(vn, d), _fmt(vu, d),
                _fmt(se, d), _fmt(sn, d), _fmt(su, d), _fmt(r.corr_en, d),
                str(min(r.n_epochs)), f"{r.window[0]:.4f}", f"{r.window[1]:.4f}",
                r.ref, r.frame, r.plate or "", r.sigma_method,
                _fmt(ke, 2), _fmt(kn, 2), _fmt(ku, 2),
                "|".join(r.flags), (r.refused or "").replace(",", ";"),
                r.tier, _fmt(r.snr[0], 1), _fmt(r.snr[1], 1),
                "|".join(f"{k}:{v}" for k, v in sorted(r.parts_refused.items())),
            ]  # fmt: skip
            fh.write(",".join(cells) + "\n")
    with open(paths["jsonl"], "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
    return paths


def read_velocity_records(path: str | Path) -> list[VelocityRecord]:
    """The records of a ``.jsonl`` velocity set (a ``.csv``/``.gmt`` path
    reads its ``.jsonl`` sibling)."""
    p = Path(path).expanduser()
    if p.suffix != ".jsonl":
        p = p.with_suffix(".jsonl")
    with open(p, encoding="utf-8") as fh:
        return [
            VelocityRecord.from_dict(json.loads(line)) for line in fh if line.strip()
        ]


def set_name(settings: VelocitySettings, start: dt.date, end: dt.date) -> str:
    """``window-<start>-<end>_<ref>_<frame>``, the dates as YYYYMMDD
    (both included)."""
    import geo_dataread.gps_read as gpsr
    from geo_dataread.pos import pos_frame

    ref = settings.ref
    if settings.pre_window is not None:  # the reference window in the name
        p0, p1 = settings.pre_window
        dts = gpsr.toDateTime([p0 + 1e-6, p1 - 1e-6])  # type: ignore[no-untyped-call]
        days = [d.date() for d in dts]
        ref = f"{ref}-{days[0]:%Y%m%d}-{days[1]:%Y%m%d}"
    return f"window-{start:%Y%m%d}-{end:%Y%m%d}_{ref}_{pos_frame(settings.frame)}"
