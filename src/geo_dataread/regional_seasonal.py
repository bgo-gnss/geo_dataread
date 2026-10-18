"""gps-regional-seasonal: seasonal-only records interpolated from neighbours.

A station with no detrend record of its own (on Reykjanes: every station
installed in the unrest) gets its seasonal from nearby stations' estimated
ones (:func:`gps_analysis.regional.interpolate_seasonal`, BGÓ 2026-10-17;
design: ``docs/regional-seasonal.md``). The result goes into the same
detrend document the estimator writes (``detrend_params.<frame>.json``), as
a *seasonal-only* record:

- ``model: "periodic"``, the four periodic ``param_names``, no offset/rate;
- ``seasonal_only: true`` and ``detrend_method: "regional"`` — the leaf
  refuses anything but ``terms="periodic"`` on it, so it never passes for
  the station's background;
- ``groups.periodic``: provenance ``regional:…``, the donors with their
  weights, distances, ``fitted_at`` and a digest, the leave-one-out
  prediction σ.

Every other record gains explicit ``groups.<g>.provenance = "estimated"``
(its terms were fitted at the station). Donors are records whose periodic
group was estimated — never a regional or borrowed one — so predictions
cannot chain and do not depend on order.

The input document is read, never written: ``--out`` gets the merged
document, deployed by the operator (``cp -f`` after a backup).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

#: the periodic parameter names, in the order the coefficients are stored
PERIODIC_NAMES = ("cos_annual", "sin_annual", "cos_semiannual", "sin_semiannual")
#: the area the method was validated on (leave-one-out, 2026-10-17): W/E/S/N
REYKJANES_REGION = (-22.9, -21.7, 63.75, 64.1)
ESTIMATED = "estimated"

Coords = Callable[[str], tuple[float, float]]


#: the fields a donor's seasonal comes from: a change here is a changed
#: donor; provenance bookkeeping (``groups``) is not
_DIGEST_KEYS = ("model", "param_names", "components", "fitted_at", "window", "frame")


def _digest(record: Mapping[str, Any]) -> str:
    fit = {k: record.get(k) for k in _DIGEST_KEYS}
    raw = json.dumps(fit, sort_keys=True, default=str).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()[:16]


def group_provenance(record: Mapping[str, Any], group: str) -> str:
    """How a record's term group was derived: ``estimated`` (also for a
    record without a ``groups`` block and the staged spelling ``self``),
    else the stored provenance (``regional:…``, ``store:…``, ``donor:…``)."""
    entry = (record.get("groups") or {}).get(group) or {}
    prov = str(entry.get("provenance") or ESTIMATED)
    return ESTIMATED if prov == "self" else prov


def periodic_coeffs(record: Mapping[str, Any]) -> np.ndarray | None:
    """(3, 4) periodic coefficients of a record by NAME, or None."""
    names = list(record.get("param_names") or ())
    comps = record.get("components") or ()
    if len(comps) != 3 or not all(n in names for n in PERIODIC_NAMES):
        return None
    idx = [names.index(n) for n in PERIODIC_NAMES]
    out = np.array([[float(c["params"][i]) for i in idx] for c in comps])
    return out if np.all(np.isfinite(out)) else None


def is_donor(record: Mapping[str, Any]) -> bool:
    """A station's own estimated seasonal: not seasonal-only, not borrowed
    (a prior-frame carry-over or a cross-station hold), periodic estimated."""
    return (
        not record.get("seasonal_only")
        and not record.get("borrowed")
        and group_provenance(record, "periodic") == ESTIMATED
        and periodic_coeffs(record) is not None
    )


def collect_donors(
    stations: Mapping[str, Mapping[str, Any]], coords: Coords
) -> tuple[dict[str, tuple[float, float, np.ndarray]], dict[str, str]]:
    """(code → (lon, lat, coeffs)) of every donor, and their digests."""
    donors: dict[str, tuple[float, float, np.ndarray]] = {}
    digests: dict[str, str] = {}
    for sta, rec in sorted(stations.items()):
        if not rec or not is_donor(rec):
            continue
        try:
            lon, lat = coords(sta)
        except Exception:  # noqa: BLE001 - gps_parser raises bare Exception
            continue
        coeffs = periodic_coeffs(rec)
        assert coeffs is not None
        donors[sta] = (float(lon), float(lat), coeffs)
        digests[sta] = _digest(rec)
    return donors, digests


def regional_record(
    sta: str,
    target: tuple[float, float],
    donors: Mapping[str, tuple[float, float, np.ndarray]],
    *,
    stations: Mapping[str, Mapping[str, Any]],
    digests: Mapping[str, str],
    sigma_pred: Sequence[float],
    frame: str | None,
    fitted_at: str | None = None,
    length_km: float | None = None,
    radius_km: float | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """A seasonal-only record for ``sta``, or (None, why not)."""
    from gps_analysis.detrend import RECORD_VERSION
    from gps_analysis.regional import (
        DEFAULT_LENGTH_KM,
        DEFAULT_RADIUS_KM,
        interpolate_seasonal,
    )

    L = DEFAULT_LENGTH_KM if length_km is None else length_km
    R = DEFAULT_RADIUS_KM if radius_km is None else radius_km
    got = interpolate_seasonal(target, donors, length_km=L, radius_km=R, exclude=sta)
    if got.refused is not None:
        return None, got.refused
    sp = [float(x) for x in sigma_pred]
    components = []
    for c in range(3):
        cov = np.diag([sp[c] ** 2] * 4)
        components.append(
            {
                "params": [round(float(v), 6) for v in got.coeffs[c]],
                "cov_upper": [float(v) for v in cov[np.triu_indices(4)]],
            }
        )
    donor_rows = [
        {
            "station": code,
            "weight": w,
            "distance_km": got.distances_km[code],
            "fitted_at": (stations.get(code) or {}).get("fitted_at"),
            "digest": digests.get(code),
        }
        for code, w in got.weights.items()
    ]
    record = {
        "record_version": RECORD_VERSION,
        "model": "periodic",
        "param_names": list(PERIODIC_NAMES),
        "step_epochs": [],
        "frame": frame,
        "detrend_method": "regional",
        "seasonal_only": True,
        "fitted_at": fitted_at,
        "window": None,
        "segments": [],
        "borrowed": None,
        "components": components,
        "groups": {
            "periodic": {
                "provenance": f"regional:gauss L={L:g}km R={R:g}km",
                "donors": donor_rows,
                "sigma_pred": [round(x, 4) for x in sp],
                "L_km": L,
                "R_km": R,
            }
        },
    }
    return record, f"{len(donor_rows)} donors, top {next(iter(got.weights))}"


def borrowed_provenance(record: Mapping[str, Any]) -> str | None:
    """``borrowed:<from>@<prior frame>`` of a carried-over record, else None."""
    b = record.get("borrowed")
    if not b:
        return None
    b = b if isinstance(b, Mapping) else {}
    return f"borrowed:{b.get('from') or '?'}@{b.get('prior_frame') or '?'}"


def mark_estimated(record: dict[str, Any]) -> None:
    """Give a record explicit group provenance: ``estimated`` for a fitted
    one, ``borrowed:…`` for a carried-over one (its terms were NOT fitted
    here); a store/donor hold's own provenance is kept."""
    if record.get("seasonal_only"):
        return
    label = borrowed_provenance(record) or ESTIMATED
    generic = (None, "", "self", ESTIMATED)
    groups = record.setdefault("groups", {})
    for group in ("secular", "periodic"):
        entry = groups.setdefault(group, {})
        if entry.get("provenance") in generic:
            entry["provenance"] = label


def add_regional_seasonals(
    doc: Mapping[str, Any],
    targets: Sequence[str],
    coords: Coords,
    *,
    validate: Sequence[str] = (),
    fitted_at: str | None = None,
    length_km: float | None = None,
    radius_km: float | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """The document with seasonal-only records for ``targets`` that have no
    record (an existing regional one is re-made), every fitted record
    marked ``estimated``; and a per-target report.

    ``validate``: donor codes the leave-one-out σ is measured over (the
    validation area's own-seasonal stations).
    """
    from gps_analysis.regional import leave_one_out_sigma

    out = json.loads(json.dumps(doc))  # deep copy, JSON-safe
    stations: dict[str, Any] = out.setdefault("stations", {})
    donors, digests = collect_donors(stations, coords)
    kw: dict[str, float] = {}
    if length_km is not None:
        kw["length_km"] = length_km
    if radius_km is not None:
        kw["radius_km"] = radius_km
    sigma, n_val = leave_one_out_sigma(donors, list(validate) or sorted(donors), **kw)
    if n_val == 0:
        raise ValueError("no donor could be predicted: no leave-one-out σ")
    report: dict[str, str] = {}
    for sta in sorted({t.upper() for t in targets}):
        rec = stations.get(sta)
        if rec and not rec.get("seasonal_only"):
            report[sta] = "has its own record"
            continue
        try:
            target = coords(sta)
        except Exception as exc:  # noqa: BLE001 - unknown station
            report[sta] = f"no coordinates ({exc})"
            continue
        new, why = regional_record(
            sta, (float(target[0]), float(target[1])), donors,
            stations=stations, digests=digests, sigma_pred=sigma.tolist(),
            frame=out.get("frame"), fitted_at=fitted_at,
            length_km=length_km, radius_km=radius_km,
        )  # fmt: skip
        if new is None:
            report[sta] = f"refused: {why}"
            if rec and rec.get("seasonal_only"):  # an old regional one goes
                del stations[sta]
            continue
        stations[sta] = new
        report[sta] = f"regional: {why}"
    for rec in stations.values():
        if rec:
            mark_estimated(rec)
    out.setdefault("regional_seasonal", {}).update(
        {"sigma_pred": [round(float(x), 4) for x in sigma], "n_validation": n_val}
    )
    return out, report


def regional_k_entry(
    doc: Mapping[str, Any],
    validate: Sequence[str],
    coords: Coords,
    settings: Any,
    *,
    days: Sequence[int] | None = None,
) -> dict[str, Any]:
    """The k(T) table's ``regional`` entry, by leave-one-out.

    At each validation station (own estimated seasonal), its quiet span with
    only its secular removed, less the seasonal PREDICTED from the other
    stations: what a regional-seasonal station's residual looks like. k(T)
    per window length as :func:`geo_dataread.velocity.build_k_table`
    measures it; the per-component median over the stations.
    """
    from gps_analysis import empirical_rate_scale
    from gps_analysis.regional import interpolate_seasonal
    from gps_analysis.staged import evaluate_group_values

    from geo_dataread import velocity as vel

    days = list(days or vel.K_TABLE_DAYS)
    stations = doc.get("stations") or {}
    donors, _ = collect_donors(stations, coords)
    per: dict[int, list[np.ndarray]] = {}
    used = []
    for sta in validate:
        rec = stations.get(sta)
        if sta not in donors or not rec or not rec.get("window"):
            continue
        lon, lat, _c = donors[sta]
        got = interpolate_seasonal((lon, lat), donors, exclude=sta)
        if got.refused is not None:
            continue
        w0, w1 = (float(v) for v in rec["window"])
        try:
            t, y, s, _ = vel._read(sta, w0, w1, settings, terms="secular", doc=doc)
        except ValueError:
            continue
        seas = np.vstack(
            [evaluate_group_values(PERIODIC_NAMES, got.coeffs[c], t) for c in range(3)]
        )
        resid = y - seas
        any_k = False
        for d in days:
            rs = empirical_rate_scale(t, resid, s, window_years=d / 365.25)
            if np.all(np.isfinite(rs.k)):
                per.setdefault(d, []).append(np.asarray(rs.k, dtype=float))
                any_k = True
        if any_k:
            used.append(sta)
    rows = [(d, np.asarray(per[d])) for d in days if d in per]
    return {
        "days": [d for d, _ in rows],
        "k": [[round(float(v), 4) for v in np.nanmedian(a, axis=0)] for _, a in rows],
        "n_stations": [int(a.shape[0]) for _, a in rows],
        "stations": used,
        "method": "leave-one-out regional seasonal (gps-regional-seasonal)",
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gps-regional-seasonal",
        description="Seasonal-only detrend records interpolated from nearby "
        "stations' estimated seasonals (docs/regional-seasonal.md).",
    )
    p.add_argument("--params", type=Path, required=True,
                   help="the detrend document to read (e.g. the deployed "
                   "detrend_params.itrf14.json); never written")  # fmt: skip
    p.add_argument("--out", type=Path, required=True, help="merged document")
    p.add_argument("--region", default="/".join(map(str, REYKJANES_REGION)),
                   help="targets: stations.cfg markers in W/E/S/N "
                   "(default: Reykjanes, where it was validated)")  # fmt: skip
    p.add_argument("--stations", nargs="*", default=(),
                   help="limit the targets to these codes")  # fmt: skip
    p.add_argument("--stamp", default=None, help="fitted_at of the new records")
    p.add_argument("--k-table", type=Path, default=None,
                   help="also measure the k(T) `regional` entry by leave-one-out "
                   "and write it into this k table (gps-velocities k-table)")  # fmt: skip
    p.add_argument("--frame", default="itrf2014",
                   help="--k-table: the .pos frame of the series")  # fmt: skip
    return p


def main(argv: Sequence[str] | None = None) -> int:
    from geo_dataread.velocity import _coords, select_stations

    args = build_parser().parse_args(argv)
    region = tuple(float(v) for v in args.region.split("/"))
    if len(region) != 4:
        print("--region must be W/E/S/N", file=sys.stderr)
        return 2
    doc = json.loads(args.params.read_text(encoding="utf-8"))
    targets = select_stations(list(args.stations), region)
    # validated where it is used: the donors within 2L of a target
    from gps_analysis.regional import DEFAULT_LENGTH_KM, great_circle_km

    donors, _ = collect_donors(doc.get("stations") or {}, _coords)
    pos = [_coords(t) for t in targets]
    validate = [
        code
        for code, (lon, lat, _c) in donors.items()
        if pos
        and great_circle_km(lon, lat, [p[0] for p in pos], [p[1] for p in pos]).min()
        <= 2 * DEFAULT_LENGTH_KM
    ]
    out, report = add_regional_seasonals(
        doc, targets, _coords, validate=validate, fitted_at=args.stamp
    )
    for sta, line in report.items():
        print(f"{sta}: {line}")
    rs = out["regional_seasonal"]
    print(
        f"prediction σ (annual, N/E/U mm) {rs['sigma_pred']} from {rs['n_validation']} stations"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    n_new = sum(v.startswith("regional") for v in report.values())
    if args.k_table is not None:
        from geo_dataread.velocity import K_RESIDUAL_REGIONAL, VelocitySettings

        table = json.loads(args.k_table.read_text(encoding="utf-8"))
        entry = regional_k_entry(
            doc, validate, _coords, VelocitySettings(frame=args.frame)
        )
        table.setdefault("residuals", {})[K_RESIDUAL_REGIONAL] = entry
        args.k_table.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
        print(f"k(T) regional from {len(entry['stations'])} stations: "
              f"{dict(zip(entry['days'], entry['k'], strict=True))} -> {args.k_table}")  # fmt: skip
    print(f"wrote {n_new} regional record(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
