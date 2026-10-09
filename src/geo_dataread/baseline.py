"""Baselines between two GNSS stations: pure arrays, numpy only.

A baseline A − B is the difference of two stations' N/E/U series on the
days both observed. This module is the ONE place that defines it, so every
consumer forms it the same way: plots (``gps_plot``), the products API
(``gps_api``), notebooks. It deliberately imports nothing but numpy — no
plotting, no config, no ``gps_analysis`` — so a light runtime (the API)
can use it; a test pins that.

Inputs are per-station ``(yearf, data, sigma)`` triplets that are ALREADY
in the wanted view and frame (both plate, both detrended, …). Producing
those is the caller's job: ``geo_dataread.gps_views.read_gps_view`` for
data consumers, ``gps_plot.timesmatplt.station_series`` for plots (which
predates and partly duplicates ``read_gps_view``; unifying them is a
separate refactor). :func:`check_same_view` refuses the mix this module
cannot detect from arrays alone.

Pieces, each usable on its own:

- :func:`split_baseline` — ``"VFLN-VFLS"`` → ``("VFLN", "VFLS")``.
- :func:`match_epochs` — index pairs of the common days.
- :func:`baseline` — the baseline AND both ends referenced to the same
  epoch, as a :class:`Baseline`.
- :func:`baseline_arrays` — just ``(yearf, data, sigma)`` of it.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

__all__ = [
    "BASELINE_EPOCH_TOLERANCE_YEARS",
    "Baseline",
    "baseline",
    "baseline_arrays",
    "check_same_view",
    "match_epochs",
    "split_baseline",
]

FloatArray = npt.NDArray[np.float64]
Triplet = tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]

#: Two epochs belong to the same day when they are closer than half a day.
BASELINE_EPOCH_TOLERANCE_YEARS = 0.5 / 365.25

#: ``AAAA-BBBB``: a baseline from station AAAA to station BBBB.
_BASELINE_RE = re.compile(r"^([A-Za-z0-9]{4})-([A-Za-z0-9]{4})$")


def split_baseline(spec: str) -> tuple[str, str] | None:
    """``"VFLN-VFLS"`` → ``("VFLN", "VFLS")`` (upper-cased); else ``None``.

    A plain 4-character station code returns ``None``, so callers can
    accept both forms in one argument.
    """
    m = _BASELINE_RE.match(spec)
    return (m.group(1).upper(), m.group(2).upper()) if m else None


def match_epochs(
    t_a: npt.ArrayLike,
    t_b: npt.ArrayLike,
    *,
    tol: float = BASELINE_EPOCH_TOLERANCE_YEARS,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Index pairs ``(ia, ib)`` of the epochs A and B share, by day.

    Each A epoch is paired with the nearest B epoch when it is within
    ``tol`` [fractional yr] (default half a day). Float equality is never
    used: both series come from the same GLOBK runs, but nothing guarantees
    bit-identical epochs. ``ia`` is increasing when ``t_a`` is; ``t_b`` may
    be in any order.
    """
    ta = np.asarray(t_a, dtype=np.float64)
    tb = np.asarray(t_b, dtype=np.float64)
    if ta.size == 0 or tb.size == 0:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    order = np.argsort(tb, kind="stable")
    tbs = tb[order]
    pos = np.clip(np.searchsorted(tbs, ta), 0, tbs.size - 1)
    left = np.clip(pos - 1, 0, tbs.size - 1)
    nearest = np.where(np.abs(tbs[left] - ta) <= np.abs(tbs[pos] - ta), left, pos)
    keep = np.abs(tbs[nearest] - ta) < tol
    return np.flatnonzero(keep).astype(np.intp), order[nearest[keep]].astype(np.intp)


@dataclasses.dataclass(frozen=True)
class Baseline:
    """A baseline A − B and its two ends, referenced to the same epoch.

    Attributes:
        stations: ``(A, B)`` markers, or ``None`` when not given.
        yearf: Common epochs (A's), shape (N,) [fractional yr].
        data: A − B per component, shape (C, N).
        sigma: shape (C, N). By default the quadrature √(σ_A² + σ_B²) —
            conservative: errors common to both stations cancel in the
            difference but are counted twice. With ``rho`` (and optionally
            ``sigma_scale``) given to :func:`baseline`, the empirical
            √(σ_A′² + σ_B′² − 2ρ·σ_A′·σ_B′), σ′ = k·σ.
        ends: ``((yearf_A, data_A), (yearf_B, data_B))`` — each station's
            FULL series (all its own epochs, not only the common ones),
            shifted so that on every common epoch ``data_A − data_B``
            equals :attr:`data` exactly. Draw them behind the baseline and
            it is their visible difference.
    """

    stations: tuple[str, str] | None
    yearf: FloatArray
    data: FloatArray
    sigma: FloatArray
    ends: tuple[tuple[FloatArray, FloatArray], tuple[FloatArray, FloatArray]]
    sigma_quadrature: FloatArray | None = None
    rho: FloatArray | None = None


def baseline(
    station_a: Triplet,
    station_b: Triplet,
    *,
    stations: tuple[str, str] | None = None,
    tol: float = BASELINE_EPOCH_TOLERANCE_YEARS,
    rezero: bool = True,
    rho: npt.ArrayLike | None = None,
    sigma_scale: tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
) -> Baseline:
    """Baseline A − B on common days, with both ends referenced alike.

    Equation (per component c, per common epoch i):
        ``b_c(tᵢ) = a_c(tᵢ) − b_c(tᵢ) − [a_c(t₀) − b_c(t₀)]``,
        ``σ_c(tᵢ) = √(σ_a,c(tᵢ)² + σ_b,c(tᵢ)²)``

    Symbols → args:
        - ``tᵢ`` → common days, :func:`match_epochs` (within ``tol``)
        - ``a``, ``σ_a`` → ``station_a = (yearf, data(C,N), sigma(C,N))``
        - ``b``, ``σ_b`` → ``station_b``, same layout and units
        - ``t₀`` → per component, the first common epoch where the
          difference is finite; the bracket is applied only when ``rezero``

    Empirical σ (opt-in): formal GLOBK σ count the daily error the two
    stations SHARE twice, so short baselines get σ several times their real
    scatter. Pass ``rho`` — the shared fraction per component, shape (C,) or
    (C, N), e.g. ``gps_analysis.empirical_sigma.shared_fraction_model`` at
    the pair's distance — and optionally ``sigma_scale = (k_A, k_B)``, each
    (C,), the stations' ``sigma_scale_factor``:
        ``σ_c = √(σ_A′² + σ_B′² − 2ρ_c·σ_A′·σ_B′)``, ``σ′ = k·σ``
    The quadrature value stays in ``Baseline.sigma_quadrature``. Validated
    out of sample on IMO TOT 2025–26: NRMS 0.52/0.51/0.33 → 0.91/0.91/0.94.

    Each single-station series was referenced to its own window start, so
    the raw difference carries an arbitrary constant; re-zeroing (default)
    starts the baseline at 0 like any single-station plot. The ends are
    shifted by their own values at ``t₀``, so ``end_A − end_B`` equals the
    baseline exactly on every common epoch — whatever ``rezero`` is.

    The components are each station's local N/E/U, which for the short
    baselines this is used on (tens of km) is the local frame of either end
    to well below the noise. Both inputs must be in the same view and frame
    (see :func:`check_same_view`); this function only differences.

    Raises:
        ValueError: when the two stations share no epoch, or the shapes of
            a triplet disagree.
    """
    ta, da, sa = (np.asarray(v, dtype=np.float64) for v in station_a)
    tb, db, sb = (np.asarray(v, dtype=np.float64) for v in station_b)
    for name, t, d, s in (("A", ta, da, sa), ("B", tb, db, sb)):
        if d.ndim != 2 or d.shape != s.shape or d.shape[1] != t.size:
            raise ValueError(
                f"baseline: station {name} data{d.shape}/sigma{s.shape} do not "
                f"match {t.size} epochs as (components, epochs)"
            )
    ia, ib = match_epochs(ta, tb, tol=tol)
    if ia.size == 0:
        raise ValueError(
            "baseline: the two stations share no epoch "
            f"(A {ta.min():.3f}-{ta.max():.3f}, B {tb.min():.3f}-{tb.max():.3f})"
        )
    raw = da[:, ia] - db[:, ib]
    end_a, end_b = da.copy(), db.copy()
    for c in range(raw.shape[0]):
        finite = np.flatnonzero(np.isfinite(raw[c]))
        if not finite.size:
            continue
        k = finite[0]
        end_a[c] -= da[c, ia[k]]
        end_b[c] -= db[c, ib[k]]
    data = end_a[:, ia] - end_b[:, ib] if rezero else raw
    sa_c, sb_c = sa[:, ia], sb[:, ib]
    quad = np.sqrt(sa_c**2 + sb_c**2)
    if rho is None:
        if sigma_scale is not None:
            raise ValueError("baseline: sigma_scale needs rho (the shared fraction)")
        return Baseline(
            stations=stations,
            yearf=ta[ia],
            data=data,
            sigma=quad,
            ends=((ta, end_a), (tb, end_b)),
        )
    n_comp = raw.shape[0]
    r = np.asarray(rho, dtype=np.float64)
    r = r.reshape(-1, 1) if r.ndim == 1 else r
    if r.shape not in ((n_comp, 1), raw.shape) or np.any(np.abs(r) > 1):
        raise ValueError(
            f"baseline: rho must be ({n_comp},) or {raw.shape} within [-1, 1], "
            f"got shape {np.shape(rho)}"
        )
    if sigma_scale is not None:
        ka, kb = (np.asarray(k, dtype=np.float64).reshape(-1, 1) for k in sigma_scale)
        if ka.shape[0] != n_comp or kb.shape[0] != n_comp:
            raise ValueError(f"baseline: sigma_scale needs ({n_comp},) per station")
        sa_c, sb_c = ka * sa_c, kb * sb_c
    sigma = np.sqrt(np.maximum(sa_c**2 + sb_c**2 - 2.0 * r * sa_c * sb_c, 0.0))
    return Baseline(
        stations=stations,
        yearf=ta[ia],
        data=data,
        sigma=sigma,
        ends=((ta, end_a), (tb, end_b)),
        sigma_quadrature=quad,
        rho=np.broadcast_to(r, raw.shape).copy(),
    )


def baseline_arrays(
    station_a: Triplet,
    station_b: Triplet,
    *,
    tol: float = BASELINE_EPOCH_TOLERANCE_YEARS,
    rezero: bool = True,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``(yearf, data, sigma)`` of :func:`baseline` — for callers that need
    only the baseline itself."""
    bl = baseline(station_a, station_b, tol=tol, rezero=rezero)
    return bl.yearf, bl.data, bl.sigma


def check_same_view(
    stations: Sequence[str], detrend_applied: Sequence[bool], *, spec: str = ""
) -> None:
    """Refuse a detrended baseline with an end that is NOT detrended.

    A station without a stored detrend record degrades to its plate series
    (a warning that is tolerable on a single-station plot). In a baseline
    that silently becomes "detrended minus plate". Call this whenever the
    detrended view was requested; ``detrend_applied[i]`` says whether
    station ``i`` really got its trajectory subtracted.

    Raises:
        ValueError: naming the station(s) without a record.
    """
    missing = [s for s, ok in zip(stations, detrend_applied, strict=True) if not ok]
    if missing:
        label = spec or "-".join(stations)
        raise ValueError(
            f"baseline {label}: the detrended view needs a stored record for "
            f"BOTH stations; {', '.join(missing)} has none, so the baseline "
            "would difference a detrended series against a plate one. Use the "
            "plate frame, or curate the missing record."
        )
