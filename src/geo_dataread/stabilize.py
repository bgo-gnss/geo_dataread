"""Frame stabilization of a loose SINEX day, reproducing GLOBK glorg (numpy only).

The daily archive keeps the LOOSE GLOBK combination (site sigmas ~1 m, from
``glbtosnx``); glorg then ties it to the reference frame with a 6-parameter
(pole-x, pole-y, UT1 rotations + 3 translations; optional scale) fit to the
a priori positions of a reference-site list, iterated with reweighting and
site rejection, and APPLIES the result as a condition on the full solution —
every site moves through its correlations, and the covariance becomes the
frame-defined one. :func:`stabilize` re-implements that from the glorg 10.71
source (``kf/glorg/glorg.f`` GLORS, ``kf/glorg/apply_cfull.f``,
``kf/glfor/pmu_part.f``); on archived days it reproduces glorg's printed
iteration results and the final positions/sigmas of every site to the org
file's print resolution (≤ 0.01 mm). The SINEX ``apriori`` column is the
glorg a priori at the solution epoch.

glorg details kept on purpose (they change the numbers):

- residuals are estimate − a priori; the fit uses the a priori positions as
  geometry, rotations are polar-motion/UT1 partials in mas (``pmu_part``:
  pole-x ``(−Z, 0, X)``, pole-y ``(0, Z, −Y)``, UT1 ``(−Y, X, 0)``, over
  rad→mas), translations in m, scale in ppb;
- per-site weight ``W = (Rᵀ diag(a, a, h) R · s)⁻¹`` with the GEOCENTRIC
  N/E/U rotation ``R``, ``h`` the height variance factor and
  ``a = (3 − 1/h)/2``; ``s`` the relative variance, all 1 in iteration 1,
  then ``(1 − ρ²) + ρ²·v_NE/mean(v_NE)`` (the header's "50 % / 50 %" is
  ρ = 0.5, which is 75 % / 25 %);
- each iteration restarts from the loose solution; the height screen (loose
  covariance), NE screen and n-sigma deletions (stabilized solution) only
  change the site set and weights of the NEXT iteration — the output is the
  last iteration's conditioning, and its deletions are reported only;
- the transformation is imposed with :func:`geofunc.local.condition`
  (``L = N⁻¹AᵀW`` on the reference sites' X/Y/Z, exact), the reported
  parameter sigma is ``√(N⁻¹ χ²)`` with ``χ² = (bᵀWb − uᵀθ)/(3·n)``.

``StabilizeConfig()`` holds glorg's defaults with the IMO ``icel.org.cmd``
settings (``stab_ite 4 0.5 4.0``, ``cnd_hgtv 10 10 4 4``);
:func:`geo_dataread.org.read_org_setup` reads them from an org file.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
from geofunc import local

from geo_dataread.sinex import SinexSolution

__all__ = [
    "RAD_TO_MAS",
    "IterationReport",
    "StabilizeConfig",
    "Stabilized",
    "design",
    "stabilize",
]

FloatArray = npt.NDArray[np.float64]

RAD_TO_MAS = 648000e3 / np.pi
_ALL = ("xrot", "yrot", "zrot", "xtran", "ytran", "ztran", "scale")


@dataclasses.dataclass(frozen=True)
class StabilizeConfig:
    """glorg stabilization settings (defaults: IMO ``icel.org.cmd``)."""

    params: tuple[str, ...] = ("xrot", "yrot", "zrot", "xtran", "ytran", "ztran")
    iterations: int = 4
    rel: float = 0.5  # ρ of stab_ite: site-dependent share (squared in use)
    nsig: float = 4.0
    height_var: float = 10.0
    use_ratio: float = 4.0  # cnd_hgtv 3rd arg: height/NE sigma screens
    min_rms: float = 0.003
    min_dh: float = 0.005
    min_dne: float = 0.0005
    ut1_mas: float = 0.0  # UT1-AT rotation of the pmu partials (glorg: ~0)

    def __post_init__(self) -> None:
        bad = [p for p in self.params if p not in _ALL]
        if bad or not self.params:
            raise ValueError(f"params must be a non-empty subset of {_ALL}: {bad}")


@dataclasses.dataclass(frozen=True)
class IterationReport:
    """What glorg prints for one position stabilization iteration."""

    iteration: int
    sites: tuple[str, ...]  # "CODE:soln" used in the fit
    rel_sigma: tuple[float, ...]  # √s per used site (before this iteration's update)
    theta: FloatArray  # in params order: mas / m / ppb
    theta_sigma: FloatArray
    pre_rms: float
    post_rms: float
    removed_height: tuple[str, ...]
    removed_ne: tuple[str, ...]
    deleted: tuple[
        tuple[str, float, float, float], ...
    ]  # name, error m, rel var, nsigma


@dataclasses.dataclass(frozen=True)
class Stabilized:
    """The stabilized day: a SinexSolution with new estimates and covariance."""

    solution: SinexSolution
    params: tuple[str, ...]
    theta: FloatArray
    theta_sigma: FloatArray
    iterations: tuple[IterationReport, ...]


def design(
    xyz: npt.ArrayLike, params: Iterable[str], *, ut1_mas: float = 0.0
) -> FloatArray:
    """glorg's partials ``(3, k)`` of a site's X/Y/Z w.r.t. ``params``.

    Rotations per mas (polar motion x/y, UT1; ``pmu_part``), translations per
    m, scale per ppb.
    """
    x, y, z = np.asarray(xyz, dtype=np.float64)
    d = ut1_mas / RAD_TO_MAS
    xr = x * np.cos(d) + y * np.sin(d)
    yr = -x * np.sin(d) + y * np.cos(d)
    col = {
        "xrot": np.array([-z, 0.0, xr]) / RAD_TO_MAS,
        "yrot": np.array([0.0, z, -yr]) / RAD_TO_MAS,
        "zrot": np.array([-yr, xr, 0.0]) / RAD_TO_MAS,
        "xtran": np.array([1.0, 0.0, 0.0]),
        "ytran": np.array([0.0, 1.0, 0.0]),
        "ztran": np.array([0.0, 0.0, 1.0]),
        "scale": np.array([x, y, z]) * 1e-9,
    }
    order = [p for p in _ALL if p in set(params)]
    return np.stack([col[p] for p in order], axis=1)


def _geocentric_neu(p: FloatArray) -> FloatArray:
    th = np.arctan2(np.hypot(p[0], p[1]), p[2])
    la = np.arctan2(p[1], p[0])
    return np.array(
        [
            [-np.cos(th) * np.cos(la), -np.cos(th) * np.sin(la), np.sin(th)],
            [-np.sin(la), np.cos(la), 0.0],
            p / np.linalg.norm(p),
        ]
    )


def _screen(values: dict[str, float], floor: float, ratio: float) -> list[str]:
    """Sites whose sigma exceeds the median by more than ratio·tol (glorg screens)."""
    v = np.array(list(values.values()))
    pos = v[v > 0]
    if pos.size == 0:
        return []
    med = float(np.median(v))
    tol = max(np.sqrt(med) - np.sqrt(pos.min()), floor)
    if np.sqrt(v.max()) - np.sqrt(med) <= ratio * tol:
        return []
    return [k for k, val in values.items() if np.sqrt(val) - np.sqrt(med) > ratio * tol]


def stabilize(
    sol: SinexSolution,
    reference: Iterable[str],
    config: StabilizeConfig | None = None,
) -> Stabilized:
    """Tie a loose SINEX day to its a priori frame the way glorg does.

    ``reference`` are 4-char site codes (every SINEX solution of the code
    takes part, like glorg ``use_site``). Sites without an a priori are
    skipped. Raises ``ValueError`` when fewer reference sites remain than the
    parameters need.
    """
    cfg = config or StabilizeConfig()
    params = tuple(p for p in _ALL if p in set(cfg.params))
    codes = {c.upper()[:4] for c in reference}
    stax = np.flatnonzero(sol.ptype == "STAX")
    row = {
        f"{sol.code[i]}:{sol.soln[i]}": int(i)
        for i in stax
        if sol.code[i] in codes and np.isfinite(sol.apriori[i : i + 3]).all()
    }
    est = sol.estimate
    x0 = np.where(np.isfinite(sol.apriori), est - np.nan_to_num(sol.apriori), 0.0)
    pos = {k: sol.apriori[i : i + 3] for k, i in row.items()}
    rhat = {k: p / np.linalg.norm(p) for k, p in pos.items()}
    h = cfg.height_var
    a = (3.0 - 1.0 / h) / 2.0
    k_par = len(params)

    used = list(row)
    ss = dict.fromkeys(used, 1.0)
    reports: list[IterationReport] = []
    x, cov = x0, sol.cov
    theta = sig = np.zeros(k_par)
    for it in range(1, cfg.iterations + 1):
        blk = {k: sol.cov[row[k] : row[k] + 3, row[k] : row[k] + 3] for k in used}
        vh = {k: float(rhat[k] @ blk[k] @ rhat[k]) for k in used}
        used = [k for k in used if vh[k] > 0]
        removed_h = _screen({k: vh[k] for k in used}, cfg.min_dh, cfg.use_ratio)
        used = [k for k in used if k not in removed_h]
        if 3 * len(used) < k_par:
            raise ValueError(
                f"iteration {it}: {len(used)} reference sites for {k_par} parameters"
            )

        nmat = np.zeros((k_par, k_par))
        u = np.zeros(k_par)
        pre = 0.0
        parts = {}
        for k in used:
            amat = design(pos[k], params, ut1_mas=cfg.ut1_mas)
            r = _geocentric_neu(pos[k])
            w = np.linalg.inv(r.T @ np.diag([a, a, h]) @ r * ss[k])
            b = x0[row[k] : row[k] + 3]
            nmat += amat.T @ w @ amat
            u += amat.T @ w @ b
            pre += float(b @ w @ b)
            parts[k] = (amat, w)
        ninv = np.linalg.inv(nmat)
        theta = ninv @ u
        nr = 3 * len(used)
        chi2 = (pre - float(u @ theta)) / nr
        sig = np.sqrt(np.diag(ninv) * chi2)
        lrows = np.zeros((k_par, est.size))
        for k, (amat, w) in parts.items():
            lrows[:, row[k] : row[k] + 3] = ninv @ amat.T @ w
        out = local.condition(x0, sol.cov, lrows)
        assert out.cov is not None
        x, cov = out.d, out.cov

        rel_sigma = tuple(float(np.sqrt(ss[k])) for k in used)
        vhor = {}
        for k in used:
            c = cov[row[k] : row[k] + 3, row[k] : row[k] + 3]
            vhor[k] = float(np.trace(c) - rhat[k] @ c @ rhat[k])
        removed_ne = _screen(vhor, cfg.min_dne, cfg.use_ratio)
        kept = [k for k in used if k not in removed_ne]
        mean_var = float(np.mean([vhor[k] for k in kept]))
        var_post = max(chi2, cfg.min_rms**2)
        nxt, deleted = [], []
        for k in kept:
            ss[k] = (1 - cfg.rel**2) + cfg.rel**2 * vhor[k] / mean_var
            d = x[row[k] : row[k] + 3]
            rr = float(d @ rhat[k])
            err = float(d @ d - rr**2 + rr**2 / h)
            ns = np.sqrt(err / var_post / ss[k])
            if ns > cfg.nsig:
                deleted.append((k, float(np.sqrt(err)), float(ss[k]), float(ns)))
            else:
                nxt.append(k)
        reports.append(
            IterationReport(
                it,
                tuple(used),
                rel_sigma,
                theta,
                sig,
                float(np.sqrt(pre / nr)),
                float(np.sqrt(chi2)),
                tuple(removed_h),
                tuple(removed_ne),
                tuple(deleted),
            )
        )
        used = nxt

    new = dataclasses.replace(
        sol, estimate=est + (x - x0), sigma=np.sqrt(np.diag(cov)), cov=cov
    )
    return Stabilized(new, params, theta, sig, tuple(reports))
