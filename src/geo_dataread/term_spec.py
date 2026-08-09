"""The ``--term`` grammar: add a transient to a station's model.

Companion to :mod:`geo_dataread.stage_plan`, and a separate module for the
same reason: the CLI, the batch estimator and the graphical picker must all
speak ONE grammar, and a grammar living inside one CLI's ``argparse`` block
is a grammar the others reimplement.

Spelling::

    --term log@2008.4085,tau=1.0        A·log(1 + Δt/T)     Bevis & Brown eq. (9)
    --term exp@2010.0,tau=0.33          Φ·(1 − e^(−Δt/τ))   Reverso eq. (20)

``tau`` is **required and operator-fixed**, which is the whole design rather
than a limitation: with τ fixed the model stays linear in its parameters, so
``select_terms``, the closed-form fit path and staged holds all keep working.
It is also what the operator recipe actually does — Bevis & Brown fix ``T``
as station metadata (default 1 yr, the variant they call the ELTM) and refine
it per station only later, by a 1-D grid search, because Appendix 1 shows the
fit is strikingly insensitive to it: the other coefficients absorb the error.
Profiling τ is :func:`gps_analysis.profile_transient_tau`, deliberately not
reachable from this flag.

Why this exists at all: a transient is the ONLY thing that fixes an
excess-candidate abort on a deforming station. Measured on NYLA
(2026-08-03), re-judging the epochs against a staged trajectory does not
help — candidate fractions stay at [0.089, 0.116, 0.004] because the staged
fit is the same model with a differently-estimated seasonal, and a straight
line cannot follow that deformation. The abort is a model-adequacy problem,
so only a model that can follow the signal resolves it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "TERM_KINDS",
    "build_trajectory_model",
    "parse_term_spec",
]

#: CLI kind → the registered term kind of :mod:`gps_analysis.terms`.
TERM_KINDS: Mapping[str, str] = {
    "log": "log_transient",
    "exp": "exp_transient",
}

#: Registry model code → the terms it is made of.
_MODEL_TERMS: Mapping[str, tuple[dict[str, Any], ...]] = {
    "linear": ({"kind": "polynomial", "degree": 1},),
    "lineperiodic": (
        {"kind": "polynomial", "degree": 1},
        {"kind": "seasonal", "n_harmonics": 2},
    ),
    "periodic": ({"kind": "seasonal", "n_harmonics": 2},),
}


def parse_term_spec(spec: str) -> dict[str, Any]:
    """Parse one ``--term KIND@EPOCH,tau=VALUE`` into a term spec dict.

    Args:
        spec: The raw flag value.

    Returns:
        A spec dict consumable by :func:`gps_analysis.term_from_spec`.

    Raises:
        ValueError: On any malformed spelling, naming the valid kinds and
            the required ``tau``.  ``tau`` is refused rather than defaulted:
            Bevis & Brown's own default of 1 yr is a reasonable STARTING
            guess for a per-station search, not a value to store silently
            behind an operator's back.
    """
    raw = spec.strip()
    if "@" not in raw:
        raise ValueError(
            f"--term {spec!r} must be 'KIND@EPOCH,tau=VALUE'; "
            f"kinds: {', '.join(sorted(TERM_KINDS))}"
        )
    kind_s, _, rest = raw.partition("@")
    kind_s = kind_s.strip().lower()
    if kind_s not in TERM_KINDS:
        raise ValueError(
            f"--term {spec!r}: unknown kind {kind_s!r}; "
            f"valid kinds are {', '.join(sorted(TERM_KINDS))} "
            f"(log = afterslip / rate-and-state, exp = relaxation toward a "
            f"steady state)"
        )

    fields = [f.strip() for f in rest.split(",") if f.strip()]
    if not fields:
        raise ValueError(f"--term {spec!r} names no epoch")
    try:
        epoch = float(fields[0])
    except ValueError:
        raise ValueError(
            f"--term {spec!r}: epoch {fields[0]!r} is not a fractional year"
        ) from None

    extra: dict[str, float] = {}
    for field in fields[1:]:
        if "=" not in field:
            raise ValueError(f"--term {spec!r}: expected 'name=value', got {field!r}")
        key, _, value = field.partition("=")
        key = key.strip().lower()
        if key != "tau":
            raise ValueError(
                f"--term {spec!r}: unknown parameter {key!r}; only 'tau' is "
                f"settable (the amplitude is ESTIMATED, not declared)"
            )
        try:
            extra["tau"] = float(value)
        except ValueError:
            raise ValueError(
                f"--term {spec!r}: tau {value!r} is not a number"
            ) from None

    if "tau" not in extra:
        raise ValueError(
            f"--term {spec!r} needs an explicit tau, e.g. "
            f"'{kind_s}@{epoch},tau=1.0' [yr]. It is operator-fixed on "
            f"purpose: with tau fixed the model stays linear in its "
            f"parameters. Bevis & Brown start from tau = 1 yr and refine per "
            f"station; a stored record should say which value was used."
        )
    if extra["tau"] <= 0:
        raise ValueError(f"--term {spec!r}: tau must be positive [yr]")

    return {"kind": TERM_KINDS[kind_s], "epoch": epoch, "tau": extra["tau"]}


def build_trajectory_model(
    model: str,
    term_specs: Sequence[str] | None,
) -> Any:
    """Compose a registry model with any ``--term`` transients.

    Returns a ``gps_analysis.TrajectoryModel``, whose ``as_modelfunc()`` is a
    drop-in for the registry callable ``estimate_detrend`` normally receives.
    Canonical ordering (polynomial → seasonal → transient) is the model's
    own, which is what keeps ``param_names[1] == "rate"`` true and therefore
    ``velocity._RATE_INDEX`` correct.

    **Steps are deliberately NOT composed in here**, and this used to be the
    bug rather than the design. Baking them in meant passing
    ``step_epochs=None`` to ``estimate_detrend``, which silently disabled the
    two protections that live on that argument: the filter dropping steps
    outside the fit window, and the refusal of two steps with no fitted epoch
    between them. A station with a declared step OUTSIDE its window then got
    an all-ones step column, collinear with the intercept, and stored a
    trajectory metres away from the truth on nothing louder than an
    ``OptimizeWarning``. Steps now travel as ``step_epochs`` on both paths,
    so there is ONE screen and ONE augmentation site
    (``fitting.with_steps``), and the ``--term`` path cannot drift from the
    registry path again.

    Args:
        model: Registry code the station is fitted with.
        term_specs: Raw ``--term`` values; None or empty returns None, so a
            caller with no transients keeps the ordinary registry path and
            its byte-identical version-1 record.

    Raises:
        ValueError: For a model code whose composition is unknown, or any
            malformed term.
    """
    if not term_specs:
        return None
    from gps_analysis import TrajectoryModel

    if model not in _MODEL_TERMS:
        raise ValueError(
            f"--term needs to know what model {model!r} is composed of; "
            f"known: {sorted(_MODEL_TERMS)}"
        )
    specs: list[dict[str, Any]] = [dict(d) for d in _MODEL_TERMS[model]]
    specs += [parse_term_spec(s) for s in term_specs]
    return TrajectoryModel.from_spec(specs)
