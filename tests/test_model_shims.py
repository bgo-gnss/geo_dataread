"""Shim-boundary tests for the model evaluators (refactor-B slice 7).

gps_read's legacy model functions (``line``/``periodic``/``xf``/``expxf``/
``expf``/``dexpf``/``secondorder``) are now thin shims over
``gps_analysis.models``. These functions feed ``scipy.optimize.curve_fit``
on golden-pinned paths, so the delegation contract is BITWISE equality
against the legacy expressions (pinned verbatim below) — not allclose.

``lineperiodic`` is the deliberate exception: the leaf's
``linear + periodic`` association differs from the legacy single
expression by up to 1 ulp, so the evaluation stays local in gps_read;
the test pins it against its own legacy expression to catch any future
accidental delegation.
"""

import numpy as np
import pytest

from geo_dataread import gps_read

RNG = np.random.default_rng(20260712)

# Representative time inputs: absolute fractional years (the production
# domain), event-referenced small times, and the unit interval.
X_CASES = [
    RNG.uniform(2000.0, 2027.0, 5000),
    RNG.uniform(-5.0, 5.0, 2000),
    RNG.uniform(0.0, 1.0, 2000),
    np.array([2015.0]),
]

# Wide parameter sweeps (order of magnitude spans what curve_fit visits).
PARAMS_6 = [RNG.normal(0.0, 50.0, 6) for _ in range(20)]
PARAMS_SMALL = [RNG.normal(0.0, 2.0, 4) for _ in range(20)]


# --- legacy reference expressions (verbatim from pre-slice-7 gps_read) ----


def _legacy_line(x, p0, p1):
    return p0 + p1 * x


def _legacy_lineperiodic(x, p0, p1, p2, p3, p4, p5):
    return (
        p0
        + p1 * x
        + p2 * np.cos(2 * np.pi * x)
        + p3 * np.sin(2 * np.pi * x)
        + p4 * np.cos(4 * np.pi * x)
        + p5 * np.sin(4 * np.pi * x)
    )


def _legacy_periodic(x, p0, p1, p2, p3, p4, p5):
    return (
        p2 * np.cos(2 * np.pi * x)
        + p3 * np.sin(2 * np.pi * x)
        + p4 * np.cos(4 * np.pi * x)
        + p5 * np.sin(4 * np.pi * x)
    )


def _legacy_xf(x, p0, p1, p2, tau=4.8):
    return p0 + p1 * x + p2 * np.exp(-tau * x)


def _legacy_expxf(x, p0, p1, p2, p3):
    return p0 + p1 * x + p2 * np.exp(-p3 * x)


def _legacy_expf(x, p0, p1, p2):
    return p0 + p1 * np.exp(-p2 * x)


def _legacy_dexpf(x, p1, p2):
    return -p1 * p2 * np.exp(-p2 * x)


def _legacy_secondorder(x, p0, p1, p2):
    return p0 + p1 * x + p2 * x**2


def _assert_bitwise(got, want, label):
    got = np.asarray(got)
    want = np.asarray(want)
    assert got.shape == want.shape, f"{label}: shape {got.shape} != {want.shape}"
    assert got.dtype == np.float64, f"{label}: dtype {got.dtype}"
    assert np.array_equal(got, want, equal_nan=True), (
        f"{label}: shim output is not bit-identical to the legacy expression"
    )


@pytest.mark.parametrize("x", X_CASES, ids=["yearf", "event", "unit", "scalar-ish"])
def test_line_bitwise(x):
    for p in PARAMS_6:
        _assert_bitwise(
            gps_read.line(x, p[0], p[1]), _legacy_line(x, p[0], p[1]), "line"
        )


@pytest.mark.parametrize("x", X_CASES, ids=["yearf", "event", "unit", "scalar-ish"])
def test_lineperiodic_stays_legacy_expression(x):
    """lineperiodic is evaluation-order-pinned — NOT delegated to the leaf."""
    for p in PARAMS_6:
        _assert_bitwise(
            gps_read.lineperiodic(x, *p), _legacy_lineperiodic(x, *p), "lineperiodic"
        )


@pytest.mark.parametrize("x", X_CASES, ids=["yearf", "event", "unit", "scalar-ish"])
def test_periodic_bitwise(x):
    for p in PARAMS_6:
        _assert_bitwise(gps_read.periodic(x, *p), _legacy_periodic(x, *p), "periodic")


def test_periodic_ignores_linear_params():
    """Legacy contract: p0/p1 are accepted so lineperiodic vectors reuse."""
    x = X_CASES[0]
    p = PARAMS_6[0]
    a = gps_read.periodic(x, p[0], p[1], p[2], p[3], p[4], p[5])
    b = gps_read.periodic(x, 999.0, -999.0, p[2], p[3], p[4], p[5])
    assert np.array_equal(a, b)


@pytest.mark.parametrize("x", X_CASES, ids=["yearf", "event", "unit", "scalar-ish"])
def test_xf_bitwise_and_default_tau(x):
    for p in PARAMS_SMALL:
        with np.errstate(over="ignore"):
            _assert_bitwise(
                gps_read.xf(x, p[0], p[1], p[2]),
                _legacy_xf(x, p[0], p[1], p[2]),
                "xf (default tau=4.8)",
            )
            _assert_bitwise(
                gps_read.xf(x, p[0], p[1], p[2], tau=p[3]),
                _legacy_xf(x, p[0], p[1], p[2], tau=p[3]),
                "xf (explicit tau)",
            )


@pytest.mark.parametrize("x", X_CASES, ids=["yearf", "event", "unit", "scalar-ish"])
def test_expxf_bitwise(x):
    for p in PARAMS_SMALL:
        with np.errstate(over="ignore"):
            _assert_bitwise(gps_read.expxf(x, *p), _legacy_expxf(x, *p), "expxf")


@pytest.mark.parametrize("x", X_CASES, ids=["yearf", "event", "unit", "scalar-ish"])
def test_expf_bitwise(x):
    for p in PARAMS_SMALL:
        with np.errstate(over="ignore"):
            _assert_bitwise(
                gps_read.expf(x, p[0], p[1], p[2]),
                _legacy_expf(x, p[0], p[1], p[2]),
                "expf",
            )


@pytest.mark.parametrize("x", X_CASES, ids=["yearf", "event", "unit", "scalar-ish"])
def test_dexpf_bitwise(x):
    for p in PARAMS_SMALL:
        with np.errstate(over="ignore"):
            _assert_bitwise(
                gps_read.dexpf(x, p[0], p[1]), _legacy_dexpf(x, p[0], p[1]), "dexpf"
            )


@pytest.mark.parametrize("x", X_CASES, ids=["yearf", "event", "unit", "scalar-ish"])
def test_secondorder_bitwise(x):
    for p in PARAMS_SMALL:
        _assert_bitwise(
            gps_read.secondorder(x, p[0], p[1], p[2]),
            _legacy_secondorder(x, p[0], p[1], p[2]),
            "secondorder",
        )
