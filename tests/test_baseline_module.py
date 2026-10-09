"""geo_dataread.baseline: the numpy-only baseline layer, used by every consumer."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from geo_dataread import baseline as bl
from geo_dataread import gps_views

DAY = 1.0 / 365.25


def _series(t, rate, level, s=1.0):
    t = np.asarray(t, dtype=float)
    data = np.vstack([level + r * (t - t[0]) for r in rate])
    return t, data, np.full_like(data, s)


def test_imports_nothing_heavy() -> None:
    """The point of the module: a light runtime (gps_api) can use it."""
    code = (
        "import sys, geo_dataread.baseline; "
        "bad = [m for m in ('matplotlib', 'gps_analysis', 'gps_parser', 'pandas') "
        "if m in sys.modules]; print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == ""


def test_gps_views_re_exports_the_same_function() -> None:
    assert gps_views.baseline_arrays is bl.baseline_arrays


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("VFLN-VFLS", ("VFLN", "VFLS")),
        ("vfln-vfls", ("VFLN", "VFLS")),
        ("VFLN", None),
        ("VFLN-VFLS-X", None),
        ("VFL-VFLS", None),
    ],
)
def test_split_baseline(spec, expected) -> None:
    assert bl.split_baseline(spec) == expected


def test_match_epochs_by_day_any_order() -> None:
    ta = 2020.0 + np.arange(5) * DAY
    tb = ta[[4, 0, 2]] + 1e-6  # unsorted, jittered, every other day
    ia, ib = bl.match_epochs(ta, tb)
    assert ia.tolist() == [0, 2, 4]
    np.testing.assert_allclose(tb[ib], ta[ia], atol=1e-5)
    assert bl.match_epochs(ta, np.array([]))[0].size == 0


def test_ends_differ_exactly_to_the_baseline() -> None:
    """Drawn behind the baseline, the ends' gap IS the baseline."""
    ta = 2020.0 + np.arange(10) * DAY
    tb = 2020.0 + np.arange(2, 14) * DAY  # B starts later and runs longer
    a = _series(ta, (5.0, -1.0, 2.0), 40.0, s=3.0)
    b = _series(tb, (1.0, 1.0, 1.0), -7.0, s=4.0)
    for rezero in (True, False):
        res = bl.baseline(a, b, stations=("AAAA", "BBBB"), rezero=rezero)
        (ya, ea), (yb, eb) = res.ends
        assert ya.size == 10 and yb.size == 12  # FULL series, not just common
        ia, ib = bl.match_epochs(ya, yb)
        np.testing.assert_allclose(
            ea[:, ia] - eb[:, ib], res.data if rezero else res.data - res.data[:, :1]
        )
    res = bl.baseline(a, b)
    np.testing.assert_allclose(res.data[:, 0], 0.0)
    np.testing.assert_allclose(res.sigma, 5.0)
    np.testing.assert_allclose(res.yearf, ta[2:])


def test_shape_mismatch_and_no_overlap_refuse() -> None:
    t = 2020.0 + np.arange(3) * DAY
    good = _series(t, (0, 0, 0), 0.0)
    with pytest.raises(ValueError, match="do not match"):
        bl.baseline(good, (t, np.zeros((3, 2)), np.zeros((3, 2))))
    with pytest.raises(ValueError, match="share no epoch"):
        bl.baseline(good, _series(t + 1.0, (0, 0, 0), 0.0))


def test_check_same_view() -> None:
    bl.check_same_view(("AAAA", "BBBB"), (True, True))
    with pytest.raises(ValueError, match="BBBB has none"):
        bl.check_same_view(("AAAA", "BBBB"), (True, False))


def _pair(n: int = 5):
    t = 2026.0 + np.arange(n) / 365.25
    sa = np.full((3, n), 2.0)
    sb = np.full((3, n), 2.0)
    return (t, np.zeros((3, n)), sa), (t, np.ones((3, n)), sb)


def test_default_sigma_is_quadrature_and_unchanged() -> None:
    a, b = _pair()
    out = bl.baseline(a, b)
    np.testing.assert_allclose(out.sigma, np.sqrt(8.0))
    assert out.sigma_quadrature is None and out.rho is None


def test_empirical_sigma_with_shared_fraction_and_scale() -> None:
    a, b = _pair()
    out = bl.baseline(
        a, b, rho=[0.75, 0.0, 1.0], sigma_scale=([1.0, 1.0, 0.5], [1.0, 1.0, 0.5])
    )
    np.testing.assert_allclose(out.sigma[0], 2.0 * np.sqrt(0.5))  # √(4+4−2·0.75·4)
    np.testing.assert_allclose(out.sigma[1], np.sqrt(8.0))  # ρ = 0 → quadrature
    np.testing.assert_allclose(out.sigma[2], 0.0)  # fully shared, equal σ
    np.testing.assert_allclose(out.sigma_quadrature, np.sqrt(8.0))
    assert out.rho.shape == (3, 5)
    np.testing.assert_allclose(out.data, bl.baseline(a, b).data)  # values untouched


def test_empirical_sigma_refusals() -> None:
    a, b = _pair()
    with pytest.raises(ValueError, match="needs rho"):
        bl.baseline(a, b, sigma_scale=([1, 1, 1], [1, 1, 1]))
    with pytest.raises(ValueError, match="rho must"):
        bl.baseline(a, b, rho=[0.5, 0.5])
    with pytest.raises(ValueError, match="rho must"):
        bl.baseline(a, b, rho=[0.5, 0.5, 1.5])
    with pytest.raises(ValueError, match="sigma_scale needs"):
        bl.baseline(a, b, rho=[0.5] * 3, sigma_scale=([1, 1], [1, 1, 1]))
