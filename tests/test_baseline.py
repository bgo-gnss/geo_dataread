"""Baseline A − B between two stations (gps_views.baseline_arrays)."""

from __future__ import annotations

import numpy as np
import pytest

from geo_dataread.gps_views import baseline_arrays

DAY = 1.0 / 365.25


def _series(t: np.ndarray, rate: tuple[float, float, float], level: float, s=1.0):
    data = np.vstack([level + r * (t - t[0]) for r in rate])
    return t, data, np.full_like(data, s)


def test_difference_and_combined_sigma() -> None:
    t = 2020.0 + np.arange(10) * DAY
    a = _series(t, (10.0, -5.0, 2.0), 3.0, s=3.0)
    b = _series(t, (4.0, -5.0, 1.0), 7.0, s=4.0)
    tt, d, s = baseline_arrays(a, b)
    np.testing.assert_allclose(tt, t)
    np.testing.assert_allclose(d[0], 6.0 * (t - t[0]))  # rate difference only
    np.testing.assert_allclose(d[1], 0.0, atol=1e-12)
    np.testing.assert_allclose(s, 5.0)  # sqrt(3² + 4²)


def test_rezero_removes_the_arbitrary_offset_and_can_be_disabled() -> None:
    t = 2020.0 + np.arange(5) * DAY
    a, b = _series(t, (0, 0, 0), 10.0), _series(t, (0, 0, 0), 4.0)
    assert np.all(baseline_arrays(a, b)[1] == 0.0)
    np.testing.assert_allclose(baseline_arrays(a, b, rezero=False)[1], 6.0)


def test_epochs_are_matched_by_day_not_float_equality() -> None:
    t = 2020.0 + np.arange(6) * DAY
    a = _series(t, (1.0, 1.0, 1.0), 0.0)
    jitter = t + 1e-6  # same days, not bit-identical
    b = _series(jitter[::2], (0.0, 0.0, 0.0), 0.0)  # B has every other day
    tt, d, _ = baseline_arrays(a, b)
    np.testing.assert_allclose(tt, t[::2])
    assert d.shape == (3, 3)


def test_nan_first_epoch_does_not_poison_rezero() -> None:
    t = 2020.0 + np.arange(4) * DAY
    a = _series(t, (365.25, 0, 0), 0.0)
    a[1][0, 0] = np.nan  # masked outlier on the first common epoch
    b = _series(t, (0, 0, 0), 0.0)
    _, d, _ = baseline_arrays(a, b)
    assert np.isnan(d[0, 0])
    np.testing.assert_allclose(d[0, 1:], [0.0, 1.0, 2.0])


def test_unsorted_b_is_handled() -> None:
    t = 2020.0 + np.arange(4) * DAY
    a = _series(t, (365.25, 0, 0), 0.0)
    tb, db, sb = _series(t, (0, 0, 0), 0.0)
    order = np.array([2, 0, 3, 1])
    _, d, _ = baseline_arrays(a, (tb[order], db[:, order], sb[:, order]))
    np.testing.assert_allclose(d[0], [0.0, 1.0, 2.0, 3.0])


def test_no_common_epoch_refuses() -> None:
    a = _series(2020.0 + np.arange(3) * DAY, (0, 0, 0), 0.0)
    b = _series(2021.0 + np.arange(3) * DAY, (0, 0, 0), 0.0)
    with pytest.raises(ValueError, match="share no epoch"):
        baseline_arrays(a, b)
