"""gpsprod step 6: the stored record's secular rate (was gps_plot record_velocity)."""

from __future__ import annotations

import numpy as np
import pytest

from geo_dataread.products import secular_rate


def _record(rates, sigmas, borrowed=None):
    comps = [
        {"params": [0.0, r], "cov_upper": [1.0, 0.0, s**2], "component": c}
        for r, s, c in zip(rates, sigmas, ("north", "east", "up"), strict=True)
    ]
    return {"record_version": 1, "model": "linear", "step_epochs": [],
            "param_names": ["offset", "rate"], "components": comps[::-1],
            "borrowed": borrowed, "n_epochs": 900, "window": [2010.0, 2019.0]}  # fmt: skip


def test_the_secular_rate_of_the_record() -> None:
    r = secular_rate("hofn", _record([-2.4, 1.1, 3.0], [0.1, 0.2, 0.3]))
    np.testing.assert_allclose(r.rate, [-2.4, 1.1, 3.0])
    np.testing.assert_allclose(r.sigma, [0.1, 0.2, 0.3])
    assert r.n == 900 and r.station == "HOFN" and r.window == (2010.0, 2019.0)


def test_refusals() -> None:
    with pytest.raises(ValueError, match="record borrowed from SKSH"):
        secular_rate("ELDC", _record([1, 1, 1], [1, 1, 1], "SKSH"))
    with pytest.raises(ValueError, match="no stored detrend record"):
        secular_rate("ELDC", None)
    rec = _record([1, 1, 1], [1, 1, 1])
    rec["param_names"] = ["offset", "slope"]
    with pytest.raises(ValueError, match="no secular rate"):
        secular_rate("ELDC", rec)
