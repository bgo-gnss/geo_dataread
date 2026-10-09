"""geo_dataread.baseline_noise: the fitted inputs for empirical baseline σ."""

from __future__ import annotations

import json

import numpy as np
import pytest

from geo_dataread import baseline as bl
from geo_dataread import baseline_noise as bn

DAY = 1.0 / 365.25
RHO0, RHO_INF, L = 0.95, 0.5, 10.0


def _network(n_sta: int = 12, n: int = 900, seed: int = 0):
    """Stations on a line; daily errors = network-wide + distance-correlated + own."""
    rng = np.random.default_rng(seed)
    t = 2024.0 + np.arange(n) * DAY
    x_km = np.r_[0.0, 0.2, np.geomspace(1, 45, n_sta - 2)]
    # shared fraction between i, j: RHO_INF + (RHO0 - RHO_INF) exp(-d/L)
    d = np.abs(x_km[:, None] - x_km[None, :])
    corr = RHO_INF + (RHO0 - RHO_INF) * np.exp(-d / L)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)
    s_true = 2.0
    series, pos = {}, {}
    noise = np.stack([chol @ rng.normal(size=(n_sta, n)) for _ in range(3)]) * s_true
    for i in range(n_sta):
        code = f"S{i:03d}"
        series[code] = (
            t,
            noise[:, i, :],
            np.full((3, n), 4.0),
        )  # formal σ 2× too large
        pos[code] = np.array([x_km[i], 0.0, 0.0])
    return series, pos


@pytest.fixture(scope="module")
def record() -> bn.BaselineNoiseRecord:
    series, pos = _network()
    return bn.fit_baseline_noise(
        series, pos, window=(2024.0, 2027.0), source="synthetic"
    )


def test_fit_recovers_scale_and_law(record) -> None:
    np.testing.assert_allclose(record.median_scale, 0.5, rtol=0.08)
    np.testing.assert_allclose(record.model.rho_inf, RHO_INF, atol=0.08)
    np.testing.assert_allclose(record.model.rho0, RHO0, atol=0.05)
    assert record.n_stations == 12 and record.n_pairs == 66


def test_record_round_trip_and_atomic_write(tmp_path, record) -> None:
    out = bn.write_baseline_noise(record, tmp_path / "noise" / "baseline_noise.json")
    assert [p.name for p in out.parent.iterdir()] == [
        "baseline_noise.json"
    ]  # no temp left
    back = bn.read_baseline_noise(out)
    assert back == record
    with pytest.raises(ValueError, match="record version"):
        bn.BaselineNoiseRecord.from_record(
            {**json.loads(out.read_text()), "record_version": 9}
        )


def test_default_path_follows_the_store(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GPS_API_STORE", str(tmp_path))
    assert bn.default_path() == tmp_path / "noise" / "baseline_noise.json"


def test_pair_inputs_prefers_own_rho_and_falls_back(record) -> None:
    rho, (ka, kb), how = bn.pair_inputs(record, "S000", "S001", distance_km=0.2)
    assert how.startswith("ρ(d") and "median" not in how
    np.testing.assert_allclose(rho, bn.es.shared_fraction_model(0.2, record.model))
    own = np.array([0.7, 0.8, 0.9])
    rho2, _, how2 = bn.pair_inputs(record, "S000", "S001", distance_km=0.2, own_rho=own)
    np.testing.assert_allclose(rho2, own)
    assert how2 == "own ρ"
    _, (ka3, _), how3 = bn.pair_inputs(
        record, "NEW1", "S001", distance_km=0.2, own_rho=[np.nan] * 3
    )
    np.testing.assert_allclose(ka3, record.median_scale)
    assert "network-median k for NEW1" in how3


def test_empirical_sigma_matches_the_baseline_scatter(record) -> None:
    """End to end: record → baseline() σ ≈ the observed A − B scatter."""
    series, _ = _network(seed=1)  # fresh noise: out of sample
    a, b = series["S000"], series["S001"]
    rho, k, _ = bn.pair_inputs(record, "S000", "S001", distance_km=0.2)
    out = bl.baseline(a, b, rho=rho, sigma_scale=k)
    obs = bn.es.daily_scatter(out.yearf, out.data)
    np.testing.assert_allclose(np.median(out.sigma, axis=1), obs, rtol=0.2)
    assert np.all(np.median(out.sigma_quadrature, axis=1) > 4 * obs)


def test_fit_refuses_without_data() -> None:
    t = 2024.0 + np.arange(10) * DAY
    with pytest.raises(ValueError, match="no station"):
        bn.fit_baseline_noise(
            {"A": (t, np.zeros((3, 10)), np.ones((3, 10)))},
            {"A": [0, 0, 0]},
            window=(2024.0, 2025.0),
        )


def test_network_stations_skips_non_station_sections() -> None:
    import configparser
    from types import SimpleNamespace

    raw = configparser.ConfigParser()
    raw.read_string(
        "[PATHS]\ntot = /x\n[DEFAULTS]\na = 1\n[FILES]\nb = 2\n"
        "[reyk]\nlatitude = 64.1\nlongitude = -21.9\n"
        "[VFLN]\nlatitude = 63.9\nlongitude = -19.1\n"
    )
    cfg = SimpleNamespace(config=raw, getStationInfo=raw.sections)
    assert bn.network_stations(cfg) == ["REYK", "VFLN"]
