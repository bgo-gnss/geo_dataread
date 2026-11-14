"""Event matching between catalogues (ticket T15 of PLAN-seismicity.md).

Synthetic catalogues with known pairs (exact geodesic distances from
``Geod.fwd``, exact ns times), swarms, empty sides, tolerance boundaries, id
join precedence and a 100k-event performance run; then one real check on the
recorded fixtures (sil-files day vs quakes-sil / skjalftalisa / SeisComP).
"""

from __future__ import annotations

import json
import time
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pyproj import Geod

from geo_dataread.seismic import compare as cm
from geo_dataread.seismic import quakes_api as qa
from geo_dataread.seismic import skjalftalisa_api as sk
from geo_dataread.seismic.sil_catalog import read_sil_day

FIX = Path(__file__).parent / "fixtures"
GEOD = Geod(ellps="GRS80")
T0 = pd.Timestamp("2024-08-22T00:00:00Z")
LAT0, LON0 = 63.88, -22.40


def offset(lat: float, lon: float, az: float, km: float) -> tuple[float, float]:
    lon2, lat2, _ = GEOD.fwd(lon, lat, az, km * 1000.0)
    return float(lat2), float(lon2)


def catalog(
    times: list[pd.Timestamp] | pd.DatetimeIndex,
    lats: list[float] | np.ndarray[Any, Any],
    lons: list[float] | np.ndarray[Any, Any],
    *,
    source: str = "quakes-seiscomp",
    **cols: Any,
) -> pd.DataFrame:
    n = len(lats)
    data: dict[str, Any] = {
        "source": [source] * n,
        "event_id": cols.pop("event_id", [f"{source}-{i}" for i in range(n)]),
        "latitude": np.asarray(lats, dtype=float),
        "longitude": np.asarray(lons, dtype=float),
        "depth_km": cols.pop("depth_km", np.full(n, 5.0)),
        "magnitude": cols.pop("magnitude", np.full(n, 1.0)),
        "magnitude_type": cols.pop("magnitude_type", ["Mlw"] * n),
    }
    data.update(cols)
    idx = pd.DatetimeIndex(times, name="time")
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return pd.DataFrame(data, index=idx)


def secs(*s: float) -> list[pd.Timestamp]:
    return [T0 + pd.Timedelta(seconds=x) for x in s]


# --------------------------------------------------------------------------
# Ids
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expect"),
    [
        ("SIL1352381", "1352381"),
        ("sil1352381", "1352381"),
        ("1352381", "1352381"),
        (" 001352381 ", "1352381"),
        (1352381, "1352381"),
        (np.int64(7), "7"),
        ("IMO2024qnogqh", None),
        ("/2024/aug/22/00:05:00/10:05:430", None),
        ("", None),
        (None, None),
        (float("nan"), None),
        (pd.NA, None),
        (True, None),
        (-3, None),
    ],
)
def test_sil_number(value: object, expect: str | None) -> None:
    assert cm.sil_number(value) == expect


@pytest.mark.parametrize(
    ("value", "expect"),
    [
        (" /2024/aug/22/00:05:00/10:05:430 ", "/2024/aug/22/00:05:00/10:05:430"),
        ("/2024/aug/22/aut.mag#3", "/2024/aug/22/aut.mag#3"),
        ("SIL1", None),
        ("", None),
        (None, None),
        (float("nan"), None),
    ],
)
def test_sil_path_key(value: object, expect: str | None) -> None:
    assert cm.sil_path_key(value) == expect


def test_key_sil_number_only_for_sil_sources() -> None:
    df = catalog(secs(0, 1), [LAT0] * 2, [LON0] * 2, event_id=["SIL5", "5"])
    assert list(cm.key_sil_number(df)) == [None, None]  # quakes-seiscomp
    df["source"] = ["quakes-sil", "skjalftalisa"]
    assert list(cm.key_sil_number(df)) == ["5", "5"]


# --------------------------------------------------------------------------
# Core on arrays
# --------------------------------------------------------------------------


def test_time_kinds_agree() -> None:
    ns = np.array([0, 1_500_000_000, 10_000_000_000], dtype=np.int64)
    ref, _ = cm.as_time_ns(ns)
    dt64, _ = cm.as_time_ns(ns.astype("datetime64[ns]").astype("datetime64[us]"))
    fl, _ = cm.as_time_ns(ns / 1e9)
    assert (ref == dt64).all() and (ref == fl).all()
    _, valid = cm.as_time_ns(np.array([1.0, np.nan]))
    assert valid.tolist() == [True, False]
    _, valid = cm.as_time_ns(np.array(["2024-01-01", "NaT"], dtype="datetime64[ns]"))
    assert valid.tolist() == [True, False]
    with pytest.raises(cm.MatchError):
        cm.as_time_ns(np.array(["a"]))


def test_known_pairs_arrays() -> None:
    ta = np.array([0.0, 100.0, 200.0, 300.0])
    lat = np.full(4, LAT0)
    lon = np.full(4, LON0)
    p1 = offset(LAT0, LON0, 90.0, 3.0)
    p3 = offset(LAT0, LON0, 0.0, 20.0)
    tb = np.array([100.5, 0.2, 303.0, 999.0])
    latb = np.array([p1[0], LAT0, p3[0], LAT0])
    lonb = np.array([p1[1], LON0, p3[1], LON0])
    m = cm.match_arrays(ta, lat, lon, tb, latb, lonb)
    got = {(int(a), int(b), m.labels[int(s)]) for a, b, s in zip(m.ia, m.ib, m.step)}
    assert got == {(0, 1, "tolerance"), (1, 0, "tolerance"), (3, 2, "loose")}
    k = int(np.flatnonzero(m.ia == 1)[0])
    assert m.dt_s[k] == pytest.approx(0.5)
    assert m.dist_km[k] == pytest.approx(3.0, abs=1e-9)
    assert m.azimuth_deg[k] == pytest.approx(90.0, abs=1e-6)
    assert m.labels == ("tolerance", "loose")


@pytest.mark.parametrize(
    ("dt_ns", "km", "matched"),
    [
        (2_000_000_000, 0.0, True),  # |dt| = 2 s exactly: inclusive
        (-2_000_000_000, 0.0, True),
        (2_000_000_001, 0.0, False),  # 1 ns beyond
        (0, 9.999, True),
        (0, 10.0, True),  # exact (within float of the geodesic)
        (0, 10.001, False),
        (2_000_000_000, 10.0, True),  # both at the bound: cost 2
    ],
)
def test_boundaries_inclusive(dt_ns: int, km: float, matched: bool) -> None:
    lat2, lon2 = offset(LAT0, LON0, 45.0, km) if km else (LAT0, LON0)
    if km == 10.0:
        # nudge inward by 1 µm so float noise of fwd/inv cannot flip the test
        lat2, lon2 = offset(LAT0, LON0, 45.0, km - 1e-9)
    m = cm.match_arrays(
        np.array([0], dtype=np.int64),
        [LAT0],
        [LON0],
        np.array([dt_ns], dtype=np.int64),
        [lat2],
        [lon2],
        tolerances=[(2.0, 10.0)],
    )
    assert (len(m) == 1) is matched


def test_swarm_one_second_apart() -> None:
    """A swarm at 1 s spacing, B with small jitter: every true pair wins."""
    rng = np.random.default_rng(1)
    n = 200
    ta = np.arange(n, dtype=float)
    lat = LAT0 + rng.uniform(-0.02, 0.02, n)
    lon = LON0 + rng.uniform(-0.04, 0.04, n)
    tb = ta + rng.uniform(-0.15, 0.15, n)
    latb = lat + rng.normal(0, 0.002, n)
    lonb = lon + rng.normal(0, 0.004, n)
    perm = rng.permutation(n)
    m = cm.match_arrays(ta, lat, lon, tb[perm], latb[perm], lonb[perm])
    assert len(m) == n
    assert (perm[m.ib] == m.ia).all()
    assert set(m.step.tolist()) == {0}
    assert m.candidates[0] > n  # wrong neighbours were candidates


def test_chain_swarm_shift() -> None:
    """Same place, A at 0,1,2,…, B shifted +0.4 s: greedy keeps the shift."""
    n = 50
    ta = np.arange(n, dtype=float)
    tb = ta + 0.4
    lat, lon = np.full(n, LAT0), np.full(n, LON0)
    m = cm.match_arrays(ta, lat, lon, tb, lat, lon)
    assert len(m) == n and (m.ia == m.ib).all()
    assert np.allclose(m.dt_s, 0.4)


def test_greedy_closest_first_not_max_cardinality() -> None:
    """A0=0, A1=1 and B0=0.6, B1=1.6 at one spot: closest first takes
    (A1, B0) at 0.4 s, then (A0, B1) at 1.6 s (still inside 2 s)."""
    lat, lon = np.full(2, LAT0), np.full(2, LON0)
    m = cm.match_arrays(
        np.array([0.0, 1.0]),
        lat,
        lon,
        np.array([0.6, 1.6]),
        lat,
        lon,
        tolerances=[(2.0, 10.0)],
    )
    assert sorted(zip(m.ia.tolist(), m.ib.tolist())) == [(0, 1), (1, 0)]


def test_tie_rules() -> None:
    """Equal cost: smaller |dt| wins; then smaller distance; then positions."""
    # one A, two B at the same cost: B0 dt=1 s d=0; B1 dt=0 s d=sqrt(1/4)*10 km…
    # cost B0 = (1/2)^2 = 0.25; cost B1 = (5/10)^2 = 0.25 → B1 (|dt| 0) wins.
    lat5, lon5 = offset(LAT0, LON0, 0.0, 5.0)
    m = cm.match_arrays(
        np.array([10.0]),
        [LAT0],
        [LON0],
        np.array([11.0, 10.0]),
        [LAT0, lat5],
        [LON0, lon5],
        tolerances=[(2.0, 10.0)],
    )
    assert m.ib.tolist() == [1]
    # identical candidates: the lower B position wins, then the lower A
    m = cm.match_arrays(
        np.array([10.0, 10.0]),
        [LAT0] * 2,
        [LON0] * 2,
        np.array([10.0, 10.0]),
        [LAT0] * 2,
        [LON0] * 2,
    )
    assert list(zip(m.ia.tolist(), m.ib.tolist())) == [(0, 0), (1, 1)]


def test_invalid_rows_never_candidates() -> None:
    m = cm.match_arrays(
        np.array([0.0, np.nan, 2.0]),
        [LAT0, LAT0, np.nan],
        [LON0] * 3,
        np.array([0.0, 1.0, 2.0]),
        [LAT0] * 3,
        [LON0] * 3,
    )
    assert m.ia.tolist() == [0] and m.invalid_a == 2 and m.invalid_b == 0


def test_exclude_and_errors() -> None:
    lat, lon = [LAT0, LAT0], [LON0, LON0]
    m = cm.match_arrays([0.0, 1.0], lat, lon, [0.0, 1.0], lat, lon, exclude_a=[0])
    assert m.ia.tolist() == [1]
    m = cm.match_arrays(
        [0.0, 1.0],
        lat,
        lon,
        [0.0, 1.0],
        lat,
        lon,
        exclude_b=np.array([False, True]),
    )
    assert m.ia.tolist() == [0]
    with pytest.raises(cm.MatchError, match="shape"):
        cm.match_arrays([0.0], [LAT0, LAT0], [LON0], [0.0], [LAT0], [LON0])
    with pytest.raises(cm.MatchError, match="> 0"):
        cm.Tolerance(0.0, 10.0)
    with pytest.raises(cm.MatchError, match="unique"):
        cm.match_arrays(
            [0.0],
            [LAT0],
            [LON0],
            [0.0],
            [LAT0],
            [LON0],
            tolerances=[(1, 1, "x"), (2, 2, "x")],
        )
    with pytest.raises(cm.MatchError, match="max_candidates"):
        cm.match_arrays(
            np.zeros(10),
            np.full(10, LAT0),
            np.full(10, LON0),
            np.zeros(10),
            np.full(10, LAT0),
            np.full(10, LON0),
            max_candidates=50,
        )


def test_empty_sides_arrays() -> None:
    e = np.zeros(0)
    m = cm.match_arrays(e, e, e, [0.0], [LAT0], [LON0])
    assert len(m) == 0 and m.candidates == (0, 0)
    m = cm.match_arrays(e, e, e, e, e, e)
    assert len(m) == 0


# --------------------------------------------------------------------------
# Frame wrapper
# --------------------------------------------------------------------------


def test_match_catalogs_known_pairs_and_output() -> None:
    p = offset(LAT0, LON0, 90.0, 4.0)
    a = catalog(
        secs(0, 50, 100),
        [LAT0] * 3,
        [LON0] * 3,
        source="quakes-sil",
        magnitude=[1.0, 2.0, 0.5],
        magnitude_type=["Mlw", "Mlw", "aut.mag"],
        mag_mlw=[1.0, 2.0, np.nan],
        evaluation_mode=["manual"] * 2 + ["automatic"],
    )
    b = catalog(
        secs(0.5, 104, 400),
        [p[0], LAT0, LAT0],
        [p[1], LON0, LON0],
        magnitude=[0.6, 1.5, 3.0],
        magnitude_type=["ML_SIL"] * 3,
        depth_km=[7.0, 5.0, 5.0],
        evaluation_mode=["manual"] * 3,
    )
    a.attrs["catalog"] = {"source": "quakes-sil", "location": "x", "window": {}}
    res = cm.match_catalogs(a, b)
    pr = res.pairs
    assert pr[["pos_a", "pos_b"]].values.tolist() == [[0, 0], [2, 1]]
    assert pr["step"].astype(str).tolist() == ["tolerance", "loose"]
    r0 = pr.iloc[0]
    assert r0["dt_s"] == pytest.approx(0.5)
    assert r0["dist_km"] == pytest.approx(4.0, abs=1e-6)
    assert r0["de_km"] == pytest.approx(4.0, abs=1e-6)
    assert r0["dn_km"] == pytest.approx(0.0, abs=1e-6)
    assert r0["ddepth_km"] == pytest.approx(2.0)
    assert r0["dmag"] == pytest.approx(-0.4)
    assert r0["mag_types"] == "Mlw|ML_SIL"
    assert np.isnan(r0["dmag_mlw"])  # B has no Mlw
    assert pr.iloc[1]["mag_types"] == "aut.mag|ML_SIL"
    assert r0["event_id_a"] == "quakes-sil-0"
    assert r0["time_a"] == T0
    assert res.unmatched_a["pos"].tolist() == [1]
    assert res.unmatched_b["pos"].tolist() == [2]
    assert res.unmatched_a.attrs["catalog"]["source"] == "quakes-sil"
    s = res.summary
    assert s["matched"] == {
        "id:sil_path": 0,
        "id:sil_aut_path": 0,
        "id:sil_number": 0,
        "tolerance": 1,
        "loose": 1,
    }
    assert (s["n_a"], s["n_b"], s["matched_total"]) == (3, 3, 2)
    assert (s["unmatched_a"], s["unmatched_b"]) == (1, 1)
    assert s["mag_types"] == {"Mlw|ML_SIL": 1, "aut.mag|ML_SIL": 1}
    assert s["source_a"] == "quakes-sil" and s["source_b"] == "quakes-seiscomp"
    json.dumps(s)  # JSON-safe
    for frame in (pr, res.unmatched_a, res.unmatched_b):
        prov = frame.attrs[cm.MATCH_ATTRS_KEY]
        assert prov["a"]["location"] == "x"
        assert prov["tolerances"][0] == {
            "label": "tolerance",
            "max_dt_s": 2.0,
            "max_km": 10.0,
        }
        json.dumps(prov)


def test_both_empty_and_one_empty() -> None:
    e = catalog([], [], [])
    one = catalog(secs(0), [LAT0], [LON0])
    for a, b in ((e, e), (e, one), (one, e)):
        res = cm.match_catalogs(a, b)
        assert len(res.pairs) == 0
        assert "dt_s" in res.pairs.columns
        assert res.summary["matched_total"] == 0
        assert res.summary["unmatched_a"] == len(a)
        assert res.summary["unmatched_b"] == len(b)
        json.dumps(res.summary)


def test_twin_split_reviewed_wins() -> None:
    """B holds a reviewed event and its automatic twin 0.5 s / 1 km off
    (the quakes-sil twin split); A only the reviewed one."""
    tw = offset(LAT0, LON0, 180.0, 1.0)
    a = catalog(secs(10.0), [LAT0], [LON0])
    b = catalog(
        secs(10.5, 10.0),
        [tw[0], LAT0],
        [tw[1], LON0],
        evaluation_mode=["automatic", "manual"],
    )
    res = cm.match_catalogs(a, b)
    assert res.pairs["pos_b"].tolist() == [1]
    assert res.unmatched_b["evaluation_mode"].tolist() == ["automatic"]


def test_id_join_precedence() -> None:
    """The sil_number join pairs A0↔B1 although a tolerance step would pair
    A0↔B0 (identical time and place); B0 then pairs A1 by tolerance."""
    a = catalog(
        secs(0, 1),
        [LAT0] * 2,
        [LON0] * 2,
        source="quakes-sil",
        event_id=["SIL100", "SIL101"],
    )
    b = catalog(
        secs(0, 30),
        [LAT0] * 2,
        [LON0] * 2,
        source="skjalftalisa",
        event_id=["555", "100"],
    )
    res = cm.match_catalogs(a, b)
    pairs = res.pairs.set_index("pos_a")
    assert pairs.loc[0, "pos_b"] == 1 and pairs.loc[0, "step"] == "id:sil_number"
    assert np.isnan(pairs.loc[0, "cost"]) and pairs.loc[0, "dt_s"] == 30.0
    assert pairs.loc[1, "pos_b"] == 0 and pairs.loc[1, "step"] == "tolerance"
    assert res.summary["id_outside_tolerance"] == 1  # 30 s > 5 s
    off = cm.match_catalogs(a, b, id_join="none")
    assert off.pairs["pos_b"].tolist() == [0]
    assert list(off.summary["matched"]) == ["tolerance", "loose"]


def test_sil_path_before_sil_number_and_aut_path() -> None:
    a = catalog(
        secs(0, 60, 120),
        [LAT0] * 3,
        [LON0] * 3,
        source="sil-files",
        event_id=["/p/a", "/p/b", "/p/c"],
        sil_path=["/p/a", "/p/b", "/p/c"],
        sil_aut_path=["/p/a", "/p/b-aut", "/p/c"],
    )
    b = catalog(
        secs(0, 61, 200),
        [LAT0] * 3,
        [LON0] * 3,
        source="skjalftalisa",
        event_id=["1", "2", "3"],
        sil_path=["/p/a", "/p/b-aut", None],
    )
    res = cm.match_catalogs(a, b)
    steps = dict(zip(res.pairs["pos_a"], res.pairs["step"].astype(str), strict=True))
    assert steps == {0: "id:sil_path", 1: "id:sil_aut_path"}
    assert res.unmatched_a["pos"].tolist() == [2]
    only = cm.match_catalogs(a, b, id_join=["sil_path"])
    assert only.pairs["step"].astype(str).tolist() == ["id:sil_path", "tolerance"]
    with pytest.raises(cm.MatchError, match="id join"):
        cm.match_catalogs(a, b, id_join=["nope"])


def test_duplicate_id_falls_through() -> None:
    a = catalog(
        secs(0, 100),
        [LAT0] * 2,
        [LON0] * 2,
        source="quakes-sil",
        event_id=["SIL7", "SIL7"],
    )
    b = catalog(secs(0.2), [LAT0], [LON0], source="skjalftalisa", event_id=["7"])
    res = cm.match_catalogs(a, b)
    assert res.summary["id_ambiguous"]["id:sil_number"] == 3
    assert res.pairs["step"].astype(str).tolist() == ["tolerance"]
    assert res.pairs["pos_a"].tolist() == [0]


def test_custom_idjoin_and_exact_step() -> None:
    a = catalog(secs(0, 10), [LAT0] * 2, [LON0] * 2, event_id=["x", "y"])
    b = catalog(secs(0.01, 11), [LAT0] * 2, [LON0] * 2, event_id=["q", "y"])

    def by_id(df: pd.DataFrame) -> Any:
        return df["event_id"].astype(object).to_numpy()

    res = cm.match_catalogs(
        a,
        b,
        id_join=[cm.IdJoin("eid", by_id, by_id)],
        tolerances=(cm.EXACT_TOLERANCE, *cm.DEFAULT_TOLERANCES),
    )
    assert res.pairs["step"].astype(str).tolist() == ["exact", "id:eid"]


def test_bad_frames() -> None:
    a = catalog(secs(0), [LAT0], [LON0])
    with pytest.raises(cm.MatchError, match="naive"):
        cm.match_catalogs(a.tz_localize(None), a)
    with pytest.raises(cm.MatchError, match="DatetimeIndex"):
        cm.match_catalogs(a.reset_index(), a)
    with pytest.raises(cm.MatchError, match="latitude"):
        cm.match_catalogs(a.drop(columns="latitude"), a)


def test_nat_and_microsecond_index() -> None:
    a = catalog(secs(0, 5), [LAT0] * 2, [LON0] * 2)
    a.index = pd.DatetimeIndex([a.index[0], pd.NaT], name="time").as_unit("us")
    b = catalog(secs(0.25), [LAT0], [LON0])
    res = cm.match_catalogs(a, b)
    assert res.pairs["dt_s"].tolist() == [0.25]
    assert res.summary["invalid_a"] == 1


def test_performance_100k() -> None:
    """~100k events per side over 30 days, with a dense swarm (1 s spacing
    for 20k events): seconds, and every planted pair found."""
    rng = np.random.default_rng(7)
    n_bg, n_sw = 80_000, 20_000
    t_bg = np.sort(rng.uniform(0, 30 * 86400, n_bg))
    t_sw = 10 * 86400 + np.arange(n_sw, dtype=float)
    ta = np.concatenate([t_bg, t_sw])
    lat = np.concatenate(
        [rng.uniform(63.3, 66.5, n_bg), LAT0 + rng.uniform(-0.05, 0.05, n_sw)]
    )
    lon = np.concatenate(
        [rng.uniform(-24, -13, n_bg), LON0 + rng.uniform(-0.1, 0.1, n_sw)]
    )
    n = len(ta)
    keep = rng.random(n) < 0.9
    tb = ta[keep] + rng.uniform(-0.2, 0.2, int(keep.sum()))
    latb = lat[keep] + rng.normal(0, 0.003, int(keep.sum()))
    lonb = lon[keep] + rng.normal(0, 0.006, int(keep.sum()))
    extra = 10_000
    tb = np.concatenate([tb, rng.uniform(0, 30 * 86400, extra)])
    latb = np.concatenate([latb, rng.uniform(63.3, 66.5, extra)])
    lonb = np.concatenate([lonb, rng.uniform(-24, -13, extra)])
    truth = np.concatenate([np.flatnonzero(keep), np.full(extra, -1)])
    to_ns = (T0.value + (ta * 1e9)).astype("datetime64[ns]")
    a = catalog(pd.DatetimeIndex(to_ns).tz_localize("UTC"), lat, lon)
    order = np.argsort(tb)
    b = catalog(
        pd.DatetimeIndex(
            (T0.value + tb[order] * 1e9).astype("datetime64[ns]")
        ).tz_localize("UTC"),
        latb[order],
        lonb[order],
    )
    truth = truth[order]
    t1 = time.perf_counter()
    res = cm.match_catalogs(a, b)
    elapsed = time.perf_counter() - t1
    assert elapsed < 30.0, elapsed
    pr = res.pairs
    correct = (truth[pr["pos_b"].to_numpy()] == pr["pos_a"].to_numpy()).mean()
    assert len(pr) >= int(keep.sum())
    assert correct > 0.999
    print(
        f"\n100k match: {len(a)} x {len(b)} → {len(pr)} pairs in {elapsed:.2f} s; "
        f"{res.summary['candidates']}; correct {correct:.5f}"
    )


# --------------------------------------------------------------------------
# Real fixtures
# --------------------------------------------------------------------------


class _Serve:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def get(self, url: str, *, timeout: float) -> sk.HttpResponse:
        urllib.parse.urlsplit(url)
        return sk.HttpResponse(200, self.body, url)


def _quakes(name: str, system: qa.System) -> pd.DataFrame:
    feats = json.loads((FIX / "quakes" / name).read_text())["features"] or []
    return qa.features_to_frame(feats, system=system)


def _cut(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    return df.loc[(df.index >= lo) & (df.index < hi)]


def _skjalftalisa(name: str, start: str, end: str) -> pd.DataFrame:
    body = (FIX / "skjalftalisa" / name).read_bytes()
    df, _ = sk.read_skjalftalisa(start, end, session=_Serve(body), pause_s=0.0)
    return df


WIN_A = ("2024-08-22T00:00:00Z", "2024-08-22T01:00:00Z")
DAY_B = ("2026-09-10T00:00:00Z", "2026-09-11T00:00:00Z")


def test_real_window_a() -> None:
    """2024-08-22 00–01 h. The SIL fixture day is trimmed (8 of 250 reviewed
    events kept), so sil-files has 4 events in the hour against 7 in the
    APIs: the 3 unmatched API events are a trimming artefact."""
    files, _ = read_sil_day(FIX / "sil" / "2024" / "aug" / "22")
    files = _cut(files, *WIN_A)
    qsil = _cut(_quakes("internal_sil_20240822T00.json", "sil"), *WIN_A)
    qsc = _cut(_quakes("internal_seiscomp_20240822T00.json", "seiscomp"), *WIN_A)
    legacy = _cut(_skjalftalisa("internal_20240822T00.json", *WIN_A), *WIN_A)
    assert (len(files), len(qsil), len(qsc), len(legacy)) == (4, 7, 6, 7)

    r = cm.match_catalogs(files, qsil)
    assert r.summary["matched"]["tolerance"] == 4 and r.summary["matched_total"] == 4
    assert r.summary["unmatched_b"] == 3
    assert r.pairs["dt_s"].abs().max() == 0.0
    assert r.pairs["dist_km"].max() < 1e-6

    r = cm.match_catalogs(qsil, legacy)
    assert r.summary["matched"]["id:sil_number"] == 7
    assert r.summary["id_outside_tolerance"] == 0
    assert r.pairs["dist_km"].max() < 1e-6

    # sil_path from the two recorded details: one reviewed, one automatic
    for eid in ("1352381", "1352541"):
        body = json.loads(
            (FIX / "skjalftalisa" / f"internal_detail_{eid}.json").read_text()
        )
        legacy.loc[legacy["event_id"] == eid, "sil_path"] = body["slungapath"]
    r = cm.match_catalogs(files, legacy)
    assert r.summary["matched"]["id:sil_path"] == 2
    assert r.summary["matched"]["tolerance"] == 2
    ids = r.pairs.loc[r.pairs["step"] == "id:sil_path", "event_id_b"].tolist()
    assert sorted(ids) == ["1352381", "1352541"]

    r = cm.match_catalogs(qsil, qsc)  # back-imported SeisComP (period B)
    assert r.summary["matched"]["tolerance"] == 6
    assert r.unmatched_a["event_id"].tolist() == ["SIL1352541"]  # automatic
    assert r.pairs["dt_s"].abs().max() < 1e-3
    assert r.pairs["dist_km"].max() < 0.1  # SeisComP rounds to 0.001°
    assert set(r.pairs["mag_types"]) == {"Mlw|ML_SIL"}
    print(
        "\nwindow A:",
        json.dumps(
            {k: r.summary[k] for k in ("matched", "unmatched_a", "unmatched_b")}
        ),
    )


def test_real_day_b() -> None:
    """2026-09-10, period C: SIL automatic only, SeisComP native."""
    files, _ = read_sil_day(FIX / "sil" / "2026" / "sep" / "10")
    files = _cut(files, *DAY_B)
    qsil = _cut(_quakes("internal_sil_20260910.json", "sil"), *DAY_B)
    qsc = _cut(_quakes("internal_seiscomp_20260910.json", "seiscomp"), *DAY_B)
    legacy = _cut(_skjalftalisa("internal_20260910.json", *DAY_B), *DAY_B)
    out: dict[str, Any] = {"n": [len(files), len(qsil), len(qsc), len(legacy)]}
    for name, (x, y) in {
        "files-quakes_sil": (files, qsil),
        "quakes_sil-legacy": (qsil, legacy),
        "files-legacy": (files, legacy),
        "quakes_sil-seiscomp": (qsil, qsc),
    }.items():
        r = cm.match_catalogs(x, y)
        out[name] = {
            "matched": {k: v for k, v in r.summary["matched"].items() if v},
            "unmatched": [r.summary["unmatched_a"], r.summary["unmatched_b"]],
            "dt_p95": r.summary["steps"]["tolerance"]["abs_dt_s_p95"],
            "dist_p95": r.summary["steps"]["tolerance"]["dist_km_p95"],
        }
    print("\nday B:", json.dumps(out))
    # the SIL fixture day keeps the first 6 of 19 automatic lines (README)
    assert out["n"] == [6, 19, 48, 19]
    assert out["files-quakes_sil"]["matched"] == {"tolerance": 6}
    assert out["files-quakes_sil"]["unmatched"] == [0, 13]
    assert out["files-legacy"]["matched"] == {"tolerance": 6}
    assert out["quakes_sil-legacy"]["matched"] == {"id:sil_number": 19}
    # period C: SIL automatic vs SeisComP's own solutions (incl. rejected)
    assert out["quakes_sil-seiscomp"]["matched"] == {"tolerance": 18, "loose": 1}
    assert out["quakes_sil-seiscomp"]["unmatched"] == [0, 29]
