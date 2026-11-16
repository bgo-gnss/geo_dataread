"""Fault planes from the legacy Skjálftalísa API (ticket T12b).

Offline: requests go to a fake transport serving the T9 recordings
(``tests/fixtures/skjalftalisa``) or synthetic bodies. The planes are checked
against the SIL files of the same events (``tests/fixtures/sil``), joined
on the exact origin time (the list has no ``slungapath``). The live test
runs only with ``GPS_LIVE_API=1``.
"""

from __future__ import annotations

import json
import os
import urllib.parse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from geo_dataread.seismic import mechanisms as mm
from geo_dataread.seismic import skjalftalisa_api as sk
from geo_dataread.seismic.schema import (
    ATTRS_KEY,
    TIME_DTYPE,
    auxiliary_plane,
    column_names,
    validate_mechanisms,
)

FIXTURES = Path(__file__).parent / "fixtures"
LEGACY = FIXTURES / "skjalftalisa"
SIL = FIXTURES / "sil"
WIN_A = ("2024-08-22T00:00:00Z", "2024-08-22T01:00:00Z")

Route = Callable[[str], tuple[int, bytes]]


class Fake:
    """A transport routing on the URL path; records the URLs."""

    def __init__(self, route: Route) -> None:
        self.route = route
        self.urls: list[str] = []

    def get(self, url: str, *, timeout: float) -> sk.HttpResponse:
        self.urls.append(url)
        status, body = self.route(urllib.parse.urlsplit(url).path)
        return sk.HttpResponse(status, body, url)


def load(name: str) -> bytes:
    return (LEGACY / name).read_bytes()


def legacy_route(
    details: dict[str, bytes] | None = None, listing: bytes | None = None
) -> Route:
    details = details or {"1352381": load("internal_detail_1352381.json")}
    body = listing if listing is not None else load("internal_20240822T00.json")

    def route(path: str) -> tuple[int, bytes]:
        if path.endswith("/quakes"):
            return 200, body
        eid = path.rsplit("/", 1)[-1]
        if eid in details:
            return 200, details[eid]
        return 404, load("error_detail_404.json")

    return route


def read(fake: Fake, *window: str, **kw: Any) -> tuple[pd.DataFrame, Any]:
    kw.setdefault("pause_s", 0.0)
    a, b = window or WIN_A
    return sk.read_skjalftalisa_mechanisms(a, b, session=fake, **kw)


def picks(eid: int, t: str, **props: Any) -> dict[str, Any]:
    base = {
        "event_id": eid,
        "time": t,
        "originating_system": sk.SYSTEM_PICKS,
        "depth": 5.0,
        "m_ml": 0.1,
        "m_mlw": 0.5,
        "seismic_moment": 1e10,
        "event_type": "qu",
    }
    lon, lat = props.pop("lon", -22.4), props.pop("lat", 63.9)
    base.update(props)
    return {
        "type": "Feature",
        "properties": base,
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
    }


def collection(features: list[dict[str, Any]]) -> bytes:
    return json.dumps({"type": "FeatureCollection", "features": features}).encode()


# --------------------------------------------------------------------------
# From the list
# --------------------------------------------------------------------------


def test_window_a_list_planes() -> None:
    fake = Fake(legacy_route())
    df, rep = read(fake)
    assert len(df) == 6 and len(fake.urls) == 1
    assert list(df.columns) == list(mm.output_columns()) + list(
        sk.MECHANISM_EXTRA_COLUMNS
    )
    assert df.index.dtype == TIME_DTYPE and df.index.is_monotonic_increasing
    assert set(df["source"]) == {"skjalftalisa"}
    assert set(df["method"]) == {"sil_fps"}
    assert set(df["planes_from"]) == {"list"}
    assert df.iloc[0]["event_id"] == "1352381"
    assert (df.iloc[0]["strike1"], df.iloc[0]["dip1"], df.iloc[0]["rake1"]) == (
        341,
        74,
        170,
    )
    # plane 2 is computed
    s2, d2, r2 = auxiliary_plane(df["strike1"], df["dip1"], df["rake1"])
    assert np.allclose(df["strike2"], s2) and np.allclose(df["rake2"], r2)
    assert np.allclose(df["dip2"], d2)
    # nothing but planes: the API's scalars are truncated (T3b)
    for col in ("quality", "misfit", "stress_drop_mpa", "corner_freq_hz"):
        assert df[col].isna().all()
    assert df["n_polarities"].isna().all() and df["sil_faulting_class"].isna().all()
    assert df["sil_path"].isna().all()
    assert not df[mm.PLANES_INCONSISTENT].any()
    assert validate_mechanisms(df).ok
    assert rep.get("mech:planes") == 6 and rep.get("mechanisms") == 6
    assert rep.get("rows:aut") == 7  # automatic rows carry no plane
    assert df.attrs[ATTRS_KEY]["frame"] == "mechanisms"
    assert df.attrs[ATTRS_KEY]["planes_from"] == "list"


def test_list_strikes_are_normalised() -> None:
    feats = [
        picks(1, "2024-08-22T00:10:00.000", strike=-72.0, dip=60.0, slip=-80.0),
        picks(2, "2024-08-22T00:11:00.000", strike=434.0, dip=80.0, slip=190.0),
        picks(3, "2024-08-22T00:12:00.000", strike=10.0, dip=95.0, slip=0.0),
        picks(4, "2024-08-22T00:13:00.000", strike=10.0, dip=None, slip=0.0),
        picks(5, "2024-08-22T00:14:00.000"),  # no plane
        picks(5, "2024-08-22T00:14:00.000"),  # chunk-edge repeat
    ]
    df, rep = read(Fake(legacy_route(listing=collection(feats))))
    assert list(df["event_id"]) == ["1", "2"]
    assert (df.iloc[0]["strike1"], df.iloc[0]["dip1"], df.iloc[0]["rake1"]) == (
        288,
        60,
        -80,
    )
    assert (df.iloc[1]["strike1"], df.iloc[1]["rake1"]) == (74, -170)
    assert rep.get("mech:bad_plane") == 2  # dip 95, partial plane
    assert rep.get("mech:no_plane") == 1 and rep.get("mech:duplicate_rows") == 1
    # the class does not see the wrap
    assert list(mm.faulting_classes(df).astype(str)) == ["normal", "strike-slip"]


def test_list_window_half_open_and_region() -> None:
    feats = [
        picks(1, "2024-08-22T00:00:00.000", strike=1.0, dip=45.0, slip=90.0),
        picks(2, "2024-08-22T00:59:59.999", strike=2.0, dip=45.0, slip=90.0),
        picks(3, "2024-08-22T01:00:00.000", strike=3.0, dip=45.0, slip=90.0),
        picks(4, "2024-08-22T00:30:00.000", strike=4.0, dip=45.0, slip=90.0, lon=-18.0),
    ]
    route = legacy_route(listing=collection(feats))
    df, rep = read(Fake(route))
    assert list(df["event_id"]) == ["1", "4", "2"]
    assert rep.get("mech:outside_window") == 1
    df, rep = read(Fake(route), bbox=(-23.0, 63.5, -22.0, 64.5))
    assert list(df["event_id"]) == ["1", "2"] and rep.get("mech:outside_region") == 1
    poly = [(-23.0, 63.5), (-22.0, 63.5), (-22.0, 64.5), (-23.0, 64.5)]
    df, _ = read(Fake(route), polygon=poly)
    assert list(df["event_id"]) == ["1", "2"]


def test_empty_window() -> None:
    df, rep = read(Fake(legacy_route(listing=load("internal_empty.json"))))
    assert len(df) == 0 and df.index.dtype == TIME_DTYPE
    assert list(df.columns)[: len(column_names("mechanisms"))] == list(
        column_names("mechanisms")
    )
    assert validate_mechanisms(df).ok


def test_mechanisms_from_rows_pure() -> None:
    rows = sk.rows_from_features(
        json.loads(load("internal_20240822T00.json"))["features"]
    )
    out, tally = sk.mechanisms_from_rows(rows)
    assert len(out) == 6 and tally["mech:picks_rows"] == 6
    assert {"latitude", "longitude", "time"} <= set(out.columns)


# --------------------------------------------------------------------------
# Agreement with the SIL files (same events)
# --------------------------------------------------------------------------


def test_list_planes_equal_sil_files() -> None:
    """Joined on the exact origin time: plane 1 is fps plane a; the computed
    plane 2 is fps plane b within its whole-degree rounding."""
    api, _ = read(Fake(legacy_route()))
    files = mm.read_sil_day_mechanisms(SIL / "2024/aug/22").frame
    j = api.join(files, how="inner", lsuffix="_api", rsuffix="_files")
    assert len(j) == 3  # the trimmed fixture day holds three of the six
    p1 = mm.same_plane_deg(
        j["strike1_api"], j["dip1_api"], j["rake1_api"],
        j["strike1_files"], j["dip1_files"], j["rake1_files"],
    )  # fmt: skip
    p2 = mm.same_plane_deg(
        j["strike2_api"], j["dip2_api"], j["rake2_api"],
        j["strike2_files"], j["dip2_files"], j["rake2_files"],
    )  # fmt: skip
    assert np.max(p1) < 1e-9
    assert np.max(p2) < 1.5
    assert (
        mm.faulting_classes(api.loc[j.index]) == mm.faulting_classes(files.loc[j.index])
    ).all()


def test_detail_planes_equal_sil_files() -> None:
    df, rep = read(Fake(legacy_route()), detail=True)
    row = df.loc[df["event_id"] == "1352381"].iloc[0]
    assert row["planes_from"] == "detail"
    assert row["sil_path"] == "/2024/aug/22/00:05:00/10:05:430"
    files = mm.read_sil_day_mechanisms(SIL / "2024/aug/22").frame
    f = files.loc[files["event_id"] == row["sil_path"]].iloc[0]
    for c in ("strike1", "dip1", "rake1", "strike2", "dip2", "rake2"):
        assert row[c] == f[c], c
    # the detail's label is empty (the ETL drops it); the file block has one
    assert (
        pd.isna(row["sil_faulting_class"]) and f["sil_faulting_class"] == "strike-slip"
    )
    assert rep.get("mech:details") == 1 and rep.get("mech:detail_not_found") == 5
    assert rep.get("mech:no_sil_label") == 1
    assert rep.get("requests") == 7
    assert (df["planes_from"] == "list").sum() == 5


# --------------------------------------------------------------------------
# Detail
# --------------------------------------------------------------------------


def detail_body(sp3: dict[str, Any] | None, **kw: Any) -> bytes:
    body = json.loads(load("internal_detail_1352381.json"))
    body["sp3"] = None if sp3 is None else json.dumps(sp3)
    body.update(kw)
    return json.dumps(body).encode()


def sp3_planes(
    a: tuple[float | None, ...], b: tuple[float | None, ...], **kw: Any
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for w, p in (("a", a), ("b", b)):
        for k, v in zip(("strike", "dip", "slip"), p, strict=True):
            out[f"fp{w}_{k}_AR"] = v
    out.update(kw)
    return out


def test_detail_inconsistent_planes_kept_and_marked() -> None:
    """The 2023 placeholder pair (180/39/−79, 14/52/−81) as a detail."""
    body = detail_body(
        sp3_planes((180, 39, -79), (14, 52, -81), faulting_mechanism="Normal")
    )
    df, rep = read(Fake(legacy_route({"1352381": body})), detail=True)
    row = df.loc[df["event_id"] == "1352381"].iloc[0]
    assert (row["strike2"], row["dip2"], row["rake2"]) == (14, 52, -81)
    assert bool(row[mm.PLANES_INCONSISTENT]) is True
    assert row["sil_faulting_class"] == "normal"
    assert rep.get("mech:detail_planes_inconsistent") == 1
    assert rep.get("mech:detail_differs_from_list") == 1  # list says 341/74/170
    assert not df.loc[df["event_id"] != "1352381", mm.PLANES_INCONSISTENT].any()
    mm.validate_marked_mechanisms(df, strict=True)
    rep_schema = validate_mechanisms(df)
    assert "planes_inconsistent" in rep_schema.kinds()


def test_detail_planes_function() -> None:
    d = sk.parse_detail(json.loads(load("internal_detail_1352381.json")))
    dp = sk.detail_planes(d)
    assert dp is not None and dp.consistent is True
    assert dp.plane1 == (341, 74, 170) and dp.plane2 == (74, 81, 16)
    assert dp.mismatch_deg < 1.5
    assert dp.time == pd.Timestamp("2024-08-22 00:10:04.454", tz="UTC")
    for name in ("internal_detail_1352541.json", "public_detail_1352381.json"):
        assert sk.detail_planes(sk.parse_detail(json.loads(load(name)))) is None
    one = sk.parse_detail(
        json.loads(detail_body(sp3_planes((341, 74, 170), (None, None, None))))
    )
    dp1 = sk.detail_planes(one)
    assert dp1 is not None and dp1.plane2 is None and dp1.consistent is None


def test_detail_without_sp3_keeps_list_row() -> None:
    df, rep = read(Fake(legacy_route({"1352381": detail_body(None)})), detail=True)
    row = df.loc[df["event_id"] == "1352381"].iloc[0]
    assert row["planes_from"] == "list" and row["strike1"] == 341
    assert row["sil_path"] == "/2024/aug/22/00:05:00/10:05:430"
    assert rep.get("mech:detail_without_planes") == 1


def test_max_details_guard() -> None:
    fake = Fake(legacy_route())
    with pytest.raises(ValueError, match="max_details"):
        read(fake, detail=True, max_details=5)
    assert len(fake.urls) == 1  # the list only, no detail fetched
    with pytest.raises(ValueError):
        read(fake, max_details=-1)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Strike slip faulting dominates ", "strike-slip"),
        ("Normal", "normal"),
        ("reverse faulting dominates", "reverse"),
        ("Dip-slip faulting dominates", "dip-slip"),
        ("", None),
        ("  ", None),
        (None, None),
        ("Oblique", None),
    ],
)
def test_sil_faulting_label(text: Any, expected: str | None) -> None:
    assert sk.sil_faulting_label(text) == expected


# --------------------------------------------------------------------------
# Live (opt-in)
# --------------------------------------------------------------------------


@pytest.mark.skipif(os.environ.get("GPS_LIVE_API") != "1", reason="GPS_LIVE_API=1")
def test_live_window_a() -> None:
    df, rep = sk.read_skjalftalisa_mechanisms(*WIN_A)
    assert len(df) == 6 and rep.get("requests") >= 1
    assert df.iloc[0]["event_id"] == "1352381"
