"""Tests for the Quakes API detail reader (ticket T11).

Offline: a fake session serves the recorded detail and error responses in
``tests/fixtures/quakes/`` (by URL path). The default urllib transport is
disabled for the whole module, so no test can touch the network by accident.
One live test runs only with ``GPS_LIVE_API=1``.
"""

from __future__ import annotations

import copy
import json
import math
import os
import urllib.parse
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from geo_dataread.seismic import quakes_api as qa
from geo_dataread.seismic.schema import SchemaError, validate
from geo_dataread.seismic.sil_catalog import KM_PER_DEG

FIX = Path(__file__).parent / "fixtures" / "quakes"
NO_PAUSE = qa.FetchPolicy(pause_s=0.0, backoff_s=0.0)
Z90 = 1.6448536269514722  # two-sided 90 % normal quantile

QNOGQH = "IMO2024qnogqh"  # back-imported SIL event: one SIL origin (degrees)
RUSIBB = "IMO2026rusibb"  # native: 5 LOCSAT origins + 1 SIL origin
RUTIQI = "IMO2026rutiqi"  # native: manual + rejected, "not existing"


# --------------------------------------------------------------------------
# Fixtures and fake transport
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeResponse:
    status_code: int
    content: bytes


def _body(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIX / name).read_text())
    return data


def _detail(kind: str, event_id: str) -> qa.QuakeDetail:
    return qa.parse_quake_detail(
        _body(f"{kind}_detail_{event_id}.json"),
        event_id=event_id,
        internal=kind == "internal",
    )


class DetailSession:
    """Serves every recorded ``detail``/``error`` response by URL path, and
    404 for any other ``/events/<id>`` path."""

    def __init__(self, overrides: Mapping[str, FakeResponse] | None = None) -> None:
        self.urls: list[str] = []
        self.table: dict[str, FakeResponse] = {}
        manifest: list[dict[str, Any]] = json.loads((FIX / "manifest.json").read_text())
        for e in manifest:
            if e["kind"] not in ("detail", "error"):
                continue
            parts = urllib.parse.urlsplit(e["url"])
            self.table[parts.netloc + parts.path] = FakeResponse(
                e["status"], (FIX / e["file"]).read_bytes()
            )
        self.table.update(overrides or {})

    def get(
        self, url: str, *, timeout: float, headers: Mapping[str, str]
    ) -> FakeResponse:
        self.urls.append(url)
        parts = urllib.parse.urlsplit(url)
        hit = self.table.get(parts.netloc + parts.path)
        if hit is None:
            return FakeResponse(404, b'{"message":"Event not found"}')
        return hit


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    monkeypatch.delenv(qa.ENV_ENDPOINT, raising=False)
    if request.node.name.startswith("test_live"):
        return

    def _blocked(self: object, url: str, **_: object) -> None:
        raise AssertionError(f"network access in an offline test: {url}")

    monkeypatch.setattr(qa.UrllibSession, "get", _blocked)


@pytest.fixture()
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    calls: list[float] = []
    monkeypatch.setattr(qa, "_sleep", calls.append)
    return calls


# --------------------------------------------------------------------------
# Parsing: shape, origins, magnitudes
# --------------------------------------------------------------------------


def test_dict_or_list_shapes() -> None:
    one = _detail("internal", QNOGQH)
    many = _detail("internal", RUSIBB)
    assert len(one.origins) == 1 and len(one.magnitudes) == 4
    assert len(many.origins) == 6 and len(many.magnitudes) == 13
    single = copy.deepcopy(_body(f"internal_detail_{QNOGQH}.json"))
    single["magnitude"] = single["magnitude"][2]  # the preferred ML_SIL, as a dict
    d = qa.parse_quake_detail(single, event_id=QNOGQH)
    assert len(d.magnitudes) == 1
    assert d.preferred_magnitude is not None
    assert d.preferred_magnitude.magnitude_type == "ML_SIL"


def test_event_fields() -> None:
    d = _detail("internal", RUSIBB)
    assert d.event_id == RUSIBB
    assert d.event_type_code == "earthquake"
    assert d.region == "Langjökull"
    assert d.internal
    assert _detail("internal", QNOGQH).event_type_code is None
    assert _detail("internal", RUTIQI).event_type_code == "not existing"


def test_preferred_origin_and_magnitude() -> None:
    d = _detail("internal", RUSIBB)
    o, m = d.preferred_origin, d.preferred_magnitude
    assert o is not None and m is not None
    assert o.evaluation_mode == "manual" and o.evaluation_status == "reviewed"
    assert o.author == "analyst-1"
    assert m.magnitude_type == "ML_SIL" and m.origin_id == o.origin_id
    assert m.station_count == 19
    assert m.value == pytest.approx(1.6556706083052721, abs=0)
    assert m.uncertainty == pytest.approx(0.3642445934731451, abs=0)
    assert [x.magnitude_type for x in d.magnitudes_of(o.origin_id)] == [
        "ML_SIL",
        "Mpgv_w",
    ]


def test_all_magnitudes_fields() -> None:
    d = _detail("internal", QNOGQH)
    by_type = {m.magnitude_type: m for m in d.magnitudes}
    assert set(by_type) == {"Msil_l", "Msil_lw", "ML_SIL", "Mpgv_w"}
    assert by_type["Msil_lw"].value == 0.25
    assert math.isnan(by_type["Msil_lw"].uncertainty)
    assert by_type["Msil_lw"].station_count is None
    assert by_type["Msil_lw"].evaluation_status == "reviewed"
    assert by_type["Mpgv_w"].station_count == 5
    assert by_type["Mpgv_w"].uncertainty == pytest.approx(0.27874051902016933)
    assert by_type["ML_SIL"].method_id is not None
    assert by_type["ML_SIL"].method_id.endswith("trimmed_mean(25)")
    assert by_type["ML_SIL"].creation_time == pd.Timestamp("2026-02-04T10:09:48.17046Z")


def test_origin_quality_fields() -> None:
    o = _detail("internal", QNOGQH).preferred_origin
    assert o is not None
    assert o.n_phases == 16 and o.n_phases_associated == 16 and o.n_stations == 9
    assert o.azimuthal_gap == 102.0
    assert o.standard_error == pytest.approx(0.19366530406864313)
    assert o.minimum_distance_deg == pytest.approx(0.035073542630817467)
    assert o.maximum_distance_deg == pytest.approx(0.09173080380367646)
    assert o.origin_type == "hypocenter"
    assert o.time == pd.Timestamp("2024-08-22T00:10:04.454Z")
    assert o.origin_uncertainty["preferredDescription"] == "uncertainty ellipse"


# --------------------------------------------------------------------------
# Uncertainty units
# --------------------------------------------------------------------------


def test_method_classification() -> None:
    d = _detail("internal", RUSIBB)
    methods = Counter(o.uncertainty_method for o in d.origins)
    assert methods == {"locsat": 5, "sil": 1}
    assert {o.uncertainty_unit for o in d.origins} == {"km", "deg"}
    assert _detail("internal", QNOGQH).origins[0].uncertainty_method == "sil"
    pub = _detail("public", QNOGQH).origins[0]
    assert pub.uncertainty_method == "unknown" and pub.uncertainty_unit is None


def test_sil_origin_degrees_to_km_with_cos_lat() -> None:
    o = _detail("internal", QNOGQH).origins[0]
    assert o.latitude_uncertainty_raw == 0.002
    assert o.longitude_uncertainty_raw == 0.003
    assert math.isnan(o.confidence_level) and o.sigma_factor == 1.0
    assert o.lat_err_km == pytest.approx(0.002 * KM_PER_DEG)
    lon_km = 0.003 * KM_PER_DEG * math.cos(math.radians(63.888))
    assert o.lon_err_km == pytest.approx(lon_km)
    assert o.h_err_km == pytest.approx(max(0.002 * KM_PER_DEG, lon_km))
    assert o.depth_err_km == 0.6 and o.time_err_s == 0.02
    # SeisComP's originUncertainty has no cos(lat): kept raw, not used
    assert float(o.origin_uncertainty["maxHorizontalUncertainty"]) > 1e3 * o.h_err_km
    assert o.uncertainty_consistent is True


def test_locsat_90_percent_to_one_sigma() -> None:
    o = _detail("internal", RUSIBB).preferred_origin
    assert o is not None and o.uncertainty_method == "locsat"
    assert o.confidence_level == pytest.approx(90.0, abs=1e-5)
    assert o.sigma_factor == pytest.approx(Z90, rel=1e-6)
    assert o.lat_err_km == pytest.approx(0.23684374814149395 / Z90, rel=1e-6)
    assert o.lon_err_km == pytest.approx(0.16370442099496263 / Z90, rel=1e-6)
    assert o.depth_err_km == pytest.approx(1.2296392802114708 / Z90, rel=1e-6)
    assert o.time_err_s > 0
    assert o.h_err_km == max(o.lat_err_km, o.lon_err_km)
    assert o.uncertainty_consistent is True


def test_locsat_axes_match_the_90_percent_ellipse() -> None:
    """The evidence for the 1-D 90 % reading: the 2-D 90 % ellipse axes /
    2.146 equal the per-axis values / 1.645 when the ellipse is near N–S
    (rusibb manual, az 7.9°) or E–W (rutiqi first origin, az 93°)."""
    z2 = math.sqrt(-2.0 * math.log(0.1))  # 2-D 90 %: 2.1460
    rus = _detail("internal", RUSIBB).preferred_origin
    assert rus is not None
    ou = rus.origin_uncertainty
    assert float(ou["maxHorizontalUncertainty"]) / z2 / 1e3 == pytest.approx(
        rus.lat_err_km, rel=0.02
    )
    assert float(ou["minHorizontalUncertainty"]) / z2 / 1e3 == pytest.approx(
        rus.lon_err_km, rel=0.02
    )
    rut = _detail("internal", RUTIQI).origins[0]
    ou = rut.origin_uncertainty
    assert float(ou["maxHorizontalUncertainty"]) / z2 / 1e3 == pytest.approx(
        rut.lon_err_km, rel=0.02
    )
    assert float(ou["minHorizontalUncertainty"]) / z2 / 1e3 == pytest.approx(
        rut.lat_err_km, rel=0.02
    )


def test_every_fixture_origin_passes_the_unit_self_check() -> None:
    for eid in (QNOGQH, RUSIBB, RUTIQI):
        for o in _detail("internal", eid).origins:
            assert o.uncertainty_consistent is True, (eid, o.origin_id)


def test_unit_self_check_flags_a_changed_unit() -> None:
    body = copy.deepcopy(_body(f"internal_detail_{QNOGQH}.json"))
    body["origin"]["latitude"]["uncertainty"] = "0.222"  # km, not degrees
    d = qa.parse_quake_detail(body, event_id=QNOGQH)
    assert d.origins[0].uncertainty_consistent is False
    notes = d.to_catalog().attrs["catalog"]["notes"]
    assert any("units may have changed" in n for n in notes)


def test_operator_assigned_depth_has_no_error() -> None:
    first = _detail("internal", RUSIBB).origins[0]
    assert first.depth_type == "operator assigned" and first.depth_fixed
    assert first.depth_km == 0.0 and math.isnan(first.depth_err_km)


def test_public_detail_is_unknown_method() -> None:
    d = _detail("public", QNOGQH)
    assert d.preferred_origin_id is None and d.preferred_magnitude_id is None
    o = d.preferred_origin
    assert o is not None and o.origin_id is None
    assert math.isnan(o.h_err_km) and math.isnan(o.depth_err_km)
    assert o.latitude_uncertainty_raw == 0.002  # raw kept
    df = d.to_catalog()
    assert df["h_err_km"].isna().all()
    assert df.attrs["catalog"]["endpoint"] == "public"
    assert any("internal=True" in n for n in df.attrs["catalog"]["notes"])


def test_sigma_divisor() -> None:
    assert qa.sigma_divisor(math.nan) == 1.0
    assert qa.sigma_divisor(68.26894921370859) == pytest.approx(1.0)
    assert qa.sigma_divisor(95.0) == pytest.approx(1.959964, rel=1e-6)
    with pytest.raises(ValueError):
        qa.sigma_divisor(100.0)


# --------------------------------------------------------------------------
# to_catalog
# --------------------------------------------------------------------------

_LIST_COLUMNS = (
    "event_id",
    "latitude",
    "longitude",
    "depth_km",
    "magnitude",
    "magnitude_type",
    "evaluation_mode",
    "evaluation_status",
    "event_type",
    "event_type_code",
    "n_phases",
)


@pytest.mark.parametrize(
    ("event_id", "list_file"),
    [
        (QNOGQH, "internal_seiscomp_20240822T00.json"),
        (RUSIBB, "internal_seiscomp_20260910.json"),
        (RUTIQI, "internal_seiscomp_20260910.json"),
    ],
)
def test_to_catalog_reproduces_the_list_row(event_id: str, list_file: str) -> None:
    """The preferred origin and magnitude are the ones the T10 list shows."""
    listed = qa.features_to_frame(_body(list_file)["features"], system="seiscomp")
    row = listed[listed["event_id"] == event_id]
    assert len(row) == 1
    got = _detail("internal", event_id).to_catalog()
    assert validate(got).ok and len(got) == 1
    assert abs(got.index[0] - row.index[0]) < pd.Timedelta(microseconds=1)
    for col in _LIST_COLUMNS:
        a, b = got[col].iloc[0], row[col].iloc[0]
        if isinstance(b, float):
            assert a == pytest.approx(b, abs=1e-6, nan_ok=True), col
        else:
            assert (pd.isna(a) and pd.isna(b)) or a == b, col


def test_to_catalog_detail_columns() -> None:
    df = _detail("internal", QNOGQH).to_catalog()
    r = df.iloc[0]
    assert df["source"].iloc[0] == "quakes-seiscomp"
    assert r["h_err_km"] == pytest.approx(0.002 * KM_PER_DEG)
    assert r["depth_err_km"] == 0.6
    assert r["azimuthal_gap"] == 102.0
    assert r["mag_mlw"] == 0.25 and r["mag_ml"] == -0.5  # Msil_lw / Msil_l
    assert math.isnan(r["mag_auto"])
    assert pd.isna(r["updated_time"]) and pd.isna(r["sil_path"])
    assert pd.isna(r["event_type"])
    prov = df.attrs["catalog"]
    assert prov["detail"] is True and prov["uncertainty_method"] == "sil"
    assert prov["sigma_factor"] == 1.0 and prov["confidence_level"] is None
    assert "region" not in df.columns
    assert _detail("internal", QNOGQH).to_catalog(extras=True)["region"].iloc[0] == (
        "Reykjanes"
    )


def test_to_catalog_scale_magnitudes_from_other_origins() -> None:
    """``Msil_auto`` sits on the SIL automatic origin, not the preferred one."""
    d = _detail("internal", RUSIBB)
    assert d.scale_magnitudes() == {
        "mag_mlw": pytest.approx(math.nan, nan_ok=True),
        "mag_ml": pytest.approx(math.nan, nan_ok=True),
        "mag_auto": 1.5,
    }
    df = d.to_catalog()
    assert df["mag_auto"].iloc[0] == 1.5
    assert df["magnitude_type"].iloc[0] == "ML_SIL"
    assert df.attrs["catalog"]["uncertainty_method"] == "locsat"
    assert df.attrs["catalog"]["confidence_level"] == pytest.approx(90.0, abs=1e-5)


def test_to_catalog_rejects_detail_only_magnitude_type() -> None:
    body = copy.deepcopy(_body(f"internal_detail_{QNOGQH}.json"))
    body["preferredMagnitudeID"] = body["magnitude"][1]["publicID"]  # Msil_lw
    with pytest.raises(SchemaError):
        qa.parse_quake_detail(body, event_id=QNOGQH).to_catalog()


def test_parse_errors() -> None:
    body = _body(f"internal_detail_{QNOGQH}.json")
    with pytest.raises(qa.QuakesApiError, match="asked for"):
        qa.parse_quake_detail(body, event_id=RUSIBB)
    bad = copy.deepcopy(body)
    bad["preferredOriginID"] = "smi:nowhere"
    with pytest.raises(qa.QuakesApiError, match="preferredOriginID"):
        qa.parse_quake_detail(bad, event_id=QNOGQH)
    bad = copy.deepcopy(body)
    bad["preferredMagnitudeID"] = "smi:nowhere"
    with pytest.raises(qa.QuakesApiError, match="preferredMagnitudeID"):
        qa.parse_quake_detail(bad, event_id=QNOGQH)
    with pytest.raises(qa.QuakesApiError, match="not a QuakeML"):
        qa.parse_quake_detail({"message": "x"})
    bad = copy.deepcopy(body)
    bad["origin"] = "x"
    with pytest.raises(qa.QuakesApiError, match="origin"):
        qa.parse_quake_detail(bad, event_id=QNOGQH)
    pub = _body(f"public_detail_{QNOGQH}.json")
    with pytest.raises(qa.QuakesApiError, match="without an event id"):
        qa.parse_quake_detail(pub)


# --------------------------------------------------------------------------
# read_quake_detail (fake transport)
# --------------------------------------------------------------------------


def test_read_quake_detail_internal() -> None:
    s = DetailSession()
    d = qa.read_quake_detail(RUSIBB, session=s, policy=NO_PAUSE)
    assert s.urls == [f"{qa.INTERNAL_BASE}/events/{RUSIBB}?"]
    assert d.internal and d.url == f"{qa.INTERNAL_BASE}/events/{RUSIBB}"
    assert d.to_catalog().attrs["catalog"]["location"] == d.url


def test_read_quake_detail_public_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(qa.ENV_ENDPOINT, "public")
    s = DetailSession()
    d = qa.read_quake_detail(QNOGQH, session=s, policy=NO_PAUSE)
    assert s.urls[0].startswith(qa.PUBLIC_BASE + "/events/")
    assert not d.internal and d.event_id == QNOGQH


def test_read_quake_detail_not_found() -> None:
    s = DetailSession()
    with pytest.raises(qa.QuakeDetailNotFound) as exc:
        qa.read_quake_detail("IMO2099zzzzzz", session=s, policy=NO_PAUSE)
    assert isinstance(exc.value, LookupError) and exc.value.status == 404
    assert "Event not found" in str(exc.value)


def test_sil_ids_refused_before_any_request() -> None:
    s = DetailSession()
    for kwargs in ({"event_id": "SIL1352381"}, {"event_id": QNOGQH, "system": "sil"}):
        with pytest.raises(ValueError, match="read_skjalftalisa_detail"):
            qa.read_quake_detail(session=s, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not a SeisComP event id"):
        qa.read_quake_detail("a/b", session=s)
    with pytest.raises(ValueError, match="system"):
        qa.read_quake_detail(QNOGQH, system="bogus", session=s)  # type: ignore[arg-type]
    assert s.urls == []


def test_the_recorded_sil_404_is_not_found() -> None:
    """The recorded 404 body for /events/SIL1352381, served for an IMO id."""
    body = (FIX / "error_sil_detail_404.json").read_bytes()
    s = DetailSession(
        {"vi-api.vedur.is/quakes/internal/events/IMO1": FakeResponse(404, body)}
    )
    with pytest.raises(qa.QuakeDetailNotFound, match="Event not found"):
        qa.read_quake_detail("IMO1", session=s, policy=NO_PAUSE)


def test_read_quake_detail_retries_5xx(sleeps: list[float]) -> None:
    good = (FIX / f"internal_detail_{QNOGQH}.json").read_bytes()
    answers = [FakeResponse(503, b"{}"), FakeResponse(200, good)]

    class Flaky:
        urls: list[str] = []

        def get(
            self, url: str, *, timeout: float, headers: Mapping[str, str]
        ) -> FakeResponse:
            self.urls.append(url)
            return answers.pop(0)

    policy = qa.FetchPolicy(pause_s=0.5, backoff_s=1.0)
    d = qa.read_quake_detail(QNOGQH, session=Flaky(), policy=policy)
    assert d.event_id == QNOGQH
    assert sleeps == [1.0, 0.5]  # backoff, then the pause before the resend


def test_read_quake_detail_unreachable() -> None:
    class Down:
        def get(
            self, url: str, *, timeout: float, headers: Mapping[str, str]
        ) -> FakeResponse:
            raise OSError("no route")

    with pytest.raises(qa.QuakesApiUnreachable, match="internal=False"):
        qa.read_quake_detail(QNOGQH, session=Down(), policy=NO_PAUSE)


def test_detail_not_json() -> None:
    s = DetailSession(
        {"vi-api.vedur.is/quakes/internal/events/IMO1": FakeResponse(200, b"<html>")}
    )
    with pytest.raises(qa.QuakesApiError, match="not JSON"):
        qa.read_quake_detail("IMO1", session=s, policy=NO_PAUSE)


# --------------------------------------------------------------------------
# read_quake_details (sample)
# --------------------------------------------------------------------------


def test_sample_pauses_between_every_request(sleeps: list[float]) -> None:
    s = DetailSession()
    policy = qa.FetchPolicy(pause_s=0.7)
    sample = qa.read_quake_details(
        [RUSIBB, QNOGQH, RUTIQI, RUSIBB], max_events=3, session=s, policy=policy
    )
    assert [d.event_id for d in sample.details] == [RUSIBB, QNOGQH, RUTIQI]
    assert sleeps == [0.7, 0.7]  # N ids → N−1 pauses (duplicates fetched once)
    assert sample.counts["requests"] == 3 and sample.counts["detail_requests"] == 3
    assert sample.missing == ()


def test_sample_to_catalog() -> None:
    sample = qa.read_quake_details(
        pd.Series([RUTIQI, QNOGQH, RUSIBB]),
        max_events=5,
        session=DetailSession(),
        policy=NO_PAUSE,
    )
    df = sample.to_catalog()
    assert validate(df, strict=True).ok
    assert list(df["event_id"]) == [QNOGQH, RUSIBB, RUTIQI]  # sorted by time
    assert df["h_err_km"].notna().all()
    prov = df.attrs["catalog"]
    assert prov["location"] == f"{qa.INTERNAL_BASE}/events"
    assert prov["sample"]["events"] == 3
    assert prov["sample"]["uncertainty_methods"] == {"sil": 1, "locsat": 2}
    assert "preferred_origin_id" not in prov


def test_sample_empty_to_catalog() -> None:
    sample = qa.read_quake_details([], max_events=1, session=DetailSession())
    df = sample.to_catalog()
    assert len(df) == 0 and validate(df).ok


def test_sample_missing(sleeps: list[float]) -> None:
    s = DetailSession()
    with pytest.raises(qa.QuakeDetailNotFound):
        qa.read_quake_details(
            [QNOGQH, "IMO2099zzzzzz"], max_events=2, session=s, policy=NO_PAUSE
        )
    sample = qa.read_quake_details(
        ["IMO2099zzzzzz", QNOGQH],
        max_events=2,
        session=s,
        policy=NO_PAUSE,
        missing="skip",
    )
    assert sample.missing == ("IMO2099zzzzzz",)
    assert [d.event_id for d in sample.details] == [QNOGQH]
    assert sample.counts["missing"] == 1


def test_sample_limits_checked_before_any_request() -> None:
    s = DetailSession()
    with pytest.raises(ValueError, match="max_events"):
        qa.read_quake_details([QNOGQH, RUSIBB], max_events=1, session=s)
    with pytest.raises(ValueError, match=str(qa.MAX_DETAIL_SAMPLE)):
        qa.read_quake_details([QNOGQH], max_events=qa.MAX_DETAIL_SAMPLE + 1, session=s)
    with pytest.raises(ValueError):
        qa.read_quake_details([QNOGQH], max_events=0, session=s)
    with pytest.raises(TypeError):
        qa.read_quake_details([QNOGQH], max_events=2.0, session=s)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        qa.read_quake_details(QNOGQH, max_events=20, session=s)
    with pytest.raises(ValueError, match="read_skjalftalisa_detail"):
        qa.read_quake_details([QNOGQH, "SIL1352381"], max_events=2, session=s)
    with pytest.raises(ValueError, match="missing"):
        qa.read_quake_details(
            [QNOGQH], max_events=2, session=s, missing="ignore"  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        qa.read_quake_details([QNOGQH], session=s)  # type: ignore[call-arg]
    assert s.urls == []


# --------------------------------------------------------------------------
# Live (opt-in)
# --------------------------------------------------------------------------


def test_live_internal_detail() -> None:
    """One detail request for the back-imported event on the internal host."""
    if os.environ.get("GPS_LIVE_API") != "1":
        pytest.skip("live Quakes API test: set GPS_LIVE_API=1")
    sample = qa.read_quake_details([QNOGQH], max_events=1)
    assert sample.counts["requests"] == 1
    d = sample.details[0]
    assert d.event_id == QNOGQH and len(d.origins) >= 1
    assert all(o.uncertainty_consistent is not False for o in d.origins)
    df = d.to_catalog()
    assert validate(df, strict=True).ok
    assert df["h_err_km"].iloc[0] == pytest.approx(0.002 * KM_PER_DEG)
    assert df["mag_mlw"].iloc[0] == 0.25


# --------------------------------------------------------------------------
# Cross-checks against the other sources' fixtures
# --------------------------------------------------------------------------


def test_sil_origin_matches_the_sil_files() -> None:
    """IMO2024qnogqh is lib.mag row 1 of the T0 day: ``Msil_lw`` = Mlw,
    ``Msil_l`` = ML, the degree errors, depth error, gap and P+S phases."""
    from geo_dataread.seismic.sil_files import read_lib_mag

    lib = read_lib_mag(
        Path(__file__).parent / "fixtures" / "sil" / "2024" / "aug" / "22" / "lib.mag"
    ).frame
    row = lib[lib["time"] == pd.Timestamp("2024-08-22T00:10:04.454Z")].iloc[0]
    d = _detail("internal", QNOGQH)
    o = d.origins[0]
    scales = d.scale_magnitudes()
    assert scales["mag_mlw"] == row["mlw"] and scales["mag_ml"] == row["ml"]
    assert o.latitude_uncertainty_raw == row["d_latitude"]
    assert o.longitude_uncertainty_raw == row["d_longitude"]
    assert o.depth_err_km == row["d_depth"] and o.time_err_s == row["d_time"]
    assert o.azimuthal_gap == row["max_azim_gap"]
    assert o.n_phases == row["num_of_p_pha"] + row["num_of_s_pha"]


def test_msil_auto_matches_the_sil_list() -> None:
    """rusibb's ``Msil_auto`` 1.5 is SIL1453671's ``aut.mag`` (same event)."""
    sil = qa.features_to_frame(
        _body("internal_sil_20260910.json")["features"], system="sil"
    )
    aut = sil.loc[sil["event_id"] == "SIL1453671", "mag_auto"].iloc[0]
    assert _detail("internal", RUSIBB).scale_magnitudes()["mag_auto"] == aut
