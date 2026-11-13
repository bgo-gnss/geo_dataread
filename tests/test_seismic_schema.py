"""Tests for the common seismicity schema (PLAN-seismicity.md T2)."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from geo_dataread.seismic import (
    EVENT_COLUMNS,
    EVENT_TYPES,
    INDEX_NAME,
    MECHANISM_COLUMNS,
    SCHEMA_VERSION,
    SIL_EVENT_TYPES,
    SOURCES,
    CoercionError,
    SchemaError,
    auxiliary_plane,
    catalog_provenance,
    coerce_catalog,
    coerce_mechanisms,
    column_names,
    column_spec,
    cumulative_moment,
    empty_catalog,
    empty_mechanisms,
    moment_magnitude,
    plane_mismatch_deg,
    sil_event_type,
    validate,
    validate_mechanisms,
    with_provenance,
)
from geo_dataread.seismic import schema as sch

# -------------------------------------------------------------- fixtures


def _raw_events() -> pd.DataFrame:
    """What a reader might build before coercion: strings, Z times, gaps."""
    return pd.DataFrame(
        {
            "time": [
                "2024-08-22T00:19:11.927Z",
                "2024-08-22T00:10:04.454Z",
                "2024-08-22T00:10:37.669Z",
            ],
            "event_id": ["/2024/aug/22/00:15:00/19:13:327", "e1", "e2"],
            "latitude": [63.87, 63.88752, 64.04466],
            "longitude": [-22.4074, -22.37545, -21.16406],
            "depth_km": [5.344, 2.111, 2.998],
            "magnitude": [0.69, 0.25, None],
            "magnitude_type": ["Mlw", "Mlw", None],
            "moment_nm": [4.93e10, 1.77e10, None],
            "evaluation_mode": ["manual", "manual", "automatic"],
            "event_type": ["earthquake", "earthquake", None],
            "event_type_code": ["qu", "qu", None],
            "n_phases": [12, 9, None],
        }
    )


@pytest.fixture
def events() -> pd.DataFrame:
    return coerce_catalog(_raw_events(), source="sil-files")


def _raw_mechs() -> pd.DataFrame:
    # real events.fps rows from 2024-08-22 (strike/dip/slip a and b)
    return pd.DataFrame(
        {
            "time": ["2024-08-22T00:10:04.454Z", "2024-08-22T00:19:11.927Z"],
            "event_id": ["e1", "e3"],
            "method": ["sil_fps", "sil_fps"],
            "strike1": [341.0, 112.0],
            "dip1": [74.0, 41.0],
            "rake1": [170.0, -39.0],
            "strike2": [74.0, 234.0],
            "dip2": [81.0, 65.0],
            "rake2": [16.0, -124.0],
            "n_polarities": [0, 0],
            "n_amplitudes": [24, 21],
        }
    )


@pytest.fixture
def mechs() -> pd.DataFrame:
    return coerce_mechanisms(_raw_mechs(), source="sil-files")


# -------------------------------------------------------------- tables


def test_column_tables_are_unique_and_complete() -> None:
    for table in (EVENT_COLUMNS, MECHANISM_COLUMNS):
        names = [c.name for c in table]
        assert len(names) == len(set(names))
        assert all(c.meaning for c in table)
        for c in table:
            if c.kind == "category":
                assert c.allowed
    assert column_names()[:2] == ("source", "event_id")
    assert column_spec("depth_km").unit == "km"
    with pytest.raises(KeyError):
        column_spec("nope")
    assert isinstance(SCHEMA_VERSION, int)


def test_plan_columns_present() -> None:
    plan = {
        "source", "event_id", "sil_path", "latitude", "longitude", "depth_km",
        "h_err_km", "depth_err_km", "magnitude", "magnitude_type", "mag_mlw",
        "mag_ml", "mag_auto", "moment_nm", "evaluation_mode",
        "evaluation_status", "event_type", "quality", "n_phases",
        "azimuthal_gap", "updated_time",
    }  # fmt: skip
    assert plan <= set(column_names("events"))
    mplan = {"source", "event_id", "method"} | {
        f"{a}{n}" for a in ("strike", "dip", "rake") for n in (1, 2)
    }
    assert mplan <= set(column_names("mechanisms"))


def test_sources_include_q11_names() -> None:
    for name in ("quakes-seiscomp", "quakes-sil", "skjalftalisa", "sil-files"):
        assert name in SOURCES


# -------------------------------------------------------------- empty frames


@pytest.mark.parametrize(
    ("make", "check", "table"),
    [
        (empty_catalog, validate, EVENT_COLUMNS),
        (empty_mechanisms, validate_mechanisms, MECHANISM_COLUMNS),
    ],
)
def test_empty_frames_valid_and_typed(make, check, table) -> None:  # type: ignore[no-untyped-def]
    df = make()
    assert len(df) == 0
    assert list(df.columns) == [c.name for c in table]
    assert df.index.name == INDEX_NAME
    assert df.index.dtype == sch.TIME_DTYPE
    for c in table:
        assert sch._dtype_ok(df[c.name].dtype, c), c.name
    rep = check(df)
    assert rep.ok and not rep.issues, rep.summary()


def test_empty_frames_concat_keeps_dtypes(events: pd.DataFrame) -> None:
    both = pd.concat([empty_catalog(), events])
    assert validate(both).ok
    assert isinstance(both["source"].dtype, pd.CategoricalDtype)


def test_coerce_empty_rangeindex_frame() -> None:
    df = coerce_catalog(pd.DataFrame())
    assert len(df) == 0 and validate(df).ok


# -------------------------------------------------------------- coercion


def test_coerce_makes_valid_frame(events: pd.DataFrame) -> None:
    rep = validate(events, strict=True)
    assert rep.ok and rep.n_rows == 3
    assert list(events.columns) == list(column_names())
    assert events.index.is_monotonic_increasing
    assert events.index.dtype == sch.TIME_DTYPE
    # missing columns present and NA
    assert events["sil_path"].isna().all()
    assert events["updated_time"].isna().all()
    assert events["n_phases"].dtype == pd.Int64Dtype()
    assert (events["source"] == "sil-files").all()


def test_coerce_does_not_mutate_input() -> None:
    raw = _raw_events()
    before = raw.copy()
    raw.attrs["x"] = 1
    out = coerce_catalog(raw, source="sil-files")
    pd.testing.assert_frame_equal(raw, before)
    assert out.attrs == {"x": 1}


def test_coerce_keeps_extra_columns_and_warns() -> None:
    raw = _raw_events().assign(MLW_old=1.0)
    out = coerce_catalog(raw, source="sil-files")
    assert list(out.columns)[-1] == "MLW_old"
    rep = validate(out)
    assert rep.ok
    assert rep.kinds() == {"extra_column"}
    assert rep.warnings[0].examples == ("MLW_old",)


def test_coerce_keeps_unknown_category_for_validate() -> None:
    raw = _raw_events()
    raw["evaluation_mode"] = ["manual", "bogus", "automatic"]
    out = coerce_catalog(raw, source="sil-files")
    # not silently NaN'd
    assert "bogus" in set(out["evaluation_mode"].astype(str))
    rep = validate(out)
    assert not rep.ok
    (issue,) = rep.errors
    assert issue.kind == "unknown_category" and issue.column == "evaluation_mode"
    assert issue.count == 1 and issue.examples == ("bogus",)


def test_coerce_rejects_bad_source_argument() -> None:
    with pytest.raises(CoercionError):
        coerce_catalog(_raw_events(), source="bogus")


def test_bogus_source_column_is_caught() -> None:
    raw = _raw_events().assign(source="bogus")
    out = coerce_catalog(raw)
    rep = validate(out)
    assert "unknown_category" in rep.kinds()
    assert (out["source"].astype(str) == "bogus").all()


def test_coerce_conflicting_source_raises() -> None:
    raw = _raw_events().assign(source="quakes-sil")
    with pytest.raises(CoercionError, match="already holds"):
        coerce_catalog(raw, source="sil-files")


def test_coerce_non_numeric_raises() -> None:
    raw = _raw_events()
    raw["latitude"] = ["63.9", "abc", "64.0"]
    with pytest.raises(CoercionError, match="latitude"):
        coerce_catalog(raw, source="sil-files")


def test_coerce_naive_times_need_assume_utc() -> None:
    raw = _raw_events()
    raw["time"] = raw["time"].str.rstrip("Z")
    with pytest.raises(CoercionError, match="assume_utc"):
        coerce_catalog(raw, source="skjalftalisa")
    out = coerce_catalog(raw, source="skjalftalisa", assume_utc=True)
    assert out.index[0] == pd.Timestamp("2024-08-22T00:10:04.454Z")


def test_coerce_converts_other_zone_to_utc() -> None:
    raw = _raw_events()
    raw["time"] = [
        "2024-08-22T02:19:11+02:00",
        "2024-08-22T00:10:04Z",
        "2024-08-22T00:10:37Z",
    ]
    out = coerce_catalog(raw, source="sil-files")
    assert str(out.index.tz) == "UTC"
    assert out.index[-1] == pd.Timestamp("2024-08-22T00:19:11Z")


def test_coerce_mixed_iso_precision() -> None:
    raw = _raw_events()
    raw["time"] = [
        "2024-08-22T00:19:11Z",
        "2024-08-22T00:10:04.454Z",
        "2024-08-22T00:10:37.6Z",
    ]
    assert validate(coerce_catalog(raw, source="sil-files")).ok


def test_coerce_keeps_duplicate_times() -> None:
    raw = _raw_events()
    raw["time"] = ["2024-08-22T00:10:04Z"] * 3
    out = coerce_catalog(raw, source="sil-files")
    assert validate(out).ok
    # stable sort: original order kept for equal times
    assert list(out["event_id"]) == list(raw["event_id"])


def test_coerce_index_already_datetime() -> None:
    raw = _raw_events()
    raw.index = pd.to_datetime(raw.pop("time"), utc=True).dt.as_unit("us")
    out = coerce_catalog(raw, source="sil-files")
    assert out.index.dtype == sch.TIME_DTYPE and validate(out).ok


# -------------------------------------------------------------- validate failures


def test_missing_column(events: pd.DataFrame) -> None:
    rep = validate(events.drop(columns=["quality"]))
    assert not rep.ok
    assert [i.column for i in rep.errors if i.kind == "missing_column"] == ["quality"]


def test_wrong_dtype(events: pd.DataFrame) -> None:
    bad = events.assign(latitude=events["latitude"].astype(object))
    rep = validate(bad)
    assert {(i.kind, i.column) for i in rep.errors} == {("wrong_dtype", "latitude")}


def test_null_in_required_column(events: pd.DataFrame) -> None:
    bad = events.copy()
    bad.iloc[1, bad.columns.get_loc("latitude")] = np.nan
    rep = validate(bad)
    (issue,) = rep.errors
    assert issue.kind == "null_values" and issue.count == 1
    assert issue.examples[0].startswith("row 1 ")


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("latitude", 91.0),
        ("longitude", -181.0),
        ("depth_km", -6.0),
        ("magnitude", 11.0),
        ("quality", 100.5),
        ("azimuthal_gap", 361.0),
        ("moment_nm", -1.0),
        ("h_err_km", -0.1),
    ],
)
def test_out_of_range(events: pd.DataFrame, column: str, value: float) -> None:
    bad = events.copy()
    bad.iloc[0, bad.columns.get_loc(column)] = value
    rep = validate(bad)
    (issue,) = rep.errors
    assert issue.kind == "out_of_range" and issue.column == column and issue.count == 1


def test_depth_slightly_above_datum_allowed(events: pd.DataFrame) -> None:
    ok = events.copy()
    ok.iloc[0, ok.columns.get_loc("depth_km")] = -1.5
    assert validate(ok).ok


def test_int_range(events: pd.DataFrame) -> None:
    bad = events.copy()
    bad["n_phases"] = pd.array([-1, 3, None], dtype="Int64")
    assert "out_of_range" in validate(bad).kinds()


def test_tz_naive_index_rejected(events: pd.DataFrame) -> None:
    bad = events.copy()
    bad.index = bad.index.tz_localize(None)
    bad.index.name = INDEX_NAME
    rep = validate(bad)
    assert "index_not_utc" in rep.kinds()
    with pytest.raises(SchemaError) as exc:
        validate(bad, strict=True)
    assert exc.value.report is not None
    assert "tz-naive" in str(exc.value)


def test_non_utc_index_rejected(events: pd.DataFrame) -> None:
    bad = events.copy()
    bad.index = bad.index.tz_convert("Atlantic/Reykjavik").rename(INDEX_NAME)
    assert "index_not_utc" in validate(bad).kinds()


def test_index_unit_checked(events: pd.DataFrame) -> None:
    bad = events.copy()
    bad.index = bad.index.as_unit("us")
    assert validate(bad).kinds() == {"index_unit"}


def test_unsorted_index(events: pd.DataFrame) -> None:
    bad = events.iloc[[2, 0, 1]]
    rep = validate(bad)
    (issue,) = rep.errors
    assert issue.kind == "index_unsorted" and issue.count == 1


def test_index_not_datetime(events: pd.DataFrame) -> None:
    bad = events.reset_index(drop=True)
    assert "index_not_datetime" in validate(bad).kinds()


def test_index_name_and_nat(events: pd.DataFrame) -> None:
    bad = events.rename_axis("t")
    assert validate(bad).kinds() == {"index_name"}
    nat = events.copy()
    nat.index = pd.DatetimeIndex(
        [pd.NaT, *events.index[1:]], dtype=sch.TIME_DTYPE, name=INDEX_NAME
    )
    assert "index_nat" in validate(nat).kinds()


def test_report_summary_and_raise(events: pd.DataFrame) -> None:
    bad = events.drop(columns=["quality"]).iloc[[1, 0, 2]]
    rep = validate(bad)
    text = rep.summary()
    assert "missing_column [quality]" in text and "index_unsorted" in text
    with pytest.raises(SchemaError):
        rep.raise_for_errors()
    assert (
        validate(events)
        .summary()
        .startswith(f"events schema v{SCHEMA_VERSION}: 3 rows")
    )


# -------------------------------------------------------------- mechanisms


def test_mechanisms_valid(mechs: pd.DataFrame) -> None:
    rep = validate_mechanisms(mechs, strict=True)
    assert rep.ok and not rep.issues
    assert mechs["stress_drop_mpa"].isna().all()


def test_mechanisms_one_plane_only(mechs: pd.DataFrame) -> None:
    one = mechs.assign(strike2=np.nan, dip2=np.nan, rake2=np.nan)
    assert validate_mechanisms(one).ok


def test_mechanisms_inconsistent_planes(mechs: pd.DataFrame) -> None:
    bad = mechs.copy()
    bad.iloc[0, bad.columns.get_loc("strike2")] = 100.0
    rep = validate_mechanisms(bad)
    (issue,) = rep.errors
    assert issue.kind == "planes_inconsistent" and issue.count == 1
    # a looser tolerance accepts it
    assert validate_mechanisms(bad, plane_tol_deg=40.0).ok


def test_mechanisms_ranges_and_sentinels(mechs: pd.DataFrame) -> None:
    # SIL writes -9/-999 for "no solution"; a reader must map them to NA
    bad = mechs.copy()
    bad.iloc[0, bad.columns.get_loc("dip1")] = -9.0
    bad.iloc[0, bad.columns.get_loc("rake1")] = -999.0
    rep = validate_mechanisms(bad)
    assert {(i.kind, i.column) for i in rep.errors if i.kind == "out_of_range"} == {
        ("out_of_range", "dip1"),
        ("out_of_range", "rake1"),
    }


def test_mechanisms_unknown_method_and_missing_plane1(mechs: pd.DataFrame) -> None:
    raw = _raw_mechs()
    raw["method"] = ["sil_fps", "guess"]
    raw.loc[1, "strike1"] = np.nan
    rep = validate_mechanisms(coerce_mechanisms(raw, source="sil-files"))
    assert {"unknown_category", "null_values"} <= rep.kinds()


def test_mechanisms_duplicate_key_warns(mechs: pd.DataFrame) -> None:
    dup = pd.concat([mechs, mechs]).sort_index(kind="stable")
    rep = validate_mechanisms(dup)
    assert rep.ok
    assert rep.kinds() == {"duplicate_key"} and rep.warnings[0].count == 4


# -------------------------------------------------------------- attrs


def test_provenance(events: pd.DataFrame) -> None:
    prov = catalog_provenance(
        source="quakes-seiscomp",
        location="https://vi-api.vedur.is/quakes/internal/events",
        start=datetime(2024, 8, 22),
        end="2024-08-23T00:00:00+00:00",
        filters={"depth_min": 0, "depth_max": 50, "size_min": -3},
        fetched_at=datetime(2026, 11, 13, 12, 0, tzinfo=UTC),
        extra={"system": "seiscomp"},
    )
    assert prov["schema_version"] == SCHEMA_VERSION
    assert prov["frame"] == "events"
    assert prov["window"] == {
        "start": "2024-08-22T00:00:00Z",
        "end": "2024-08-23T00:00:00Z",
    }
    assert prov["fetched_at"] == "2026-11-13T12:00:00Z"
    assert prov["filters"]["depth_max"] == 50 and prov["system"] == "seiscomp"
    assert "geo_dataread" in prov["software"]
    json.dumps(prov)  # JSON-safe
    out = with_provenance(events, prov)
    assert out is events and events.attrs["catalog"]["source"] == "quakes-seiscomp"
    # coercion carries attrs
    assert coerce_catalog(events).attrs["catalog"] == prov


def test_provenance_defaults_and_bad_source() -> None:
    prov = catalog_provenance(
        source="sil-files", location="/mnt_data/sildata", frame="mechanisms"
    )
    assert prov["window"] == {"start": None, "end": None}
    assert prov["fetched_at"].endswith("Z") and prov["frame"] == "mechanisms"
    with pytest.raises(ValueError):
        catalog_provenance(source="sil_api", location="x")


# -------------------------------------------------------------- event-type map


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("qu", "earthquake"),
        ("ex", "explosion"),
        (" qu ", "earthquake"),
        ("QU", "earthquake"),
        ("sm", "other"),
        ("un", "other"),
        ("ql", "other"),
        ("sf", "other"),
        ("te", "other"),
        ("sc", "other"),
        ("zz", "other"),
        ("", None),
        (None, None),
        (float("nan"), None),
    ],
)
def test_sil_event_type(code: object, expected: str | None) -> None:
    assert sil_event_type(code) == expected


def test_sil_event_type_map_targets_vocabulary() -> None:
    assert set(SIL_EVENT_TYPES.values()) <= set(EVENT_TYPES)
    for code in ("qu", "ex", "sm", "un", "ql", "sf", "te", "sc"):
        assert code in SIL_EVENT_TYPES


# -------------------------------------------------------------- derived values


def test_moment_magnitude_known_values() -> None:
    # Mw 0 ↔ M0 = 10^(1.5·6.0333) = 10^9.05 N·m; Mw 6 ↔ 10^18.05
    assert float(moment_magnitude(10**9.05)) == pytest.approx(0.0, abs=1e-12)
    assert float(moment_magnitude(10**18.05)) == pytest.approx(6.0, abs=1e-12)
    # 1e7 dyn·cm form: Mw = 2/3 log10(M0·1e7) − 10.7
    m0 = 2.78e10
    assert float(moment_magnitude(m0)) == pytest.approx(
        2 / 3 * math.log10(m0 * 1e7) - 10.7
    )
    # the plan's check: 1.77e10 N·m (Mlw 0.25 row) → ~0.79; legacy 6.03 is 0.0033 above
    assert float(moment_magnitude(1.77e10)) == pytest.approx(
        2 / 3 * math.log10(1.77e10) - 6.03 - 0.00333333, abs=1e-6
    )


def test_moment_magnitude_array_and_invalid() -> None:
    with np.errstate(all="raise"):
        out = moment_magnitude([1e15, 0.0, -1.0, np.nan])
    assert out.shape == (4,)
    assert out[0] == pytest.approx(2 / 3 * 15 - 6.0333333, abs=1e-6)
    assert np.isnan(out[1:]).all()


def test_cumulative_moment(events: pd.DataFrame) -> None:
    cm = cumulative_moment(events)
    assert cm.name == "cumulative_moment_nm"
    assert cm.index.equals(events.index)
    # sorted order: e1 (1.77e10), e2 (NA → +0), path id (4.93e10)
    assert list(cm) == pytest.approx([1.77e10, 1.77e10, 1.77e10 + 4.93e10])
    shuffled = events.iloc[[2, 0, 1]]
    assert list(cumulative_moment(shuffled)) == pytest.approx(list(cm))
    assert len(cumulative_moment(empty_catalog())) == 0


# -------------------------------------------------------------- double couple


def _close_plane(
    got: tuple[np.ndarray, np.ndarray, np.ndarray], want: tuple[float, float, float]
) -> None:
    assert [float(x) for x in got] == pytest.approx(list(want), abs=1e-9)


@pytest.mark.parametrize(
    ("plane", "aux"),
    [
        ((0.0, 90.0, 0.0), (90.0, 90.0, 180.0)),  # vertical strike-slip
        ((90.0, 90.0, 180.0), (0.0, 90.0, 0.0)),
        ((0.0, 45.0, 90.0), (180.0, 45.0, 90.0)),  # 45° thrust
        ((0.0, 45.0, -90.0), (180.0, 45.0, -90.0)),  # 45° normal
        ((360.0, 45.0, 90.0), (180.0, 45.0, 90.0)),  # strike wrap 360 ≡ 0
        ((-20.0, 45.0, 90.0), (160.0, 45.0, 90.0)),  # negative strike
        ((350.0, 45.0, 90.0), (170.0, 45.0, 90.0)),
        ((0.0, 0.0, 90.0), (0.0, 90.0, -90.0)),  # horizontal plane
    ],
)
def test_auxiliary_plane_textbook(
    plane: tuple[float, float, float], aux: tuple[float, float, float]
) -> None:
    _close_plane(auxiliary_plane(*plane), aux)
    assert float(plane_mismatch_deg(*plane, *aux)) == pytest.approx(0.0, abs=1e-6)


def test_auxiliary_plane_real_sil_row() -> None:
    # events.fps 2024-08-22 row 1: 341/74/170 and 74/81/16 (whole degrees)
    s, d, r = (float(x) for x in auxiliary_plane(341.0, 74.0, 170.0))
    assert (s, d, r) == pytest.approx((74.0, 81.0, 16.0), abs=1.0)
    assert float(plane_mismatch_deg(341, 74, 170, 74, 81, 16)) < 2.0


def test_auxiliary_plane_round_trip_random() -> None:
    rng = np.random.default_rng(1)
    s = rng.uniform(0, 360, 2000)
    d = rng.uniform(5, 85, 2000)
    r = rng.uniform(-179, 179, 2000)
    s2, d2, r2 = auxiliary_plane(s, d, r)
    assert ((s2 >= 0) & (s2 < 360)).all()
    assert ((d2 >= 0) & (d2 <= 90)).all()
    assert ((r2 > -180) & (r2 <= 180)).all()
    s3, d3, r3 = auxiliary_plane(s2, d2, r2)
    wrap = lambda x: (x + 180.0) % 360.0 - 180.0  # noqa: E731
    assert np.abs(wrap(s3 - s)).max() < 1e-7
    assert np.abs(d3 - d).max() < 1e-7
    assert np.abs(wrap(r3 - r)).max() < 1e-7
    assert np.nanmax(plane_mismatch_deg(s, d, r, s2, d2, r2)) < 1e-5


def test_plane_mismatch_vertical_descriptions_equal() -> None:
    # (φ, 90, λ) and (φ+180, 90, −λ) are the same plane and slip
    assert float(plane_mismatch_deg(0, 90, 0, 270, 90, 180)) == pytest.approx(
        0.0, abs=1e-6
    )


def test_plane_mismatch_detects_reversed_slip() -> None:
    # same planes, opposite sense of motion: not the same double couple
    assert float(plane_mismatch_deg(0, 45, 90, 180, 45, -90)) == pytest.approx(90.0)


def test_auxiliary_plane_nan() -> None:
    s, d, r = auxiliary_plane(np.nan, 45.0, 90.0)
    assert np.isnan(s) and np.isnan(d)
    assert np.isnan(plane_mismatch_deg(np.nan, 45, 90, 0, 45, 90))


# -------------------------------------------------------------- review follow-ups


def test_all_na_updated_time_coerces() -> None:
    out = coerce_catalog(_raw_events().assign(updated_time=np.nan), source="quakes-sil")
    assert validate(out).ok and out["updated_time"].isna().all()


def test_read_csv_with_empty_columns_coerces(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "q.csv"
    path.write_text(
        "time,event_id,latitude,longitude,evaluation_mode,updated_time,magnitude_type\n"
        "2024-08-22T00:10:04.454Z,SIL1,63.9,-22.4,manual,,\n"
        "2024-08-22T00:01:00Z,SIL2,64.0,-21.2,automatic,,\n"
    )
    out = coerce_catalog(pd.read_csv(path), source="quakes-sil")
    rep = validate(out)
    assert rep.ok, rep.summary()
    assert list(out["event_id"]) == ["SIL2", "SIL1"]


def test_numeric_index_without_time_refused() -> None:
    raw = _raw_events().drop(columns=["time"])
    with pytest.raises(CoercionError, match="numbers are not times"):
        coerce_catalog(raw, source="sil-files", assume_utc=True)


def test_partial_plane2(mechs: pd.DataFrame) -> None:
    bad = mechs.copy()
    bad.iloc[0, bad.columns.get_loc("dip2")] = np.nan
    bad.iloc[0, bad.columns.get_loc("rake2")] = np.nan
    rep = validate_mechanisms(bad)
    assert rep.kinds() == {"partial_plane"} and rep.errors[0].count == 1


def test_parquet_round_trip(events: pd.DataFrame, tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("pyarrow")
    path = tmp_path / "e.parquet"
    events.to_parquet(path)
    back = pd.read_parquet(path)
    assert validate(back).ok, validate(back).summary()
    both = pd.concat([back, events])
    assert isinstance(both["source"].dtype, pd.CategoricalDtype)
