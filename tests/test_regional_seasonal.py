"""gps-regional-seasonal: seasonal-only records from neighbours' seasonals."""

import json

import numpy as np
import pytest

from geo_dataread import regional_seasonal as rs

NAMES = ["offset", "rate", *rs.PERIODIC_NAMES]
POS = {"A": (-22.40, 63.90), "B": (-22.30, 63.95), "C": (-22.35, 63.85),
       "D": (-22.32, 63.88), "E": (-22.38, 63.87), "T": (-22.36, 63.90), "FAR": (-16.0, 65.0)}  # fmt: skip


def _own(seasonal: float, **extra) -> dict:
    comp = {"params": [0.0, 1.0, seasonal, seasonal, 0.1, 0.1], "cov_upper": [0.0] * 21}
    return {"record_version": 1, "model": "lineperiodic", "param_names": NAMES,
            "components": [dict(comp) for _ in range(3)], "step_epochs": [],
            "fitted_at": "2026-01-01", "window": [2015.0, 2020.0], **extra}  # fmt: skip


def _doc(**stations) -> dict:
    return {"frame": "plate_removed@itrf14", "stations": stations}


def _coords(sta: str) -> tuple[float, float]:
    return POS[sta]


def test_a_station_without_a_record_gets_a_seasonal_only_regional_one() -> None:
    doc = _doc(A=_own(2.0), B=_own(2.0), C=_own(2.0))
    out, report = rs.add_regional_seasonals(doc, ["T", "A"], _coords)
    rec = out["stations"]["T"]
    assert report["A"] == "has its own record" and report["T"].startswith("regional")
    assert rec["seasonal_only"] and rec["model"] == "periodic"
    assert (
        rec["param_names"] == list(rs.PERIODIC_NAMES) and rec["frame"] == doc["frame"]
    )
    assert np.allclose(rec["components"][0]["params"], [2.0, 2.0, 0.1, 0.1])
    g = rec["groups"]["periodic"]
    assert g["provenance"].startswith("regional:") and {
        d["station"] for d in g["donors"]
    } == {"A", "B", "C"}
    assert all(d["digest"].startswith("sha256:") for d in g["donors"])
    # every fitted record now says so
    for code in "ABC":
        grp = out["stations"][code]["groups"]
        assert (
            grp["secular"]["provenance"] == grp["periodic"]["provenance"] == "estimated"
        )
    assert doc["stations"]["A"].get("groups") is None  # the input is untouched


def test_borrowed_and_regional_records_are_never_donors() -> None:
    doc = _doc(
        A=_own(2.0), B=_own(9.0, borrowed={"from": "B"}), C=_own(2.0), D=_own(2.0)
    )
    out, _ = rs.add_regional_seasonals(doc, ["T"], _coords)
    donors = {
        d["station"] for d in out["stations"]["T"]["groups"]["periodic"]["donors"]
    }
    assert donors == {"A", "C", "D"}
    # a second run does not use T's regional record as a donor for anything
    assert not rs.is_donor(out["stations"]["T"])


def test_too_few_near_donors_refuses_and_drops_an_old_regional_record() -> None:
    doc = _doc(A=_own(2.0), B=_own(2.0), C=_own(2.0), T={"seasonal_only": True})
    far_t = lambda s: (-18.0, 66.0) if s == "T" else POS[s]  # noqa: E731
    out, report = rs.add_regional_seasonals(doc, ["T"], far_t)
    assert report["T"].startswith("refused") and "T" not in out["stations"]


def test_cli_writes_the_merged_document_and_never_the_input(
    tmp_path, monkeypatch
) -> None:
    import geo_dataread.velocity as vel

    src = tmp_path / "in.json"
    src.write_text(json.dumps(_doc(A=_own(2.0), B=_own(2.0), C=_own(2.0))))
    before = src.read_bytes()
    monkeypatch.setattr(vel, "_coords", _coords)
    monkeypatch.setattr(
        vel, "select_stations", lambda pats, region: ["A", "B", "C", "T"]
    )
    out = tmp_path / "out.json"
    assert (
        rs.main(["--params", str(src), "--out", str(out), "--stamp", "2026-10-17"]) == 0
    )
    assert src.read_bytes() == before
    got = json.loads(out.read_text())
    assert got["stations"]["T"]["fitted_at"] == "2026-10-17"
    assert got["regional_seasonal"]["n_validation"] == 3


def test_group_provenance_reads_self_and_missing_as_estimated() -> None:
    assert rs.group_provenance({}, "periodic") == "estimated"
    assert (
        rs.group_provenance(
            {"groups": {"periodic": {"provenance": "self"}}}, "periodic"
        )
        == "estimated"
    )
    assert rs.group_provenance(
        {"groups": {"periodic": {"provenance": "store:SENG@x"}}}, "periodic"
    ).startswith("store:")


def test_a_regional_record_applies_its_seasonal_only() -> None:
    from gps_analysis.detrend import apply_detrend

    out, _ = rs.add_regional_seasonals(
        _doc(A=_own(2.0), B=_own(2.0), C=_own(2.0)), ["T"], _coords
    )
    rec = out["stations"]["T"]
    t = np.linspace(2024.0, 2025.0, 20)
    y = np.zeros((3, t.size))
    assert np.isfinite(apply_detrend(rec, t, y, terms="periodic")).all()
    with pytest.raises(ValueError, match="seasonal-only"):
        apply_detrend(rec, t, y)


def _fake_read(true_amp: dict[str, float], rng: np.random.Generator):
    """A ``velocity._read`` stand-in: 6 years of daily white noise (σ 1 mm)
    plus each station's TRUE annual (cos = sin = amp), secular removed."""
    from gps_analysis.staged import evaluate_group_values

    def read(sta, start, end, settings, *, terms, doc):
        assert terms == "secular"
        t = np.arange(start, end, 1 / 365.25)
        a = true_amp[sta]
        seas = evaluate_group_values(rs.PERIODIC_NAMES, np.array([a, a, 0, 0]), t)
        y = np.vstack([seas + rng.normal(0, 1, t.size) for _ in range(3)])
        return t, y, np.ones_like(y), {}

    return read


def test_the_regional_k_entry_is_measured_on_the_predicted_seasonal(
    monkeypatch,
) -> None:
    import geo_dataread.velocity as vel

    stations = {c: _own(2.0) for c in "ABCD"}
    doc = _doc(**stations)
    # the stored seasonals all say 2 mm; the truth agrees: k ~ white-noise k
    monkeypatch.setattr(vel, "_read", _fake_read(dict.fromkeys("ABCD", 2.0),
                                                 np.random.default_rng(1)))  # fmt: skip
    good = rs.regional_k_entry(doc, list("ABCD"), _coords, None, days=[91])
    # A's true seasonal is 8 mm: the neighbours' prediction leaves 6 mm in
    monkeypatch.setattr(vel, "_read", _fake_read({**dict.fromkeys("BCD", 2.0),
                                                  "A": 8.0},
                                                 np.random.default_rng(1)))  # fmt: skip
    bad = rs.regional_k_entry(doc, ["A"], _coords, None, days=[91])
    assert good["days"] == [91] and good["n_stations"] == [4]
    assert good["stations"] == list("ABCD")
    assert np.all(np.asarray(bad["k"][0]) > 2 * np.asarray(good["k"][0]))


def test_the_regional_k_entry_skips_a_station_it_cannot_predict(monkeypatch) -> None:
    import geo_dataread.velocity as vel

    monkeypatch.setattr(vel, "_read", _fake_read(dict.fromkeys(POS, 2.0),
                                                 np.random.default_rng(2)))  # fmt: skip
    doc = _doc(A=_own(2.0), B=_own(2.0), FAR=_own(2.0))
    got = rs.regional_k_entry(doc, ["FAR", "T"], _coords, None, days=[91])
    assert got["stations"] == [] and got["days"] == []


def test_a_second_run_over_its_own_output_changes_nothing() -> None:
    doc = _doc(A=_own(2.0), B=_own(2.0), C=_own(2.0), D=_own(2.0))
    once, _ = rs.add_regional_seasonals(doc, ["T"], _coords, fitted_at="x")
    twice, _ = rs.add_regional_seasonals(once, ["T"], _coords, fitted_at="x")
    assert twice == once  # digests ignore the provenance bookkeeping


def test_borrowed_records_say_borrowed_and_holds_keep_their_provenance() -> None:
    held = _own(2.0, groups={"periodic": {"provenance": "store:SENG"}})
    carried = _own(2.0, borrowed={"from": "BAUG", "prior_frame": "plate_removed"})
    # a document a first version marked `estimated` everywhere is corrected
    carried["groups"] = {"secular": {"provenance": "estimated"}}
    doc = _doc(A=_own(2.0), B=held, C=carried, D=_own(2.0), E=_own(2.0))
    out, _ = rs.add_regional_seasonals(doc, ["T"], _coords)
    st = out["stations"]
    assert st["B"]["groups"]["periodic"]["provenance"] == "store:SENG"
    assert st["B"]["groups"]["secular"]["provenance"] == "estimated"
    for g in ("secular", "periodic"):
        assert st["C"]["groups"][g]["provenance"] == "borrowed:BAUG@plate_removed"
    donors = {d["station"] for d in st["T"]["groups"]["periodic"]["donors"]}
    assert donors == {"A", "D", "E"}  # neither the hold nor the carry-over


def test_a_regional_record_is_no_stage_plan_donor() -> None:
    from geo_dataread.stage_plan import donor_group_values

    out, _ = rs.add_regional_seasonals(
        _doc(A=_own(2.0), B=_own(2.0), C=_own(2.0)), ["T"], _coords
    )
    with pytest.raises(ValueError, match="seasonal-only"):
        donor_group_values(out["stations"]["T"], "periodic", component=0, donor="T")


def test_a_k_table_rebuild_keeps_the_regional_entry(tmp_path, monkeypatch) -> None:
    import geo_dataread.velocity as vel

    out = tmp_path / "k.json"
    regional = {"days": [91], "k": [[3.0, 2.6, 2.7]], "n_stations": [20]}
    out.write_text(json.dumps({"residuals": {vel.K_RESIDUAL_REGIONAL: regional}}))
    fresh = {"stations": ["A"], "residuals": {"all": {"days": [91],
             "k": [[2.0, 2.0, 2.0]], "n_stations": [1]}}}  # fmt: skip
    # the command line moved beside the core (gpsprod step 3): patch it there
    import geo_dataread.products.velocity_cli as vcli

    monkeypatch.setattr(
        vcli, "build_k_table", lambda *a, **k: json.loads(json.dumps(fresh))
    )
    monkeypatch.setattr(vcli, "select_stations", lambda *a: ["A"])
    assert vel.main(["k-table", "--out", str(out)]) == 0
    got = json.loads(out.read_text())["residuals"]
    assert got["all"] == fresh["residuals"]["all"]
    assert got[vel.K_RESIDUAL_REGIONAL] == {**regional, "carried_over": True}
