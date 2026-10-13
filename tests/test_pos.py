"""geo_dataread.pos: the PBO .pos reader and the openGlobkTimes(tType="POS") path."""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import numpy as np
import pytest

from geo_dataread import pos

FIX = Path(__file__).parent / "fixtures" / "pos"


def test_read_pos_fields() -> None:
    s = pos.read_pos(FIX / "AKUR.imo.final_itr08.pos")
    assert s.station == "AKUR" and len(s) == 5
    assert str(s.epochs[0]) == "2026-09-01T11:59:00"
    np.testing.assert_allclose(s.xyz[0], [2502918.04409, -819167.02669, 5789715.19547])
    np.testing.assert_allclose(s.sxyz[0], [0.00212, 0.00116, 0.00422])
    np.testing.assert_allclose(s.rxyz[0], [-0.453, 0.818, -0.460])
    np.testing.assert_allclose(s.ref_xyz, [2502918.04456, -819167.02792, 5789715.19606])
    assert s.soln[0] == "final"
    assert s.header["Format Version"] == "1.1.0"


def test_yearf_is_globk_epoch_convention() -> None:
    # 2026-09-01 11:59 = doy 244: 2026 + (243 + 11/24 + 59/1440) / 365; mb_ TOT prints 2026.66712
    s = pos.read_pos(FIX / "AKUR.imo.final_itr08.pos")
    assert s.yearf[0] == pytest.approx(2026 + (243 + 11 / 24 + 59 / 1440) / 365)
    assert round(s.yearf[0], 5) == 2026.66712


def test_covariance_from_sigmas_and_correlations() -> None:
    s = pos.read_pos(FIX / "AKUR.imo.final_itr08.pos")
    c = s.cov_neu()[0]
    sn, se, su = s.sneu[0]
    rne, rnu, reu = s.rneu[0]
    np.testing.assert_allclose(np.diag(c), [sn**2, se**2, su**2])
    assert c[0, 1] == pytest.approx(rne * sn * se) and c[1, 0] == c[0, 1]
    assert c[0, 2] == pytest.approx(rnu * sn * su) and c[1, 2] == pytest.approx(
        reu * se * su
    )


def test_gz_and_open_pos_times(tmp_path: Path) -> None:
    with (
        open(FIX / "AKUR.imo.final_itr08.pos", "rb") as fi,
        gzip.open(tmp_path / "AKUR.imo.final_itr08.pos.gz", "wb") as fo,
    ):
        shutil.copyfileobj(fi, fo)
    yearf, data, ddata = pos.open_pos_times("akur", tmp_path, ref="tssum")
    s = pos.read_pos(FIX / "AKUR.imo.final_itr08.pos")
    assert data.shape == ddata.shape == (3, 5)
    np.testing.assert_array_equal(data, s.dneu.T)
    np.testing.assert_array_equal(ddata, s.sneu.T)
    np.testing.assert_array_equal(yearf, s.yearf)


def test_missing_station_and_bad_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        pos.pos_path("NONE", tmp_path)
    (tmp_path / "BAD.imo.final_itr08.pos").write_text("not a pos file\n")
    with pytest.raises(pos.PosError):
        pos.read_pos(tmp_path / "BAD.imo.final_itr08.pos")


def test_pos_path_prefers_new_id_then_legacy(tmp_path: Path) -> None:
    legacy = tmp_path / f"AKUR.{pos.LEGACY_PROD_ID}.pos"
    shutil.copy(FIX / "AKUR.imo.final_itr08.pos", legacy)
    assert pos.pos_path("akur", tmp_path) == legacy
    okada = tmp_path / "AKUR.imo.daily_itr08.pos"  # okada's cron still writes it
    shutil.copy(legacy, okada)
    assert pos.pos_path("AKUR", tmp_path) == okada
    new = tmp_path / f"AKUR.{pos.DEFAULT_PROD_ID}.pos"
    shutil.copy(legacy, new)
    assert pos.pos_path("AKUR", tmp_path) == new
    # the laptop's sets under their earlier names, asked for by old or new id
    igb14 = tmp_path / "AKUR.imo.daily_igb14.pos"
    shutil.copy(legacy, igb14)
    assert pos.pos_path("AKUR", tmp_path, prod_id="imo.24hr_itrf14") == igb14
    assert pos.pos_path("AKUR", tmp_path, prod_id="imo.daily_igb14") == igb14
    # an id with no earlier names gets no fallback
    with pytest.raises(FileNotFoundError):
        pos.pos_path("AKUR", tmp_path, prod_id="imo.08hr_itrf20")


def test_openGlobkTimes_pos_scheme() -> None:
    from geo_dataread.gps_read import openGlobkTimes

    yearf, data, ddata = openGlobkTimes("AKUR", Dir=str(FIX), tType="POS")
    assert data.shape == (3, 5) and yearf[0] == pytest.approx(2026.667122, abs=1e-6)


def test_overflow_fields_are_nan(tmp_path: Path) -> None:
    # SEY1 on okada: reference height and recent dN/dE/dU overflow (2026-10)
    lines = (FIX / "AKUR.imo.final_itr08.pos").read_text().splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("NEU Reference position"))
    lines[i] = (
        lines[i][:25] + "   63.9956463003  352.7515334168 ********** (Unknown/WGS84)"
    )
    row = lines[-1].split()
    row[15:18] = ["*********"] * 3
    lines[-1] = " " + " ".join(row)
    f = tmp_path / "AKUR.imo.final_itr08.pos"
    f.write_text("\n".join(lines) + "\n")
    s = pos.read_pos(f)
    assert np.isnan(s.ref_llh[2]) and not np.isnan(s.ref_llh[:2]).any()
    assert np.isnan(s.dneu[-1]).all() and not np.isnan(s.dneu[:-1]).any()
    assert not np.isnan(s.sneu).any()


def test_numpy_only() -> None:
    # like sinex.py: usable from gps_api / notebooks without the legacy stack
    imports = [
        ln
        for ln in Path(pos.__file__).read_text().splitlines()
        if ln.startswith(("import ", "from "))
    ]
    assert not [
        ln for ln in imports if "pandas" in ln or "gps_parser" in ln or "gps_read" in ln
    ]


# --- N/E/U from X/Y/Z (pos_local, 2026-10-11) ---------------------------------


def _akur() -> pos.PosSeries:
    return pos.read_pos(FIX / "AKUR.imo.final_itr08.pos")


def test_pos_local_globk_reproduces_tssum_columns() -> None:
    # tssum's dN/dE/dU = glorg Loc. convention from the header reference;
    # equal to its 0.01 mm print resolution, sigmas likewise
    s = _akur()
    out, ref = pos.pos_local(s, s.ref_xyz, junctions=(), convention="globk")
    np.testing.assert_allclose(out.d, s.dneu, atol=2e-5)
    assert out.cov is not None
    sig = np.sqrt(np.diagonal(out.cov, axis1=1, axis2=2))
    np.testing.assert_allclose(sig, s.sneu, atol=2e-5)
    np.testing.assert_array_equal(ref, s.ref_xyz)


def test_pos_local_reference_modes() -> None:
    s = _akur()
    first, ref = pos.pos_local(s, "first", junctions=())
    np.testing.assert_allclose(first.d[0], 0.0, atol=1e-9)
    np.testing.assert_array_equal(ref, s.xyz[0])
    wmean, _ = pos.pos_local(s, junctions=())
    # least-squares mean as reference: the weighted residuals balance out
    assert wmean.cov is not None
    w = np.linalg.inv(np.asarray(wmean.cov, dtype=float))
    np.testing.assert_allclose(
        np.einsum("nij,nj->i", w, wmean.d), 0.0, atol=1e-6 * np.abs(w).max()
    )
    # every reference gives the same displacement differences
    np.testing.assert_allclose(
        np.diff(first.d, axis=0), np.diff(wmean.d, axis=0), atol=1e-9
    )
    with pytest.raises(ValueError, match="ref must be"):
        pos.pos_local(s, "middle", junctions=())


def test_pos_local_apriori_from_coord_file(tmp_path: Path) -> None:
    s = _akur()
    apr = s.xyz[2] + [0.01, -0.02, 0.03]
    coord = tmp_path / "station_coord.xyz"
    coord.write_text(
        f" 1.0 2.0 3.0 REYK\n {apr[0]:.5f} {apr[1]:.5f} {apr[2]:.5f} AKUR\n"
    )
    got = pos.apriori_xyz("akur", coord)
    assert got is not None
    np.testing.assert_allclose(got, apr, atol=1e-5)
    assert pos.apriori_xyz("NONE", coord) is None
    out, ref = pos.pos_local(s, "apriori", junctions=(), coord_file=coord)
    assert ref is not None
    np.testing.assert_allclose(ref, apr, atol=1e-5)
    # not listed: falls back to the least-squares mean
    other = tmp_path / "other.xyz"
    other.write_text(" 1.0 2.0 3.0 REYK\n")
    _, ref2 = pos.pos_local(s, "apriori", junctions=(), coord_file=other)
    _, ref3 = pos.pos_local(s, "wmean", junctions=())
    np.testing.assert_array_equal(ref2, ref3)


def test_pos_local_applies_junction_offsets_along_up() -> None:
    from geo_dataread.globk_tot import JunctionOffset

    s = _akur()
    cut = float(s.yearf[2])
    rule = JunctionOffset("AKUR", "U", cut, 1.093, "test")
    plain, ref = pos.pos_local(s, "first", junctions=())
    assert ref is not None
    fixed, _ = pos.pos_local(s, ref, junctions={"AKUR": [rule]})
    np.testing.assert_allclose(fixed.d[:2, 2] - plain.d[:2, 2], 1.093, atol=1e-6)
    np.testing.assert_allclose(fixed.d[:2, :2], plain.d[:2, :2], atol=1e-6)
    np.testing.assert_array_equal(fixed.d[2:], plain.d[2:])
    np.testing.assert_array_equal(fixed.cov, plain.cov)
    d = pos.junction_dlocal(s.yearf, [rule])
    assert d[:2, 2].tolist() == [1.093, 1.093] and not d[2:].any()


def test_getData_pos_reference_stays_zero(tmp_path: Path) -> None:
    from geo_dataread.gps_read import getData, openGlobkTimes

    s = _akur()
    y, data, _, off = getData(  # type: ignore[no-untyped-call]
        "AKUR", Dir=str(FIX), tType="POS", ref_position=s.ref_xyz, uncert=1e3
    )
    out, _ = pos.pos_local(s, s.ref_xyz)
    np.testing.assert_allclose(data, out.d.T * 1000, atol=1e-9)  # mm
    np.testing.assert_array_equal(off, np.zeros(3))
    with pytest.raises(ValueError, match="POS.? only"):
        openGlobkTimes(  # type: ignore[no-untyped-call]
            "AKUR", Dir=str(FIX), tType="TOT", ref_position="wmean"
        )


# --- product ids (pos_id, 2026-10-12) ------------------------------------------


def test_pos_product_names() -> None:
    assert pos.pos_product(None) == pos.DEFAULT_PROD_ID == "imo.24hr_itrf08"
    assert pos.pos_product("igb14") == pos.pos_product("ITRF2014") == "imo.24hr_itrf14"
    assert pos.pos_product("itrf14") == "imo.24hr_itrf14"
    assert pos.pos_product("itrf2020") == pos.pos_product("IGb20") == "imo.24hr_itrf20"
    assert pos.pos_product("itrf20", session="8h") == "imo.08hr_itrf20"
    assert pos.pos_product("itrf2008") == pos.pos_product("itr08") == "imo.24hr_itrf08"
    # earlier ids map to the new ones; any other full id passes through
    assert pos.pos_product("imo.daily_igb14") == "imo.24hr_itrf14"
    assert pos.pos_product("imo.final_igb20") == "imo.24hr_itrf20"
    assert pos.pos_product("xyz.final_foo") == "xyz.final_foo"
    assert pos.pos_frame("imo.daily_itr08") == "itrf08"
    assert pos.pos_frame("imo.08hr_itrf20") == "itrf20"
    with pytest.raises(ValueError, match="unknown frame"):
        pos.pos_product("wgs84")
    assert {
        pos.POS_PRODUCTS[pos.pos_product(n)][1] for n in ("itr08", "igb14", "itrf2020")
    } == {
        "ITRF2008",
        "ITRF2014",
        "ITRF2020",
    }


def test_tssum_id_and_width() -> None:
    # tssum keeps 15 characters for the file name and reads chars 17-21 as
    # the solution type it writes in the last column
    for f in pos.FRAMES:
        for sess in pos.SESSIONS:
            assert len(pos.product_id(f, sess)) == 15
    assert pos.tssum_id("imo.24hr_itrf14") == "imo.24hr_itrf14+24hrs"
    assert pos.tssum_id("imo.08hr_itrf20") == "imo.08hr_itrf20+08hrs"
    with pytest.raises(ValueError, match="15 characters"):
        pos.tssum_id("imo.24hrs_itrf14")


def test_openGlobkTimes_picks_the_product(tmp_path: Path) -> None:
    from geo_dataread.gps_read import openGlobkTimes

    src = (FIX / "AKUR.imo.final_itr08.pos").read_text()
    (tmp_path / "AKUR.imo.daily_itr08.pos").write_text(src)
    # same file shifted 1 m in X, under the IGb14 product id
    lines = src.splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.startswith(" 2026"))
    shifted = []
    for ln in lines[k:]:
        tok = ln.split()
        tok[3] = f"{float(tok[3]) + 1.0:.5f}"
        shifted.append(" " + " ".join(tok))
    (tmp_path / "AKUR.imo.daily_igb14.pos").write_text(
        "\n".join(lines[:k] + shifted) + "\n"
    )
    _, d08, _ = openGlobkTimes(
        "AKUR", Dir=str(tmp_path), tType="POS", ref_position="first"
    )  # type: ignore[no-untyped-call]
    _, d14, _ = openGlobkTimes(
        "AKUR", Dir=str(tmp_path), tType="POS", ref_position="first", pos_id="igb14"
    )  # type: ignore[no-untyped-call]
    np.testing.assert_allclose(d14, d08, atol=1e-6)  # relative to each own first epoch
    _, a08, _ = openGlobkTimes(
        "AKUR",
        Dir=str(tmp_path),
        tType="POS",
        ref_position=pos.read_pos(tmp_path / "AKUR.imo.daily_itr08.pos").xyz[0],
    )  # type: ignore[no-untyped-call]
    _, a14, _ = openGlobkTimes(
        "AKUR",
        Dir=str(tmp_path),
        tType="POS",
        ref_position=pos.read_pos(tmp_path / "AKUR.imo.daily_itr08.pos").xyz[0],
        pos_id="imo.daily_igb14",
    )  # type: ignore[no-untyped-call]
    assert (
        np.abs(a14 - a08).max() > 0.5
    )  # the 1 m shift shows against a common reference
    with pytest.raises(FileNotFoundError):
        openGlobkTimes("AKUR", Dir=str(tmp_path), tType="POS", pos_id="itrf2020")  # type: ignore[no-untyped-call]
    with pytest.raises(ValueError, match="POS.? only"):
        openGlobkTimes("AKUR", Dir=str(tmp_path), tType="TOT", pos_id="igb14")  # type: ignore[no-untyped-call]


def test_junctions_only_for_okadas_products() -> None:
    # GUSK's stale-height junction corrects okada's processing; gamit-p01's
    # itrf20 and 8 h series never had the error, so the rule must not apply
    assert pos.product_junctions("imo.24hr_itrf08") is None
    assert pos.product_junctions("imo.daily_itr08") is None
    assert pos.product_junctions("imo.24hr_itrf14") is None
    assert pos.product_junctions("imo.24hr_itrf20") == ()
    assert pos.product_junctions("imo.08hr_itrf20") == ()
    assert pos.product_junctions("imo.final_igb20") == ()
    assert pos.product_junctions("xyz.other_thing") is None


def test_frame_name_is_the_full_itrf_name():
    from geo_dataread.pos import frame_name

    assert frame_name("itrf14") == "itrf2014"
    assert frame_name("igb20") == "itrf2020"
    assert frame_name("imo.24hr_itrf08") == "itrf2008"
    assert frame_name("itrf2014") == "itrf2014"
