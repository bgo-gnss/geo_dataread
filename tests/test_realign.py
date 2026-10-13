"""write_pos and realign: the .pos writer and the day re-alignment."""

from __future__ import annotations

import datetime
import re
from pathlib import Path

import numpy as np
import pytest

from geo_dataread import org, pos, realign, sinex, stabilize

FIX = Path(__file__).parent / "fixtures"
SNX = FIX / "sinex" / "IMO0OPSFIN_20141500000_01D_01D_SOL.SNX.gz"
AKUR = FIX / "pos" / "AKUR.imo.final_itr08.pos"


def test_write_pos_reproduces_a_tssum_file(tmp_path: Path) -> None:
    src = pos.read_pos(AKUR)
    out = pos.write_pos(
        tmp_path / "AKUR.x.pos",
        "AKUR",
        src.epochs,
        src.xyz,
        src.cov_xyz(),
        frame="Unknown",
        ref_xyz=src.ref_xyz,
    )
    got = pos.read_pos(out)
    np.testing.assert_array_equal(got.epochs, src.epochs)
    np.testing.assert_array_equal(got.yearf, src.yearf)
    np.testing.assert_allclose(got.mjd, src.mjd, atol=1e-4)
    np.testing.assert_allclose(got.xyz, src.xyz, atol=6e-6)
    np.testing.assert_allclose(got.sxyz, src.sxyz, atol=6e-6)
    np.testing.assert_allclose(got.rxyz, src.rxyz, atol=6e-4)
    np.testing.assert_allclose(got.llh[:, :2], src.llh[:, :2], atol=1e-9)
    np.testing.assert_allclose(got.dneu, src.dneu, atol=2e-5)  # GLOBK Loc. like tssum
    np.testing.assert_allclose(got.sneu, src.sneu, atol=2e-5)
    np.testing.assert_allclose(got.ref_xyz, src.ref_xyz, atol=6e-6)
    assert got.header["Format Version"] == "1.1.1"
    # the reference lines have tssum's layout, column for column (the NEU
    # reference's last digits differ: tssum takes it from another epoch)
    for key in ("XYZ Reference position", "NEU Reference position"):
        want = next(ln for ln in AKUR.read_text().splitlines() if ln.startswith(key))
        line = next(ln for ln in out.read_text().splitlines() if ln.startswith(key))
        assert re.sub(r"\d", "0", line) == re.sub(r"\d", "0", want)


def test_write_pos_reference_height_over_1000_m(tmp_path: Path) -> None:
    # AMC2 (1911 m): a 10-wide height ran into the longitude, unreadable
    x = np.array([[-1248596.38009, -4819428.20400, 3976505.92963]])
    t = np.array(["2025-01-01T12:00"], dtype="datetime64[m]")
    c = np.eye(3)[None] * 1e-6
    got = pos.read_pos(
        pos.write_pos(tmp_path / "A.pos", "AMC2", t, x, c, frame="IGb20")
    )
    assert got.ref_llh[2] == pytest.approx(1911.384, abs=1e-3)
    assert got.ref_llh[1] == pytest.approx(255.4754, abs=1e-4)


def test_write_pos_truncates_to_the_minute_and_sorts(tmp_path: Path) -> None:
    t = np.array(["2024-04-10T11:59:45", "2024-04-09T11:59:45"], dtype="datetime64[s]")
    x = np.array([[2587384.0, -1043034.0, 5716564.0]] * 2) + [[0.001, 0, 0], [0, 0, 0]]
    c = np.broadcast_to(np.diag([1e-6, 2e-6, 4e-6]), (2, 3, 3))
    got = pos.read_pos(pos.write_pos(tmp_path / "S.pos", "s", t, x, c, frame="IGb14"))
    assert [str(e) for e in got.epochs] == [
        "2024-04-09T11:59:00",
        "2024-04-10T11:59:00",
    ]
    assert got.station == "S" and got.header["4-character ID"] == "S"
    np.testing.assert_allclose(got.xyz[1] - got.xyz[0], [0.001, 0, 0], atol=1e-9)


def test_realign_day_glorg_frame_and_external_identity(tmp_path: Path) -> None:
    day = realign.realign_day(SNX, "itrf2008")
    pos_org = org.read_org_positions(str(SNX).replace(".SNX.gz", ".org.gz"))
    assert day.n_ref == 47 and not day.dropped and not day.error
    assert day.epoch == datetime.datetime(2014, 5, 30, 11, 59, 45)
    np.testing.assert_allclose(
        day.sites["ONSA"][0], pos_org["ONSA_1PS"].xyz, atol=1.5e-5
    )
    # an external apriori equal to the SINEX one is the same fit
    sol = sinex.read_sinex(SNX)
    setup = org.read_org_setup(str(SNX).replace(".SNX.gz", ".org.gz"))
    own = {
        c: sol.apriori[sol.xyz_index(c)]
        for c in setup.reference
        if (sol.code == c).sum() == 3
    }
    ext = stabilize.stabilize(sol, setup.reference, setup.config, apriori=own)
    np.testing.assert_allclose(
        ext.solution.estimate[sol.xyz_index("ONSA")], day.sites["ONSA"][0], atol=1e-12
    )


def test_archive_days(tmp_path: Path) -> None:
    (tmp_path / "2014").mkdir()
    for doy in (149, 150, 151):
        (tmp_path / "2014" / f"IMO0OPSFIN_2014{doy}0000_01D_01D_SOL.SNX.gz").touch()
    got = realign.archive_days(
        tmp_path, datetime.date(2014, 5, 30), datetime.date(2014, 5, 31)
    )
    assert [p.name[11:18] for p in got] == ["2014150", "2014151"]


def _series(tmp_path: Path, name: str, days: list[str], dx: float = 0.0) -> Path:
    t = np.array([f"{d}T11:59:45" for d in days], dtype="datetime64[s]")
    x = np.array([[2587384.0, -1043034.0, 5716564.0]] * len(days))
    x[:, 0] += np.arange(len(days)) * 1e-3 + dx
    c = np.broadcast_to(np.diag([1e-6, 2e-6, 4e-6]), (len(days), 3, 3))
    return pos.write_pos(tmp_path / name, "REYK", t, x, c, frame="IGb14")


def test_merge_pos_replaces_days_and_keeps_the_rest(tmp_path: Path) -> None:
    path = _series(tmp_path, "R.pos", ["2024-04-01", "2024-04-02", "2024-04-03"])
    before = path.read_text().splitlines()
    ref = pos.read_pos(path).ref_xyz
    t = np.array(["2024-04-02T11:59:45", "2024-04-04T11:59:45"], dtype="datetime64[s]")
    x = np.array([[2587384.5, -1043034.0, 5716564.0]] * 2)
    c = np.broadcast_to(np.diag([1e-6, 2e-6, 4e-6]), (2, 3, 3))
    new = pos.write_pos(tmp_path / "R.new", "REYK", t, x, c, frame="IGb14", ref_xyz=ref)
    pos.merge_pos(path, new, ["20240402", "20240404"])
    after = path.read_text().splitlines()
    got = pos.read_pos(path)
    assert [str(e)[:10] for e in got.epochs] == [
        "2024-04-01",
        "2024-04-02",
        "2024-04-03",
        "2024-04-04",
    ]
    assert got.xyz[1, 0] == 2587384.5 and not new.exists()
    # untouched days are byte for byte the old rows, the header follows the merge
    assert after[-4] == before[-3] and after[-2] == before[-1]
    assert got.header["Last Epoch"] == "20240404 115900"
    np.testing.assert_array_equal(got.ref_xyz, ref)
    # dropping without new rows, and dropping the last rows removes the file
    pos.merge_pos(path, None, ["20240401"])
    assert pos.read_pos(path).header["First Epoch"] == "20240402 115900"
    assert pos.merge_pos(path, None, ["20240402", "20240403", "20240404"]) is None
    assert not path.exists()


def _archive(tmp_path: Path) -> Path:
    arch = tmp_path / "arch" / "2014"
    arch.mkdir(parents=True)
    for suffix in (".SNX.gz", ".org.gz"):
        src = Path(str(SNX).replace(".SNX.gz", suffix))
        (arch / src.name).symlink_to(src)
    return arch.parent


def test_main_update_and_rejected_days(tmp_path: Path) -> None:
    arch, out = _archive(tmp_path), tmp_path / "pos"
    day = ["--start", "2014-05-30", "--end", "2014-05-30", "--archive", str(arch)]
    common = [*day, "--frame", "itrf2008", "--out", str(out), "-j", "1"]
    assert realign.main([*common, "--stations", "REYK", "HOFN"]) == 0
    reyk = out / "REYK.imo.realign_itrf2008.pos"
    assert pos.read_pos(reyk).xyz.shape == (1, 3)
    # an older day kept in the file is not touched by an update of 2014-05-30
    old = pos.read_pos(reyk)
    keep = pos.write_pos(
        tmp_path / "k.pos",
        "REYK",
        np.array(["2014-05-29T11:59:45"], dtype="datetime64[s]"),
        old.xyz,
        old.cov_xyz(),
        frame="ITRF2008",
        ref_xyz=old.ref_xyz,
    )
    pos.merge_pos(reyk, keep, [])
    first = reyk.read_text().splitlines()[-2]
    assert realign.main([*common, "--stations", "REYK", "HOFN", "--update"]) == 0
    assert reyk.read_text().splitlines()[-2] == first
    assert len(pos.read_pos(reyk).epochs) == 2
    # the day rejected: its rows go, HOFN (only that day) disappears
    assert realign.main([*common, "--update", "--min-ref", "999"]) == 0
    assert [str(e)[:10] for e in pos.read_pos(reyk).epochs] == ["2014-05-29"]
    assert not (out / "HOFN.imo.realign_itrf2008.pos").exists()
    log = (out / "days.imo.realign_itrf2008.csv").read_text().splitlines()
    assert len(log) == 2 and "rejected: 49 reference site(s), need 999" in log[1]


def test_too_few_reference_sites_rejected() -> None:
    r = realign.realign_day(SNX, "itrf2008", min_ref=48)
    assert r.sites == {} and r.error.startswith("rejected: 47")
