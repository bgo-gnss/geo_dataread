"""Unit tests for the slice-6 JOIN revival (refactor-B, decision D4).

Not goldens — behavior tests for the NEW multi-processing-scheme reader
`read_join` and for `openGlobkTimes`' generalized scheme handling (the
slice-4 tType="08h" guard is gone; any scheme with mb_STA_<scheme>.dat{1,2,3}
files reads; a scheme whose files don't exist is rejected up front).

Runs on the frozen synthetic fixtures (fixtures/TOT/, where the
mb_SENG_08h.dat* files are TOT copies — ideal for equality assertions).
The genuine 8-hourly values are pinned by the realdata goldens
(real_ogt_SENG_08h, real_join_SENG_TOT_08h). Every test takes gm_env so
GPS_CONFIG_PATH points at the hermetic config dir, never ~/.config.
"""

import shutil

import numpy as np
import pandas as pd
import pytest

import geo_dataread.gps_read as gpsr

from cases import FIXTURES, TOT

JOIN_COLUMNS = ["north", "east", "up", "Dnorth", "Deast", "Dup", "yearf", "scheme"]


def _copy_scheme_fixture(dest, schemes, sta="SENG"):
    """Materialize mb_<sta>_<scheme>.dat{1,2,3} in dest for each scheme
    (copies of the frozen SENG TOT files)."""
    for scheme in schemes:
        for i in (1, 2, 3):
            shutil.copy(
                FIXTURES / "TOT" / f"mb_{sta}_TOT.dat{i}",
                dest / f"mb_{sta}_{scheme}.dat{i}",
            )
    return str(dest) + "/"


# --- openGlobkTimes: generalized scheme handling ---------------------------


def test_08h_reads(gm_env):
    """tType="08h" reads again (slice-4 guard removed) — and identically to
    TOT on the fixture copies: same plain read, no sub-daily special-casing."""
    yearf_8h, data_8h, Ddata_8h = gpsr.openGlobkTimes("SENG", Dir=TOT, tType="08h")
    yearf_tot, data_tot, Ddata_tot = gpsr.openGlobkTimes("SENG", Dir=TOT, tType="TOT")
    np.testing.assert_array_equal(yearf_8h, yearf_tot)
    np.testing.assert_array_equal(data_8h, data_tot)
    np.testing.assert_array_equal(Ddata_8h, Ddata_tot)


def test_missing_scheme_raises_clear_error(gm_env):
    with pytest.raises(FileNotFoundError, match="scheme '04h'.*mb_SENG_04h"):
        gpsr.openGlobkTimes("SENG", Dir=TOT, tType="04h")


# --- read_join: the multi-scheme JOIN reader --------------------------------


def test_join_two_schemes(gm_env):
    joined = gpsr.read_join("SENG", ("TOT", "08h"), Dir=TOT)

    assert list(joined.columns) == JOIN_COLUMNS
    assert isinstance(joined.index, pd.DatetimeIndex)
    assert joined.index.is_monotonic_increasing

    # per-scheme row counts == the plain per-scheme reads
    n_tot = len(gpsr.openGlobkTimes("SENG", Dir=TOT, tType="TOT")[0])
    n_08h = len(gpsr.openGlobkTimes("SENG", Dir=TOT, tType="08h")[0])
    counts = joined["scheme"].value_counts()
    assert counts["TOT"] == n_tot
    assert counts["08h"] == n_08h
    assert len(joined) == n_tot + n_08h

    # stable time sort: coincident epochs (fixture 08h == TOT copies) keep
    # the schemes listing order
    per_stamp = joined.groupby(level=0)["scheme"].apply(list)
    assert all(labels == ["TOT", "08h"] for labels in per_stamp)


def test_join_scheme_extensibility_three_schemes(gm_env, tmp_path):
    """Adding a scheme is just listing it: a 3-scheme JOIN needs no new code."""
    d = _copy_scheme_fixture(tmp_path, ("TOT", "08h", "04h"))
    joined = gpsr.read_join("SENG", ("TOT", "08h", "04h"), Dir=d)
    counts = joined["scheme"].value_counts()
    assert sorted(counts.index) == ["04h", "08h", "TOT"]
    assert counts["TOT"] == counts["08h"] == counts["04h"]
    assert joined.index.is_monotonic_increasing


def test_join_single_scheme_degenerate(gm_env):
    """A single-scheme list is the plain read plus the scheme label."""
    joined = gpsr.read_join("SENG", ("TOT",), Dir=TOT)
    plain = gpsr.convGlobktopandas(*gpsr.openGlobkTimes("SENG", Dir=TOT, tType="TOT"))
    assert (joined["scheme"] == "TOT").all()
    pd.testing.assert_frame_equal(joined.drop(columns="scheme"), plain)


def test_join_missing_scheme_skipped_with_warning(gm_env, capsys):
    """Default missing="warn": absent schemes are skipped, the rest load."""
    joined = gpsr.read_join("SENG", ("TOT", "04h"), Dir=TOT)
    assert set(joined["scheme"]) == {"TOT"}
    assert "WARNING: skipping scheme '04h'" in capsys.readouterr().out


def test_join_missing_scheme_raise_mode(gm_env):
    with pytest.raises(FileNotFoundError, match="scheme '04h'"):
        gpsr.read_join("SENG", ("TOT", "04h"), Dir=TOT, missing="raise")


def test_join_all_schemes_missing_raises(gm_env):
    with pytest.raises(FileNotFoundError, match="any of the schemes"):
        gpsr.read_join("SENG", ("04h", "02h"), Dir=TOT)


def test_join_invalid_missing_mode(gm_env):
    with pytest.raises(ValueError, match='"warn" or "raise"'):
        gpsr.read_join("SENG", ("TOT",), Dir=TOT, missing="ignore")


# --- getData back-compat -----------------------------------------------------


def test_getdata_join_alias_unchanged(gm_env):
    """Decision (slice 6): getData(tType="JOIN") keeps its legacy meaning —
    an alias for TOT returning arrays. The DataFrame JOIN lives in read_join."""
    j = gpsr.getData("SENG", ref="plate", Dir=TOT, tType="JOIN")
    t = gpsr.getData("SENG", ref="plate", Dir=TOT, tType="TOT")
    for got, want in zip(j, t):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
