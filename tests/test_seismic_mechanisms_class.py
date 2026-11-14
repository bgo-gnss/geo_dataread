"""P/B/T axes, the computed faulting class, SIL's label and the
inconsistent-planes marker (ticket T12b of PLAN-seismicity.md).

Hand cases, symmetry properties (plane swap, strike/rake wrap), the SIL
fixture blocks (``tests/fixtures/sil``: SIL's own P/T axes and labels), and
a synthetic inconsistent plane pair as in the two 2023 fps rows. No network,
no mount.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geo_dataread.seismic import mechanisms as mm
from geo_dataread.seismic.schema import (
    SIL_FAULTING_CLASSES,
    SchemaError,
    auxiliary_plane,
    validate_mechanisms,
)
from geo_dataread.seismic.sil_files import read_events_fps

SIL = Path(__file__).parent / "fixtures" / "sil"
REVIEWED_DAYS = ("1995/jun/15", "2022/dec/12", "2024/aug/22", "2025/may/03")


def axes(*plane: float) -> list[float]:
    return [float(x) for x in mm.pt_axes(*plane)]


def cls(*plane: float, rule: str = "frohlich") -> object:
    return mm.faulting_class(*plane, rule=rule).item()


def vec(az: float, plunge: float) -> np.ndarray:
    a, p = np.radians(az), np.radians(plunge)
    return np.array([np.cos(p) * np.cos(a), np.cos(p) * np.sin(a), np.sin(p)])


def axis_angle(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Sign-free angle between two axes given as (azimuth, plunge)."""
    c = abs(float(vec(*a) @ vec(*b)))
    return float(np.degrees(np.arccos(min(c, 1.0))))


# --------------------------------------------------------------------------
# pt_axes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("plane", "expected"),
    [
        # left-lateral on a N-S vertical plane: P 135, T 45, B vertical
        ((0, 90, 0), (135, 0, 0, 90, 45, 0)),
        # pure normal on a N-striking plane dipping 45 E: P vertical, T E-W
        ((0, 45, -90), (0, 90, 0, 0, 90, 0)),
        # pure thrust: T vertical, P E-W
        ((0, 45, 90), (90, 0, 0, 0, 0, 90)),
    ],
)
def test_pt_axes_hand_cases(
    plane: tuple[float, float, float], expected: tuple[float, ...]
) -> None:
    got = axes(*plane)
    assert got == pytest.approx(list(expected), abs=1e-6)


def test_pt_axes_are_orthogonal_and_lower_hemisphere() -> None:
    rng = np.random.default_rng(12)
    s, d, r = (
        rng.uniform(-400, 400, 200),
        rng.uniform(0, 90, 200),
        rng.uniform(-400, 400, 200),
    )
    ax = mm.pt_axes(s, d, r)
    for pl in (ax.p_plunge, ax.b_plunge, ax.t_plunge):
        assert ((pl >= 0) & (pl <= 90)).all()
    for az in (ax.p_azimuth, ax.b_azimuth, ax.t_azimuth):
        assert ((az >= 0) & (az < 360)).all()
    total = sum(
        np.sin(np.radians(p)) ** 2 for p in (ax.p_plunge, ax.b_plunge, ax.t_plunge)
    )
    assert total == pytest.approx(np.ones(200), abs=1e-9)
    for i in range(0, 200, 17):
        p = vec(ax.p_azimuth[i], ax.p_plunge[i])
        b = vec(ax.b_azimuth[i], ax.b_plunge[i])
        t = vec(ax.t_azimuth[i], ax.t_plunge[i])
        assert abs(p @ b) < 1e-9 and abs(p @ t) < 1e-9 and abs(b @ t) < 1e-9


def test_pt_axes_plane_swap_and_wrap() -> None:
    rng = np.random.default_rng(3)
    s, d, r = (
        rng.uniform(0, 360, 100),
        rng.uniform(1, 89, 100),
        rng.uniform(-179, 179, 100),
    )
    s2, d2, r2 = auxiliary_plane(s, d, r)
    a = mm.pt_axes(s, d, r)
    b = mm.pt_axes(s2, d2, r2)
    w = mm.pt_axes(s - 360.0, d, r + 720.0)
    for x, y, z in zip(a, b, w, strict=True):
        assert np.allclose(x, z, atol=1e-7)
    for i in range(100):
        for k in (0, 2, 4):
            assert axis_angle((a[k][i], a[k + 1][i]), (b[k][i], b[k + 1][i])) < 1e-5


def test_pt_axes_nan_and_scalars() -> None:
    ax = mm.pt_axes(np.nan, 45, 90)
    assert all(np.isnan(float(v)) for v in ax)
    assert np.ndim(mm.pt_axes(10, 45, 90).p_plunge) == 0
    arr = mm.pt_axes([10, np.nan], 45, [90, 0])
    assert np.isnan(arr.t_plunge[1]) and arr.t_plunge[0] == pytest.approx(90)


def test_pt_axes_match_sil_blocks() -> None:
    """SIL's own P/T axes (dip may be negative = the upward end) agree with
    the axes computed from the converted planes, within the whole-degree
    rounding of the block."""
    n = 0
    for day in REVIEWED_DAYS:
        fps = read_events_fps(SIL / day / "events.fps").frame
        for _, row in fps.loc[fps["has_solution"].to_numpy(bool)].iterrows():
            b = mm.read_sp3(
                mm.block_path(SIL / day, row["id"], row["source_file"])
            ).block
            assert b.has_planes and b.p_axis_azimuth is not None
            (s, d, r), _ = b.planes_aki_richards()
            ax = mm.pt_axes(s, d, r)
            assert b.p_axis_dip is not None and b.t_axis_dip is not None
            p_off = axis_angle(
                (float(ax.p_azimuth), float(ax.p_plunge)),
                (b.p_axis_azimuth, b.p_axis_dip),
            )
            t_off = axis_angle(
                (float(ax.t_azimuth), float(ax.t_plunge)),
                (float(b.t_axis_azimuth or 0.0), b.t_axis_dip),
            )
            assert p_off < 2.0 and t_off < 2.0, (row["id"], p_off, t_off)
            n += 1
    assert n == 22


# --------------------------------------------------------------------------
# faulting_class
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rule", sorted(mm.FAULTING_RULES))
@pytest.mark.parametrize(
    ("plane", "expected"),
    [
        ((0, 90, 0), "strike-slip"),
        ((37, 90, 180), "strike-slip"),
        ((0, 45, -90), "normal"),
        ((210, 60, -90), "normal"),
        ((0, 45, 90), "reverse"),
        ((120, 30, 90), "reverse"),
    ],
)
def test_pure_mechanisms(
    rule: str, plane: tuple[float, float, float], expected: str
) -> None:
    assert cls(*plane, rule=rule) == expected


@pytest.mark.parametrize(
    ("plane", "frohlich", "wsm"),
    [
        # oblique reverse: plunges P 19, B 42, T 42
        ((10, 45, 20), "odd", "reverse-oblique"),
        # oblique normal: plunges P 49, B 38, T 15
        ((10, 45, -30), "odd", "normal-oblique"),
        # steep oblique: P 51, B 29, T 24 — odd under both
        ((10, 75, -60), "odd", "odd"),
        # B plunge 65 > 60: strike-slip under both
        ((10, 75, 20), "strike-slip", "strike-slip"),
        # Frohlich's asymmetry: T > 50 is reverse, P needs > 60 for normal
        ((10, 30, 30), "reverse", "reverse"),
        ((10, 30, -30), "odd", "normal"),
    ],
)
def test_oblique_mechanisms(
    plane: tuple[float, float, float], frohlich: str, wsm: str
) -> None:
    assert cls(*plane) == frohlich
    assert cls(*plane, rule="wsm") == wsm


def test_class_plane_swap_and_strike_wrap() -> None:
    rng = np.random.default_rng(7)
    s, d, r = (
        rng.uniform(0, 360, 300),
        rng.uniform(1, 89, 300),
        rng.uniform(-179, 179, 300),
    )
    s2, d2, r2 = auxiliary_plane(s, d, r)
    for rule in mm.FAULTING_RULES:
        a = mm.faulting_class(s, d, r, rule=rule)
        assert list(a) == list(mm.faulting_class(s2, d2, r2, rule=rule))
        assert list(a) == list(mm.faulting_class(s + 360.0, d, r - 360.0, rule=rule))
        assert list(a) == list(mm.faulting_class(s - 720.0, d, r, rule=rule))
        assert set(a) <= set(mm.FAULTING_RULES[rule])
    # the legacy list's raw strikes (−72, 434) classify as their wraps
    assert cls(-72, 60, -80) == cls(288, 60, -80) == "normal"
    assert cls(434, 80, 10) == cls(74, 80, 10) == "strike-slip"


def test_frohlich_classes_are_exclusive() -> None:
    """sin² of the plunges sums to 1, so at most one threshold can hold."""
    rng = np.random.default_rng(9)
    s, d, r = (
        rng.uniform(0, 360, 2000),
        rng.uniform(0, 90, 2000),
        rng.uniform(-180, 180, 2000),
    )
    ax = mm.pt_axes(s, d, r)
    hits = (
        (ax.b_plunge > 60).astype(int)
        + (ax.p_plunge > 60).astype(int)
        + (ax.t_plunge > 50).astype(int)
    )
    assert hits.max() <= 1


def test_class_nan_bad_rule_and_shapes() -> None:
    assert cls(np.nan, 45, 90) is None
    out = mm.faulting_class([0, np.nan], [90, 45], [0, 90])
    assert out.dtype == object and list(out) == ["strike-slip", None]
    assert np.ndim(mm.faulting_class(0, 90, 0)) == 0
    with pytest.raises(ValueError, match="rule"):
        mm.faulting_class(0, 90, 0, rule="aki")


def test_faulting_classes_series() -> None:
    df = mm.read_sil_day_mechanisms(SIL / "2024/aug/22").frame
    for rule, cats in mm.FAULTING_RULES.items():
        s = mm.faulting_classes(df, rule=rule)
        assert s.index.equals(df.index) and s.name == "faulting_class"
        assert list(s.cat.categories) == list(cats)
        assert s.notna().all()
    empty = mm.faulting_classes(df.iloc[0:0])
    assert len(empty) == 0 and isinstance(empty.dtype, pd.CategoricalDtype)


# --------------------------------------------------------------------------
# SIL's label: sil_faulting_class
# --------------------------------------------------------------------------


def fixture_mechanisms() -> pd.DataFrame:
    return pd.concat(
        [mm.read_sil_day_mechanisms(SIL / d, extras=True).frame for d in REVIEWED_DAYS]
    )


def test_sil_faulting_class_populated() -> None:
    df = fixture_mechanisms()
    assert len(df) == 22
    assert df["sil_faulting_class"].notna().all()
    assert set(df["sil_faulting_class"].astype(str)) <= set(SIL_FAULTING_CLASSES)
    assert validate_mechanisms(df).ok
    # the deprecated extra repeats the schema column
    assert list(df["faulting_class"].astype(str)) == list(
        df["sil_faulting_class"].astype(str)
    )
    counts = df["sil_faulting_class"].value_counts().to_dict()
    assert counts == {"strike-slip": 13, "normal": 4, "reverse": 4, "dip-slip": 1}


def test_sil_faulting_class_without_extras() -> None:
    df = mm.read_sil_day_mechanisms(SIL / "2024/aug/22").frame
    assert "faulting_class" not in df.columns
    assert df.iloc[0]["sil_faulting_class"] == "strike-slip"


def test_sil_faulting_class_na_without_block() -> None:
    fps = read_events_fps(SIL / "2024/aug/22/events.fps").frame
    df = mm.mechanisms_from_fps(fps)
    assert df["sil_faulting_class"].isna().all()


def test_sil_vs_computed_cross_table() -> None:
    """Pins the fixture cross-table (the report's numbers). SIL calls more
    mechanisms strike-slip than Frohlich's B > 60° does; it never puts a
    mechanism in a different pure class."""
    df = fixture_mechanisms()
    sil = df["sil_faulting_class"].astype(str)
    fro = mm.faulting_classes(df).astype(str)
    tab = pd.crosstab(sil, fro)
    assert tab.loc["strike-slip"].to_dict() == {
        "normal": 0,
        "odd": 7,
        "reverse": 0,
        "strike-slip": 6,
    }
    pure = {"strike-slip", "normal", "reverse"}
    both = sil.isin(pure) & fro.isin(pure)
    assert (sil[both] == fro[both]).all()


# --------------------------------------------------------------------------
# planes_inconsistent: kept and marked
# --------------------------------------------------------------------------


def inconsistent_fps() -> pd.DataFrame:
    """The 2024-08-22 fps with row 0 given the 2023 placeholder pair
    (180/39/−79 and 14/52/−81, 22° apart) and row 1 without plane b."""
    fps = read_events_fps(SIL / "2024/aug/22/events.fps").frame.copy()
    fps.loc[0, ["strike_a", "dip_a", "rake_a"]] = (180, 39, -79)
    fps.loc[0, ["strike_b", "dip_b", "rake_b"]] = (14, 52, -81)
    fps.loc[1, ["strike_b", "dip_b", "rake_b"]] = np.nan
    return fps


def test_inconsistent_row_is_kept_and_marked() -> None:
    fps = inconsistent_fps()
    n = int(fps["has_solution"].sum())
    df = mm.mechanisms_from_fps(fps)
    assert len(df) == n
    flag = df[mm.PLANES_INCONSISTENT]
    assert flag.dtype == "boolean"
    assert bool(flag.iloc[0]) is True
    assert pd.isna(flag.iloc[1])
    assert not flag.iloc[2:].any()
    assert (df.iloc[0]["strike2"], df.iloc[0]["dip2"], df.iloc[0]["rake2"]) == (
        14,
        52,
        -81,
    )
    # marked: strict validation passes, the issue stays visible as a warning
    rep = mm.validate_marked_mechanisms(df, strict=True)
    assert rep.ok and "planes_inconsistent" in {i.kind for i in rep.warnings}


def test_unmarked_inconsistent_row_still_fails() -> None:
    df = mm.mechanisms_from_fps(inconsistent_fps())
    df[mm.PLANES_INCONSISTENT] = pd.array([False] * len(df), dtype="boolean")
    with pytest.raises(SchemaError, match="auxiliary"):
        mm.validate_marked_mechanisms(df, strict=True)


def test_marker_tolerance_and_columns() -> None:
    df = mm.mechanisms_from_fps(inconsistent_fps(), plane_tol_deg=30.0)
    assert not bool(df[mm.PLANES_INCONSISTENT].iloc[0])
    assert list(df.columns) == list(mm.output_columns())
    assert mm.PLANES_INCONSISTENT in mm.output_columns()


def test_every_fixture_day_carries_the_marker() -> None:
    for day in (*REVIEWED_DAYS, "1993/jan/29", "2026/sep/10"):
        for extras in (False, True):
            f = mm.read_sil_day_mechanisms(SIL / day, extras=extras).frame
            assert mm.PLANES_INCONSISTENT in f.columns
            assert not f[mm.PLANES_INCONSISTENT].fillna(False).any()
            mm.validate_marked_mechanisms(f, strict=True)


def test_day_with_inconsistent_row_reads_strictly(tmp_path: Path) -> None:
    """A whole day directory whose fps row has a mutually inconsistent pair:
    the reader keeps it, marks it, and a strict (marked) check passes."""
    day = tmp_path / "2024/aug/22"
    shutil.copytree(SIL / "2024/aug/22", day)
    fps = day / "events.fps"
    lines = fps.read_text(encoding="latin-1").splitlines(keepends=True)
    first = next(i for i, ln in enumerate(lines) if "10:05:430" in ln)
    row = read_events_fps(SIL / "2024/aug/22/events.fps").frame.iloc[0]
    old = f"{int(row['strike_b']):4d}{int(row['dip_b']):4d}{int(row['rake_b']):5d}"
    if old not in lines[first]:
        pytest.skip("fps plane-b field layout differs from the assumed width")
    lines[first] = lines[first].replace(old, f"{14:4d}{52:4d}{-81:5d}", 1)
    fps.write_text("".join(lines), encoding="latin-1")
    res = mm.read_sil_day_mechanisms(day, extras=True)
    f = res.frame
    hit = f[f["event_id"].str.endswith("10:05:430")].iloc[0]
    assert (hit["strike2"], hit["dip2"], hit["rake2"]) == (14, 52, -81)
    assert bool(hit[mm.PLANES_INCONSISTENT]) is True
    assert len(f) == 8
    mm.validate_marked_mechanisms(f, strict=True)
