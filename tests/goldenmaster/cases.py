"""Golden-master case registry for gps_read (plan §15.6, refactor risk R1/R5).

Every case pins the CURRENT behavior of `openGlobkTimes` / `getData` /
`gamittoNEU` / `gamittooneuf` / `read_gps_data` on frozen fixture data, so the
Phase 1 geo_dataread/gps_analysis refactor can prove it changed nothing.

The argument profiles mirror the 2026-07-08 call-site survey:

- `gps-savetimes` (LIVE, feeds cdn.vedur.is .NEU files consumed by aflogun):
  ``gamittooneuf(sta, outfile, mm=True, ref=..., dstring=None, rhour=True)``
- `gps-displacemnts` / `gps_displ.simpleDisp` (LIVE):
  ``gamittoNEU(sta, mm=True, ref="plate", dstring="%Y/%m/%d 12:00:00.000")``
- `gps_plot.timesmatplt.plotTime` (LIVE):
  ``getData(sta, fstart=..., fend=..., ref=..., Dir=..., tType="TOT", uncert=...)``
- research workflows (detrend-*, svartsengi-model, pygmt_test, oraefajokull):
  the `read_gps_data` profiles, including the useSTA/useFIT borrowing branch
  and the missing-detrend-CSV fallback branch.

Known quirks pinned ON PURPOSE (do not "fix" without a decision):

- `openGlobkTimes` does ``skiprows=3`` although the fixture mb_ files have no
  header lines — the first 3 epochs are silently dropped.
- `gamittoNEU` subtracts plate velocity in METERS before the optional mm
  conversion; `getData` subtracts ``plateVelo * 1000`` on mm data (post
  `iprep`). Same idea, different unit order — both pinned as-is.
- With gps_parser 0.3.0 (the pinned git dep) `detrendFile` resolves relative
  to the CURRENT WORKING DIRECTORY, so `read_gps_data` cases chdir explicitly.
- The detrend CSV columns are ``UseSTA``/``Fit`` while getDetrFit writes
  lowercase ``useSTA``/``fit`` — the useSTA branch therefore grows extra
  columns. Pinned as-is.
"""

import contextlib
import datetime as dt
import os
from pathlib import Path

import numpy as np

import geo_dataread.gps_read as gpsr

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
TOT = str(FIXTURES / "TOT") + os.sep
EXPECTED = HERE / "expected"

POSTPROCESS_TEMPLATE = """\
[Configs]
totDir = {totdir}

[FILES]
detrendFile = detrend_itrf2008.csv
coordFile = {config_dir}/station_coord.llh
platefile = {config_dir}/station-plate
"""


def build_config_dir(dest: Path) -> Path:
    """Render a hermetic gpsconfig dir (for GPS_CONFIG_PATH) from fixtures."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in (
        "stations.cfg",
        "detrend_itrf2008.csv",
        "station-plate",
        "station_coord.llh",
    ):
        dest.joinpath(name).write_bytes((FIXTURES / "gpsconfig" / name).read_bytes())
    dest.joinpath("postprocess.cfg").write_text(
        POSTPROCESS_TEMPLATE.format(totdir=TOT, config_dir=dest)
    )
    return dest


def _structured_to_dict(neu):
    """Split a gamittoNEU structured array into plain arrays for .npz storage."""
    out = {}
    for field in neu.dtype.names:
        key = field.replace("[", "_").replace("]", "")
        out[key] = np.asarray(neu[field])
    return out


def _tuple_to_dict(yearf, data, Ddata, offset=None):
    out = {"yearf": yearf, "data": data, "Ddata": Ddata}
    if offset is not None:
        out["offset"] = np.asarray(offset)
    return out


def _frame_to_dict(df, const):
    out = {"index_ns": df.index.asi8}
    for col in df.columns:
        out[f"col_{col}"] = df[col].to_numpy()
    float_cols = [c for c in const.columns if const[c].dtype.kind == "f"]
    str_cols = [c for c in const.columns if c not in float_cols]
    out["const_float_names"] = np.array(float_cols)
    out["const_floats"] = const[float_cols].to_numpy(dtype=float)
    out["const_str_names"] = np.array(str_cols)
    out["const_strs"] = np.array(
        [[str(v) for v in const[c].to_numpy()] for c in str_cols]
    )
    return out


# --- case runners ----------------------------------------------------------
# Each runner: env -> dict[str, np.ndarray]. `env` has "config_dir" (Path to
# the rendered gpsconfig dir, which also holds the detrend CSV) and
# "empty_dir" (a directory WITHOUT the CSV, for the fallback branch).


def _ogt_seng(env):
    yearf, data, Ddata = gpsr.openGlobkTimes("SENG", Dir=TOT)
    return _tuple_to_dict(yearf, data, Ddata)


def _getdata(sta, **kwargs):
    def run(env):
        yearf, data, Ddata, offset = gpsr.getData(sta, Dir=TOT, **kwargs)
        return _tuple_to_dict(yearf, data, Ddata, offset)

    return run


def _gamittoneu(sta, **kwargs):
    def run(env):
        return _structured_to_dict(gpsr.gamittoNEU(sta, Dir=TOT, **kwargs))

    return run


def _read_gps_data(sta, csv_present=True, **kwargs):
    def run(env):
        cwd = env["config_dir"] if csv_present else env["empty_dir"]
        with contextlib.chdir(cwd):
            df, const = gpsr.read_gps_data(sta, Dir=TOT, **kwargs)
        return _frame_to_dict(df, const)

    return run


CASES = {
    # -- openGlobkTimes: raw reader incl. the skiprows=3 quirk
    "ogt_SENG": _ogt_seng,
    # -- getData: legacy-dominant and gps_plot profiles
    "getdata_SENG_plate": _getdata("SENG", ref="plate"),
    "getdata_SENG_plate_uncert20": _getdata(
        "SENG", ref="plate", tType="TOT", uncert=20
    ),
    "getdata_SENG_window": _getdata(
        "SENG", fstart=2020.0, fend=2024.5, ref="plate"
    ),
    "getdata_SENG_itrf2008": _getdata("SENG"),
    "getdata_ELDC_plate": _getdata("ELDC", ref="plate"),
    "getdata_SKSH_plate": _getdata("SKSH", ref="plate"),
    # -- gamittoNEU: LIVE simpleDisp + gps-savetimes inner profiles
    "neu_SENG_simpledisp": _gamittoneu(
        "SENG", mm=True, ref="plate", dstring="%Y/%m/%d 12:00:00.000"
    ),
    "neu_SENG_yearf": _gamittoneu("SENG", mm=True, ref="plate", dstring="yearf"),
    "neu_SENG_savetimes": _gamittoneu(
        "SENG", mm=True, ref="plate", dstring=None, rhour=True
    ),
    "neu_SENG_meters_itrf": _gamittoneu(
        "SENG", mm=False, ref="itrf2008", dstring="yearf"
    ),
    # ENTC has 3 epochs with sigma in (1.1 m, 20 m) — the only case that
    # pins gamittoNEU's hardcoded vshift(uncert=1.1) outlier threshold.
    "neu_ENTC_savetimes": _gamittoneu(
        "ENTC", mm=True, ref="plate", dstring=None, rhour=True
    ),
    # -- read_gps_data: research-workflow profiles
    "rgd_SENG_defaults": _read_gps_data("SENG"),
    "rgd_SENG_svartsengi": _read_gps_data(
        "SENG",
        start=dt.datetime(2020, 11, 1),
        end=dt.datetime(2024, 7, 1),
        uncert=20.0,
        detrend_periodic=True,
    ),
    "rgd_SENG_nodetrend": _read_gps_data("SENG", detrend_periodic=False),
    "rgd_SENG_no_csv": _read_gps_data("SENG", csv_present=False),
    "rgd_SENG_detrendline_list": _read_gps_data(
        "SENG", detrend_line=[2020.0, 2024.0]
    ),
    "rgd_OLAC_useSTA_lineperiodic": _read_gps_data(
        "OLAC", useSTA="DYNG", useFIT="lineperiodic"
    ),
    "rgd_SKSH_fit_lineperiodic": _read_gps_data("SKSH", fit="lineperiodic"),
}

# Behavior pinned as a CRASH: getDetrFit builds lowercase 'fit'/'useSTA'
# columns while read_gps_data reads const["Fit"]/const["useSTA"] — so the
# useFIT="periodic" borrowing branch raises KeyError('Fit') today. The pygmt
# scripts that exercised it are broken against current geo_dataread. If the
# refactor fixes this, delete this pin deliberately and note it in the log.
CRASH_CASES = {
    "rgd_useSTA_periodic_raises": (
        _read_gps_data("OLAC", useSTA="DYNG", useFIT="periodic"),
        KeyError,
    ),
    # openGlobkTimes' 08h branch references undefined names (todatetime,
    # shiftime, timetoyearf — wrong case) — so gps_plot's tType="JOIN"
    # sub-daily branch is dead in current code. NameError pinned; the
    # fixture mb_SENG_08h.dat* files (TOT copies) exist only so execution
    # reaches the broken block instead of dying on a missing file.
    "ogt_08h_raises": (
        lambda env: gpsr.openGlobkTimes("SENG", Dir=TOT, tType="08h"),
        NameError,
    ),
}

# .NEU file cases: the exact production gps-savetimes call. Golden = the file.
NEU_FILE_CASES = {
    "neufile_SENG_plate": ("SENG", "plate"),
    "neufile_ELDC_plate": ("ELDC", "plate"),
}


def run_neu_file_case(case_id, out_dir: Path) -> Path:
    sta, ref = NEU_FILE_CASES[case_id]
    out = out_dir / f"{sta}-{ref}.NEU"
    gpsr.gamittooneuf(
        sta, str(out), mm=True, ref=ref, dstring=None, rhour=True, Dir=TOT
    )
    return out
