"""
This module contains functions for reading and returning GPS data. It includes the following functions

"""

import logging
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import geofunc.geofunc as gf
import numpy as np
import pandas as pd
from gps_analysis import baseline as ga_baseline
from gps_analysis import fitting as ga_fitting
from gps_analysis import models as ga_models
from gps_analysis import preprocess as ga_preprocess
from gps_analysis.models import TrajectoryParams
from gps_parser import ConfigParser

from gtimes.timefunc import (
    TimefromYearf,
    TimetoYearf,
    convfromYearf,
    currYearfDate,
    round_to_hour,
)

#
# time series filtering
#


def line(x, p0, p1):
    """Deprecated shim: linear (secular) model.

    The math lives in :func:`gps_analysis.models.linear` (refactor-B
    slice 7); this wrapper keeps the legacy name/signature. Bit-identical
    to the old ``p0 + p1*x`` expression (same float64 arithmetic, verified
    bitwise over a wide parameter sweep); the only change is the return
    type for scalar ``x`` (0-d ndarray instead of a Python scalar — all
    callers pass arrays).

    Args:
        x: input value in yearf or ordinal (or any numeric time)
        p0: intercept
        p1: slope

    Returns:
        output value, float64 (same shape as ``x``)
    """
    return ga_models.linear(x, p0, p1)


def lineperiodic(
    x: float, p0: float, p1: float, p2: float, p3: float, p4: float, p5: float
) -> Optional[float]:
    """
    linear function with a periodic superimposed

    examples:
        >>> lineperiodic(1, 0, 1)
        1

    args:
        x: input value in yearf or ordinal (or any numeric time)
        p0: intercept
        p1: slope of linear term
        p2: slope of cosine semiannual term
        p3: slope of sine semiannual term
        p4: slope of cosine annual term
        p5: slope of sine annual term

    returns:
        output value

    Note:
        NOT delegated to :func:`gps_analysis.models.lineperiodic`
        (refactor-B slice 7 decision): the leaf evaluates
        ``linear(t, …) + periodic(t, …)``, whose float64 association
        differs from this legacy single expression by up to 1 ulp —
        measured 300/300 mismatching trials on representative
        yearf/parameter sweeps. This function feeds
        ``scipy.optimize.curve_fit`` on golden-pinned paths
        (``read_gps_data`` detrend/fit cases), so the evaluation order is
        pinned verbatim here for bit-parity. Consolidating it into the
        leaf is a follow-up for whenever the goldens are deliberately
        re-baselined (or the leaf grows a legacy-association evaluator).
    """

    return (
        p0
        + p1 * x
        + p2 * np.cos(2 * np.pi * x)
        + p3 * np.sin(2 * np.pi * x)
        + p4 * np.cos(4 * np.pi * x)
        + p5 * np.sin(4 * np.pi * x)
    )


def periodic(x, p0, p1, p2, p3, p4, p5):
    """Deprecated shim: seasonal (annual + semiannual) model.

    The math lives in :func:`gps_analysis.models.periodic` (refactor-B
    slice 7); this wrapper keeps the legacy 7-argument signature, in which
    ``p0``/``p1`` are accepted — and ignored — so that full
    :func:`lineperiodic` parameter vectors can be reused (the
    ``read_gps_data`` fit paths rely on this). Bit-identical to the old
    expression: the leaf's ``2.0 * two_pi_t`` semiannual argument equals
    the legacy ``4*np.pi*x`` exactly (power-of-two scaling) and the terms
    sum in the same order (verified bitwise over a wide parameter sweep).

    Args:
        x: input value in yearf or ordinal (or any numeric time)
        p0: intercept — IGNORED (legacy signature compatibility)
        p1: slope — IGNORED (legacy signature compatibility)
        p2: annual cosine amplitude
        p3: annual sine amplitude
        p4: semiannual cosine amplitude
        p5: semiannual sine amplitude

    Returns:
        output value, float64 (same shape as ``x``)
    """
    return ga_models.periodic(x, p2, p3, p4, p5)


def xf(x, p0, p1, p2, tau=4.8):
    """Deprecated shim: linear + exponential transient, fixed decay.

    The math lives in :func:`gps_analysis.models.exp_linear` (refactor-B
    slice 7); this wrapper keeps the legacy name/signature including the
    ``tau=4.8`` default decay constant. Bit-identical to the old
    ``p0 + p1*x + p2*np.exp(-tau*x)`` expression (identical arithmetic
    and association, verified bitwise over a wide parameter sweep).

    Args:
        x: input value in yearf or ordinal (or any numeric time)
        p0: intercept
        p1: slope
        p2: exponential term coefficient
        tau: exponential decay constant [1/yr]. Default 4.8

    Returns:
        output value, float64 (same shape as ``x``)
    """
    return ga_models.exp_linear(x, p0, p1, p2, tau)


def expxf(x, p0, p1, p2, p3):
    """Deprecated shim: linear + exponential transient, fitted decay.

    The math lives in :func:`gps_analysis.models.exp_linear` (refactor-B
    slice 7) — identical signature semantics, so this is a direct
    pass-through. Bit-identical to the old
    ``p0 + p1*x + p2*np.exp(-p3*x)`` expression (identical arithmetic and
    association, verified bitwise over a wide parameter sweep).

    Args:
        x: input value in yearf or ordinal (or any numeric time)
        p0: intercept
        p1: slope
        p2: exponential term coefficient (amplitude)
        p3: exponential decay constant

    Returns:
        output value, float64 (same shape as ``x``)
    """
    return ga_models.exp_linear(x, p0, p1, p2, p3)


def expf(x, p0, p1, p2):
    """Deprecated shim: pure exponential relaxation (no secular term).

    The math lives in :func:`gps_analysis.models.exp_linear` (refactor-B
    slice 7), called with ``rate=0.0``. Bit-identical to the old
    ``p0 + p1*np.exp(-p2*x)`` expression for finite ``x`` (the leaf's
    extra ``0.0*x`` term is exact, verified bitwise over a wide parameter
    sweep). This is the legacy compatibility alias the leaf deliberately
    did not adopt (CONSOLIDATION_MAP decision 3 — new names only there,
    shims here).

    Args:
        x: input value in yearf or ordinal (or any numeric time)
        p0: offset
        p1: exponential amplitude
        p2: exponential decay constant

    Returns:
        output value, float64 (same shape as ``x``)
    """
    return ga_models.exp_linear(x, p0, 0.0, p1, p2)


def dexpf(x, p1, p2):
    """Deprecated shim: derivative of ``p0 + p1*exp(-p2*x)``.

    The math lives in :func:`gps_analysis.models.exp_linear_rate`
    (refactor-B slice 7), called with ``rate=0.0``. Bit-identical to the
    old ``-p1*p2*np.exp(-p2*x)`` expression (IEEE-exact sign/zero
    manipulation, verified bitwise over a wide parameter sweep).

    Args:
        x: input value in yearf or ordinal (or any numeric time)
        p1: exponential amplitude
        p2: exponential decay constant

    Returns:
        instantaneous rate, float64 (same shape as ``x``)
    """
    return ga_models.exp_linear_rate(x, 0.0, p1, p2)


def secondorder(x, p0, p1, p2):
    """Deprecated shim: degree-2 polynomial.

    The math lives in :func:`gps_analysis.models.poly2` (refactor-B
    slice 7); this wrapper keeps the legacy name. Bit-identical to the
    old ``p0 + p1*x + p2*x**2`` expression (identical arithmetic and
    association, verified bitwise over a wide parameter sweep). This is
    the legacy compatibility alias the leaf deliberately did not adopt
    (CONSOLIDATION_MAP decision 3 — new names only there, shims here).

    Args:
        x: input value in yearf or ordinal (or any numeric time)
        p0: offset
        p1: initial rate
        p2: quadratic coefficient (curvature)

    Returns:
        output value, float64 (same shape as ``x``)
    """
    return ga_models.poly2(x, p0, p1, p2)


def getDetrFit(
    sta: str,
    useSTA=None,
    useFIT=None,
    onlyPeriodic=False,
    detrendFile="",
    logging_level=logging.WARNING,
):
    """
    this function defines the type of detrending to be applied. it is based on the detrending parameters file, detrfile, which is stored in the gpsconfig directory of the gpsplot server.

    examples:
        >>> getDetrFit('gric', useSTA=None, useFIT=None, onlyperiodic=false, detrfile='detrend_itrf2008.csv')

    args:
        sta: station name
        useSTA: station name
        useFIT: fit name
        onlyperiodic: boolean
        detrfile: detrending file name, csv format
    """

    if useFIT == "periodic":
        # Deliberate dead-branch removal (refactor-B slice 4, decision D4):
        # the periodic-borrowing branch never worked — downstream
        # read_gps_data crashed with KeyError('Fit') on the detrend-CSV
        # column-case mismatch. Replaced with an explicit error.
        raise ValueError(
            "unsupported useFIT='periodic': the periodic-borrowing branch was "
            "dead (it crashed with KeyError('Fit')) and was removed in the "
            "refactor-B slice-4 cleanup; use 'lineperiodic' or 'line'"
        )

    # create an instance for the configparser
    config = ConfigParser()

    # handling logging
    logging.basicConfig(
        format="%(asctime)-15s [%(levelname)s] %(funcname)s: %(message)s",
        level=logging_level,
    )

    logging.getLogger().setLevel(logging_level)
    module_logger = logging.getLogger()

    # read the detrending file
    north = ["Nrate", "Nacos", "Nasin", "Nscos", "Nssin"]
    east = ["Erate", "Eacos", "Easin", "Escos", "Essin"]
    up = ["Urate", "Uacos", "Uasin", "Uscos", "Ussin"]

    # read the detrending file
    # headers
    info = ["sitename", "starttime", "endtime", "useSTA", "fit"]
    columns = north + east + up + info

    # grab the station name from the station.cfg file
    # Use station_name if available, otherwise fall back to station_id or station code
    station_info = config.getStationInfo(sta)["station"]
    sta_name = station_info.get("station_name", station_info.get("station_id", sta))

    # initialize the table as an array with nan values
    data = [[np.nan] * 15 + [sta_name, np.nan, np.nan, np.nan, np.nan]]

    # convert to dataframe
    table = pd.DataFrame(data, index=[sta], columns=columns)

    # read the detrending file
    if detrendFile == "":
        detrendFile = config.getPostProcessConfig("detrendFile")

    try:
        table = pd.read_csv(detrendFile, index_col="STA")
    except pd.errors.EmptyDataError:
        module_logger.warning("{} does not contain any data".format(detrendFile))
    except FileNotFoundError:
        module_logger.warning("detrend file not found")

    # set the fit constants from the file within const variable
    try:
        const = pd.DataFrame(index=[sta], columns=columns)
        const.loc[sta] = tuple(table.loc[sta].values)
    except KeyError:
        module_logger.warning("{} not in table making an empty row".format(sta))
        const = pd.DataFrame(data, index=[sta], columns=table.columns)

    # set the fit constants from the other station fit
    if useSTA:
        module_logger.warning(
            "setting {0} {1} constants as {1} parameters for {2}".format(
                useSTA, useFIT, sta
            )
        )
        module_logger.info("current {} constants are:\n {} ".format(sta, const))

        # set the fit constants if the fit is done from another station
        const.loc[sta, ["useSTA", "fit"]] = useSTA, useFIT

        # set the fit constants
        if useFIT == "lineperiodic":
            const.loc[sta, north + east + up] = table.loc[useSTA, north + east + up]
        elif useFIT == "line":
            const.loc[sta, north[:1] + east[:1] + up[:1]] = table.loc[
                useSTA, north[:1] + east[:1] + up[:1]
            ]

        module_logger.info("new parameters for {} are:\n {}".format(sta, const))

    # set the linear part of the fit constants as nan for periodic only fit
    if onlyPeriodic is True:
        const["Nrate"] = const["Erate"] = const["urate"] = np.nan

    return const


def convconst(const, pb=None):
    """
    this function converts detrending constants from pandas dataframe to list of the form
    [[], [], []] for optimize.curv_fit

    examples:
        >>> convconst()

    args:
        const: detrend constants dataframe
        pb: periodic constants

    returns:
        const: detrending constants

    """

    north = ["Nrate", "Nacos", "Nasin", "Nscos", "Nssin"]
    east = ["Erate", "Eacos", "Easin", "Escos", "Essin"]
    up = ["Urate", "Uacos", "Uasin", "Uscos", "Ussin"]

    if pb:
        # const = pd.dataframe(data, index=[sta], columns=table.columns)
        stat = const.index.tolist()[0]
        for i, dimention in zip(range(3), [north, east, up]):
            const.loc[stat, dimention] = list(pb[i][1:])

        return const

    else:
        if const[north].isna().sum(axis=1).values == 5:
            p0 = [None, None, None]
        else:
            p0 = [[], [], []]
            p0[0] = [0] + const[north].values.tolist()[0]
            p0[1] = [0] + const[east].values.tolist()[0]
            p0[2] = [0] + const[up].values.tolist()[0]

            p0 = np.nan_to_num(p0)

        return p0


def save_detrend_const(const, detrendFile=""):
    """
    this function saves constants in a file for detrending gps stations

    examples:
        >>> save_detrend_const(const, detrfile='detrend_itrf2008.csv')

    args:
        const: detrending constants dataframe
        detrfile: detrending file name

    returns:
        None


    """

    north = ["Nrate", "Nacos", "Nasin", "Nscos", "Nssin"]
    east = ["Erate", "Eacos", "Easin", "Escos", "Essin"]
    up = ["Urate", "Uacos", "Uasin", "Uscos", "Ussin"]
    info = ["Sitename", "Starttime", "Endtime", "useSTA", "Fit"]

    columns = north + east + up + info
    stationlist = const.index.values

    # read the detrending file
    if detrendFile == "":
        config = ConfigParser()
        detrendFile = config.getPostProcessConfig("detrendFile")

    # path = path(path.cwd(), detrendFile)

    if not os.path.isfile(detrendFile):
        open(detrendFile, "w").close()

    try:
        table = pd.read_csv(detrendFile, index_col="STA")
    except pd.errors.EmptyDataError:
        table = pd.DataFrame(index=["STA"], columns=columns)

    print(table)
    if table.dropna(how="all").empty:
        print("no data in file")
        const.to_csv(detrendFile, mode="w", header=True, index_label="STA")

        return

    else:
        for stat in stationlist:
            if stat in table.index:
                print("station {} already in table updating".format(stat))
                table.loc[stat, :] = const.loc[stat]
                print("saving the new table")
                table.to_csv(detrendFile, mode="w", header=True, index_label="STA")
            else:
                print("stations {} are not in detrend file, adding it".format(stat))
                table.loc[stat, :] = const.loc[stat]
                table.to_csv(detrendFile, mode="w", header=True, index_label="STA")

        return


def fitDataFrame(func, df, p0=[None, None, None]):
    """
    this function finds the fitting parameters for a dataframe of gps time series through the scipy.optimize.curve_fit function

    examples:
        >>> fitdataframe(line, df, p0=[None, None, None])

    args:
        func: fitting function (line, periodic, lineperiodic, others)
        df: dataframe of gps time series, with x in fractional year, y the components dn, de or du and yd, the uncertainties dn, de, du.
        p0: initial guess for the fitting parameters

    returns:
        pb: fitting parameters
        pcov: covariance matrix

    """

    x = df["yearf"].to_numpy()
    y = df[["north", "east", "up"]].to_numpy().T
    yd = df[["Dnorth", "Deast", "Dup"]].to_numpy().T

    return fittimes(func, x, y, yd, p0=p0)


def fittime(func, x, y, yd=None, p0=None):
    """Deprecated shim: single-component fit.

    The math lives in :func:`gps_analysis.fitting.fit_components`
    (refactor-B slice 1); this wrapper keeps the legacy name/signature and
    the legacy ``maxfev=100000``. Bit-identical to the old direct
    ``scipy.optimize.curve_fit`` call (same invocation).
    """
    fit = ga_fitting.fit_components(func, x, y, sigma=yd, p0=p0, maxfev=100000)[0]

    return fit.params, fit.covariance


def fittimes(func, x, y, yd=[None, None, None], p0=[None, None, None]):
    """Deprecated shim: per-component (N/E/U) fit loop.

    The math lives in :func:`gps_analysis.fitting.fit_components`
    (refactor-B slice 1); this wrapper keeps the legacy name/signature,
    the legacy ``maxfev=100000``, the (pb, pcov) list-of-3 return shape,
    and the legacy per-component ``p0``/``yd`` semantics (each entry may
    independently be None). Bit-identical to the old loop — the underlying
    ``scipy.optimize.curve_fit`` invocation is unchanged.
    """

    pb = [[], [], []]
    pcov = [[], [], []]

    for i in range(3):
        fit = ga_fitting.fit_components(
            func, x, y[i], sigma=yd[i], p0=p0[i], maxfev=100000
        )[0]
        pb[i], pcov[i] = fit.params, fit.covariance

    return pb, pcov


# routines to extract and save coordinates and time series from gamit


def gamittooneuf(
    sta,
    outfile,
    mm=True,
    ref="plate",
    dstring=None,
    outformat=True,
    reference="ITRF2008",
    Dir=None,
    rhour=False,
):
    """
    extract gamit timeseries from standard format to one file formated time string

    examples:
        >>> gamittooneuf('gric', outfile, mm=True, ref='plate', dstring=None, outformat=True)

    args:
        sta: station four letter short name
        outfile: file object
        mm: boolean True return data in mm, else in m
        ref: subtract plate velocity from the time series.
        dstring: format of the time string (e.g dstring)="%y%m%d-%h%m%s"), defaults to decimal year (yyyy.yyyy)
        reference: reference frame to use. Default="ITRF2008"


    returns: #not finished but determines the output order of the data into the file
        .outformat file

    """
    # "%y/%m/%d 12:00:00.000"

    neudata = gamittoNEU(
        sta, mm=mm, ref=ref, dstring=dstring, reference=reference, Dir=Dir, rhour=rhour
    )

    gamittoFile(neudata, outfile, mm=mm, ref=ref, dstring=dstring, outformat=outformat)


def gamittoFile(neudata, outfile, mm=True, ref="plate", dstring=None, outformat=True):
    """
    this function converts gamit timeseries files to neu files

    examples:
        >>> gamittoFile(neudata, outfile, mm=True, ref='plate', dstring=None, outformat=True)

    args:
        neudata: neu data
        outfile: file object
        mm: boolean True return data in mm, else in m
        ref: subtract plate velocity from the time series if = ref.
        dstring: format of the time string (e.g dstring)="%y%m%d-%h%m%s"), defaults to decimal year (yyyy.yyyy)
        outformat: file format (True) or string format (false)

    returns:
        .outformat file


    """

    # formatting time column
    if dstring == "yearf":
        timef = "{0: 8.5f}\t"
        timeh = "#\"yyyy.ddddd'    "
    else:
        timef = " {0:s}\t"
        timeh = "#yyyy/mm/dd hh:mm:ss.sss          "

    # unit conversion formatting
    if mm:
        if outformat:
            header = timeh + "dn[mm] dn[mm]\tde[mm] de[mm]\tdu[mm]  du[mm]"
            formatstr = (
                timef + "{1: 7.2f} {4: 7.2f}\t{2: 7.2f} {5: 7.2f}\t{3: 7.2f} {6: 7.2f}"
            )
        else:
            header = timeh + "dn[mm]  de[mm] du[mm]\t\t  dn[mm]  de[mm]  du[mm]"
            formatstr = (
                timef
                + "{1: 7.2f} {2: 7.2f} {3: 7.2f}\t\t{4: 7.2f} {5: 7.2f}\t{6: 7.2f}"
            )
    else:
        if outformat:
            header = timeh + "dn[m] dn[m]\tde[m] de[m]\tdu[m]  du[m]"
            formatstr = (
                timef + "{1: 7.5f} {4: 7.5f}\t{2: 7.5f} {5: 7.5f}\t{3: 7.5f} {6: 7.5f}"
            )
        else:
            header = timeh + "dn[m]    de[m]    du[m]          dn[m]   de[m]    du[m]"
            formatstr = (
                timef + "{1: 7.5f} {2: 7.5f} {3: 7.5f}\t{4: 7.5f} {5: 7.5f} {6: 7.5f}"
            )

    if outfile == sys.stdout:
        f = outfile
        print("printing header and file \n", header, file=f)

    else:
        f = open(outfile, "w")
        print("printing header and file \n", header, file=f)

    # applying formatting to each column of the neudata file
    for x in neudata:
        print(
            formatstr.format(
                x["yearf"].decode() if isinstance(x["yearf"], bytes) else x["yearf"],
                x["data[0]"],
                x["data[1]"],
                x["data[2]"],
                x["Ddata[0]"],
                x["Ddata[1]"],
                x["Ddata[2]"],
            ),
            file=f,
        )
    if f is not sys.stdout:
        f.close()


def extractfromgamitbakf(cfile, stations):
    """
    function to extract data from a gamit .bak file

    examples:
        >>> extractfromgamitbakf(cfile, stations)

    args:
        cfile: .bak file name
        stations: station name

    returns:
        slines: list of lines
    """

    slines = []

    site = re.compile(stations)
    tim = re.compile("solution refers to", re.ignorecase)
    f = open(cfile, "r")

    for line in f:
        if site.search(line):  # or tim.search(line):
            slines.append(line.rstrip())

    return slines


def openGlobkTimes(sta, Dir=None, tType="TOT"):
    """
    function to import data from globk time series files into a numpy arrays

    dir is the directory containing the time series if left blank the default path will be the path
    defined in the config file postprossesing.cfg, totDir

    Any GLOBK processing scheme whose files follow the
    ``mb_<STA>_<tType>.dat{1,2,3}`` naming convention can be read: ``TOT``
    (24 h daily solutions), ``08h`` (8-hourly solutions, ~3 epochs/day,
    revived in refactor-B slice 6), and extensibly e.g. ``04h``. All schemes
    share the identical file format — 3 header lines followed by
    ``yearf value sigma`` rows — so no scheme needs special time handling:
    sub-daily epochs are already encoded as fractional years.

    args:
        sta: station four letter short name in captial letters
        Dir: optional alternative directory of the gamit time series data.
        tType: GLOBK processing scheme, i.e. the ``<tType>`` part of the
            ``mb_<STA>_<tType>.dat{1,2,3}`` file names. Default "TOT"
            (daily); "08h" reads the 8-hourly solutions. A scheme whose
            files are absent raises FileNotFoundError (only ``tType="TOT"``
            retains the legacy fallback to lowercase ``mb_<STA>_tot.dat*``).

    returns:

        yearf: is array with time  (usually) in fractional year format (i.e. 2014.62328)
        data:  three arrays containing gps data in north,east, up
        ddata: respective uncertainty values

    """

    config = ConfigParser()

    yearf = []

    # loading the data
    # first, if no dir is input, find the path in the postprocess config file in the "totDir" instance
    if Dir is None:
        Dir = config.getPostProcessDir("totDir")
    else:  # this is still in development
        # print(f"DIR: {Dir}")
        # print( os.path.isdir(Dir) )
        # check if the path given as argument indeed exists
        pass

    # constructing the full path filenames and parsing parameters
    # define the file name to be looked for into dir
    filepre = "mb_{0:s}_{1:s}.dat".format(sta, tType)

    if os.path.isfile(
        os.path.join(Dir, filepre + "1")
    ):  # use dat1, dat2, dat3 as file format
        pass
    elif tType == "TOT":  # legacy fallback: lowercase tot as file format
        filepre = "mb_{0:s}_{1:s}.dat".format(sta, "tot")

    if not os.path.isfile(os.path.join(Dir, filepre + "1")):
        # a scheme whose files genuinely don't exist is rejected up front
        # with a clear error instead of a bare np.loadtxt failure (and a
        # non-TOT scheme is never silently substituted with TOT data).
        raise FileNotFoundError(
            "no GLOBK time series for station '{0}', scheme '{1}': "
            "expected {2}{{1,2,3}} in {3}".format(sta, tType, filepre, Dir)
        )

    # construct the file names for each component
    datafile1 = os.path.join(Dir, filepre + "1")
    datafile2 = os.path.join(Dir, filepre + "2")
    datafile3 = os.path.join(Dir, filepre + "3")
    # load the data for files and store it in arrays. use converters defined in the __converter functions
    yearf, d1, D1 = np.loadtxt(
        datafile1, unpack=True, skiprows=3, converters={1: __converter, 2: __converter}
    )
    d2, D2 = np.loadtxt(
        datafile2,
        usecols=(1, 2),
        unpack=True,
        skiprows=3,
        converters={1: __converter, 2: __converter},
    )
    d3, D3 = np.loadtxt(
        datafile3,
        usecols=(1, 2),
        unpack=True,
        skiprows=3,
        converters={1: __converter, 2: __converter},
    )

    # stack the arrays for the three components in the data array and for their uncertainties in ddata array
    data = np.vstack([d1, d2, d3])
    ddata = np.vstack([D1, D2, D3])

    return yearf, data, ddata


def open3DataFiles(STA, Dir=None, comp=["-N", "-E", "-U"]):
    """
    This function opens data contained in 3 files, one file for each component E, N and U

    Examples:
        >>> open3DataFiles(STA, Dir=None, comp=["-N", "-E", "-U"])

    Args:
        STA: Station four letter short name in capital letters
        Dir: optional alternative Directory of the files.
        comp: list of components to be read. Default is ["-N", "-E", "-U"]

    Returns:
        data:  three arrays containing GPS data in north, east and up

    """

    if Dir is None:
        Dir = os.getcwd()

    compdict = {}
    compdict[comp[0]] = ["north", "Dnorth"]
    compdict[comp[1]] = ["east", "Deast"]
    compdict[comp[2]] = ["up", "Dup"]
    components = {"north": None, "east": None, "up": None}

    for item in compdict.keys():
        dfile = "{0}{1}".format(STA, item)
        components[compdict[item][0]] = pd.read_csv(
            dfile, sep=r"\s+", index_col=0, header=None, names=compdict[item]
        )

    columnreorder = ["north", "east", "up", "Dnorth", "Deast", "Dup"]
    data = pd.concat(components.values(), axis=1)[columnreorder]
    data.set_index(pd.DatetimeIndex(toDateTime(data.index)), inplace=True)
    data.index = data.index.round("1h")

    return data


def convGlobktopandas(yearf, data, Ddata):
    """
    This function converts Globk files (from OpenGlobkTimes function) into pandas dataframes

    Examples:
        >>> convGlobktopandas(yearf, data, Ddata)

    Args:
        yearf: is array with time  (usually) in fractional year format (i.e. 2014.62328)
        data:  three arrays containing GPS data in north, east and up
        Ddata: respective uncertainty values for each component

    Returns:
           Index [datetime], north, east, up, Dnorth, Deast, Dup, yearf


    """

    # reduce(lambda x, y: pd.merge(x, y, on = 'Date'), dfList)
    names = ["north", "east", "up", "Dnorth", "Deast", "Dup", "yearf"]

    # !!!!! from_item depricated will remove
    # data = pd.DataFrame.from_items(zip(names[:3],data))
    # data = data.join( pd.DataFrame.from_items(zip(names[3:],Ddata) ) )
    # Using from_dict instead
    data = pd.DataFrame.from_dict(OrderedDict(zip(names[:3], data)))
    data = data.join(pd.DataFrame.from_dict(OrderedDict(zip(names[3:], Ddata))))
    data["yearf"] = yearf
    data.set_index(pd.DatetimeIndex(toDateTime(yearf)), inplace=True)
    data.index = data.index.round("1h")

    return data


def read_join(sta, schemes=("TOT", "08h"), Dir=None, missing="warn"):
    """
    Read a station's MULTIPLE GLOBK processing schemes into ONE DataFrame.

    "JOIN" = joining different processing schemes (refactor-B slice 6
    revival, decision D4): the daily TOT solutions and the sub-daily 08h
    solutions (and extensibly e.g. 04h) are held together in one long-format
    DataFrame, labeled by scheme, so they can be worked with jointly. This
    is NOT a merge with overlap resolution — the schemes coexist; every
    epoch of every scheme is kept, tagged with its scheme name.

    Each scheme is read with openGlobkTimes (all schemes share the identical
    mb_<STA>_<scheme>.dat{1,2,3} format — 3 header lines, then
    ``yearf value sigma`` rows, sub-daily epochs already encoded as
    fractional years), converted with convGlobktopandas, tagged, and
    concatenated. Adding a new scheme is just listing it, e.g.
    ``schemes=("TOT", "08h", "04h")``.

    NOTE on getData(tType="JOIN"): that legacy alias is left unchanged — it
    maps JOIN -> TOT and returns the (yearf, data, Ddata, offset) arrays.
    read_join is the real JOIN: its natural return is a DataFrame (a
    different shape), so it is a dedicated reader rather than a getData
    mode.

    Examples:
        >>> read_join("SENG")
        >>> read_join("SENG", schemes=("TOT", "08h"), Dir="/mnt_data/gpsdata")

    Args:
        sta: station four letter short name in capital letters
        schemes: iterable of GLOBK processing-scheme names, i.e. the
            ``<scheme>`` part of ``mb_<STA>_<scheme>.dat{1,2,3}``.
            Default ("TOT", "08h"). A single-scheme list is the degenerate
            JOIN (one label, same rows as the plain read).
        Dir: optional alternative directory of the time series; default is
            totDir from postprocess.cfg (same as openGlobkTimes).
        missing: how to treat a scheme whose files don't exist —
            "warn" (default): print a WARNING and skip the scheme;
            "raise": re-raise the FileNotFoundError. If NO scheme yields
            data, FileNotFoundError is always raised.

    Returns:
        pandas DataFrame, sorted by time (stable — coincident epochs keep
        the schemes order), with
            Index [datetime]  (rounded to 1h, as convGlobktopandas)
            north, east, up   [m]  raw GLOBK solution values
            Dnorth, Deast, Dup [m] respective uncertainties
            yearf             fractional year (e.g. 2014.62328)
            scheme            processing-scheme label (e.g. "TOT", "08h")
    """

    if missing not in ("warn", "raise"):
        raise ValueError(
            'missing must be "warn" or "raise", got {0!r}'.format(missing)
        )

    frames = []
    for scheme in schemes:
        try:
            yearf, data, Ddata = openGlobkTimes(sta, Dir=Dir, tType=scheme)
        except FileNotFoundError as e:
            if missing == "raise":
                raise
            print("WARNING: skipping scheme {0!r} for station {1}: {2}".format(scheme, sta, e))
            continue

        frame = convGlobktopandas(yearf, data, Ddata)
        frame["scheme"] = scheme
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            "no GLOBK time series found for station '{0}' in any of the "
            "schemes {1}".format(sta, tuple(schemes))
        )

    joined = pd.concat(frames)
    joined.sort_index(inplace=True, kind="stable")

    return joined


def pvel(pl, pcov):
    """
    This function prints the velocity for the three components and their uncertainty to numpy arrays

    Examples:
        >>> pvel(pl, pcov)

    Args:
        pl: list of the three components of the velocity and their uncertainties
        pcov: covariance matrix of the three components

    Returns:
        vel: list of the three components of the velocity
        vunc: list of the three components of the uncertainties
    """

    vunc = [None, None, None]
    vel = [None, None, None]

    for i in range(3):
        vel[i] = pl[i][1]
        vunc[i] = np.sqrt(np.diag(pcov[i]))[1]
        # print("{0:0.2f} {1:0.2f}".format( pl[i][1],vunc[i]) )

    return vel, vunc


def printvelocity(sta, ll, vel, vfile, pheader=False):
    """
    This function prints the velocity for a station to a file

    Examples:
        >>> printvelocity('GRIC', ll, vel, vfile)

    Args:
        sta: station name
        ll: list of the longitude and latitude
        vel: list of the three components of the velocity
        vfile: file to print the velocity
        pheader: print the header

    Returns:
        gpsvelo: string of the velocity

    """

    header = (
        "#lon       lat\t\t   N[mm]   DN[mm]  E[mm]  DE[mm]  U[mm]  DU[mm]\t\tStation"
    )

    if pheader is True:
        print(header, file=vfile)

    gpsvelo = "{0:5.6f} {1:5.6f}\t{2:7.2f} {5:7.2f}\t{3:7.2f} {6:7.2f}\t{4:7.2f} {7:7.2f}\t\t{8:s}".format(
        ll[1], ll[0], vel[0], vel[1], vel[2], vel[3], vel[4], vel[5], sta
    )

    print(gpsvelo, file=vfile)

    return gpsvelo


def detrend(
    x,
    y,
    Dy=None,
    fitfunc=lineperiodic,
    p=None,
    STA=None,
    onlyPeriodic=True,
    zref=False,
):
    """
    This function returns the detrending parameters for a specific fit function and station, for the GPS components

    Examples:
        >>> detrend(x, y, Dy, fitfunc=lineperiodic, p=None, pcov=None, STA='GRIC', onlyPeriodic=True, zref=False)

    Args:
        x: list of the years fractional for the timeseries
        y: list of the three components measurements for the timeseries
        Dy: list of the uncertainties of the three components
        fitfunc: fit function for the detrending. Default is lineperiodic
        p: list with initial guess for the fitting coefficients. Default is None.
        pcov: covariance matrix of the initial guess. Default is None.
        STA: station name, four letters in upper case. Default is None.
        onlyPeriodic: True or False. Default is True.
        zref: Reference z. Default is False. - Not implemented fully yet.

    Returns:



    Note:
        Deprecated shim (refactor-B slice 1): the math lives in
        :mod:`gps_analysis.fitting` — the fit is
        ``fit_components`` (via the :func:`fittimes` shim) and the
        subtraction is ``remove_trend``. Legacy semantics preserved
        exactly: ``y`` is still mutated IN PLACE and returned, and the
        seeding via :func:`getDetrFit`/:func:`convconst` (file I/O) stays
        here in geo_dataread.
    """

    if Dy is None:
        Dy = np.ones(y.shape)

    # Handling parameters
    if p is not None:
        pass
    else:
        if STA:
            const = getDetrFit(STA, onlyPeriodic=onlyPeriodic)
            p0 = convconst(const)
        else:
            p0 = [None, None, None]

        p, _ = fittimes(fitfunc, x, y, Dy, p0=p0)

    fits = [
        TrajectoryParams(
            params=np.asarray(p[i], dtype=np.float64),
            covariance=np.zeros((len(p[i]), len(p[i]))),
        )
        for i in range(3)
    ]
    y[:] = ga_fitting.remove_trend(fitfunc, x, y, fits)

    if zref:
        _, y, _, _ = vshift(x, y, Dy, uncert=20.0, refdate=None, Period=5)

    return y


def dPeriod(yearf, data, Ddata, startyear=None, endyear=None):
    """
    update( dict( [ [ line.split(',')[0], line.split(',')[1:] ] for line in args.eventf.read().splitlines() ] ) )

    This function returns the data for a specific period. Its default behavior is to do nothing.

    Examples:
        >>> dPeriod(yearf, data, Ddata, startyear=None, endyear=None)

    Args:
        yearf: time array with numeric time values
        data: data array
        Ddata: same form as data but containing the uncertainties
        startyear: Initial time. Default=None
        endyear: Final time. Default=None

    Returns:
        yearf: Time array within the period defined by startyear and endyear
        data:  Data within the period defined by startyear and endyear
        Ddata: Data within the period defined by startyear and endyear

    Note:
        Deprecated shim (refactor-B slice 2): the window math lives in
        :func:`gps_analysis.baseline.slice_window` (same delete
        conditions, ±0.001 yr tolerance). Legacy semantics preserved
        exactly: falsy bounds (``None``/0) are open, and with both
        bounds open the inputs are returned unchanged (same objects).
    """
    if not startyear and not endyear:
        return yearf, data, Ddata

    mask = ga_baseline.slice_window(
        yearf,
        start=startyear if startyear else None,
        end=endyear if endyear else None,
    )
    return yearf[mask], data[:, mask], Ddata[:, mask]


def vshift(yearf, data, Ddata, uncert=20.0, refdate=None, Period=5, offset=None):
    """
    This function shifts time series data by the average value of the interval defined by reday and the number of days specified ()

    Examples:
        >>> vshift(yearf, data, Ddata, uncert=20.0, refdate=None, Period=5, offset=None)

    Args:
        yearf: time array with numeric time values
        data: data array
        Ddata: same form as data but containing the uncertainties
        uncert: Maximum uncertainty of the data. Default=20.0
        refdate: reference date. Default=None
        Period: number of days. Default=5
        offset: initial offset. Default=None

    Returns:
        yearf: Time array
        data:  Data
        Ddata: Data uncertainty

    Note:
        Deprecated shim (refactor-B slice 2): the math lives in
        :func:`gps_analysis.preprocess.prep_neu_series` — the
        **.NEU/gamittoNEU profile** (decision D1; ``gamittoNEU``
        hardcodes ``uncert=1.1``, pinned by the ``real_neu_*`` golden
        masters). The legacy ``refdate``/``Period`` pair is converted
        to a fractional-year reference window here (date handling is
        caller policy, not leaf math). Degenerate-branch crashes
        (``TypeError`` on all-screened data with no offset, the
        ``NameError`` extrapolation branch) are now an explicit
        ``ValueError`` from the leaf.
    """
    ref_start, ref_end = _refdate_window(refdate, Period)
    if refdate:
        return ga_preprocess.prep_neu_series(
            yearf,
            data,
            Ddata,
            max_sigma=uncert,
            offset=offset,
            ref_start=ref_start,
            ref_end=ref_end,
        )
    return ga_preprocess.prep_neu_series(
        yearf, data, Ddata, max_sigma=uncert, offset=offset, ref_samples=Period
    )


def _refdate_window(refdate, Period):
    """Convert the legacy refdate/Period pair to a fractional-year window.

    Returns ``(None, None)`` when no refdate is given (count mode); a
    negative ``Period`` counts backwards from ``refdate`` (legacy swap).
    """
    if not refdate:
        return None, None
    startdate = currYearfDate(0, refdate)
    enddate = currYearfDate(Period, refdate)
    if Period < 0:
        return enddate, startdate
    return startdate, enddate


def estimate_offset(yearf, data, Ddata, refdate=None, Period=5):
    """
    Estimating offset of a time series at a reference (refdate) point for a given interval (Period)
    defaults at 5 days at the start of the time series

    Note:
        Deprecated shim (refactor-B slice 2): the math lives in
        :func:`gps_analysis.baseline.estimate_offset` (1/σ-weighted
        mean; window mode for ``refdate``, first-``Period``-samples
        count mode otherwise). The legacy empty-window branch (a
        ``NameError`` on an undefined variable) is an explicit
        ``ValueError`` in the leaf.
    """
    ref_start, ref_end = _refdate_window(refdate, Period)
    if refdate:
        return ga_baseline.estimate_offset(
            yearf, data, Ddata, start=ref_start, end=ref_end
        )
    return ga_baseline.estimate_offset(
        yearf[0:Period], data[0:3, 0:Period], Ddata[0:3, 0:Period]
    )


def iprep(yearf, data, Ddata, uncert=20.0, offset=None):
    """
    This function is a wrapper for vshift intendet for initializing the time series. It converts to mm and initializes the start of the time series to zero.

    Examples:
        >>> iprep(yearf, data, Ddata, uncert=20.0, offset=None)

    Args:
        yearf: time array with numeric time values
        data: data array
        Ddata: same form as data but containing the uncertainties
        uncert: Maximum uncertainty of the data. Default=20.0
        offset: initial offset. Default=None

    Returns:
        yearf: Time array
        data:  Data
        Ddata: Data uncertainty

    Note:
        Deprecated shim (refactor-B slice 2): the math lives in
        :func:`gps_analysis.preprocess.prep_plot_series` — the
        **plot/getData profile** (decision D1; ``getData`` passes
        ``uncert=15``). The m→mm conversion is unit policy and stays
        here — INCLUDING its legacy side effect of scaling the caller's
        ``data``/``Ddata`` arrays IN PLACE. The returned ``offset`` (in
        mm, weighted mean of the first 5 kept samples unless supplied)
        is the offset-reuse contract ``getData`` exposes to callers.
    """

    # converting to mm (in place — legacy contract preserved)
    data *= 1000
    Ddata *= 1000
    return ga_preprocess.prep_plot_series(
        yearf, data, Ddata, max_sigma=uncert, offset=offset
    )


def _remove_plate_velocity(sta, yearf, data, plate=None, reference="ITRF2008", scale=1):
    """Subtract secular plate motion from the horizontal components in place.

    Single shared implementation of the plate-velocity removal that was
    duplicated verbatim between the two production paths (refactor-B
    slice 5): ``getData`` (plot path — data already in mm after ``iprep``,
    so ``scale=1000`` converts ``gf.plateVelo``'s m/yr) and ``gamittoNEU``
    (.NEU path — data still in m, ``scale=1``; the m→mm conversion happens
    afterwards as caller unit policy). Bit-identical behavior is pinned by
    the ``getdata_*`` / ``real_getdata_*`` plate cases and the
    ``neu_*`` / ``neufile_*`` / ``real_neu_*`` / ``real_neufile_*`` golden
    masters.

    Note the long-standing axis swap relative to ``gf.plateVelo``'s
    documented (N, E, U) column order: data row 0 gets velocity column 1
    and data row 1 gets velocity column 0 — preserved verbatim from both
    legacy call sites (which agreed with each other).

    Args:
        sta: station four-letter name (plate velocity is looked up per station)
        yearf: fractional-year time array; motion accumulates from ``yearf[0]``
        data: 3xN component array; rows 0 and 1 are modified IN PLACE
        plate: explicit tectonic plate forwarded to ``gf.plateVelo``
            (default None = per-station plate lookup)
        reference: reference frame forwarded to ``gf.plateVelo``.
            Default="ITRF2008"
        scale: unit factor applied to the velocity — 1 for the m-unit .NEU
            path, 1000 for the mm-unit plot path

    Returns:
        data: the same (mutated) array, for assignment-style call sites

    """
    plateVel = gf.plateVelo([sta], plate, reference=reference)
    data[0, :] = data[0, :] - plateVel[0, 1] * scale * (yearf - yearf[0])
    data[1, :] = data[1, :] - plateVel[0, 0] * scale * (yearf - yearf[0])
    return data


def gamittoNEU(
    sta,
    mm=False,
    ref="plate",
    dstring=None,
    reference="ITRF2008",
    Dir=None,
    rhour=False,
):
    """
    This function convert a gamit time series to a single np.array with readable time tag.

    Examples:
        >>> gamittoNEU(sta, mm=False, ref="plate", dstring=None)

    Args:
        sta: station name
        mm: convert to mm. Default=False
        ref: reference. Default="plate"
        dstring: time tag. Default=None
        reference: reference frame to use. Default="ITRF2008"

    Returns:
        yearf: Time array
        data:  Data
        Ddata: Data uncertainty

    """

    yearf, data, Ddata = openGlobkTimes(sta, Dir=Dir)
    yearf, data, Ddata, _ = vshift(
        yearf, data, Ddata, uncert=1.1, refdate=None, Period=5, offset=None
    )

    # remove plate velocity (shared helper — data still in m, so scale=1)
    if ref == "plate":
        data = _remove_plate_velocity(sta, yearf, data, reference=reference)

    # convert to mm
    if mm:
        data = data * 1000
        Ddata = Ddata * 1000

    return gtoNEU(yearf, data, Ddata, dstring=dstring, rhour=rhour)


def gtoNEU(yearf, data, Ddata, dstring=None, rhour=False):
    """
    This function converts a gamit time series to a single np.array with readable time tag.

    Examples:
        >>> gtoNEU(yearf, data, Ddata)

    Args:
        yearf: time array with numeric time values
        data: data array
        Ddata: same form as data but containing the uncertainties
        dstring: time tag. Default=None

    Returns:
        yearf: Time array
        data:  Data
        Ddata: Data uncertainty

    """

    if dstring == "yearf":  # use the decimal year format
        NEUdata = np.array(
            list(zip(yearf, data[0], data[1], data[2], Ddata[0], Ddata[1], Ddata[2])),
            dtype=[
                ("yearf", float),
                ("data[0]", float),
                ("data[1]", float),
                ("data[2]", float),
                ("Ddata[0]", float),
                ("Ddata[1]", float),
                ("Ddata[2]", float),
            ],
        )
    else:
        yearf = convfromYearf(yearf, dstring, rhour=rhour)
        NEUdata = np.array(
            list(zip(yearf, data[0], data[1], data[2], Ddata[0], Ddata[1], Ddata[2])),
            dtype=[
                ("yearf", "S23"),
                ("data[0]", float),
                ("data[1]", float),
                ("data[2]", float),
                ("Ddata[0]", float),
                ("Ddata[1]", float),
                ("Ddata[2]", float),
            ],
        )

    return NEUdata


def read_gps_data(
    sta,
    Dir=None,
    start=None,
    end=None,
    ref="plate",
    detrend_periodic=True,
    detrend_line=False,
    fit=False,
    detrend_period=[None, None],
    useSTA=None,
    useFIT=None,
    uncert=20.0,
    detrendFile="",
    logging_level=logging.WARNING,
):
    """
    This function reads gps data from a station. It can be used to get the raw data, plate velocity and detrended data.

    Examples:


    Args:
        sta: station name
        Dir: directory. Default=None
        start: start date. Default=None
        end: end date. Default=None
        ref: reference. Default="plate"
        detrend_periodic: detrend periodic. Default=True
        detrend_line: detrend line. Default=False
        fit: fit. Default=False
        detrend_period: detrend period. Default=[None, None]
        useSTA: works in conjunction with useFIT to get the fit. Default=None
        useFIT: useFIT from another station. Default= "periodic"
        uncert: Maximum uncertainty of the data. Default=20.0
        logging_level: logging level. Default=logging.WARNING


    Returns:
        const: detrending constants coefficient, for a lineperiodic function.
        data:  Data array for the three components including the uncertainties from the raw_data (measurement uncertainty)

    """

    # Handling logging
    logging.basicConfig(
        format="%(asctime)-15s [%(levelname)s] %(funcName)s: %(message)s",
        level=logging_level,
    )
    logging.getLogger().setLevel(logging_level)
    module_logger = logging.getLogger()

    yearf, data, Ddata, _ = getData(sta, Dir=Dir, ref=ref, uncert=uncert)
    syearf, sdata, sDdata = yearf, data, Ddata

    # NOTE: detrending needs to be revised.

    const = getDetrFit(sta, useSTA=useSTA, useFIT=useFIT, detrendFile=detrendFile)
    p0 = convconst(const)
    pb = p0.copy()

    module_logger.info("Fitting constants:\n {}".format(p0))
    if not np.all([p0[i][1:].dtype if p0[i] is not None else False for i in range(3)]):
        module_logger.warning("Setting fit to lineperiodic")
        fit = "lineperiodic"

    syearf, sdata, sDdata = dPeriod(
        yearf,
        data,
        Ddata,
        startyear=detrend_period[0],
        endyear=detrend_period[1],
    )

    if fit:
        module_logger.info("Fitting =====================")
        if len(syearf) == 0:
            module_logger.warning(
                "No data for interval {}-{} will try using the whole time series".format(
                    *detrend_period
                )
            )
            syearf, sdata, sDdata = yearf, data, Ddata

        module_logger.info("Fitting the data to a {}".format(fit))

        if fit == "line":
            pt = [p0[i][0:2] for i in range(3)]
            module_logger.info("initial parameters p0 {}".format(pt))
            pb, _ = fittimes(line, syearf, sdata, sDdata, p0=pt)
            module_logger.info("Estimated parameters for {} {}".format(sta, pb))
            pb = [np.concatenate((pb[i], p0[i][2:])) for i in range(3)]
            pt = [p0[i][0:2] for i in range(3)]

        elif fit == "periodic":
            pb, _ = fittimes(periodic, syearf, sdata, sDdata, p0=p0)

        elif fit == "lineperiodic":
            pb, _ = fittimes(lineperiodic, syearf, sdata, sDdata, p0=p0)

        const = convconst(const, pb)
        const.loc[sta, ["Starttime", "Endtime"]] = [syearf[0], syearf[-1]]

    else:
        module_logger.info("Not fitting")

    if np.all([pb[i][1:] if pb[i] is not None else False for i in range(3)]):
        if detrend_periodic:
            module_logger.info("Periodic detrending")
            module_logger.debug("Fitt parameters: {}".format(pb))
            data = detrend(yearf, data.copy(), Ddata, fitfunc=periodic, p=pb)

        if detrend_line:
            module_logger.info("linear detrending")
            pc = [pb[i][0:2] for i in range(3)]
            module_logger.debug("Fitt parameters: {}".format(pc))
            data = detrend(yearf, data.copy(), Ddata, fitfunc=line, p=pc)
    else:
        module_logger.warning("Fitt parameters are unknown pb={}".format(pb))
        if detrend_periodic or detrend_line:
            module_logger.warning(
                "Will try to estimate the parameters based the whole dataset and detrend"
            )
            data = detrend(yearf, data.copy(), Ddata, fitfunc=lineperiodic)
    # ----------------------------------------------------

    if start:
        startf = TimetoYearf(start.year, start.month, start.day)
    else:
        startf = yearf[0]
    if end:
        endf = TimetoYearf(end.year, end.month, end.day)
    else:
        endf = yearf[-1]

    yearf, data, Ddata = dPeriod(yearf, data, Ddata, startyear=startf, endyear=endf)
    yearf, data, Ddata, _ = vshift(yearf, data, Ddata, uncert=uncert)

    data = convGlobktopandas(yearf, data, Ddata)

    data["hlength"] = np.sqrt(np.square(data[["east", "north"]]).sum(axis=1))
    data["hangle"] = np.rad2deg(np.arctan2(data["north"], data["east"]))
    data["Dhlength"] = np.sqrt(np.square(data[["Deast", "Dnorth"]]).sum(axis=1))

    module_logger.info("Dataframe columns:\n" + str(data.columns) + "\n")
    module_logger.info("Input time period: ({}, {})".format(start, end))
    module_logger.info("dataframe First and Last lines:\n" + str(data.iloc[[0, -1]]))
    module_logger.info("detrend parameter row:\n" + str(const))
    module_logger.debug("Dataframe shape: {}".format(str(data.shape)))
    module_logger.debug("dataframe types:\n" + str(data.dtypes) + "\n")

    return data, const


#
# Other functions
#


def toDateTime(yearf):
    """
    Function converting from floating point year to datetime.

    Examples:


    Args:
        yearf: floating point year array

    Returns:
        tmp: datetime equivalence of the yearf array



    """

    tmp = []

    for i in range(len(yearf)):
        tmp.append(TimefromYearf(yearf[i]))

    return tmp


def toord(yearf):
    """
    Function from floating point year to floating point ordinal.

    Examples:

    Args:
        yearf: floating point year array

    Returns:
        yearf: floating point ordinal array

    """

    for i in range(len(yearf)):
        yearf[i] = TimefromYearf(yearf[i], "ordinalf")

    return yearf


def fromord(yearf):
    """
    Function from floating point ordinal to floating point year.

    Examples:

    Args:
        yearf: floating point ordinal array

    Returns:
        yearf: floating point year array

    """

    # from floating point year to floating point ordinal

    for i in range(len(yearf)):
        yearf[i] = Timeto(yearf[i], "ordinalf")

    return yearf


def getData(
    sta,
    fstart=None,
    fend=None,
    ref="itrf2008",
    Dir=None,
    tType="TOT",
    uncert=15,
    offset=None,
):
    """
    Function extracting and filtering data to prepeare for plotting.

    Examples:
        >>>getData('GRIC', fstart='2017-01-01', fend='2017-12-31', ref='itrf2008', Dir=None, tType='TOT', uncert=15, offset=None)

    Args:
        sta: station name
        fstart: start date
        fend: end date
        ref: reference frame
        Dir: directory
        tType: type of data
        uncert: uncertainty
        offset: offset

    Returns:
        yearf: floating point year array
        data: data array
        Ddata: uncertainty array
        offset: offset




    """

    if tType == "JOIN":
        # legacy alias kept for back-compat: getData's array return can't
        # carry multiple schemes. The real multi-scheme JOIN (DataFrame,
        # revived in refactor-B slice 6) is read_join().
        tType = "TOT"

    yearf, data, Ddata = openGlobkTimes(sta, Dir=Dir, tType=tType)
    yearf, data, Ddata = dPeriod(yearf, data, Ddata, fstart, fend)
    if yearf is None or len(yearf) == 0:
        print("WARNING: no data for station {}".format(sta))
        return None, None, None, None
    yearf, data, Ddata, offset = iprep(yearf, data, Ddata, uncert=uncert, offset=offset)

    if offset is None:
        print("WARNING: offset determination failure for station {}".format(sta))

    if ref == "plate":
        # shared helper — data already in mm after iprep, so scale=1000
        data = _remove_plate_velocity(sta, yearf, data, scale=1000)

    elif ref == "detrend":
        # Deliberate dead-branch removal (refactor-B slice 4): this branch
        # called detrend() with the wrong signature and the deleted legacy
        # leastsq errfunc — it always crashed. Replaced with an explicit error.
        raise ValueError(
            "unsupported ref='detrend': the branch was dead (wrong detrend() "
            "call signature) and was removed in the refactor-B slice-4 cleanup"
        )

    elif ref == "itrf2008":
        pass

    else:
        # explicit plate name — same shared helper, mm path
        data = _remove_plate_velocity(sta, yearf, data, plate=ref, scale=1000)

    return yearf, data, Ddata, offset


#
#   --- Private functions ---
#


def __converter(x):
    """
    The data extracted are converted to float and
    ocassional ******* in the data files needs to handled as NAN

    """

    try:
        return float(x)
    except:
        return np.nan

    # if x == '********':
    #    return np.nan
    #    return float(x)
    # else:
