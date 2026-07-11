"""
This module contains functions for reading and returning GPS data. It includes the following functions

"""

import datetime as dt
import glob
import logging
import os
import re
import shutil
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Union

import geofunc.geofunc as gf
import numpy as np
import pandas as pd
from gps_analysis import fitting as ga_fitting
from gps_analysis import models as ga_models
from gps_analysis.models import TrajectoryParams
from gps_parser import ConfigParser

from gtimes.timefunc import (
    TimefromYearf,
    TimetoYearf,
    convfromYearf,
    currYearfDate,
    round_to_hour,
)
from scipy import optimize

import geo_dataread.gps_read as gdrgps

#
# time series filtering
#


def line(
    x: Union[int, float], p0: Union[int, float], p1: Union[int, float]
) -> Union[int, float]:
    """
    linear function for detrending timeseries without considering their oscillatory nature.

    examples:
        >>> line(1, 0, 1)
        1

    args:
        x (union[int, float]): input value in yearf or ordinal (or any numeric time)
        p0 (union[int, float]): intercept
        p1 (union[int, float]): slope

    returns:
        union[int, float]: output value

    """
    return p0 + p1 * x


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

    """

    return (
        p0
        + p1 * x
        + p2 * np.cos(2 * np.pi * x)
        + p3 * np.sin(2 * np.pi * x)
        + p4 * np.cos(4 * np.pi * x)
        + p5 * np.sin(4 * np.pi * x)
    )


def periodic(
    x: float, p0: float, p1: float, p2: float, p3: float, p4: float, p5: float
) -> float:
    """
    periodic function without the linear part


    examples:
        >>> periodic(1, 0, 1, 1, 1, 1, 1)


    args:
        x: input value in yearf or ordinal (or any numeric time)
        p0: coefficient
        p1: coefficient
        p2: coefficient
        p3: coefficient
        p4: coefficient
        p5: coefficient

    returns:
        output value

    """

    return (
        p2 * np.cos(2 * np.pi * x)
        + p3 * np.sin(2 * np.pi * x)
        + p4 * np.cos(4 * np.pi * x)
        + p5 * np.sin(4 * np.pi * x)
    )


def xf(x: float, p0: float, p1: float, p2: float, tau=4.8) -> float:
    """
    linear + exponential function with constant tau coefficient

    examples:
        >>> xf(1, 0, 1, 1, 1)

    args:
        x: input value in yearf or ordinal (or any numeric time)
        p0: intercept
        p1: slope
        p2: exponential term coefficient
        tau: exponential coefficient

    returns:
        y: output value


    """

    return p0 + p1 * x + p2 * np.exp(-tau * x)


def expxf(x: float, p0: float, p1: float, p2: float, p3: float) -> float:
    """

    linear + exponential function with variable coefficient

    examples:
        >>> expxf(1, 0, 1, 1, 1)

    args:
        x: input value in yearf or ordinal (or any numeric time)
        p0: intercept
        p1: slope
        p2: exponential term coefficient
        p3: exponential coefficient

    returns:
        y: output value

    """

    return p0 + p1 * x + p2 * np.exp(-p3 * x)

    """"""


def expf(x, p0, p1, p2):
    """
    exponential function

    """

    return p0 + p1 * np.exp(-p2 * x)


def dexpf(x, p1, p2):
    """
    derivative of p0 + p1*exp(-p2*x)
    """

    return -p1 * p2 * np.exp(-p2 * x)


def secondorder(x, p0, p1, p2):
    """
    Second decree polynomial function
    """

    return p0 + p1 * x + p2 * x**2


def gpsvelo_df():
    """
    dataframe for pygmt velo with three extra columns
    for date, period and boolean for vertical component.
    this is based on the output of the gpsvelo function, which estimates the gps velocity for a specific period of time.

    examples:
        >>> gpsvelo_df()

    returns:
        gpsvelo: dataframe


    """

    gpsvelo = pd.dataframe(
        columns=[
            "longitude",
            "latitude",
            "east_velo",
            "north_velo",
            "east_sigma",
            "north_sigma",
            "coorelation_en",
            "station",
            "date",
            "period",
            "vertical",
        ],
    )

    return gpsvelo


def gpsvelo(sta: str, ll, vel, vertical=False, vfile=None, pheader=False):
    """
    return gps velocities gmt velo

    examples:
        >>> gpsvelo()

    args:
        sta: station name
        ll: [longitude, latitude]
        vertical: boolean to describe
        vfile: velocity file
        pheader: header

    returns:
        gpsvelo: dataframe

    """

    gpsvelo = "{0:5.6f} {1:5.6f}\t{2:7.2f} {5:7.2f}\t{3:7.2f} {6:7.2f}\t{4:7.2f} {7:7.2f}\t\t{8:s}".format(
        ll[1], ll[0], vel[0], vel[1], vel[2], vel[3], vel[4], vel[5], sta
    )


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
        if useFIT == "periodic":
            const.loc[sta, north[1:] + east[1:] + up[1:]] = table.loc[
                useSTA, north[1:] + east[1:] + up[1:]
            ]
        elif useFIT == "lineperiodic":
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


def savedisp(datadict, fname=None, header=""):
    """
    this function saves a dictionary of data to a file

    examples:
        >>> savedisp(datadict, fname=None, header="")

    args:
        datadict: dictionary of data
        fname: file name
        header: header of the file

    returns:
        output file

    """

    valtype = type(datadict.values()[0])

    datadict = ordereddict(sorted(datadict.items()))

    if (valtype is list) or (valtype is np.ndarray):
        fmt = "% 3.8f\t% 2.8f\t% 2.8f\t%s"  # format

        ab = np.zeros(
            len(datadict.keys()),
            dtype=[
                ("var1", "float"),
                ("var2", "float"),
                ("var3", "float"),
                ("var4", "a4"),
            ],
        )

        ab["var1"] = np.squeeze(datadict.values())[:, 0]
        ab["var2"] = np.squeeze(datadict.values())[:, 1]
        ab["var3"] = np.squeeze(datadict.values())[:, 2]
        ab["var4"] = datadict.keys()

    # formatting tuples into a dictionary of arrays
    if valtype is tuple:
        fmt = "% 3.8f\t% 2.8f\t% 2.8f\t%2.8f\t%2.8f\t%s"
        ab = np.zeros(
            len(datadict.keys()),
            dtype=[
                ("var1", "float"),
                ("var2", "float"),
                ("var3", "float"),
                ("var4", "float"),
                ("var5", "float"),
                ("var6", "a4"),
            ],
        )
        ab["var1"] = np.squeeze(zip(*datadict.values()[:])[0])[:, 0]
        ab["var2"] = np.squeeze(zip(*datadict.values()[:])[0])[:, 1]
        ab["var3"] = np.squeeze(zip(*datadict.values()[:])[1])[:, 0]
        ab["var4"] = np.squeeze(zip(*datadict.values()[:])[1])[:, 1]
        ab["var5"] = np.squeeze(zip(*datadict.values()[:])[1])[:, 2]
        ab["var6"] = datadict.keys()

    if fname:
        np.savetxt(fname, ab, fmt=fmt, header=header)
    return ab


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

    args:
        sta: station four letter short name in captial letters
        Dir: optional alternative directory of the gamit time series data.

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
    else:  # use tot as file format to read in data
        filepre = "mb_{0:s}_{1:s}.dat".format(sta, "tot")

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

    # option to grab 8hr subdaily solutions from the same path
    if tType == "08h":
        shift8h = dt.timedelta(**shiftime("h8"))
        yearf = np.array(
            [
                timetoyearf(*(item + shift8h).timetuple()[:6])
                for item in todatetime(yearf)
            ]
        )

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


def compGlobkTimes(stalist="any", dirConFilePath=None, freq=None):
    """
    This function joins old and new mb_ time series files

    Examples:
        >>> compGlobkTimes(stalist="any", dirConFilePath=None, freq=None)

    Args:
        stalist: list of station names in capital letters. If "any", all stations are used. Default is "any"
        dirConFilePath: optional alternative Directory of the GAMIT time series data.
        freq: optional frequency of the data. Default is None

    Returns:
        data:  three arrays containing GPS data in north, east and up

    """

    config = ConfigParser()

    # totpath = config.getPostprocessConfig('totDir')

    if dirConFilePath:  # for custom file
        Dirs = parsedir(dirConFilePath)
    else:  # grab paths from the postprocess.cfg file in gpsconfig directory that cparser reads into a dictionary
        Dirs = {
            "figDir": config.get_config("Configs", "figDir"),
            "preDir": config.get_config("Configs", "preDir"),
            "rapDir": config.get_config("Configs", "rapDir"),
            "totDir": config.get_config("Configs", "totDir"),
        }

        print("Directory is \n", Dirs)

    # Reading into a string the paths
    PreDir = Dirs["preDir"]
    RapDir = Dirs["rapDir"]
    TotDir = Dirs["totDir"]

    # Setting the frequency to TOT if freq is None
    if freq == "TOT" or freq is None:
        freq = "TOT"
    else:  # setting the frequency
        PreDir = PreDir + "_%s" % (freq)
        RapDir = RapDir + "_%s" % (freq)

    if stalist == "any":
        FilePreL = os.path.join(PreDir, "mb_*.dat?")
        FileRapL = os.path.join(RapDir, "mb_*.dat?")

        List = glob.glob(FilePreL) + glob.glob(FileRapL)

        # listing all stations in  the Rap and Pre directories
        stalist = sorted(set([item[-13:-9] for item in List]))

    # Set up names for files
    for STA in stalist:
        FilePre = "mb_%s_?PS.dat" % STA
        OutFilePre = "mb_%s_GPS.dat" % STA
        GPS20PS = "mb_%s_0PS.dat" % STA

        for axes in range(1, 4):
            FilePreR = os.path.join(PrePath, FilePre + "%s" % (axes,))
            FileRapR = os.path.join(RapPath, FilePre + "%s" % (axes,))

            # graping the list for files for for that station
            PreFileL = glob.glob(FilePreR)  # listing files in the pre dir
            RapFileL = glob.glob(FileRapR)  # listing files in th Rap dir

            #  Sorting the file lists
            PreFileL.sort()
            if len(PreFileL) > 1:
                PreFileL.insert(0, PreFileL.pop(-1))
            RapFileL.sort()
            if len(RapFileL) > 1:
                RapFileL.insert(0, RapFileL.pop(-1))

            TotFile = os.path.join(TotPath, "mb_%s_%s.dat%s" % (STA, freq, axes))

            print("Concatenating all the %s data to %s" % (STA, TotFile))

            if os.path.exists(TotFile):
                os.remove(TotFile)

            outf = open(TotFile, "a")

            for fil in PreFileL:
                print("Processing file %s " % fil, file=sys.stderr)
                f = open(fil)
                f.seek(61)
                shutil.copyfileobj(f, outf)
                f.close()

            outf.close()

            preexist = os.stat(TotFile).st_size != 0
            if preexist:
                outf = open(TotFile, "r")
                lastline = outf.readlines()[-1]
                lastline = lastline.split()
                outf.close()

            outf = open(TotFile, "a")
            for file in RapFileL:
                formatstr = "Processing file {0:s} ".format(file)
                print(formatstr, file=sys.stderr)
                rapfile = open(file, "r")
                rapfile.seek(61)
                lines = rapfile.readlines()
                if preexist:
                    lines = "".join(
                        [line for line in lines if line.split()[0] > lastline[0]]
                    )
                else:
                    lines = "".join([line for line in lines])

            outf.close()


def TieTimes(sta1, sta2, dirConFilePath=None, freq=None, tie=[None, None, None]):
    """
    This function joins old and new mb_ time series files

    Examples:
        >>> TieTimes(sta1, sta2, dirConFilePath=None, freq=None, tie=[None, None, None])

    Args:
        sta1: first station name in capital letters
        sta2: second station name in capital letters
        dirConFilePath: optional alternative Directory of the GAMIT time series data.
        freq: optional frequency of the data. Default is None
        tie: optional tie file. Default is [None, None, None]


    """

    config = ConfigParser()
    # totpath = config.getPostprocessConfig('totpath')

    if dirConFilePath:  # for custom file
        Dirs = parsedir(dirConFilePath)
    else:
        # As the standard configparser works with dictionaries, use it to create the Dirs dictionaries
        Dirs = {
            "figDir": config.get_config("Configs", "figDir"),
            "preDir": config.get_config("Configs", "preDir"),
            "rapDir": config.get_config("Configs", "rapDir"),
            "totDir": config.get_config("Configs", "totDir"),
        }

    # parse the paths from the Dirs dictionary
    PreDir = Dirs["predDir"]
    RapDir = Dirs["rapdDir"]
    TotDir = Dirs["totdDir"]

    # Setting the frequency to TOT if freq is None
    if freq == "TOT" or freq is None:
        freq = "TOT"
    else:
        PreDir = PreDir + "_%s" % (freq)
        RapDir = RapDir + "_%s" % (freq)

    # for all the stations, create the path for PreL and RapL files
    if stalist == "any":
        FilePreL = os.path.join(PreDir, "mb_*.dat?")
        FileRapL = os.path.join(RapDir, "mb_*.dat?")

        List = glob.glob(FilePreL) + glob.glob(FileRapL)

        # listing all stations in  the Rap and Pre dir
        stalist = sorted(set([item[-13:-9] for item in List]))

    for STA in stalist:
        FilePre = "mb_%s_?PS.dat" % STA
        OutFilePre = "mb_%s_GPS.dat" % STA
        GPS20PS = "mb_%s_0PS.dat" % STA

        for axes in range(1, 4):
            FilePreR = os.path.join(PrePath, FilePre + "%s" % (axes,))
            FileRapR = os.path.join(RapPath, FilePre + "%s" % (axes,))

            # graping the list for files for for that station
            PreFileL = glob.glob(FilePreR)  # listing files in the pre dir
            RapFileL = glob.glob(FileRapR)  # listing files in th Rap dir

            #  Sorting the file lists
            PreFileL.sort()
            if len(PreFileL) > 1:
                PreFileL.insert(0, PreFileL.pop(-1))
            RapFileL.sort()
            if len(RapFileL) > 1:
                RapFileL.insert(0, RapFileL.pop(-1))

            TotFile = os.path.join(TotPath, "mb_%s_%s.dat%s" % (STA, freq, axes))
            print("Concating all the %s data to %s" % (STA, TotFile))
            if os.path.exists(TotFile):
                os.remove(TotFile)
            outf = open(TotFile, "a")
            for fil in PreFileL:
                print("Processing file %s " % fil, file=sys.stderr)
                f = open(fil)
                f.seek(61)
                shutil.copyfileobj(f, outf)
                f.close()
            outf.close()

            preexist = os.stat(TotFile).st_size != 0
            if preexist:
                outf = open(TotFile, "r")
                lastline = outf.readlines()[-1]
                lastline = lastline.split()
                outf.close()

            outf = open(TotFile, "a")
            for file in RapFileL:
                formatstr = "Processing file {0:s} ".format(file)
                print(formatstr, file=sys.stderr)
                rapfile = open(file, "r")
                rapfile.seek(61)
                lines = rapfile.readlines()
                if preexist:
                    lines = "".join(
                        [line for line in lines if line.split()[0] > lastline[0]]
                    )
                else:
                    lines = "".join([line for line in lines])

                outf.write(lines)
                rapfile.close()

            outf.close()


# def TieTimes(sta1, sta2, dirConFilePath=None, freq=None, tie=[None, None, None]):
#     """
#     This function joins old and new mb_ time series files
#
#     Examples:
#         >>> TieTimes(sta1, sta2, dirConFilePath=None, freq=None, tie=[None, None, None])
#
#     Args:
#         sta1: first station name in capital letters
#         sta2: second station name in capital letters
#         dirConFilePath: optional alternative Directory of the GAMIT time series data.
#         freq: optional frequency of the data. Default is None
#         tie: optional tie file. Default is [None, None, None]
#
#     Returns:
#         None
#
#
#     """
#
#     if dirConFilePath:  # for custom file
#         Dirs = parsedir(dirConFilePath)
#     else:
#         Dirs = cp.Parser().getPostprocessConfig()
#
#     # PrePath = Dirs['prePath'] - These paths are not used for now, but can be added later
#     # RapPath = Dirs['rapPath'] - Same case as above
#     TieFile = Dirs["tiefile"]
#     TotPath = Dirs["totDir"]
#
#     if freq == "TOT" or freq is None:
#         freq = "TOT"
#     else:
#         PrePath = PrePath + "_%s" % (freq)
#         RapPath = RapPath + "_%s" % (freq)
#
#     print(TieFile)
#
#     dtype = [
#         ("North", "<f8"),
#         ("East", "<f8"),
#         ("Up", "<f8"),
#         ("sta1", "|S5"),
#         ("sta2", "|S5"),
#     ]
#
#     const = np.genfromtxt(TieFile, dtype=dtype)
#     const = [i for i in const if i[3] == sta1 and i[4] == sta2]
#     print(const)
#
#     for axes in range(1, 4):
#         TotFile1 = os.path.join(TotDir, "mb_%s_%s.dat%s" % (sta1, freq, axes))
#         TotFile2 = os.path.join(TotDir, "mb_%s_%s.dat%s" % (sta2, freq, axes))
#         print("Concating all the %s data to %s" % (sta1, TotFile2))
#         # outf = open(TotFile, 'r')
#         data1 = read_table(
#             TotFile1, sep=r"\s+", header=None, index_col=0, names=["disp", "uncert"]
#         )
#         data2 = pd.read_csv(
#             TotFile2, sep=r"\s+", header=None, index_col=0, names=["disp", "uncert"]
#         )
#         print(const[0][axes - 1])
#         data2["disp"] -= const[0][axes - 1] / 1000
#         data = pd.concat([data1, data2])
#
#         outfile = os.path.join(TotDir, "mb_%s_%s.dat%s" % (sta2, "JON", axes))
#         data.to_csv(outfile, sep="\t", index=True, header=False)


# Legacy scipy.optimize.leastsq residual machinery (rate-first parameter
# packing, p = [rate, ...amplitudes..., intercept]) — only consumers are the
# dead `fitline` (reads ./itrf08det) path, the broken getData(ref="detrend")
# branch, and `filt_outl` (no internal callers). Slated for deletion in
# refactor-B slice 4 ("dead leastsq block").


def fitfuncl(p, x):
    # Deprecated shim (refactor-B slice 1): p = [rate, intercept] →
    # gps_analysis.models.linear(x, intercept, rate). Bit-identical
    # (IEEE addition/multiplication are commutative).
    return ga_models.linear(x, p[1], p[0])


def errfuncl(p, x, y):
    return fitfuncl(p, x) - y  # distance to the target function


def fitfunc(p, x):
    # NOT delegated to gps_analysis.models.lineperiodic: same model, but the
    # leaf evaluates linear(t) + periodic(t) whereas this legacy single
    # expression associates left-to-right — measured ≤ 2.3e-13 (~1 ulp)
    # difference. Kept verbatim (behavior frozen) until slice 4 deletes it.
    return (
        p[0] * x
        + p[1] * np.cos(2 * np.pi * x)
        + p[2] * np.sin(2 * np.pi * x)
        + p[3] * np.cos(4 * np.pi * x)
        + p[4] * np.sin(4 * np.pi * x)
        + p[5]
    )


def errfunc(p, x, y):
    return fitfunc(p, x) - y  # Distance to the target function


def fitline(yearf, data, STA):
    """
    This function fits a function through data points of a station STA

    Examples:
        >>> fitline(yearf, data, STA)

    Args:
        yearf: list of years
        data: list of data
        STA: station name

    Returns:
        parameters of the linear fitting


    """

    dtype = [
        ("Nrate", "<f8"),
        ("Erate", "<f8"),
        ("Urate", "<f8"),
        ("Nacos", "<f8"),
        ("Nasin", "<f8"),
        ("Eacos", "<f8"),
        ("Easin", "<f8"),
        ("Uacos", "<f8"),
        ("Uasin", "<f8"),
        ("Nscos", "<f8"),
        ("Nssin", "<f8"),
        ("Escos", "<f8"),
        ("Essin", "<f8"),
        ("Uscos", "<f8"),
        ("Ussin", "<f8"),
        ("shortname", "|S5"),
        ("name", "|S20"),
    ]

    const = np.genfromtxt("itrf08det", dtype=dtype)
    const = [i for i in const if i[15] == STA]

    pN = [const[0][0]]
    pE = [const[0][1]]
    pU = [const[0][2]]
    pN = [-1 * i for i in pN]
    pE = [-1 * i for i in pE]
    pU = [-1 * i for i in pU]
    # pN.append(0)
    # pE.append(0)
    # pU.append(0)

    # print "pN: %s" % p
    # print "pE: %s" % pE
    # print "pU: %s" % pU

    pb = [[0, 0], [0, 0], [0, 0]]

    # pb[0], success = optimize.leastsq(errfunc, pN[:], args=(yearf-yearf[0], data[0]))
    # pb[1], success = optimize.leastsq(errfunc, pE[:], args=(yearf-yearf[0], data[1]))
    # pb[2], success = optimize.leastsq(errfunc, pU[:], args=(yearf-yearf[0], data[2]))
    pb[0], success = optimize.leastsq(errfuncl, pb[0], args=(yearf, data[0]))
    pb[1], success = optimize.leastsq(errfuncl, pb[1], args=(yearf, data[1]))
    pb[2], success = optimize.leastsq(errfuncl, pb[2], args=(yearf, data[2]))

    return pN, pE, pU, pb


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

    """
    if startyear:
        index = np.where(yearf <= startyear - 0.001)
        yearf = np.delete(yearf, index)
        data = np.delete(data, index, 1)
        Ddata = np.delete(Ddata, index, 1)

    if endyear:
        index = np.where(yearf >= endyear + 0.001)
        yearf = np.delete(yearf, index)
        data = np.delete(data, index, 1)
        Ddata = np.delete(Ddata, index, 1)

    return yearf, data, Ddata


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

    """

    # Filtering a little, removing big outliers
    with np.errstate(invalid="ignore"):
        filt = Ddata < uncert
    filt = np.logical_and(np.logical_and(filt[0, :], filt[1, :]), filt[2, :])

    yearf = yearf[filt]
    data = np.reshape(data[np.array([filt, filt, filt])], (3, -1))
    Ddata = np.reshape(Ddata[np.array([filt, filt, filt])], (3, -1))

    if data.any():
        if not (offset is None):
            pass
        else:
            offset = estimate_offset(yearf, data, Ddata, refdate=refdate, Period=Period)

    data = np.array([data[i, :] - offset[i] for i in range(3)])

    return yearf, data, Ddata, offset


def estimate_offset(yearf, data, Ddata, refdate=None, Period=5):
    """
    Estimating offset of a time series at a reference (refdate) point for a given interval (Period)
    defaults at 5 days at the start of the time series

    """

    # averaging the first period days
    if refdate:
        startdate = currYearfDate(0, refdate)
        enddate = currYearfDate(Period, refdate)
        if Period < 0:
            tmpyearf, tmpdata, tmpDdata = dPeriod(
                yearf, data, Ddata, enddate, startdate
            )
        else:
            tmpyearf, tmpdata, tmpDdata = dPeriod(
                yearf, data, Ddata, startdate, enddate
            )

        if tmpdata.any():
            # if there are any data from this period
            offset = np.average(tmpdata[0:3, :], 1, weights=1 / tmpDdata[0:3, :])
        else:
            # We need to extrapolate
            # þarf að díla við þetta með því að módelera.
            offset = np.average(data[0:3, 0:j], 1, weights=1 / Ddata[0:3, 0:7])
    else:
        offset = np.average(data[0:3, 0:Period], 1, weights=1 / Ddata[0:3, 0:Period])

    return offset


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

    """

    # converting to mm
    data *= 1000
    Ddata *= 1000
    return vshift(yearf, data, Ddata, uncert=uncert, offset=offset)


def filt_outl(yearf, data, Ddata, pb, errfunc, outlier):
    """
    This function removes outliers from a time series.

    Examples:
        >>> filt_outl(yearf, data, Ddata, pb, errfunc, outlier)

    Args:
        yearf: time array with numeric time values
        data: data array
        Ddata: same form as data but containing the uncertainties
        pb: parameters of the error function. Default=[None, None, None]
        errfunc: error function. Default=abs
        outlier: Maximum uncertainty of the data.

    Returns:
        yearf: Time array
        data:  Data
        Ddata: Data uncertainty

    """
    # Removing big outliers
    for i in range(3):
        index = np.where(abs(errfunc(pb[i], yearf - yearf[0], data[i])) > outlier[i])
        yearf = np.delete(yearf, index)
        data = np.delete(data, index, 1)
        Ddata = np.delete(Ddata, index, 1)

    return yearf, data, Ddata


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

    # remove plate velocity
    if ref == "plate":
        plateVel = gf.plateVelo([sta], reference=reference)
        data[0, :] = data[0, :] - plateVel[0, 1] * (yearf - yearf[0])
        data[1, :] = data[1, :] - plateVel[0, 0] * (yearf - yearf[0])

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

    if useFIT == "periodic":
        module_logger.warning(
            '"{}" parameters from {} used in {}'.format(
                const["Fit"].values[0], const["useSTA"].values[0], sta
            )
        )
        module_logger.info('Setting the fit to a "line" for estimating the rate')
        fit = "line"

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
        plateVel = gf.plateVelo([sta])
        data[0, :] = data[0, :] - plateVel[0, 1] * 1000 * (yearf - yearf[0])
        data[1, :] = data[1, :] - plateVel[0, 0] * 1000 * (yearf - yearf[0])

    elif ref == "detrend":
        pN, pE, pU, pb = detrend(yearf, data, sta)
        pb_org = [pN, pE, pU]

        for i in range(3):
            data[i] = -errfunc(pb_org[i], yearf - yearf[0], data[i])

    elif ref == "itrf2008":
        pass

    else:
        plateVel = gf.plateVelo([sta], ref)
        data[0, :] = data[0, :] - plateVel[0, 1] * 1000 * (yearf - yearf[0])
        data[1, :] = data[1, :] - plateVel[0, 0] * 1000 * (yearf - yearf[0])

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
