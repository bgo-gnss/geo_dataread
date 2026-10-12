#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
from __future__ import print_function


def xyzDict():
    """
    {station: [X, Y, Z]} from the a-priori coordinate file (postprocess.cfg coordFile)
    """

    from gps_parser import ConfigParser

    xyzDict = {}
    with open(ConfigParser().getPostProcessConfig("coordFile")) as f:
        for line in f:
            tok = line.split()
            if len(tok) >= 4:
                xyzDict[tok[3]] = [float(v) for v in tok[:3]]

    return xyzDict


def llh(sta, radians=False):
    """
    (longitude, latitude, ellipsoidal height) of a GPS station from its
    a-priori ITRF2008 coordinate (GRS80)
    """

    import numpy as np
    from geofunc import local

    g = local.ecef_to_geodetic(xyzDict()[sta])
    lon, lat = (g.lon, g.lat) if radians else (np.degrees(g.lon), np.degrees(g.lat))
    return float(lon), float(lat), float(g.h)


def simpleDisp(sta, start, end):
    """
    Calculate simple displacement vector between dates start and end. Time series extracted from gamit time series
    """

    import geo_dataread.gps_read as gpsr
    import numpy as np
    import datetime as dt
    import gtimes.timefunc as timefunc

    day = 0.00273224
    start = timefunc.TimetoYearf(start.year, start.month, start.day)
    # end = timefunc.TimetoYearf(end.year,end.month,end.day)

    # Fetch the gamit time series and return them in the format
    # 'UTC (%Y/%m/%d %H:%M:%s.000) N[m] E[m] U[m] DN[m] DE[m] DU[m]
    NEUdata = gpsr.gamittoNEU(
        sta, mm=True, ref="plate", dstring="%Y/%m/%d 12:00:00.000"
    )

    # print( NEUdata[np.where( np.logical_and( NEUdata['yearf'] > start-2*day, NEUdata['yearf'] < start+day ) )] )
    # print( dt.datetime.strftime( start ,"%Y/%m/%d 12:00:00.000") )
    # print( NEUdata[np.where([ x[0] == dt.datetime.strftime( start ,"%Y/%m/%d 12:00:00.000") for x in NEUdata ] ) ] )

    print(NEUdata)
    startperiod = NEUdata[
        np.where(
            [
                x[0] == dt.datetime.strftime(start, "%Y/%m/%d 12:00:00.000")
                for x in NEUdata
            ]
        )
    ][0]
    endperiod = NEUdata[
        np.where(
            [
                x[0] == dt.datetime.strftime(end, "%Y/%m/%d 12:00:00.000")
                for x in NEUdata
            ]
        )
    ][0]
    disp = [(endperiod[i] - startperiod[i]) for i in [1, 2, 3]]
    uncert = [(np.sqrt(endperiod[i] ** 2 + startperiod[i] ** 2)) for i in [4, 5, 6]]

    return disp, uncert


def fitDisp(sta, start, end):
    """
    Calculate  displacement vector between dates start and end using a smiple linear fit. Time series extracted from gamit time series
    """

    import numpy as np
    import datetime as dt

    import geo_dataread.gps_read as gpsr
    import gtimes.timefunc as timefunc
    import matplotlib.pyplot as plt

    start = timefunc.TimetoYearf(start.year, start.month, start.day)
    end = timefunc.TimetoYearf(end.year, end.month, end.day)

    NEUdata = gpsr.gamittoNEU(sta, mm=True, ref="plate", dstring="yearf")
    fitperiod = NEUdata[
        np.where(np.logical_and(NEUdata["yearf"] > start, NEUdata["yearf"] < end))
    ]

    x = fitperiod["yearf"]
    N = fitperiod["data[0]"]
    E = fitperiod["data[1]"]
    U = fitperiod["data[2]"]
    DN = fitperiod["Ddata[0]"]
    DE = fitperiod["Ddata[1]"]
    DU = fitperiod["Ddata[2]"]

    fitdeg = 1

    fit, cov = np.polyfit(x, np.transpose([N, E, U]), fitdeg, cov=True)
    print("The fit {0}".format(fit))
    print("The cov {0}".format(np.sqrt(np.diag(cov[:, :, 0]))))
    print("The cov {0}".format(np.sqrt(np.diag(cov[:, :, 1]))))
    print("The cov {0}".format(np.sqrt(np.diag(cov[:, :, 2]))))

    fNEU = lambda x, fit: np.polyval(fit, np.vstack([x for i in range(3)]).T)
    TT = np.vstack([x ** (fitdeg - i) for i in range(fitdeg + 1)]).T
    yi = np.dot(TT, fit)
    C_yin = np.dot(TT, np.dot(cov[:, :, 0], TT.T))
    C_yie = np.dot(TT, np.dot(cov[:, :, 1], TT.T))
    C_yiu = np.dot(TT, np.dot(cov[:, :, 2], TT.T))

    sig_yin = np.sqrt(np.diag(C_yin))
    sig_yie = np.sqrt(np.diag(C_yie))
    sig_yiu = np.sqrt(np.diag(C_yiu))

    fN = lambda x: fit[1, 0] + fit[0, 0] * x
    fE = lambda x: fit[1, 1] + fit[0, 1] * x
    fU = lambda x: fit[1, 2] + fit[0, 2] * x

    # print(fNEU)

    # plt.plot(x,U,'bo-',x,fU(x),'r-')
    # plt.show()

    disp = fNEU(end, fit) - fNEU(start, fit)
    print(disp)

    plotdimens = {"N": 0, "E": 1, "U": 2}
    NEU = "N"

    # fg, ax = plt.subplots(1, 1)
    # ax.set_title("Fit for Polynomial (degree {}) with $\pm1\sigma$-interval".format(fitdeg))
    # ax.fill_between(x, yi[:,plotdimens[NEU]]+sig_yin, yi[:,plotdimens[NEU]]-sig_yin, alpha=.25)
    # ax.plot(x, yi[:,plotdimens[NEU]],'-')
    # ax.plot(x, N, 'ro')
    # ax.axis('tight')

    # fg.canvas.draw()
    # plt.show()


def main():
    """
    Program to calculete displacements/velocities between to points in time. from a list of stations
    """

    import argparse
    import sys

    import gtimes.timefunc as timefunc
    import datetime as dt
    import matplotlib.pyplot as plt

    dstr = "%Y%m%d"
    outpstr = "%Y%m%d"
    sday = dt.datetime.now(dt.timezone.utc) - dt.timedelta(1)

    parser = argparse.ArgumentParser(
        description="Return displacements/velocities of a list of stations between two given dates in NEU format."
    )
    parser.add_argument("Stations", nargs="+", help="List of stations")
    parser.add_argument("--file", action="store_true", help="write to *.NEU file")
    parser.add_argument("-s", "--start", required=True, help="reference point")
    parser.add_argument("-e", "--end", default=sday, help="displacement end")
    parser.add_argument(
        "-f",
        default=dstr,
        type=str,
        help="Format of the string passed to -s and -e. If absent, -d defaults to "
        "%%Y-%%m-%%-%%m"
        "." + " Special formating: "
        "yearf"
        " -> fractional year "
        "w-dow"
        " -> GPS Week-Day of Week." + " See datetime documentation for formating",
    )

    args = parser.parse_args()
    stations = args.Stations  # station list

    wfile = args.file

    start = timefunc.toDatetime(args.start, args.f)
    end = timefunc.toDatetime(args.end, args.f)

    if wfile:
        outFile = open(
            "{0:s}-{1:s}.NEU".format(start.strftime(outpstr), end.strftime(outpstr)),
            "w+",
        )
    else:
        outFile = sys.stdout

    header = "#lon       lat\t\t   N[mm]   DN[mm]\t  E[mm]    DE[mm]\t  U[mm]    DU[mm]\t\tStation"

    print(header, file=outFile)  # ,

    for sta in stations:
        disp, uncert = simpleDisp(sta, start, end)
        # fitDisp(sta, start, end)

        coord = llh(sta)
        print(
            "{0:5.6f} {1:5.6f}\t{2:7.2f} {5:7.2f}\t{3:7.2f} {6:7.2f}\t{4:7.2f} {6:7.2f}\t\t{8:s}".format(
                coord[0],
                coord[1],
                disp[0],
                disp[1],
                disp[2],
                uncert[0],
                uncert[1],
                uncert[2],
                sta,
            ),
            file=outFile,
        )  # ,

    # print( '{0:5.6f} {1:5.6f}\t{2:7.2f} {5:7.2f}\t{3:7.2f} {6:7.2f}\t{4:7.2f} {6:7.2f}\t\t{8:s}'.format(
    #     coord[0], coord[1], disp[0], disp[1], disp[2], uncert[0], uncert[1], uncert[2],sta), file=outFile ) #,

    outFile.close()


if __name__ == "__main__":
    main()
