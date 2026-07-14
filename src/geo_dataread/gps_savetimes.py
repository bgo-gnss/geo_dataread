#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
from __future__ import print_function

# standard lib
import argparse
import os
import sys
import traceback

# special lib
import geo_dataread.gps_read as gpsr
import geo_dataread.gps_write as gpsw
import gps_parser as cp


def main():
    """
    Program to calculete displacements/velocities between to points in time. from a list of stations
    """

    config = cp.ConfigParser()

    ref_allow = ["plate", "detrend", "itrf2008"]
    tformat = None

    parser = argparse.ArgumentParser(
        description="return gamit time series in the form of Time NEU DNEU."
    )
    parser.add_argument("Stations", nargs="+", help="List of stations")
    parser.add_argument(
        "--file",
        action="store_true",
        help="write to file Dir/STAT-ref.NEU where Dir is file directory defined by the"
        + " --Dir flag, STAT is the station four letter idendity, ref is defined by the --ref flag ",
    )
    parser.add_argument("--meters", action="store_true", help="print values in meters")
    parser.add_argument(
        "--ref",
        type=str,
        default="plate",
        choices=ref_allow,
        help="Reference frame: defaults to plate, remove plate velocity (plate), Detrend the time series (detrend), No filtering (itrf2008)",
    )
    parser.add_argument(
        "-tf",
        default=tformat,
        type=str,
        help="Format of the output time string If absent, -tf defaults to "
        "%%Y/%%m/%%d %%H:%%M:%%S"
        "." + " Special formating: "
        "yearf"
        " -> decimal year." + " See datetime documentation for formating",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="ALSO write an outlier-cleaned Dir/STA-ref_cleaned.NEU (plus a "
        "provenance sidecar .prov.json) next to the raw file. The raw file is "
        "always written unchanged. Requires --file.",
    )
    parser.add_argument(
        "-d",
        "--Dir",
        type=str,
        nargs="?",
        default="",
        const=os.getcwd(),
        help="output directory for files. Defaults to default figDir from cparser",
    )

    args = parser.parse_args()
    print(args, file=sys.stderr)

    if args.clean and not args.file:
        parser.error(
            "--clean requires --file (the cleaned .NEU and its "
            ".prov.json sidecar are file products)"
        )

    stations = args.Stations  # station list
    wfile = args.file
    ref = args.ref
    tformat = args.tf
    meters = not args.meters
    Dir = args.Dir

    if "all" in stations:  # geting a list of all the GPS stations
        stations = config.getStationInfo()

    for sta in stations:
        try:  # Trying to plot
            if wfile:
                outFile = os.path.join(Dir, f"{sta}-{ref}.NEU")
            else:
                outFile = sys.stdout

            print(f"writing to {outFile} ")
            gpsr.gamittooneuf(
                sta, outFile, mm=meters, ref=ref, dstring=tformat, rhour=True
            )
            # print "Time series of  %s using: %s, %s" % (sta, kwargs['ref'], kwargs['special'])

            if args.clean:
                # requested plain name; write_cleaned_neu inserts .DEGRADED
                # before the suffix when the station degrades (option b).
                cleanedFile = os.path.join(Dir, f"{sta}-{ref}_cleaned.NEU")
                result = gpsw.write_cleaned_neu(
                    sta, cleanedFile, ref=ref, mm=meters, dstring=tformat, rhour=True
                )
                print(
                    "wrote cleaned series to {outfile} — removed "
                    "{n_removed}/{n_total} epochs (degraded={degraded}); "
                    "provenance: {sidecar}".format(**result)
                )

        except (IndexError, TypeError) as e:
            tb = sys.exc_info()[2]
            last_trace = traceback.extract_tb(tb)[-1]
            errorstr = (
                f"{type(e).__name__}: "
                f"{os.path.basename(last_trace.filename)}, "
                f"line {last_trace.lineno} in {last_trace.name}, "
                f"For station {sta}"
            )
            print(errorstr, file=sys.stderr)
            traceback.print_exc()

        except Exception as e:
            print(traceback.format_exc(), file=sys.stderr)
            print(
                f"Unexpected error: {type(e).__name__} during processing of {sta}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
