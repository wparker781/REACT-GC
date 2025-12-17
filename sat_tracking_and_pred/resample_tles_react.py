import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import pymc as pm
from scipy.interpolate import interp1d, CubicSpline
import datetime as dt
import csv
from skyfield.api import EarthSatellite, load
import time
import pytz
from datetime import timezone
from concurrent.futures import ProcessPoolExecutor
import pickle as pkl


def main():
    """
    Read raw TLE histories for a set of satellites, extract orbital
    elements, resample onto a common hourly grid, and save the
    interpolated altitude time series.

    This script:
      - Parses TLE files in parallel
      - Extracts semi-major axis and inclination histories
      - Converts SMA to altitude assuming circular orbits
      - Resamples all objects to a common hourly timeline
      - Saves the resulting altitude matrix for downstream analysis
    """

    # ----------------------------------------------------------
    # Plotting configuration
    # ----------------------------------------------------------
    mpl.rcParams['font.family'] = 'Arial'

    # ----------------------------------------------------------
    # Physical constants
    # ----------------------------------------------------------
    earth_radius = 6378.137  # km
    mu = 398600.4418         # km^3/s^2 (unused here, but kept for consistency)

    # ----------------------------------------------------------
    # Input directory containing per-object TLE histories
    # ----------------------------------------------------------
    dir_loc = 'ref_tles_2024'
    files = os.listdir(dir_loc)

    # Containers for parsed data
    sma = []       # semi-major axis [km]
    incl = []      # inclination [rad]
    t = []         # epoch datetimes
    satcat = []    # NORAD catalog numbers

    # Remove macOS metadata files
    files = [file for file in files if '.DS_Store' not in file]

    # ----------------------------------------------------------
    # Parallel TLE parsing
    # ----------------------------------------------------------
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(process_file, os.path.join(dir_loc, file))
            for file in files
        ]

        for i, future in enumerate(futures):
            sma_col, incl_col, t_col = future.result()
            sma.append(sma_col)
            incl.append(incl_col)
            t.append(t_col)

            # Assume NORAD ID is encoded in filename
            satcat.append(int(files[i][:5]))

            print(i / len(files) * 100, '% done')

    print('Time to read in TLEs:', time.time() - t0)

    # ----------------------------------------------------------
    # Remove objects with no valid data
    # ----------------------------------------------------------
    idx_real = [i for i in range(len(sma)) if len(sma[i]) > 0]

    sma = [sma[i] for i in idx_real]
    incl = [incl[i] for i in idx_real]
    t = [t[i] for i in idx_real]
    satcat = [satcat[i] for i in idx_real]

    # ----------------------------------------------------------
    # Determine common overlapping time interval
    # ----------------------------------------------------------
    start_date = max([np.min(t[i]) for i in range(len(t))])
    end_date = min([np.max(t[i]) for i in range(len(t))])

    # Shorten interval slightly to avoid edge extrapolation
    end_date = end_date - dt.timedelta(days=1)

    # ----------------------------------------------------------
    # Construct common hourly time grid
    # ----------------------------------------------------------
    def datetime_range_ts(start, end, delta):
        """
        Generate a list of UNIX timestamps from start to end
        with fixed spacing, rounded to the nearest hour.
        """
        current = start
        dt_range = []

        while current < end:
            current += delta
            current = current.replace(minute=0, second=0, microsecond=0)
            dt_range.append(current.timestamp())

        return dt_range

    new_t_ts = datetime_range_ts(start_date, end_date, dt.timedelta(hours=1))
    new_t_dt = [dt.datetime.fromtimestamp(ts) for ts in new_t_ts]

    # ----------------------------------------------------------
    # Convert original epochs to timestamps
    # ----------------------------------------------------------
    t_ts = []
    for i in range(len(t)):
        t_ts.append([t[i][j].timestamp() for j in range(len(t[i]))])

    # ----------------------------------------------------------
    # Interpolate SMA histories onto common grid
    # ----------------------------------------------------------
    new_sma = np.zeros((len(sma), len(new_t_ts)))

    for i in range(len(sma)):
        new_sma[i, :] = np.interp(
            new_t_ts,
            t_ts[i],
            sma[i],
            left=np.nan,
            right=np.nan
        )

    # Convert semi-major axis to altitude
    new_alt = new_sma - earth_radius

    # ----------------------------------------------------------
    # Reorder satellites to a fixed reference ordering
    # ----------------------------------------------------------
    satcat_correct_order = [
        34106, 33874, 34486, 34696, 34648,
        34428, 33784, 34473, 33911, 34588,
        39452, 39451, 39453, 44714, 49172,
        53057, 25544, 43613, 43180, 54370
    ]

    idx_satcat_order = np.array(
        [
            satcat.index(satcat_correct_order[i])
            for i in range(len(satcat))
            if satcat[i] in satcat_correct_order
        ],
        dtype=int
    )

    satcat = [satcat[i] for i in idx_satcat_order]
    new_alt = new_alt[idx_satcat_order, :]

    # ----------------------------------------------------------
    # Diagnostic plot: altitude vs time
    # ----------------------------------------------------------
    plt.figure()
    for i in range(len(new_alt)):
        plt.plot(new_t_dt, new_alt[i], label=str(satcat[i]))

    plt.xlabel('Time (UTC)')
    plt.ylabel('Altitude (km)')
    plt.title('Altitude vs Time')
    plt.legend()
    plt.show()

    # ----------------------------------------------------------
    # Save resampled data
    # ----------------------------------------------------------
    op_file = 'data/example_objs_interp_temp.pkl'
    with open(op_file, 'wb') as f:
        pkl.dump([new_t_dt, new_alt.T, satcat], f)

    print('Resampled TLEs saved to:', op_file)


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def get_satellite(tle_line1, tle_line2):
    """
    Construct a Skyfield EarthSatellite object from TLE lines.
    """
    return EarthSatellite(tle_line1, tle_line2, 'Satellite', load.timescale())


def process_file(file):
    """
    Parse a single TLE file and extract:
      - Semi-major axis
      - Inclination
      - Epoch time

    Returns time-ordered histories for one satellite.
    """

    if '.DS_Store' in file:
        return [], [], []

    print('Loading file:', file)

    with open(file, encoding="utf8", errors='ignore') as csvfile:
        tle_lines = csvfile.readlines()

    tle_l1 = np.zeros(int(len(tle_lines) / 2), dtype=object)
    tle_l2 = np.zeros(int(len(tle_lines) / 2), dtype=object)

    sma_col = []
    incl_col = []
    t_col = []

    earth_radius = 6378.137  # km

    for k in range(int(len(tle_lines) / 2)):
        tle_l1[k] = tle_lines[2 * k][:-1]
        tle_l2[k] = tle_lines[2 * k + 1][:-1]

        year = int(tle_l1[k][18:20])

        # Only keep post-2017 epochs
        if year >= 17:
            satellite = get_satellite(tle_l1[k], tle_l2[k])

            a = satellite.model.a * earth_radius  # km
            inc = satellite.model.inclo           # rad
            epoch = satellite.epoch.utc_datetime()

            sma_col.append(a)
            incl_col.append(inc)
            t_col.append(epoch)

    # Reverse so time increases forward
    return sma_col[::-1], incl_col[::-1], t_col[::-1]


if __name__ == '__main__':
    main()
