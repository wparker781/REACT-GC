import numpy as np
import math
from pymsis import msis
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# -----------------------------------------------------------------------------
# Plot configuration
# -----------------------------------------------------------------------------
matplotlib.rcParams['font.family'] = 'Arial'


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
def main():

    # =====================================================================
    # 1. Time history of global-mean density during storm
    # =====================================================================

    dates = np.arange(
        np.datetime64("2024-05-09T00:00"),
        np.datetime64("2024-05-15T00:00"),
        np.timedelta64(30, "m")
    )

    # Alternative single-epoch or historic storm
    # dates = np.array(np.datetime64("2024-10-31T00:00"))
    # dates = np.arange(np.datetime64("2003-10-28T00:00"),
    #                   np.datetime64("2003-11-04T00:00"),
    #                   np.timedelta64(30, "m"))

    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 20)
    alt = np.array([500])  # km
    # alt = np.arange(200, 1000, 100)

    # geomagnetic_activity = -1 enables storm-time conditions
    data = msis.run(dates, lon, lat, alt, geomagnetic_activity=-1)

    # Global mean density at fixed altitude
    dens_avg = np.mean(data[:, :, :, 0, 0], axis=(1, 2))

    dens_expo_ref = dens_expo(alt)[0]

    plt.figure(figsize=(5, 4))
    plt.plot(dates, dens_avg, 'k')
    plt.xlabel('Date')
    plt.ylabel(r'$\rho$ [kg/m$^3$]')
    plt.xticks(rotation=45)

    ax2 = plt.gca().twinx()
    ax2.plot(dates, dens_avg / dens_expo_ref, 'k')
    ax2.set_ylabel('q')

    plt.tight_layout()
    plt.show()

    # =====================================================================
    # 2. Mean density vs altitude at storm epoch
    # =====================================================================

    # dates = np.array(np.datetime64("2024-05-09T00:00"))

    # Alternative epochs
    # dates = np.array(np.datetime64("2024-05-11T00:00"))
    dates = np.array(np.datetime64("2020-01-01T00:00"))
    # dates = np.arange(np.datetime64("2003-10-28T00:00"),
    #                   np.datetime64("2003-11-04T00:00"),
    #                   np.timedelta64(30, "m"))

    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 20)
    alt = np.arange(200, 1000, 20)

    data = msis.run(dates, lon, lat, alt, geomagnetic_activity=-1)

    dens_avg = np.mean(data[0, :, :, :, 0], axis=(0, 1))
    dens_expo_ref = dens_expo(alt)

    plt.figure(figsize=(3.6, 3))
    plt.semilogy(alt, dens_avg, 'k', label='MSIS')
    plt.semilogy(alt, dens_expo_ref, 'k--', label='Exponential ref')

    plt.xlabel('Altitude [km]')
    plt.ylabel(r'$\rho$ [kg/m$^3$]')
    # plt.legend()

    ax2 = plt.gca().twinx()
    ax2.plot(alt, dens_avg / dens_expo_ref, color='tab:blue', alpha=0.75)
    ax2.set_ylabel('q', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.tight_layout()
    plt.show()

    # =====================================================================
    # 3. Local density enhancement map at fixed altitude
    # =====================================================================

    lat = np.linspace(-90, 90, 20)
    lon = np.linspace(-180, 180, 40)
    alt = np.array([500])
    # alt = np.arange(200, 1000, 20)

    data = msis.run(dates, lon, lat, alt, geomagnetic_activity=-1)

    dens_expo_ref = dens_expo(alt)[0]

    # Create jet colormap with uniform alpha
    jet_rgba = plt.cm.jet(np.linspace(0, 1, 256))
    jet_rgba[:, -1] = 0.5
    jet_alpha = mcolors.ListedColormap(jet_rgba)

    plt.figure(figsize=(3.3, 3))

    # Note: axes intentionally swapped to match desired orientation
    lon_plt = np.linspace(-180, 180, len(lon))
    lat_plt = np.linspace(-90, 90, len(lat))

    # q_field = data[0, :, :, 0, 0] / dens_expo_ref
    q_field = (data[0, :, :, 0, 0] / dens_expo_ref).T


    plt.contourf(lon_plt, lat_plt, q_field, cmap=jet_alpha, aspect='auto')
    contours = plt.contour(lon_plt, lat_plt, q_field, colors='k', linewidths=0.5)

    matplotlib.rcParams['font.weight'] = 'bold'
    plt.clabel(contours, inline=True, fontsize=8, fmt="%.2f")

    plt.xlabel('Longitude [degrees]')
    plt.ylabel('Latitude [degrees]')

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Reference exponential density model
# -----------------------------------------------------------------------------
def dens_expo(h):

    params = [
        (0, 25, 0, 1.225, 7.249),
        (25, 30, 25, 3.899e-2, 6.349),
        (30, 40, 30, 1.774e-2, 6.682),
        (40, 50, 40, 3.972e-3, 7.554),
        (50, 60, 50, 1.057e-3, 8.382),
        (60, 70, 60, 3.206e-4, 7.714),
        (70, 80, 70, 8.77e-5, 6.549),
        (80, 90, 80, 1.905e-5, 5.799),
        (90, 100, 90, 3.396e-6, 5.382),
        (100, 110, 100, 5.297e-7, 5.877),
        (110, 120, 110, 9.661e-8, 7.263),
        (120, 130, 120, 2.438e-8, 9.473),
        (130, 140, 130, 8.484e-9, 12.636),
        (140, 150, 140, 3.845e-9, 16.149),
        (150, 180, 150, 2.070e-9, 22.523),
        (180, 200, 180, 5.464e-10, 29.74),
        (200, 250, 200, 2.789e-10, 37.105),
        (250, 300, 250, 7.248e-11, 45.546),
        (300, 350, 300, 2.418e-11, 53.628),
        (350, 400, 350, 9.518e-12, 53.298),
        (400, 450, 400, 3.725e-12, 58.515),
        (450, 500, 450, 1.585e-12, 60.828),
        (500, 600, 500, 6.967e-13, 63.822),
        (600, 700, 600, 1.454e-13, 71.835),
        (700, 800, 700, 3.614e-14, 88.667),
        (800, 900, 800, 1.17e-14, 124.64),
        (900, 1000, 900, 5.245e-15, 181.05),
        (1000, float('inf'), 1000, 3.019e-15, 268)
    ]

    dens = np.zeros(len(h))

    for i, h_i in enumerate(h):
        for h_min, h_max, h0, rho0, H in params:
            if h_min <= h_i < h_max:
                dens[i] = rho0 * math.exp(-(h_i - h0) / H)
                break

    return dens


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
