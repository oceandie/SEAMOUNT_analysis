#!/usr/bin/env python

#     |------------------------------------------------------------|
#     | Author: Diego Bruciaferri                                  |
#     | Date and place: 07-09-2021, Met Office, UK                 |
#     |------------------------------------------------------------|


import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import xarray as xr
import cmocean

def calc_rmax(depth: xr.DataArray) -> xr.DataArray:
    """
    Calculate rmax: measure of steepness
    This function returns the slope paramater field

    r = abs(Hb - Ha) / (Ha + Hb)

    where Ha and Hb are the depths of adjacent grid cells (Mellor et al 1998).

    Reference:
    *) Mellor, Oey & Ezer, J Atm. Oce. Tech. 15(5):1122-1131, 1998.

    Parameters
    ----------
    depth: DataArray
        Bottom depth (units: m).

    Returns
    -------
    DataArray
        2D slope parameter (units: None)

    Notes
    -----
    This function uses a "conservative approach" and rmax is overestimated.
    rmax at T points is the maximum rmax estimated at any adjacent U/V point.
    """
    # Mask land
    depth = depth.where(depth > 0)

    # Loop over x and y
    both_rmax = []
    for dim in depth.dims:

        # Compute rmax
        rolled = depth.rolling({dim: 2}).construct("window_dim")
        diff = rolled.diff("window_dim").squeeze("window_dim")
        rmax = np.abs(diff) / rolled.sum("window_dim")

        # Construct dimension with velocity points adjacent to any T point
        # We need to shift as we rolled twice
        rmax = rmax.rolling({dim: 2}).construct("vel_points")
        rmax = rmax.shift({dim: -1})

        both_rmax.append(rmax)

    # Find maximum rmax at adjacent U/V points
    rmax = xr.concat(both_rmax, "vel_points")
    rmax = rmax.max("vel_points", skipna=True)

    # Mask halo points
    for dim in rmax.dims:
        rmax[{dim: [0, -1]}] = 0

    return rmax.fillna(0)

# ==============================================================================
# Input parameters

# 1. INPUT FILES

meshmask = ['/data/users/dbruciaf/HPG/SEAMOUNT/N-DJC-CTR_steep/mesh_mask.nc',
            '/data/users/dbruciaf/HPG/SEAMOUNT/N-DJC-CTR_moderate/mesh_mask.nc']
conf = ['s','m']

fig_path = '/home/h01/dbruciaf/mod_dev/SEAMOUNT_analysis/plots' 

# ==============================================================================

for exp in range(len(meshmask)):

    # Loading domain geometry
    ds_dom = xr.open_dataset(meshmask[exp], drop_variables=("x", "y"))

    # Computing rmax
    rmax  = calc_rmax(ds_dom.bathymetry.squeeze())
    rmax  = rmax.expand_dims({"nav_lev": 20})
    glamt = ds_dom.glamt.expand_dims({"nav_lev": 20})
    glamu = ds_dom.glamu.expand_dims({"nav_lev": 20})

    # Plotting ----------------------------------------------------------

    levt_y = ds_dom.gdept_0.isel(y=32).squeeze()
    levw_y = ds_dom.gdepw_0.isel(y=32).squeeze()
    glamt_y = glamt.isel(y=32).squeeze()
    glamu_y = glamu.isel(y=32).squeeze()

    rmax_y = rmax.isel(y=32).squeeze()

    fig = plt.figure(figsize=(30, 20), dpi=100)
    ax = fig.add_subplot()
    ax.invert_yaxis()

    # rmax
    pc = ax.pcolormesh(glamt_y, levw_y, rmax_y, cmap='hot_r')

    # W-levels and T-points
    for k in range(len(ds_dom.nav_lev)):
        ax.plot(
            glamt_y.isel(nav_lev=k),
            levw_y.isel(nav_lev=k),
            color="k",
        )
        ax.scatter(
            glamt_y.isel(nav_lev=k),
            levt_y.isel(nav_lev=k),
            s = 5.0,
            color="k",
        )

    # Mesh-lines
    for i in range(len(ds_dom.x)):
        ax.plot(
           glamu_y.isel(x=i),
           levw_y.isel(x=i),
           color="k",
           linestyle='--'
        )
    ax.set_ylim(4500.0, 0.0)
    cb = plt.colorbar(pc)
    cb.set_label("Slope parameter", size=40)
    cb.ax.tick_params(labelsize=30)
    plt.xticks(fontsize=30.)
    plt.xlabel('Domain extension [km]', fontsize=40.)
    plt.yticks(fontsize=30.)
    plt.ylabel('Depth [m]' ,fontsize=40.)

    path = [
        [glamt_y[-1, 0], levt_y[-1, 0]],
        [glamt_y[-1, -1], levt_y[-1, -1]],
        ]
    for i in range(len(levw_y.x)):
        path.append([glamt_y[-1, i], levw_y[-1, i]])
    ax.add_patch(patches.Polygon(path, facecolor="silver", zorder=4))

    fig_name = '/seamount_'+conf[exp]+'_rmax.png'
    plt.savefig(fig_path+fig_name, bbox_inches="tight")
    plt.close()
