#!/usr/bin/env python

#     |------------------------------------------------------------|
#     | Author: Diego Bruciaferri                                  |
#     | Date and place: 28-02-2022, Met Office, UK                 |
#     |------------------------------------------------------------|


import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import xarray as xr

def calc_r0(depth: xr.DataArray) -> xr.DataArray:
    """
    Calculate slope parameter r0: measure of steepness
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
            '/data/users/dbruciaf/HPG/SEAMOUNT/N-DJC-CTR_moderate/mesh_mask.nc',
            '/data/users/dbruciaf/HPG/SEAMOUNT/N-DJC-SH94_moderate/mesh_mask.nc',
            '/data/users/dbruciaf/HPG/SEAMOUNT/N-DJC-VQS_moderate/mesh_mask.nc',
            '/data/users/dbruciaf/HPG/SEAMOUNT/N-DJC-VQS_steep/mesh_mask.nc',
            '/data/users/dbruciaf/HPG/SEAMOUNT/N-DJC-VQS_steep/mesh_mask.nc']
conf = ['ctr-s','ctr-m','sh94-m','vqs-m','vqs-s','vqs-s-zoom']

fig_path = '/home/h01/dbruciaf/mod_dev/SEAMOUNT_analysis/plots' 

# ==============================================================================

for exp in range(len(meshmask)):

    # Loading domain geometry
    ds_dom = xr.open_dataset(meshmask[exp], drop_variables=("x", "y"))

    # Computing slope paramter of model levels
    r0 = calc_r0(ds_dom.gdept_0.squeeze().isel(nav_lev=-1))
    r0 = r0.drop_vars("nav_lev")
     
    # Adding 3rd dimension for plotting
    r0    = r0.expand_dims({"nav_lev": 20})
    glamt = ds_dom.glamt.expand_dims({"nav_lev": 20})
    glamu = ds_dom.glamu.expand_dims({"nav_lev": 20})

    # Plotting ----------------------------------------------------------

    levT_y = ds_dom.gdept_0.isel(y=31).squeeze()
    levW_y = ds_dom.gdepw_0.isel(y=31).squeeze()
    glamT_y = glamt.isel(y=31).squeeze()
    tmask_y = ds_dom.tmask.isel(y=31).squeeze()
    tmask_y = tmask_y.drop_vars("nav_lev")

    r0_y = r0.isel(y=31).squeeze()
    r0_y = r0_y.where(tmask_y==1)
    
    fig = plt.figure(figsize=(30, 20), dpi=100)
    ax = fig.add_subplot()
    ax.invert_yaxis()
    ax.set_facecolor('silver')

    # RMAX -----------------------------------------------
    
    # Creating grid for plotting T-grid values
    # at the centre of the cells
    ni = len(levT_y.x)
    nk = len(levT_y.nav_lev)

    # Computing depths on U and W grid
    levUT_y = levT_y.rolling({"x": 2}).mean().fillna(levT_y[:, 0])
    levUW_y = levUT_y.rolling({"nav_lev": 2}).mean().fillna(0.0) # surface

    # Computing coords on U grid
    glamU_y = glamT_y.rolling({"x": 2}).mean()
    glamU_y[:, 0] = glamT_y[:, 0] - (glamU_y[:, 1] - glamT_y[:, 0])

    # Creating arrays with shape (nk+1,n+1) 
    levUW_yp1 = np.zeros(shape=(nk+1, ni+1))
    glamU_yp1 = np.zeros(shape=(nk+1, ni+1))

    levUW_yp1[:-1, :-1] = levUW_y.data
    glamU_yp1[:-1, :-1] = glamU_y.data

    cmap = cm.get_cmap("hot_r").copy()
 
    if '-m' in conf[exp]:
       pc = ax.pcolormesh(
                  glamU_yp1, 
                  levUW_yp1, 
                  r0_y, 
                  cmap=cmap,
                  vmin=0.,
                  vmax=0.07 
            )
    else:
       pc = ax.pcolormesh(
                  glamU_yp1, 
                  levUW_yp1,
                  r0_y,
                  cmap=cmap,
                  vmin=0.,
                  vmax=0.35
            )

    # MODEL W-levels and U-points ----------------------------
    for k in range(len(ds_dom.nav_lev)):
        if 'ctr-s' in conf[exp]:
           # To avoid visual inconsistency
           # due to the steepness 
           x = glamT_y.isel(nav_lev=k)
           z = levW_y.isel(nav_lev=k)
        else:
            x = glamU_y.isel(nav_lev=k)
            z = levUW_y.isel(nav_lev=k)
        ax.plot(
            x,
            z,
            color="k",
            zorder=5
        )
    for i in range(len(ds_dom.x)):
        ax.plot(
           [glamU_y.isel(x=i)[0],
            glamU_y.isel(x=i)[-1]],
           [np.nanmin(levW_y),
            levUW_y.isel(x=i)[-1]],
           color="k",
           linestyle='--',
           zorder=5
        )

    if conf[exp] == 'vqs-s':
       ax.plot(
          [225., 225., 280.,280.,220.],
          [500.,1800.,1800.,500.,500.],
          color="limegreen",
          linestyle='--',
          linewidth=9.,
          zorder=5
        )


    if 'ctr-s' in conf[exp]:
       # To avoid visual inconsistency
       # due to the steepness
       path = [
           [glamT_y[-1, 0], levT_y[-1, 0]],
           [glamT_y[-1, -1], levT_y[-1, -1]],
           ]
       for i in range(len(levW_y.x)):
           path.append([glamT_y[-1, i], levW_y[-1, i]])
       ax.add_patch(patches.Polygon(path, facecolor="silver", zorder=4))

    # PLOT setting ----------------------------
    if conf[exp] == 'vqs-s-zoom':
       ax.set_ylim(1800.0, 500.0)
       ax.set_xlim(225, 280)
    else:
       ax.set_ylim(4500.0, 0.0)
       ax.set_xlim(8, 500)
    cb = plt.colorbar(pc)
    cb.set_label("Slope parameter", size=40)
    cb.ax.tick_params(labelsize=30)
    plt.xticks(fontsize=30.)
    plt.xlabel('Domain extension [km]', fontsize=40.)
    plt.yticks(fontsize=30.)
    plt.ylabel('Depth [m]' ,fontsize=40.)


    fig_name = '/seamount_'+conf[exp]+'_rmax.png'
    plt.savefig(fig_path+fig_name, bbox_inches="tight")
    plt.close()
