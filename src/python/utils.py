#!/usr/bin/env python

#     |------------------------------------------------------------|
#     | Author: Diego Bruciaferri                                  |
#     | Date and place: 28-02-2022, Met Office, UK                 |
#     |------------------------------------------------------------|


import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as colors
import xarray as xr
from typing import Tuple

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

def e3_to_dep(e3W: xr.DataArray, e3T: xr.DataArray) -> Tuple[xr.DataArray, ...]:

    gdepT = xr.full_like(e3T, None, dtype=np.double).rename('gdepT')
    gdepW = xr.full_like(e3W, None, dtype=np.double).rename('gdepW')

    gdepW[{"nav_lev":0}] = 0.0
    gdepT[{"nav_lev":0}] = 0.5 * e3W[{"nav_lev":0}]
    for k in range(1, e3W.sizes["nav_lev"]):
        gdepW[{"nav_lev":k}] = gdepW[{"nav_lev":k-1}] + e3T[{"nav_lev":k-1}]
        gdepT[{"nav_lev":k}] = gdepT[{"nav_lev":k-1}] + e3W[{"nav_lev":k}]

    return tuple([gdepW, gdepT])

def compute_tmask(top_level, bottom_level, k_lev):
    """
    Compute masks from domain_cfg Dataset.
    If merge=True, merge with the input dataset.
    Parameters
    ----------
    top_level: xr.Dataarray
        domain_cfg top_level dataarray
    bottom_level: xr.Dataarray
        domain_cfg bottom_level dataarray
    k_lev: xr.Dataarray
        domain_cfg level k-index starting from 1 to nk+1
    Returns
    -------
    tmask: xr.Dataarray
        tmask dataarray
    """  
    # Page 27 NEMO book.
    # I think there's a typo though.
    # It should be:
    #                  | 0 if k < top_level(i, j)
    # tmask(i, j, k) = | 1 if top_level(i, j) ≤ k ≤ bottom_level(i, j)
    #                  | 0 if k > bottom_level(i, j)
    tmask = xr.where(np.logical_or(k_lev < top_level, k_lev > bottom_level), 0, np.nan)
    tmask = xr.where(np.logical_and(bottom_level >= k_lev, top_level <= k_lev), 1, tmask)
    tmask = tmask.rename("tmask")

    tmask = tmask.transpose("nav_lev","y","x")

    return tmask

def regrid_UV_to_T(daU, daV):
    U = daU.rolling({'x':2}).mean().fillna(0.)
    V = daV.rolling({'y':2}).mean().fillna(0.)
    return U, V

def calc_speed(daU, daV):
    if "depthu" in daU.dims:
       daU = daU.rename({"depthu": "nav_lev"})
    if "depthv" in daV.dims:
       daV = daV.rename({"depthv": "nav_lev"})
    return np.sqrt(daU**2 + daV**2)

def calc_max_vel(daU, daV):
    return np.maximum(np.nanmax(np.absolute(daU)),np.nanmax(np.absolute(daV)))

def calc_KE(daU, daV):
    if "depthu" in daU.dims:
       daU = daU.rename({"depthu": "nav_lev"})
    if "depthv" in daV.dims:
       daV = daV.rename({"depthv": "nav_lev"})
    return 0.5 * (daU**2 + daV**2)

def calc_vol_avg(da, e1, e2, e3):
    cel_vol = e1 * e2 * e3
    dom_vol = cel_vol.sum(skipna=True)
    if "nav_lev" not in da:
       cel_vol = cel_vol.sum(dim="nav_lev",skipna=True)
    return (cel_vol*da).sum(skipna=True) / dom_vol

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap    
