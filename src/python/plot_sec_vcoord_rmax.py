#!/usr/bin/env python

#     |------------------------------------------------------------|
#     | Author: Diego Bruciaferri                                  |
#     | Date and place: 28-02-2022, Met Office, UK                 |
#     |------------------------------------------------------------|


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.cm as cm
import xarray as xr
from utils import calc_r0, e3_to_dep, compute_masks

# ==============================================================================
# Input parameters

# 1. INPUT FILES

domcfg = ['/data/users/dbruciaf/HPG/SEAMOUNT/2022-07-12/s94_djc_pnt_fp4/domain_cfg_out.nc',
          '/data/users/dbruciaf/HPG/SEAMOUNT/2022-07-12/s94_djc_pnt_fp4/domain_cfg_out.nc', 
          '/data/users/dbruciaf/HPG/SEAMOUNT/2022-07-12/vqs_djc_pnt_fp4/domain_cfg_out.nc',
          '/data/users/dbruciaf/HPG/SEAMOUNT/2022-07-12/vqs_djc_pnt_fp4/domain_cfg_out.nc',
          '/data/users/dbruciaf/HPG/SEAMOUNT/2022-07-12/zco_djc_pnt_fp4/domain_cfg_out.nc']
conf = ['s94','s94-zoom','vqs','vqs-zoom', 'zco']

fig_path = '/home/h01/dbruciaf/mod_dev/SEAMOUNT_analysis/plots' 

j_sec = 31 # we plot a zonal cross section in the middle of the domain

# ==============================================================================

for exp in range(len(domcfg)):

    # Loading domain geometry
    ds_dom = xr.open_dataset(domcfg[exp], drop_variables=("x", "y","nav_lev"))
    # Computing land-sea masks
    ds_dom = compute_masks(ds_dom, merge=True)
    # Extracting variables
    e3t = ds_dom.e3t_0.squeeze()
    e3w = ds_dom.e3w_0.squeeze()
    e3u = ds_dom.e3u_0.squeeze()
    e3uw = ds_dom.e3uw_0.squeeze()  
    glamt = ds_dom.glamt.squeeze()
    glamu = ds_dom.glamu.squeeze()
    tmask = ds_dom.tmask.squeeze()   
 
    # Computing depths
    gdepw, gdept  = e3_to_dep(e3w,  e3t)
    gdepuw, gdepu = e3_to_dep(e3uw, e3u)

    print(np.nanmax(gdepw),np.nanmax(gdept),np.nanmax(gdepuw),np.nanmax(gdepu))

    # Computing slope paramter of model levels
    r0 = calc_r0(gdept.isel(nav_lev=-1))
     
    # Adding 3rd dimension for plotting
    r0    = r0.expand_dims({"nav_lev": len(ds_dom.nav_lev)})
    glamt = glamt.expand_dims({"nav_lev": len(ds_dom.nav_lev)})
    glamu = glamu.expand_dims({"nav_lev": len(ds_dom.nav_lev)})

    # Extracting arrays of the section
    gdept  = gdept.isel(y=j_sec).values
    gdepw  = gdepw.isel(y=j_sec).values
    gdepu  = gdepu.isel(y=j_sec).values
    gdepuw = gdepuw.isel(y=j_sec).values
    glamt  = glamt.isel(y=j_sec).values
    glamu  = glamu.isel(y=j_sec).values
    tmask  = tmask.isel(y=j_sec).values    
    r0     = np.copy(r0.isel(y=j_sec).values)

    r0[tmask==0] = np.nan

    # Plotting ----------------------------------------------------------

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 10))
    plt.sca(ax) # Set the current Axes to ax and the current Figure to the parent of ax.
    ax.invert_yaxis()
    ax.set_facecolor('gray')
 
    # RMAX -----------------------------------------------
    # We create a polygon patch for each T-cell of the
    # section and we colour it based on the value of the rmax
    #
    ni = r0.shape[1]
    nk = r0.shape[0]
    patches = []
    colors = []

    for k in range(0,nk-1):             
        for i in range(1,ni-1):
            x = [0.5*(glamt[k  ,i-1]+glamt[k   ,i ]), # U_(k,i-1)
                      glamt[k  ,i  ]                , # T_(k,i)
                 0.5*(glamt[k  ,i  ]+glamt[k  ,i+1]), # U_(k,i)
                 0.5*(glamt[k+1,i  ]+glamt[k+1,i+1]), # U_(k+1,i)
                      glamt[k+1,i  ]                , # T_(k+1,i)
                 0.5*(glamt[k+1,i-1]+glamt[k+1,i  ]), # U_(k+1,i-1)
                 0.5*(glamt[k  ,i-1]+glamt[k  ,i  ])] # U_(k  ,i-1)
                 
            y = [gdepuw[k  ,i-1],
                 gdepw [k  ,i  ],
                 gdepuw[k  ,i  ],
                 gdepuw[k+1,i  ],
                 gdepw [k+1,i  ],
                 gdepuw[k+1,i-1],
                 gdepuw[k  ,i-1]]

            polygon = Polygon(np.vstack((x,y)).T, True)
            patches.append(polygon)  
            colors = np.append(colors,r0[k,i])

    # MODEL W-levels and U-points ----------------------------
    for k in range(len(ds_dom.nav_lev)):
        x = glamt[k,:]
        z = gdepw[k,:]
        ax.plot(
            x,
            z,
            color="k",
            zorder=5
        )
    for i in range(len(ds_dom.x)):
        ax.plot(
           [glamu[0,i], glamu[-1,i]],
           [0., gdepuw[-1,i]],
           color="k",
           linestyle='--',
           zorder=5
        )

    # MODEL T-points ----------------------------
    ax.scatter(np.ravel(glamt[:-1,:]),
               np.ravel(gdept[:-1,:]),
               s=1,
               color='black',
               zorder=5
               )

    if conf[exp] == 's94' or conf[exp] == 'vqs':
       ax.plot(
          [225., 225., 280.,280.,220.],
          [400.,1800.,1800.,400.,400.],
          color="limegreen",
          linestyle='--',
          linewidth=9.,
          zorder=5
        )

    # PLOT setting ----------------------------
    if conf[exp] == 's94-zoom' or conf[exp] == 'vqs-zoom':
       ax.set_ylim(1800.0, 400.0)
       ax.set_xlim(225, 280)
    else:
       ax.set_ylim(4500.0, 0.0)
       ax.set_xlim(8, 500)

    p = PatchCollection(patches, alpha=0.7)
    p.set_array(np.array(colors))
    p.set_clim((0,0.36))
    cmap = cm.get_cmap("hot_r").copy()
    p.set_cmap(cmap)
    ax.add_collection(p)
    cb = fig.colorbar(p, ax=ax, extend='max')    
    cb.set_label("Slope parameter", size=40)
    cb.ax.tick_params(labelsize=30)
    plt.xticks(fontsize=30.)
    plt.xlabel('Domain extension [km]', fontsize=40.)
    plt.yticks(fontsize=30.)
    plt.ylabel('Depth [m]' ,fontsize=40.)

    fig_name = '/seamount_'+conf[exp]+'_rmax.png'
    plt.savefig(fig_path+fig_name, bbox_inches="tight")
    plt.close()
