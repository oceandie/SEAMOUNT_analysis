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
from utils import *
import cmocean

# ==============================================================================
# Input parameters

vel_component = "V"

# 1. INPUT FILES

exp_base_dir = '/data/users/dbruciaf/HPG/SEAMOUNT/2022-08-25/EAS02'
out_frq = '1h'

#hpg_list  = ['prj','sco','djc','ffl','ffq','fflr']
#vco_list  = ['s94', 'vqs', 'zco']
hpg_list  = ['djc','ffl','djr','fflr']
vco_list  = ['s94', 'vqs']

conf = ['tot']#,'zoom']

fig_path = '/home/h01/dbruciaf/mod_dev/SEAMOUNT_analysis/plots/2022-08-25/EAS02' 

j_sec = 31 # we plot a zonal cross section in the middle of the domain
tstep = (24*180)-1

# ==============================================================================

cmap    = cmocean.cm.curl
newcmap = truncate_colormap(cmap, 0.1, 0.9)

for vco in range(len(vco_list)):

    exp = vco_list[vco] + "_" + hpg_list[0] + "_pnt_fp4_" + out_frq
    domcfg = exp_base_dir + '/' + exp + '/domain_cfg_out.nc'

    # Loading domain geometry
    ds_dom = xr.open_dataset(domcfg, drop_variables=("x", "y","nav_lev")).squeeze()
    ds_dom = ds_dom.rename_dims({'nav_lev':'z'})

    # Computing land-sea masks
    ds_dom = compute_masks(ds_dom, merge=True)

    # Extracting variables
    e3t = ds_dom.e3t_0
    e3w = ds_dom.e3w_0
    e3u = ds_dom.e3u_0
    e3uw = ds_dom.e3uw_0  
    glamt = ds_dom.glamt
    glamu = ds_dom.glamu
    tmask = ds_dom.tmask   
 
    # Computing depths
    gdepw, gdept  = e3_to_dep(e3w,  e3t)
    gdepuw, gdepu = e3_to_dep(e3uw, e3u)

    print(np.nanmax(gdepw),np.nanmax(gdept),np.nanmax(gdepuw),np.nanmax(gdepu))

    # Adding 3rd dimension for plotting
    glamt = glamt.expand_dims({"z": len(ds_dom.z)})
    glamu = glamu.expand_dims({"z": len(ds_dom.z)})

    # Extracting arrays of the section
    gdept  = gdept.isel(y=j_sec).values
    gdepw  = gdepw.isel(y=j_sec).values
    gdepu  = gdepu.isel(y=j_sec).values
    gdepuw = gdepuw.isel(y=j_sec).values
    glamt  = glamt.isel(y=j_sec).values
    glamu  = glamu.isel(y=j_sec).values
    tmask  = tmask.isel(y=j_sec).values

    for hpg in range(len(hpg_list)):

        exp = vco_list[vco] + "_" + hpg_list[hpg] + "_pnt_fp4_" + out_frq
        exp_dir = exp_base_dir + "/" + exp

        dsU = xr.open_dataset(exp_dir + '/SEAMOUNT_'+exp+'_grid_'+vel_component+'.nc',
                              drop_variables=("x", "y","depth"+vel_component.lower()))
        if vel_component == "U":
           U = dsU.uoce.rolling({'x':2}).mean().fillna(0.)
        elif vel_component == "V":
           U = dsU.voce.rolling({'y':2}).mean().fillna(0.)
        U = U.rename({"depth"+vel_component.lower(): "z"})

        U = U.isel(time_counter=tstep,y=j_sec).values
        U[tmask==0] = np.nan

        # Plotting ----------------------------------------------------------
        for con in range(len(conf)): 

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 10))
            plt.sca(ax) # Set the current Axes to ax and the current Figure to the parent of ax.
            ax.invert_yaxis()
            ax.set_facecolor('gray')
 
            # -----------------------------------------------
            # We create a polygon patch for each T-cell of the
            # section and we colour it based on the value of the rmax
            #
            ni = U.shape[1]
            nk = U.shape[0]
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
                    colors = np.append(colors,U[k,i])

            # MODEL W-levels and U-points ----------------------------
            for k in range(len(ds_dom.z)):
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

            #if conf[con] == 'tot':
            #   ax.plot(
            #      [225., 225., 280.,280.,220.],
            #      [400.,1800.,1800.,400.,400.],
            #      color="limegreen",
            #      linestyle='--',
            #      linewidth=9.,
            #      zorder=5
            #   )

            # PLOT setting ----------------------------
            if conf[con] == 'zoom':
               ax.set_ylim(1800.0, 400.0)
               ax.set_xlim(225, 280)
            else:
               ax.set_ylim(4500.0, 0.0)
               ax.set_xlim(8, 500)

            p = PatchCollection(patches, alpha=1.0)
            p.set_array(np.array(colors))
            p.set_clim((-0.2,0.2))
            p.set_cmap(newcmap)
            ax.add_collection(p)
            cb = fig.colorbar(p, ax=ax, extend='both')    
            cb.set_label("u [$m\;s^{-1}$]", size=40)
            cb.ax.tick_params(labelsize=30)
            plt.xticks(fontsize=30.)
            plt.xlabel('Domain extension [km]', fontsize=40.)
            plt.yticks(fontsize=30.)
            plt.ylabel('Depth [m]' ,fontsize=40.)

            fig_name = '/'+vel_component+'_sec_j'+str(j_sec) + '_t'+str(tstep)+'_'+conf[con]+'_'+vco_list[vco]+'_'+hpg_list[hpg]+'.png'
            plt.savefig(fig_path+fig_name, bbox_inches="tight")
            plt.close()
