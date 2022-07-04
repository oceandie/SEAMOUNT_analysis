#!/usr/bin/env python

#     |------------------------------------------------------------|
#     | Author: Diego Bruciaferri                                  |
#     | Date and place: 07-09-2021, Met Office, UK                 |
#     |------------------------------------------------------------|

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
from utils import *

# ==============================================================================
# Input parameters

# 1. INPUT FILES

exp_base_dir = '/data/users/dbruciaf/HPG/SEAMOUNT/2022-07-04'

color_hpg = ['black','red','blue','limegreen','deepskyblue','gold']
style_vco = ["solid", "dotted", "dashed"]
#hpg_list  = ['sco','prj','djc','ffl','ffq','fflr']
hpg_list  = ['sco','djc','ffl','fflr']
vco_list  = ['s94', 'vqs','zco']

fig_path = '/home/h01/dbruciaf/mod_dev/SEAMOUNT_analysis/plots' 

# ==============================================================================

fig1, ax1 = plt.subplots(figsize=(16,9)) # KE
fig2, ax2 = plt.subplots(figsize=(16,9)) # KE barotropic
fig3, ax3 = plt.subplots(figsize=(16,9)) # C
fig4, ax4 = plt.subplots(figsize=(16,9)) # U
fig5, ax5 = plt.subplots(figsize=(16,9)) # U barotropic 

for hpg in range(len(hpg_list)):

    for vco in range(len(vco_list)):

        for ini in ['pnt']:

            exp_dir = exp_base_dir + "/" + vco_list[vco] + "_" + hpg_list[hpg] + "_" + ini + "_fp4"

            # Loading NEMO geometry
            ds_dom = xr.open_dataset(exp_dir + '/domain_cfg_out.nc', drop_variables=("x", "y","nav_lev")).squeeze()
            ds_dom["tmask"] = compute_tmask(ds_dom.top_level, ds_dom.bottom_level, ds_dom.nav_lev + 1)

            e1t = ds_dom.e1t.where(ds_dom.tmask==1)
            e2t = ds_dom.e2t.where(ds_dom.tmask==1)

            # Loading NEMO output
            dsT = xr.open_dataset(exp_dir + '/SEAMOUNT_xxx_1h_grid_T.nc',drop_variables=("x", "y","deptht"))
            dsU = xr.open_dataset(exp_dir + '/SEAMOUNT_xxx_1h_grid_U.nc',drop_variables=("x", "y","depthu"))
            dsV = xr.open_dataset(exp_dir + '/SEAMOUNT_xxx_1h_grid_V.nc',drop_variables=("x", "y","depthv"))

            e3t = dsT.e3t
            e3t = e3t.rename({"deptht": "nav_lev"})
            e3t = e3t.where(ds_dom.tmask==1)

            # Interp. to T-grid
            U , V  = regrid_UV_to_T(dsU.uoce, dsV.voce)
            Ub, Vb = regrid_UV_to_T(dsU.ubar, dsV.vbar)
 
            # Calc KE
            KE  = calc_KE(U, V)
            KEb = calc_KE(Ub, Vb)
            KE  = KE.where(ds_dom.tmask==1)
            KEb = KEb.where(ds_dom.tmask[0,:,:]==1)

            # Calc speed
            C  = calc_speed(U, V)
            C  = C.where(ds_dom.tmask==1)
            U  = U.where(ds_dom.tmask==1)
            Ub = Ub.where(ds_dom.tmask[0,:,:]==1)
            V  = V.where(ds_dom.tmask==1)
            Vb = Vb.where(ds_dom.tmask[0,:,:]==1)

            KE_avg  = []
            KEb_avg = []
            C_max   = []
            U_max   = []
            Ub_max  = []
            for t in range(len(KE.time_counter)):
                KE_avg.append(calc_vol_avg(KE[t,:,:,:], e1t, e2t, e3t[t,:,:,:]).values)
                KEb_avg.append(calc_vol_avg(KEb[t,:,:], e1t, e2t, e3t[t,:,:,:]).values)
                C_max.append(C[t,:,:,:].max(skipna=True))
                U_max.append(calc_max_vel(U[t,:,:,:], V[t,:,:,:]))
                Ub_max.append(calc_max_vel(Ub[t,:,:], Vb[t,:,:]))

            lab_exp = hpg_list[hpg] + "_" + vco_list[vco]
            print(lab_exp)

            ax1.plot(range(len(KE_avg)),
                     KE_avg,
                     color=color_hpg[hpg],
                     linestyle=style_vco[vco],
                     label=lab_exp,
                     linewidth=3.
            )

            ax2.plot(range(len(KEb_avg)),
                     KEb_avg,
                     color=color_hpg[hpg],
                     linestyle=style_vco[vco],
                     label=lab_exp,
                     linewidth=3.
            )

            ax3.plot(range(len(C_max)),
                     C_max,
                     color=color_hpg[hpg],
                     linestyle=style_vco[vco],
                     label=lab_exp,
                     linewidth=3.
            )

            ax4.plot(range(len(U_max)),
                     U_max,
                     color=color_hpg[hpg],
                     linestyle=style_vco[vco],
                     label=lab_exp,
                     linewidth=3.
            )
         
            ax5.plot(range(len(Ub_max)),
                     Ub_max,
                     color=color_hpg[hpg],
                     linestyle=style_vco[vco],
                     label=lab_exp,
                     linewidth=3.
            )

#ax.set_xlim(0., 120.)
#if conf == 'steep':
#   ax.set_ylim(0., 1.8)
#elif conf == 'moderate':
#   ax.set_ylim(0., 0.02)
#plt.xticks(fontsize=30.)
#plt.xlabel('Time [hours]', fontsize=40.)
#plt.yticks(fontsize=30.)
#plt.ylabel('Maximum velocity error [$m\;s^{-1}$]' ,fontsize=40.)

plt.rc('legend', **{'fontsize':10})
ax1.legend(loc=0, ncol=1, frameon=False)
ax2.legend(loc=0, ncol=1, frameon=False)
ax3.legend(loc=0, ncol=1, frameon=False)
ax4.legend(loc=0, ncol=1, frameon=False)
ax5.legend(loc=0, ncol=1, frameon=False)

fig_name = '/KE_timeseries.png'
fig1.savefig(fig_path+fig_name, bbox_inches="tight")
plt.close(fig1)
fig_name = '/KEbaro_timeseries.png'
fig2.savefig(fig_path+fig_name, bbox_inches="tight")
plt.close(fig2)
fig_name = '/Cmax_timeseries.png'
fig3.savefig(fig_path+fig_name, bbox_inches="tight")
plt.close(fig3)
fig_name = '/Umax_timeseries.png'
fig4.savefig(fig_path+fig_name, bbox_inches="tight")
plt.close(fig4)
fig_name = '/Ubaro_max_timeseries.png'
fig5.savefig(fig_path+fig_name, bbox_inches="tight")
plt.close(fig5)
