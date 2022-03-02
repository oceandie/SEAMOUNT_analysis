#!/usr/bin/env python

#     |------------------------------------------------------------|
#     | Author: Diego Bruciaferri                                  |
#     | Date and place: 07-09-2021, Met Office, UK                 |
#     |------------------------------------------------------------|


import numpy as np
from matplotlib import pyplot as plt
import xarray as xr

# ==============================================================================
# Input parameters

# 1. INPUT FILES

exp_hpg = ['/data/users/dbruciaf/HPG/SEAMOUNT/N-SCO',
           '/data/users/dbruciaf/HPG/SEAMOUNT/N-PRJ',
           '/data/users/dbruciaf/HPG/SEAMOUNT/N-DJC-CTR']

color_hpg = ['red','blue','limegreen']
label_hpg = ['N-SCO','N-PRJ','N-DJC-CTR']

exp_djc = ['/data/users/dbruciaf/HPG/SEAMOUNT/N-DJC-CTR',
           '/data/users/dbruciaf/HPG/SEAMOUNT/N-DJC-TEOS',
           '/data/users/dbruciaf/HPG/SEAMOUNT/N-DJC-SH94',
           '/data/users/dbruciaf/HPG/SEAMOUNT/N-DJC-VQS']

color_djc = ['limegreen','deepskyblue','gold','magenta']
label_djc = ['N-DJC-CTR','N-DJC-TEOS','N-DJC-SH94','N-DJC-VQS']

fig_path = '/home/h01/dbruciaf/mod_dev/SEAMOUNT_analysis/plots' 

# ==============================================================================

for conf in ['steep','moderate']:

    fig = plt.figure(figsize=(30, 20), dpi=100)
    ax = fig.add_subplot()

    for exp in range(len(exp_hpg)):

        # Loading NEMO output
        dsU = xr.open_dataset(exp_hpg[exp]+'_'+conf+'/SEAMOUNT_xxx_1h_grid_U.nc')
        dsV = xr.open_dataset(exp_hpg[exp]+'_'+conf+'/SEAMOUNT_xxx_1h_grid_V.nc')

        # Interp. to T-grid
        U = dsU.uoce.rolling({'x':2}).mean().fillna(0.)
        V = dsV.voce.rolling({'y':2}).mean().fillna(0.)
        C = np.sqrt(U**2 + V**2)
        Ubt = dsU.ubar.rolling({'x':2}).mean().fillna(0.)
        Vbt = dsV.vbar.rolling({'y':2}).mean().fillna(0.)
        Cbt = np.sqrt(Ubt**2 + Vbt**2)

        Umax = []
        Umax_bt = [0.]
        for t in range(len(dsU.time_counter)):
            Umax.append(np.nanmax(C[t,:,:,:]))
            Umax_bt.append(np.nanmax(Cbt[t,:,:]))

        print(exp_hpg[exp]+'_'+conf)
        print('  baroclinic:  ', np.nanmax(np.asarray(Umax)*100.))
        print('  barotropic:  ', np.nanmax(np.asarray(Umax_bt)*100.))

        # Plotting ----------------------------------------------------------

        ax.plot(
            range(len(Umax)),
            Umax,
            color=color_hpg[exp],
            label=label_hpg[exp],
            linewidth=5.
        )

    ax.set_xlim(0., 120.)
    if conf == 'steep':
       ax.set_ylim(0., 1.8)
    elif conf == 'moderate':
       ax.set_ylim(0., 0.02)
    plt.xticks(fontsize=30.)
    plt.xlabel('Time [hours]', fontsize=40.)
    plt.yticks(fontsize=30.)
    plt.ylabel('Maximum velocity error [$m\;s^{-1}$]' ,fontsize=40.)

    plt.rc('legend', **{'fontsize':50})
    ax.legend(loc=0, ncol=1, frameon=False)

    fig_name = '/umax_hpg_'+conf+'.png'
    plt.savefig(fig_path+fig_name, bbox_inches="tight")
    plt.close()

for conf in ['steep','moderate']:

    fig = plt.figure(figsize=(30, 20), dpi=100)
    ax = fig.add_subplot()

    for exp in range(len(exp_djc)):

        # Loading NEMO output
        dsU = xr.open_dataset(exp_djc[exp]+'_'+conf+'/SEAMOUNT_xxx_1h_grid_U.nc')
        dsV = xr.open_dataset(exp_djc[exp]+'_'+conf+'/SEAMOUNT_xxx_1h_grid_V.nc')

        # Interp. to T-grid
        U = dsU.uoce.rolling({'x':2}).mean().fillna(0.)
        V = dsV.voce.rolling({'y':2}).mean().fillna(0.)
        C = np.sqrt(U**2 + V**2)
        Ubt = dsU.ubar.rolling({'x':2}).mean().fillna(0.)
        Vbt = dsV.vbar.rolling({'y':2}).mean().fillna(0.)
        Cbt = np.sqrt(Ubt**2 + Vbt**2)

        Umax = [0.]
        Umax_bt = [0.]
        for t in range(len(dsU.time_counter)):
            Umax.append(np.nanmax(C[t,:,:,:]))
            Umax_bt.append(np.nanmax(Cbt[t,:,:]))

        print(exp_djc[exp]+'_'+conf)
        print('  baroclinic:  ', np.nanmax(np.asarray(Umax)*100.))
        print('  barotropic:  ', np.nanmax(np.asarray(Umax_bt)*100.))

        # Plotting ----------------------------------------------------------

        ax.plot(
            range(len(Umax)),
            Umax,
            color=color_djc[exp],
            label=label_djc[exp],
            linewidth=5.
        )

    ax.set_xlim(0., 120.)
    if conf == 'steep':
       ax.set_ylim(0., 0.08)
    elif conf == 'moderate':
       ax.set_ylim(0., 0.003)
    plt.xticks(fontsize=30.)
    plt.xlabel('Time [hours]', fontsize=40.)
    plt.yticks(fontsize=30.)
    plt.ylabel('Maximum velocity error [$m\;s^{-1}$]' ,fontsize=40.)

    plt.rc('legend', **{'fontsize':50})
    ax.legend(loc=0, ncol=1, frameon=False)

    fig_name = '/umax_djc_'+conf+'.png'
    plt.savefig(fig_path+fig_name, bbox_inches="tight")
    plt.close()

