#!/usr/bin/env python

#     |-----------------------------------------------------------------|
#     | This function compare several diagnostics from the SEAMOUNT     |
#     | testcase against the expected analytical values.                |
#     | This is a modified version of original code from Mike Bell.     |
#     |                                                                 |
#     | N.B: This code is useful/appropriate ONLY if the SEAMOUNT       |
#     |      test case is run with the following options:               |
#     |                                                                 |
#     |      a) nn_ini_cond = 0 (i.e., Shchepetkin & McWilliams (2002)) |
#     |      b) ln_hpg_ffr  = .TRUE. (i.e., all flavours of             |
#     |                               Forces on the Faces scheme)       |
#     |      c) 1 timestep                                              |
#     |                                                                 | 
#     | Author: Diego Bruciaferri                                       |
#     | Date and place: 12-09-2022, Met Office, UK                      |
#     |-----------------------------------------------------------------|

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
from utils import *

# ==============================================================================
# Input parameters

# 1. INPUT FILES

exp_base_dir = '/data/users/dbruciaf/HPG/SEAMOUNT/2022-08-16_debug_ff'

hpg_list  = ['ffl','fflr']
vco_list  = ['s94', 'vqs','zco']
ini_list  = ['pnt']

# 2. DIAGNOSTICS
jip = 39 ; jjp = 29

# ==============================================================================

for hpg in range(len(hpg_list)):

    for vco in range(len(vco_list)):

        for ini in range(len(ini_list)):

            exp_name = vco_list[vco] + "_" + \
                       hpg_list[hpg] + "_" + \
                       ini_list[ini] + "_fp4_1ts"
            print('   ', exp_name)
 
            exp_dir =   exp_base_dir + "/" + exp_name

            # Loading NEMO geometry
            ds_dom = xr.open_dataset(exp_dir + '/domain_cfg_out.nc', 
                                     drop_variables=("x", "y","nav_lev")
                                    ).squeeze()
            ds_dom = ds_dom.rename_dims({'nav_lev':'z'})

            # Computing land-sea masks
            ds_dom = compute_masks(ds_dom, merge=True)
 
            # Loading NEMO output
            exp_stem = exp_dir + '/SEAMOUNT_' + exp_name + '_grid_'
            dsT = xr.open_dataset(exp_stem + 'T.nc',
                                  drop_variables=("x", "y","deptht"))
            dsU = xr.open_dataset(exp_stem + 'U.nc',
                                  drop_variables=("x", "y","depthu"))
            dsV = xr.open_dataset(exp_stem + 'V.nc',
                                  drop_variables=("x", "y","depthv"))
            dsW = xr.open_dataset(exp_stem + 'W.nc',
                                  drop_variables=("x", "y","depthw"))
            dsT = dsT.rename_dims({'deptht':'z'}).squeeze()
            dsU = dsU.rename_dims({'depthu':'z'}).squeeze()
            dsV = dsV.rename_dims({'depthv':'z'}).squeeze()
            dsW = dsW.rename_dims({'depthw':'z'}).squeeze()

            # Computing gdept and gdepw
            e3t = dsT.e3t
            e3w = dsW.e3w
            gdepw, gdept  = e3_to_dep(e3w,  e3t)
 
            # Extracting variables
            tmask = ds_dom.tmask.squeeze().values
            umask = ds_dom.umask.squeeze().values
            vmask = ds_dom.umask.squeeze().values
            jpk = len(dsT.z)

            rhd   = dsT.rhd_hpg.values
            press = dsW.pressure.values

            u_Fx_ver_faces = dsU.u_force_west.values
            u_Fx_hor_faces = dsU.u_force_upper.values
            v_Fx_ver_faces = dsV.v_force_south.values
            v_Fx_hor_faces = dsV.v_force_upper.values

            # masking land points
            rhd[tmask==0] = np.nan
            press[tmask==0] = np.nan
            u_Fx_ver_faces[umask==0] = np.nan
            u_Fx_hor_faces[umask==0] = np.nan
            v_Fx_ver_faces[vmask==0] = np.nan
            v_Fx_hor_faces[vmask==0] = np.nan

            # -----------------------------------------------
            # Compute forces on faces:

            # 1) rhd profiles
            rhd_true = calc_SM03_rhd(gdept.values)

            # 2) pressure profiles as -g * int_z^0 rhd dz 
            press_true = calc_press(gdepw.values)
 
            # 3) horizontal forces Fx profiles as - int_z^0 p dz
            Fx_from_surf_true  = calc_Fx_from_surf(gdepw.values)
            
            # 4) Fx on faces of u cells
            u_Fx_ver_faces_true = np.zeros_like(Fx_from_surf_true)
            for jk in range(jpk-1) :
                u_Fx_ver_faces_true[jk,:,:] = Fx_from_surf_true[jk+1,:,:] - Fx_from_surf_true[jk,:,:]

            u_Fx_hor_faces_true = np.zeros_like(Fx_from_surf_true)
            u_Fx_hor_faces_true[:,:,:-1] = Fx_from_surf_true[:,:,:-1] - Fx_from_surf_true[:,:,1:]

            u_Fx_ver_sum_true = np.zeros_like(Fx_from_surf_true)
            u_Fx_ver_sum      = np.zeros_like(Fx_from_surf_true)
            u_Fx_ver_sum_true[:,:,:-1]  = u_Fx_ver_faces_true[:,:,:-1] - u_Fx_ver_faces_true[:,:,1:]
            u_Fx_ver_sum[:,:,:-1]       =      u_Fx_ver_faces[:,:,:-1] -      u_Fx_ver_faces[:,:,1:]

            u_Fx_hor_sum_true = np.zeros_like(Fx_from_surf_true)
            u_Fx_hor_sum      = np.zeros_like(Fx_from_surf_true)
            u_Fx_hor_sum_true[:-1,:,:] = u_Fx_hor_faces_true[:-1,:,:] - u_Fx_hor_faces_true[1:,:,:]
            u_Fx_hor_sum[:-1,:,:]      =      u_Fx_hor_faces[:-1,:,:] -      u_Fx_hor_faces[1:,:,:]

            u_Fx_tot_sum_true = u_Fx_ver_sum_true + u_Fx_hor_sum_true
            u_Fx_tot_sum      =      u_Fx_ver_sum +      u_Fx_hor_sum
 
            # 5) Fx on faces of v cells
            v_Fx_ver_faces_true = u_Fx_ver_faces_true

            v_Fx_hor_faces_true = np.zeros_like(Fx_from_surf_true)
            v_Fx_hor_faces_true[:,:-1,:] = Fx_from_surf_true[:,:-1,:] - Fx_from_surf_true[:,1:,:]

            v_Fx_ver_sum_true = np.zeros_like(Fx_from_surf_true)
            v_Fx_ver_sum      = np.zeros_like(Fx_from_surf_true)
            v_Fx_ver_sum_true[:,:-1,:]  = v_Fx_ver_faces_true[:,:-1,:] - v_Fx_ver_faces_true[:,1:,:]
            v_Fx_ver_sum[:,:-1,:]       =      v_Fx_ver_faces[:,:-1,:] -      v_Fx_ver_faces[:,1:,:]

            v_Fx_hor_sum_true = np.zeros_like(Fx_from_surf_true)
            v_Fx_hor_sum      = np.zeros_like(Fx_from_surf_true)
            v_Fx_hor_sum_true[:-1,:,:] = v_Fx_hor_faces_true[:-1,:,:] - v_Fx_hor_faces_true[1:,:,:]
            v_Fx_hor_sum[:-1,:,:]      =      v_Fx_hor_faces[:-1,:,:] -      v_Fx_hor_faces[1:,:,:]
 
            v_Fx_tot_sum_true = v_Fx_ver_sum_true + v_Fx_hor_sum_true
            v_Fx_tot_sum      =      v_Fx_ver_sum +      v_Fx_hor_sum

            # masking land points
            rhd_true[tmask==0] = np.nan
            press_true[tmask==0] = np.nan
            u_Fx_ver_faces_true[umask==0] = np.nan
            u_Fx_hor_faces_true[umask==0] = np.nan
            u_Fx_ver_sum_true[umask==0] = np.nan
            u_Fx_ver_sum[umask==0] = np.nan
            u_Fx_hor_sum_true[umask==0] = np.nan
            u_Fx_hor_sum[umask==0] = np.nan
            u_Fx_tot_sum_true[umask==0] = np.nan
            u_Fx_tot_sum[umask==0] = np.nan
            v_Fx_ver_faces_true[vmask==0] = np.nan
            v_Fx_hor_faces_true[vmask==0] = np.nan
            v_Fx_ver_sum_true[vmask==0] = np.nan
            v_Fx_ver_sum[vmask==0] = np.nan
            v_Fx_hor_sum_true[vmask==0] = np.nan
            v_Fx_hor_sum[vmask==0] = np.nan
            v_Fx_tot_sum_true[vmask==0] = np.nan
            v_Fx_tot_sum[vmask==0] = np.nan

            # -----------------------------------------------
            # Compute error

            rhd_err   = rhd - rhd_true
            press_err = press - press_true

            u_Fx_ver_faces_err = u_Fx_ver_faces - u_Fx_ver_faces_true
            u_Fx_hor_faces_err = u_Fx_hor_faces - u_Fx_hor_faces_true
            v_Fx_ver_faces_err = v_Fx_ver_faces - v_Fx_ver_faces_true
            v_Fx_hor_faces_err = v_Fx_hor_faces - v_Fx_hor_faces_true

            u_Fx_ver_sum_err = u_Fx_ver_sum - u_Fx_ver_sum_true
            u_Fx_hor_sum_err = u_Fx_hor_sum - u_Fx_hor_sum_true
            u_Fx_tot_sum_err = u_Fx_tot_sum - u_Fx_tot_sum_true
            v_Fx_ver_sum_err = v_Fx_ver_sum - v_Fx_ver_sum_true
            v_Fx_hor_sum_err = v_Fx_hor_sum - v_Fx_hor_sum_true
            v_Fx_tot_sum_err = v_Fx_tot_sum - v_Fx_tot_sum_true

            # -----------------------------------------------
            # Print diagnostics
            file_std = open(exp_dir+'/std_out.txt','w')
            diag_txt = ['gdept',
                        'rhd_true'  , 'rhd'  , 'rhd_err'  ,
                        'press_true', 'press', 'press_err']
            diag_var = [gdept.values, 
                        rhd_true  , rhd  , rhd_err  , 
                        press_true, press, press_err] 
            
            for jk in range (jpk) :
                for jj in range(jjp-2,jjp+2,1):
                    print ('jk, jj, jip = ', jk, jj, jip, 'for ji in range(jip,jip+5,2)', file=file_std)
                    for d in range(len(diag_var)):
                        print('  '+diag_txt[d]+': ',
                              [diag_var[d][jk,jj,ji] for ji in range(jip,jip+5,2)],
                              file=file_std
                             )

            print(file=file_std)

            for jk in range (0,jpk-1,2) :
                for jj in range (jjp-2,jjp+2,2):
                    print ('jk, jj, jip = ', jk, jj, jip, 'for ji in range(jip,jip+5,2)', file=file_std)          
                    diag_txt = ['u_Fx_ver_faces_true', 'u_Fx_ver_faces', 'u_Fx_ver_faces_err',
                                'u_Fx_hor_faces_true', 'u_Fx_hor_faces', 'u_Fx_hor_faces_err',
                                'u_Fx_ver_sum_true'  , 'u_Fx_ver_sum'  , 'u_Fx_ver_sum_err'  ,
                                'u_Fx_hor_sum_true'  , 'u_Fx_hor_sum'  , 'u_Fx_hor_sum_err'  ,
                                'u_Fx_tot_sum_true'  , 'u_Fx_tot_sum'  , 'u_Fx_tot_sum_err'  ]
                    diag_var = [u_Fx_ver_faces_true, u_Fx_ver_faces, u_Fx_ver_faces_err,
                                u_Fx_hor_faces_true, u_Fx_hor_faces, u_Fx_hor_faces_err,
                                u_Fx_ver_sum_true  , u_Fx_ver_sum  , u_Fx_ver_sum_err  ,
                                u_Fx_hor_sum_true  , u_Fx_hor_sum  , u_Fx_hor_sum_err  ,
                                u_Fx_tot_sum_true  , u_Fx_tot_sum  , u_Fx_tot_sum_err  ]
                    for d in range(len(diag_var)):
                        print('  '+diag_txt[d]+': ',
                              [diag_var[d][jk,jj,ji] for ji in range(jip,jip+5,2)],
                              file=file_std
                             )
                    print(file=file_std)
 
                    diag_txt = ['v_Fx_ver_faces_true', 'v_Fx_ver_faces', 'v_Fx_ver_faces_err',
                                'v_Fx_hor_faces_true', 'v_Fx_hor_faces', 'v_Fx_hor_faces_err',
                                'v_Fx_ver_sum_true'  , 'v_Fx_ver_sum'  , 'v_Fx_ver_sum_err'  ,
                                'v_Fx_hor_sum_true'  , 'v_Fx_hor_sum'  , 'v_Fx_hor_sum_err'  ,
                                'v_Fx_tot_sum_true'  , 'v_Fx_tot_sum'  , 'v_Fx_tot_sum_err'  ]
                    diag_var = [v_Fx_ver_faces_true, v_Fx_ver_faces, v_Fx_ver_faces_err,
                                v_Fx_hor_faces_true, v_Fx_hor_faces, v_Fx_hor_faces_err,
                                v_Fx_ver_sum_true  , v_Fx_ver_sum  , v_Fx_ver_sum_err  ,
                                v_Fx_hor_sum_true  , v_Fx_hor_sum  , v_Fx_hor_sum_err  ,
                                v_Fx_tot_sum_true  , v_Fx_tot_sum  , v_Fx_tot_sum_err  ]
                    for d in range(len(diag_var)):
                        print('  '+diag_txt[d]+': ',
                              [diag_var[d][jk,jj,ji] for ji in range(jip,jip+5,2)],
                              file=file_std
                             )

            # -----------------------------------------------
            # Output some fields
            dims3D = ["z","y","x"]
            ds = xr.Dataset(
                data_vars=dict(
                    tmask=(dims3D, tmask),
                    umask=(dims3D, umask),
                    vmask=(dims3D, vmask),
                    rhd=(dims3D, rhd),
                    rhd_true=(dims3D, rhd_true),
                    rhd_err=(dims3D, rhd_err),
                    press=(dims3D, press),
                    press_true=(dims3D, press_true),
                    press_err=(dims3D, press_err),
                    u_Fx_ver_faces_err=(dims3D, u_Fx_ver_faces_err),
                    v_Fx_ver_faces_err=(dims3D, v_Fx_ver_faces_err),
                    u_Fx_hor_faces_err=(dims3D, u_Fx_hor_faces_err),
                    v_Fx_hor_faces_err=(dims3D, v_Fx_hor_faces_err),
                    u_Fx_ver_sum_err=(dims3D, u_Fx_ver_sum_err),
                    u_Fx_hor_sum_err=(dims3D, u_Fx_hor_sum_err),
                    u_Fx_tot_sum_err=(dims3D, u_Fx_tot_sum_err),
                    v_Fx_ver_sum_err=(dims3D, v_Fx_ver_sum_err),
                    v_Fx_hor_sum_err=(dims3D, v_Fx_hor_sum_err),
                    v_Fx_tot_sum_err=(dims3D, v_Fx_tot_sum_err),
                    #v_Fx_ver_sum_true=(dims3D, v_Fx_ver_sum_true),
                    #v_Fx_ver_sum=(dims3D, v_Fx_ver_sum),
                    #v_Fx_ver_faces_true=(dims3D, v_Fx_ver_faces_true),
                    #v_Fx_ver_faces=(dims3D, v_Fx_ver_faces)
                ),
                coords=dict(
                    nav_lon=(["x","y"], dsT.nav_lon.values),
                    nav_lat=(["x","y"], dsT.nav_lat.values),
                    gdept=(dims3D, gdept.values)
                )
            )

            out_nc = exp_dir + "/assess_forces.nc"
            ds.to_netcdf(out_nc)
