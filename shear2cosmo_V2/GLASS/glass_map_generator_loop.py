import sys, platform, os
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import healpy as hp
import numpy as np
import copy
import camb
from camb import model, initialpower
from cosmology import Cosmology
# GLASS imports
import glass
import glass.ext.camb
from glass import shells, fields, lensing, galaxies 
import os.path
import time
import itertools
import functools
print = functools.partial(print, flush=True) # all print()s will be called with "flush=True"

'''
1st. NOTE: lmax=4*nside and lmax=5*nside give similar results (2pt and k4 improved at small scales for 5*nside, the rest is quite consistent.
2nd. NOTE: Globally, it seems better to use the healpy window function to the GLASS maps.
3rd. NOTE: Better lognormal-shift seems to be 0.7. 
4th. NOTE: using lmax=10*nside does not give improvements!
'''

#Loading the cosmo-SLICS table (without 01 cosmology)
cosmo_slics_path = '/net/GECO/pisco/SHEAR2COSMO/Leo_COSMO-SLICS/data/'
cosmo_slics = pd.read_csv(cosmo_slics_path + f'cosmo_slics_table.csv')
#n_cosmo_slics = [0,1,2,3,4,5]               #batch1
#n_cosmo_slics = [6,7,8,9,10,11]            #batch2
#n_cosmo_slics = [12,13,14,15,16,17]        #batch3
#n_cosmo_slics = [18,19,20,21,22,23,24]     #batch4
n_cosmo_slics = np.arange(0, 25)     #all cosmologies

for i in n_cosmo_slics:
    for lognorm_shift in [None]: #[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # Buttons
        compute_matter_cls = 1 # LAM-CC cpu_time: (lmax=5*1024, less than a minute); (lmax=5*6144, )                     
        generate_glass_map = 1
        
        # simulation parameters
        cosmo_mode = 'COSMOSLICS'
        ns = 0.969
        zmax = 3.0
        id = cosmo_slics["ID"][i] # ID of the cosmo model, -1 for FID
        cosmo_mode += f'_{id}' # re-name the cosmology label according to the index (e.g. COSMOSLICS_00, COSMOSLICS_01, ..)
        h = cosmo_slics["h"][i]
        w0 = cosmo_slics["w0"][i]
        sig8 = cosmo_slics["sig8"][i]
        Om = cosmo_slics["Om"][i]
        Oc = cosmo_slics["Oc"][i]
        Ob = Om - Oc
        
        # basic parameters for the GLASS simulation for the parameters above
        lmax = 5*1024 #5*6144 # 5*1024
        
        print(f'\n {"".join(itertools.repeat("-", 10))} {cosmo_mode} started {"".join(itertools.repeat("-", 10))} \n')
        
        # initial CAMB parameters set up (Ob and ns are fixed)
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=h*100, omch2=Oc*h**2, ombh2=Ob*h**2)
        pars.InitPower.set_params(As=2.1e-9, ns=ns, r=0)
        pars.DarkEnergy.set_params(w=w0)
        pars.set_for_lmax(lmax, lens_potential_accuracy=0)
        pars.set_matter_power(redshifts=[0.], kmax=2.0)
        
        # calculate sigma8 at z=0 for these parameters
        results = camb.get_results(pars)
        
        #Non-Linear spectra (Halofit)
        pars.NonLinear = model.NonLinear_both
        results.calc_power_spectra(pars)
        random_sig8 = np.array(results.get_sigma8())
        
        # get final As for the simulation parameters (sim_pars)
        # final_sig8 is the value of sigma8 for the simulation
        
        final_sig8 = sig8 * 1
        final_As = 2.1e-9*(final_sig8/random_sig8)**2.
        
        print(f'Final As matching sig8 = {sig8}: final_As = {final_As}')
        
        # updated CAMB parameters with As = final_As previously computed (Ob and ns are fixed)
        sim_pars = camb.CAMBparams()
        sim_pars.set_cosmology(H0=h*100, omch2=Oc*h**2, ombh2=Ob*h**2)
        sim_pars.InitPower.set_params(As=final_As, ns=ns, r=0)
        sim_pars.DarkEnergy.set_params(w=w0)
        sim_pars.NonLinear = model.NonLinear_both
        sim_pars.set_matter_power(redshifts=[0.], kmax=2.0)
        
        # calculate results for these parameters
        results = camb.get_results(sim_pars)
        results.calc_power_spectra(sim_pars)
        sig8 = np.array(results.get_sigma8())
        # check that this NEW sigma8 is the one you want for the simulation (final_sig8)!
        print(f'Computed sig8 from final_As = {final_As}: final_sig8 = {sig8}')
        
        # get the cosmology for CAMB
        cosmo = Cosmology.from_camb(sim_pars)
        
        # shells of 200 Mpc in comoving distance spacing
        zb = glass.shells.distance_grid(cosmo, 0., zmax, dx=200.)
        
        # uniform matter weight function
        # CAMB requires linear ramp for low redshifts
        ws = glass.shells.tophat_windows(zb, weight=glass.ext.camb.camb_tophat_weight)
        
        # GET MATTER Cls FOR CHOSEN COSMOLOGY (if not computed yet)
        if compute_matter_cls:
            
            print(f'\n {"".join(itertools.repeat("-", 20))} {cosmo_mode} matter-Cls started {"".join(itertools.repeat("-", 20))} \n')
            
            start_time = time.time()
            
            matter_path = '/net/GECO/pisco/SHEAR2COSMO/Leo_COSMO-SLICS/data/matter_Cls/'
            
            if not os.path.isfile(matter_path + f'glass_{cosmo_mode}_matter_cls_lmax_{lmax}_zmax_{str(zmax)}.npy'):
            
                # compute angular power spectra with CAMB (for later use);
                matter_cls = glass.ext.camb.matter_cls(sim_pars, lmax, ws)
                
                # save shell definitions to file for later use
                np.save(matter_path + f'glass_{cosmo_mode}_matter_cls_lmax_{lmax}_zmax_{str(zmax)}.npy', matter_cls) 
            
            end_time = time.time()
            print(f'CPU-time to generate glass_{cosmo_mode}_matter_cls_lmax_{lmax}_zmax_{str(zmax)}: \t {(end_time - start_time)/60}mins \n')
            
            print(f'\n {"".join(itertools.repeat("-", 20))} {cosmo_mode} matter-Cls finished {"".join(itertools.repeat("-", 20))} \n')  
            
        
        # GET KAPPA FIELD FOR CHOSEN COSMOLOGY
        if generate_glass_map:
            
            print(f'\n {"".join(itertools.repeat("-", 20))} {cosmo_mode} GLASS map generation started {"".join(itertools.repeat("-", 20))} \n')  
            start_time = time.time()
            
            # Assign seed to realization
            seed = 3
            
            # use the CAMB cosmology that generated the matter power spectra
            
            # Define field type of glass map
            field_type = 'gaussian' #'lognormal'
            
            # set map resolution (to best match cosmo-slics resolution of 0.589 arcmin/px)
            ''' Set here twice nside to improve any projection from Spherical to Cartesian at corresponding nside px scale'''
            nside = 2048 # nside=2048 roughly corresponds to 1.7'
        
            # Define whether applying the healpix window function to the halpix pixels of the final GLASS map or not
            apply_window_function_to_GLASS_pixels = 0
            
            if apply_window_function_to_GLASS_pixels:
            
                nside_window_function = nside * 1
                
            else:
            
                nside_window_function = None
        
            # load the angular matter power spectra previously computed with CAMB
            path = '/net/GECO/pisco/SHEAR2COSMO/Leo_COSMO-SLICS/data/matter_Cls/'
            matter_cls = np.load(path + f'glass_{cosmo_mode}_matter_cls_lmax_{lmax}_zmax_{str(zmax)}.npy')
            
            # MATTER FIELD (lognormal/gaussian)
            if field_type == 'gaussian':
            
              # Define extra label for file saving
              lognormal_label = ''
                
              # compute Gaussian/Lognormal cls for lognormal/gaussian fields for 3 correlated shells
              ''' putting nside!=None in gaussian_gls() means that the HEALPix pixel window function is applied (it takes into account lack of small-scales information wrt applied lmax '''
              gls = glass.fields.gaussian_gls(matter_cls, nside=nside_window_function, lmax=lmax, ncorr=3)
              # generator for lognormal/gaussian matter fields
              matter = glass.fields.generate_gaussian(gls, nside, ncorr=3, rng=np.random.default_rng(seed))
                
            if field_type == 'lognormal':
                
              # Set lognormal shift parameter (if needed)
              ''' default: 1.'''
        #     lognorm_shift = 0.9 #0.030620606871129982
                
              # Define extra label for file saving
              lognormal_label = f'_lognorm_shift_{np.round(lognorm_shift, 3)}'
                
              # compute Gaussian/Lognormal cls for lognormal/gaussian fields for 3 correlated shells
              ''' putting nside!=None in lognormal_gls() means that the HEALPix pixel window function is applied (it takes into account lack of small-scales information wrt applied lmax '''
              gls = glass.fields.lognormal_gls(matter_cls, nside=nside_window_function, lmax=lmax, ncorr=3, shift=lognorm_shift)
              # generator for lognormal/gaussian matter fields
              matter = glass.fields.generate_lognormal(gls, nside, ncorr=3, shift=lognorm_shift, rng=np.random.default_rng(seed))
            
            # COMPUTE KAPPA FIELD: this will compute the convergence field iteratively
            convergence = glass.lensing.MultiPlaneConvergence(cosmo)
            
            # GALAXY DISTRIBUTION n(z)
            # localised redshift distribution (the actual density per arcmin2 does not matter here, it is never used)
            nz_path = '/net/GECO/pisco/SHEAR2COSMO/Leo_COSMO-SLICS/data/nz/'
            avg_nz_fname = 'euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01'
            nz = np.loadtxt(nz_path + f'{avg_nz_fname}.cat')
            z = nz[:,0]
            dndz = nz[:,1]
            
            # Compute the convergence and shear fields for each shell and add them up using the galaxy densities as a weight
            
            # the integrated convergence field over the redshift distribution
            kappa_bar = np.zeros(12*nside**2)
            #gamma1_bar = np.zeros(12*nside**2)
            #gamma2_bar = np.zeros(12*nside**2)
            
            # main loop to simulate the matter fields iterative
            for i, delta_i in enumerate(matter):
            
                # add lensing plane from the window function of this shell
                convergence.add_window(delta_i, ws[i])
            
                # get convergence field
                kappa_i = convergence.kappa
                
                # compute shear field
                #gamma1_i, gamma2_i = glass.lensing.shear_from_convergence(kappa_i)
            
                # get the restriction of the dndz to this shell
                z_i, dndz_i = glass.shells.restrict(z, dndz, ws[i])
            
                # compute the galaxy density in this shell
                ngal = np.trapz(dndz_i, z_i)
            
                # add to mean fields using the galaxy number density as weight
                kappa_bar += ngal * kappa_i
                #gamma1_bar += ngal * gamma1_i
                #gamma2_bar += ngal * gamma2_i
            
            # compute the overall galaxy density
            ngal = np.trapz(dndz, z)
            
            # normalise mean fields by the total galaxy number density
            kappa_bar /= np.sum(ngal)
            #gamma1_bar /= np.sum(ngal)
            #gamma2_bar /= np.sum(ngal) 
            
            save_path = '/net/GECO/pisco/SHEAR2COSMO/Leo_COSMO-SLICS/data/GLASS_maps/spherical/'
            np.save(save_path + f'glass_{field_type}_kappa_{cosmo_mode}_{avg_nz_fname}_nside_{str(nside)}_lmax_{lmax}_zmax_{str(zmax)}{lognormal_label}_hpwindowfunction_{apply_window_function_to_GLASS_pixels}_seed_{seed}.npy', kappa_bar)
            
            end_time = time.time()
            print(f'CPU-time to generate glass_{field_type}_kappa_{cosmo_mode}_{avg_nz_fname}_nside_{str(nside)}_lmax_{lmax}_zmax_{str(zmax)}{lognormal_label}_hpwindowfunction_{apply_window_function_to_GLASS_pixels}_seed_{seed}: \t {(end_time - start_time)/60}mins \n')
            
            print(f'\n {"".join(itertools.repeat("-", 20))} {cosmo_mode} GLASS map generation finished {"".join(itertools.repeat("-", 20))} \n') 
        
        print(f'\n {"".join(itertools.repeat("-", 10))} {cosmo_mode} finished {"".join(itertools.repeat("-", 10))} \n')