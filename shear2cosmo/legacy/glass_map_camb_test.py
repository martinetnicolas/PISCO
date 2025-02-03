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
from multiprocessing import Process, Queue
print = functools.partial(print, flush=True) # all print()s will be called with "flush=True"

'''
1st. NOTE: lmax=4*nside and lmax=5*nside give similar results (2pt and k4 improved at small scales for 5*nside, the rest is quite consistent.
2nd. NOTE: Globally, it seems better to use the healpy window function to the GLASS maps.
3rd. NOTE: Better lognormal-shift seems to be 0.7. 
4th. NOTE: using lmax=10*nside does not give improvements!
'''

#Loading the cosmo-SLICS table (without 01 cosmology)
n_cosmo = 2000          # number of cosmologies, to change
seed = 0
cosmo_slics_path = f'/net/GECO/pisco/SHEAR2COSMO/Leo_COSMO-SLICS/LHS_comparison/lhs_2000/lhs_2000_0.csv' # path of csv, to change
print(cosmo_slics_path)
cosmo_slics = pd.read_csv(cosmo_slics_path)

n_cosmo_slics = np.arange(n_cosmo)     #all cosmologies
df = []
count_success = 0

for i in n_cosmo_slics:
    for lognorm_shift in [None]: #[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        
        # simulation parameters
        cosmo_mode = 'cosmoslics'
        ns = 0.969
        zmax = 3.0
        id = i # ID of the cosmo model, -1 for FID
        cosmo_mode += f'_{id}' # re-name the cosmology label according to the index (e.g. COSMOSLICS_00, COSMOSLICS_01, ..)
        Om = cosmo_slics['Om'][i]
        S8 = cosmo_slics['S8'][i]
        h = cosmo_slics['h'][i]
        w0 = cosmo_slics['w0'][i]
        sig8 = S8*np.sqrt(0.3/Om)
        Ob = 0.0473
        Oc = Om - Ob
        
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
        
        def compute_matter_cls_with_timeout(sim_pars, timeout, result_queue):
            try:
                matter_cls = camb.get_results(sim_pars)
                result_queue.put(matter_cls)
            except Exception as e:
                result_queue.put(e)
            
        timeout = 2

        try:
            print(f'\n {"".join(itertools.repeat("-", timeout))} {cosmo_mode} matter-Cls started {"".join(itertools.repeat("-", timeout))} \n')
            
            start_time = time.time()
        
            # Queue to store the result
            result_queue = Queue()
                
            # Create a process to compute matter_cls
            process = Process(target=compute_matter_cls_with_timeout, args=(sim_pars, timeout, result_queue))
                
            # Start the process
            process.start()
                
            # Wait for the process to finish or timeout
            process.join(timeout=timeout)
                
            # Check if the process is still alive (meaning it has timed out)
            if process.is_alive():
                process.terminate()
                process.join()
                print("The computation of matter_cls timed out.")
                res = 1
                count_success += 1
                new_line = {"Om": Om, "S8": S8, "h": h, "w0": w0,'sig8':sig8,'Oc':Oc}
                df.append(new_line)
            else:
                # Get the result from the queue
                try:
                    result = result_queue.get_nowait()
                    if isinstance(result, Exception):
                        #test if error encountered
                        print(f"An error occurred: {result}")
                        res = 0
                    else:
                        #test if success
                        matter_cls = result
                        res = 0
                except Exception as e:
                    print(f"Failed to get the result from the queue: {e}")
                    res = 0

            #
            

        except Exception as e:
            #
            print(f"An unexpected error occurred: {e}")
            res = 0   

new_df = pd.DataFrame(df, columns=['Om','S8','h','w0','sig8','Oc','zmax','lmax','final_As','final_sig8'])            

# output path (to change) :
new_df.to_csv(f'/net/GECO/pisco/SHEAR2COSMO/Leo_COSMO-SLICS/LHS_comparison/lhs_2000/lhs_{int(count_success)}_wo_crash.csv')