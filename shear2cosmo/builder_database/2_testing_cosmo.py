from pathlib import Path

import camb
import glass.ext.camb
import numpy
import pandas
from cosmology import Cosmology
from glass import shells

# --------------------------------------------------------------------------------------------------


# DBB = "DBB_KMEANS_25"
DBB = "DBB_DENSITY_25"

FOLDER_BASE = Path("/net/GECO/pisco/SHEAR2COSMO")
FOLDER_RESULTS = FOLDER_BASE / DBB


# --------------------------------------------------------------------------------------------------


cosmo_slics_path = sorted(list(FOLDER_RESULTS.glob("lhs_*.csv")))[0]
cosmo_slics = pandas.read_csv(cosmo_slics_path)
NUMBER_OF_COSMO = len(cosmo_slics)
print("NUMBER_OF_COSMO", NUMBER_OF_COSMO)
print("\n LHS \n", cosmo_slics)

for i in numpy.arange(NUMBER_OF_COSMO):
    print("i", i)
    # simulation parameters
    ns = 0.969
    zmax = 3.0
    cosmo_slics.loc[i, "zmax"] = zmax
    cosmo_mode = f"id{i:02}"
    h = cosmo_slics["h"][i]
    w0 = cosmo_slics["w0"][i]
    sig8 = cosmo_slics["sig8"][i]
    Om = cosmo_slics["Om"][i]
    Oc = cosmo_slics["Oc"][i]
    Ob = Om - Oc

    # basic parameters for the GLASS simulation for the parameters above
    lmax = 5 * 1024  # 5*6144 // 5*1024
    cosmo_slics.loc[i, "lmax"] = lmax

    print(f"\n ----- {cosmo_mode} started ------ \n")

    # initial CAMB parameters set up (Ob and ns are fixed)
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=h * 100, omch2=Oc * h**2, ombh2=Ob * h**2)
    pars.InitPower.set_params(As=2.1e-9, ns=ns, r=0)
    pars.DarkEnergy.set_params(w=w0)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars.set_matter_power(redshifts=[0.0], kmax=2.0)

    # calculate sigma8 at z=0 for these parameters
    results = camb.get_results(pars)

    # Non-Linear spectra (Halofit)
    pars.NonLinear = camb.model.NonLinear_both
    results.calc_power_spectra(pars)
    random_sig8 = numpy.array(results.get_sigma8())

    # get final As for the simulation parameters (sim_pars)
    # final_sig8 is the value of sigma8 for the simulation
    final_sig8 = sig8 * 1
    final_As = 2.1e-9 * (final_sig8 / random_sig8) ** 2.0
    cosmo_slics.loc[i, "final_As"] = final_As[0]
    print(f"Final As matching sig8 = {sig8}: final_As = {final_As}")

    # updated CAMB parameters with As = final_As previously computed (Ob and ns are fixed)
    sim_pars = camb.CAMBparams()
    sim_pars.set_cosmology(H0=h * 100, omch2=Oc * h**2, ombh2=Ob * h**2)
    sim_pars.InitPower.set_params(As=final_As, ns=ns, r=0)
    sim_pars.DarkEnergy.set_params(w=w0)
    sim_pars.NonLinear = camb.model.NonLinear_both
    sim_pars.set_matter_power(redshifts=[0.0], kmax=2.0)

    # calculate results for these parameters
    results = camb.get_results(sim_pars)
    results.calc_power_spectra(sim_pars)
    sig8 = numpy.array(results.get_sigma8())
    cosmo_slics.loc[i, "final_sig8"] = final_sig8

    # check that this NEW sigma8 is the one you want for the simulation (final_sig8)!
    print(f"Computed sig8 from final_As = {final_As}: final_sig8 = {sig8}")

    # get the cosmology for CAMB
    cosmo = Cosmology.from_camb(sim_pars)

    # shells of 200 Mpc in comoving distance spacing
    zb = shells.distance_grid(cosmo, 0.0, zmax, dx=200.0)

    # uniform matter weight function / CAMB requires linear ramp for low redshifts
    ws = shells.tophat_windows(zb, weight=glass.ext.camb.camb_tophat_weight)

    # FINAL LHS
    cosmo_slics.to_csv(FOLDER_RESULTS / f"lhs_{NUMBER_OF_COSMO}.csv", index=False)
