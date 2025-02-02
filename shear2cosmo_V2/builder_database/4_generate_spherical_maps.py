import time
from pathlib import Path

import camb
import glass.ext.camb
import numpy
import pandas
from cosmology import Cosmology
from glass import fields, lensing, shells
from utils import create_dir

# --------------------------------------------------------------------------------------------------


DBB = "DBB_KMEANS_25"
# DBB = "DBB_DENSITY_25"

NB_OF_SEEDS = 4

FOLDER_BASE = Path("/net/GECO/pisco/SHEAR2COSMO")
FOLDER_RESULTS = FOLDER_BASE / DBB
FOLDER_MATTER = create_dir(FOLDER_RESULTS / "matter")
FOLDER_SPHERICAL = create_dir(FOLDER_RESULTS / "spherical")


# --------------------------------------------------------------------------------------------------


cosmo_slics_path = sorted(list(FOLDER_RESULTS.glob("lhs_*.csv")))[0]
cosmo_slics = pandas.read_csv(cosmo_slics_path)
NUMBER_OF_COSMO = len(cosmo_slics)
avg_nz_fname = "euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01"
print("\n LHS \n", cosmo_slics)

for i in numpy.arange(NUMBER_OF_COSMO):
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

    # GET MATTER Cls FOR CHOSEN COSMOLOGY (if not computed yet)
    matter_path = FOLDER_MATTER / f"{cosmo_mode}_matter_cls.npy"

    # Define field type of glass map
    field_type = "gaussian"
    cosmo_slics.loc[i, "field_type"] = field_type

    # GET KAPPA FIELD FOR CHOSEN COSMOLOGY
    for seed in range(NB_OF_SEEDS):
        print(f"\n --------- {cosmo_mode} seed {seed} GLASS map generation started ----------- \n")
        start_time = time.time()

        # use the CAMB cosmology that generated the matter power spectra

        # set map resolution (to best match cosmo-slics resolution of 0.589 arcmin/px)
        """ Set here twice nside to improve any projection from Spherical to Cartesian
        at corresponding nside px scale"""
        nside = 2048  # nside=2048 roughly corresponds to 1.7'

        # load the angular matter power spectra previously computed with CAMB
        matter_cls = numpy.load(matter_path)

        # MATTER FIELD (lognormal/gaussian)
        if field_type == "gaussian":
            # compute Gaussian cls for gaussian fields for 3 correlated shells
            """ putting nside!=None in gaussian_gls() means that the HEALPix pixel window
            function is applied (it takes into account lack of small-scales information
            wrt applied lmax """
            gls = fields.gaussian_gls(matter_cls, nside=None, lmax=lmax, ncorr=3)
            # generator for lognormal/gaussian matter fields
            matter = fields.generate_gaussian(
                gls, nside, ncorr=3, rng=numpy.random.default_rng(seed)
            )

        # COMPUTE KAPPA FIELD: this will compute the convergence field iteratively
        convergence = lensing.MultiPlaneConvergence(cosmo)

        # GALAXY DISTRIBUTION n(z) // localised redshift distribution
        # (the actual density per arcmin2 does not matter here, it is never used)
        nz = numpy.loadtxt(FOLDER_RESULTS / f"{avg_nz_fname}.cat")
        z = nz[:, 0]
        dndz = nz[:, 1]

        # Compute the convergence and shear fields for each shell
        # and add them up using the galaxy densities as a weight

        # the integrated convergence field over the redshift distribution
        kappa_bar = numpy.zeros(12 * nside**2)

        # main loop to simulate the matter fields iterative
        for i, delta_i in enumerate(matter):
            # add lensing plane from the window function of this shell
            convergence.add_window(delta_i, ws[i])

            # get convergence field
            kappa_i = convergence.kappa

            # get the restriction of the dndz to this shell
            z_i, dndz_i = shells.restrict(z, dndz, ws[i])

            # compute the galaxy density in this shell
            ngal = numpy.trapz(dndz_i, z_i)

            # add to mean fields using the galaxy number density as weight
            kappa_bar += ngal * kappa_i

        # compute the overall galaxy density
        ngal = numpy.trapz(dndz, z)

        # normalise mean fields by the total galaxy number density
        kappa_bar /= numpy.sum(ngal)

        final_name = f"{cosmo_mode}_seed{seed}"
        numpy.save(FOLDER_SPHERICAL / final_name, kappa_bar)

        end_time = time.time()
        print(f"CPU-time to generate {final_name}: \t {(end_time - start_time) / 60} mins \n")
    print(f"\n ----- {cosmo_mode} finished ----- \n")
