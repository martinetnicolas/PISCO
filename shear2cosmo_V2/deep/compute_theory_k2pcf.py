from pathlib import Path

import numpy
import pandas
import pyccl as ccl
from matplotlib import pyplot
from tqdm import tqdm

# --------------------------------------------------------------------------------------------------


FOLDER_BASE = Path("/net/GECO/pisco/SHEAR2COSMO/")
DBB = "DBB_OPT_1000"
n_cosmo = 100
seed = 0
# --------------------------------------------------------------------------------------------------


avg_nz_fname = "euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01"
FOLDER_RESULTS = FOLDER_BASE / DBB
#cosmo_slics_path = sorted(list(FOLDER_RESULTS.glob("lhs*maximin.csv")))[0]
#cosmo_slics_path = f'/net/GECO/pisco/SHEAR2COSMO/Leo_COSMO-SLICS/LHS_comparison/lhs_{n_cosmo}/lhs_{n_cosmo}_{seed}.csv'
cosmo_slics_path = f'/net/GECO/pisco/SHEAR2COSMO/DBB_OPT_COSMOSLICS/cosmo_slics_train.csv'
cosmo_slics = pandas.read_csv(cosmo_slics_path)
NUMBER_OF_COSMO = len(cosmo_slics)
two_pcf_theta_midpoints_arcmin = [
    6.0789e00,
    1.0224e01,
    1.7194e01,
    2.8916e01,
    4.8631e01,
    8.1788e01,
    1.3755e02,
    2.3133e02,
    3.8905e02,
    6.5430e02,
]

two_pcf_theta_midpoints = [x / 60 for x in two_pcf_theta_midpoints_arcmin]
two_pcf_theta_midpoints = numpy.array(two_pcf_theta_midpoints)

for i in tqdm(numpy.arange(NUMBER_OF_COSMO), total=NUMBER_OF_COSMO):
    # Make predictions for Cell and convert to xi_plus/xi_minus
    # Set transfer function, package from which computing the non linear power spectrum
    # and its eventual extra parameters
    # CAMB non linear model (only if matter_power_spectrum_package = 'camb')
    pyccl_transfer_function = "boltzmann_camb"
    matter_power_spectrum_package = "camb"
    CAMB_pk_nonlin_model = "takahashi"
    zbin_mode = "no_tomography"

    cosmo_mode = f"id{i:04}"
    #outpath = FOLDER_RESULTS / "theory_2pcf_maximin"
    #outpath = Path(f'/net/GECO/pisco/SHEAR2COSMO/Leo_COSMO-SLICS/LHS_comparison/theory_2pcf/lhs_{n_cosmo}_{seed}')
    outpath = Path(f'/net/GECO/pisco/SHEAR2COSMO/DBB_OPT_COSMOSLICS/theory_2pcf')

    h = cosmo_slics["h"][i]
    w0 = cosmo_slics["w0"][i]
    sigma8 = cosmo_slics["sig8"][i]
    Omega_m = cosmo_slics["Om"][i]
    Omega_c = cosmo_slics["Oc"][i]
    Omega_b = Omega_m - Omega_c

    cosmo = ccl.Cosmology(
        Omega_c=Omega_c,
        sigma8=sigma8,
        Omega_b=Omega_b,
        n_s=0.969,
        h=h,
        w0=w0,
        wa=0,
        transfer_function=pyccl_transfer_function,
        matter_power_spectrum=matter_power_spectrum_package,
        extra_parameters={"camb": {"halofit_version": CAMB_pk_nonlin_model}},
    )

    # Create objects to represent tracers of the weak lensing signal with this
    # number density (with has_intrinsic_alignment=False)
    nz_file = numpy.loadtxt(FOLDER_RESULTS / f"{avg_nz_fname}.cat")
    lens = ccl.WeakLensingTracer(cosmo, dndz=(nz_file[:, 0], nz_file[:, 1]))

    # Calculate the angular cross-spectrum of the two tracers as a function of ell
    ell = numpy.arange(1, 1e6)

    # @@@@@@@ CONVERGENCE POWER SPECTRUM @@@@@@@
    cls = ccl.angular_cl(cosmo, lens, lens, ell)

    # mid-points in degs of log-bins used in HOWLS for shear-2pcfs
    theta_deg = two_pcf_theta_midpoints

    # @@@@@@@ CONVERGENCE 2PCF @@@@@@@
    xi_plus = ccl.correlation(
        cosmo, ell=ell, C_ell=cls, theta=theta_deg, type="GG+", method="FFTLog"
    )

    numpy.savetxt(
        outpath / f"{cosmo_mode}_theory_2pcf.dat",
        numpy.c_[theta_deg * 60, xi_plus],
        header="theta arcmin, xi_kappa",
    )

    # Open plot
    if i%50==0:
        fig, ax = pyplot.subplots(1, 2, figsize=(20, 10), squeeze=False)
        ax[0, 0].plot(ell, cls * ell * (ell + 1) / (2 * numpy.pi), label="z_all", color="black")
        ax[0, 1].plot(theta_deg * 60, xi_plus, label="z_all")
    
        ax[0, 0].xaxis.set_tick_params(labelsize=14)
        ax[0, 0].yaxis.set_tick_params(labelsize=14)
        ax[0, 0].set_yscale("log")
        ax[0, 0].set_xscale("log")
        ax[0, 0].set_title(f"Angular power spectrum [{cosmo_mode}]")
        ax[0, 0].legend()
        ax[0, 0].set_xlabel(r"$\ell$", fontsize=18)
        ax[0, 0].set_ylabel(r"$\frac{\ell (\ell + 1)}{2 \pi} C_{\ell}$", fontsize=18)

        ax[0, 1].xaxis.set_tick_params(labelsize=14)
        ax[0, 1].yaxis.set_tick_params(labelsize=14)
        ax[0, 1].set_yscale("log")
        ax[0, 1].set_xscale("log")
        ax[0, 1].set_title(
            "Angular correlation function" + r" $\xi_{\kappa} (\vartheta)$ " + f"[{cosmo_mode}]"
        )
        ax[0, 1].legend()
        ax[0, 1].set_xlabel(r"$\vartheta$ (arc min)", fontsize=18)
        ax[0, 1].set_ylabel(r"$\xi_{\kappa} (\vartheta)$", fontsize=18)

        fig.tight_layout()
        #fig.savefig(outpath / f"{cosmo_mode}_theory_2pcf.pdf", dpi=300)
        fig.savefig(outpath / f"/net/GECO/pisco/SHEAR2COSMO/DBB_OPT_COSMOSLICS/theory_2pcf", dpi=300)
        pyplot.close(fig)
