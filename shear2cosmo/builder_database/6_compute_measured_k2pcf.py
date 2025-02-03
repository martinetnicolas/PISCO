from pathlib import Path

import numpy
import treecorr
from matplotlib import pyplot
from tqdm import tqdm
from utils import create_dir

# --------------------------------------------------------------------------------------------------


DBB = "DBB_KMEANS_25"
# DBB = "DBB_DENSITY_25"

PLOT_STEP = 500  # better if same as spheric_to_cartesian

FOLDER_BASE = Path("/net/GECO/pisco/SHEAR2COSMO")
FOLDER_RESULTS = FOLDER_BASE / DBB
FOLDER_CARTESIAN = FOLDER_RESULTS / "cartesian"
FOLDER_MEASURED = create_dir(FOLDER_RESULTS / "measured")
FOLDER_PLOTS = FOLDER_RESULTS / "plots"


# --------------------------------------------------------------------------------------------------


# (AUTO-) Convergence-2PCF Function from flat map
cartesian_paths = sorted(list(FOLDER_CARTESIAN.glob("*.npy")))
for ind, path_kappaE_map in enumerate(tqdm(cartesian_paths, total=len(cartesian_paths))):
    # Make output folders for the considered statistics if they do not exist
    final_name = path_kappaE_map.stem
    kappaE_map = numpy.load(path_kappaE_map)

    # Define map properties
    npix = kappaE_map.shape[0]  # pixel number per side
    side_deg = 10
    pix_res = (side_deg / npix) * 60  # arcmins

    # Mean-shift map
    # kappa maps reconstructed with KS/KSP are already set to a null mean
    kappaE_map -= numpy.nanmean(kappaE_map)

    # Define map coordinates
    ra, dec = numpy.meshgrid(
        numpy.arange(len(kappaE_map)) * pix_res, numpy.arange(len(kappaE_map)) * pix_res
    )

    # Flattenise map and coordinates
    k_cat = kappaE_map.flatten()
    ra_cat = ra.flatten()
    dec_cat = dec.flatten()

    # Remove nan pixels
    sel = numpy.where(numpy.logical_not(numpy.isnan(k_cat)))
    k_cat = k_cat[sel]
    ra_cat = ra_cat[sel]
    dec_cat = dec_cat[sel]

    # Compute kappa-2pcf using Paper I settings
    twopt_min = pix_res  # arcmins
    # Fix maximum separation to the DUSTGRAIN map diagonal in order to have measurements
    # at the same scales across all cosmologies of DUSTGRAIN and SLICS
    twopt_max = side_deg * 60 * numpy.sqrt(2)
    twopt_bins = 10  # ?? DIMENSION = 20 ??

    treecormap = treecorr.Catalog(
        ra=ra_cat, dec=dec_cat, k=k_cat, ra_units="arcmin", dec_units="arcmin"
    )
    kk = treecorr.KKCorrelation(
        min_sep=twopt_min, max_sep=twopt_max, nbins=twopt_bins, sep_units="arcmin"
    )
    kk.process(treecormap)

    # Note: meanlogr represents the mean distance of all the galaxy pairs falling in the given bin
    # (it is an alternative to the simple bin mid-point rnom considered in the mean plots)
    kk_r = numpy.exp(kk.meanlogr)
    kk_xi = kk.xi
    kk_sig = numpy.sqrt(kk.varxi)

    if ind % PLOT_STEP == 0:
        # Plot kappa-2pcf using meanlogr (for debugging) for first two realisations
        fig_2pt, ax_2pt = pyplot.subplots(1, 1, squeeze=False)
        ax_2pt[0, 0].set_xscale("log")
        ax_2pt[0, 0].set_yscale("log", nonpositive="clip")
        ax_2pt[0, 0].set_xlabel(r"$\theta$ (arcmin)", fontsize=18)
        ax_2pt[0, 0].set_ylabel(r"$\xi$", fontsize=18)
        ax_2pt[0, 0].set_xlim([twopt_min, twopt_max])

        if any(kk_xi > 0):
            ax_2pt[0, 0].errorbar(
                kk_r[kk_xi > 0],
                kk_xi[kk_xi > 0],
                yerr=kk_sig[kk_xi > 0],
                color="blue",
                ls="None",
                marker="o",
            )
        if any(kk_xi < 0):
            ax_2pt[0, 0].errorbar(
                kk_r[kk_xi < 0],
                -kk_xi[kk_xi < 0],
                yerr=kk_sig[kk_xi < 0],
                color="white",
                markeredgecolor="blue",
                ls="None",
                marker="o",
            )

        fig_2pt.suptitle(f"{final_name} kappa_2pcf")
        fig_2pt.savefig(FOLDER_PLOTS / f"{final_name}_kappa_2pcf.png", dpi=300)
        fig_2pt.clf()
        pyplot.close(fig_2pt)

    # Write kappa-2pcf DV to text
    kk.write(FOLDER_MEASURED / f"{final_name}_kappa_2pcf.txt")
