import galsim
import numpy
from functions_cdfs import (
    bins_per_mag,
    compute_centers_and_edges,
    compute_which_bin,
    get_cosmos_values,
    get_projected,
    get_projected_magnitude,
    load_data_for_inverse_functions,
)
from functions_lhs import draw_lhs
from plots import (
    cdfs_cosmos,
    check_distrib_cosmos,
    check_lhs_projections,
    distrib_cosmos,
    histos_cosmos,
)
from utils import DIRS

# USER DATA ----------------------------------------------------------------------------------------
# SAMPLE = "23.5"
SAMPLE = "25.2"


# FILTER INDICES -----------------------------------------------------------------------------------
if __name__ == "__main__":
    # load all parameters with marginal exclusion
    cosmos_cat = galsim.COSMOSCatalog(
        sample=SAMPLE, dir=DIRS["cosmos"], exclusion_level="marginal", exptime=1500
    )

    # value for distribution functions
    non_filtered_param = cosmos_cat.param_cat
    mag, sersic, hlr = get_cosmos_values(non_filtered_param)

    # mask for bad fit
    filtered_indices = cosmos_cat.orig_index
    mask = numpy.zeros(non_filtered_param.shape[0], dtype=bool)
    mask[filtered_indices] = True

    # masks for wanted distribution (more than 1k sample for a specific value for statistics)
    mask &= (sersic < 6.0) & (sersic >= 0.3)
    mask &= (mag >= 20.0) & (mag <= 24.5)
    mask &= (hlr >= 0.05) & (hlr <= 1.5)

    # apply mask
    mag = mag[mask]
    # mag = numpy.clip(mag, a_min=None, a_max=24.5)
    sersic = sersic[mask]
    hlr = hlr[mask]

    # Check initial plots from cosmos
    bins = 50
    distrib_cosmos(mag, hlr, sersic, DIRS["cosmos_plots"], SAMPLE, bins)

    # bins specifics
    BIN_SPACE = 0.5
    mag_bin_edges, bin_centers = compute_centers_and_edges(SAMPLE, bin_space=0.5)

    # Plots wanted cdfs
    bins = 50
    density = False
    histos_cosmos(SAMPLE, mag, hlr, sersic, mag_bin_edges, density, DIRS["cosmos_plots"], bins)
    density = True
    histos_cosmos(SAMPLE, mag, hlr, sersic, mag_bin_edges, density, DIRS["cosmos_plots"], bins)
    cdfs_cosmos(SAMPLE, mag, hlr, sersic, mag_bin_edges, DIRS["cosmos_plots"])

    # save for inv future functions
    dict_hlr, dict_sersic = bins_per_mag(mag, hlr, sersic, mag_bin_edges, bin_centers)
    numpy.savez(DIRS["cosmos"] / f"{SAMPLE}_mag.npz", mag=mag)
    numpy.savez(DIRS["cosmos"] / f"{SAMPLE}_dict_hlr.npz", **dict_hlr)
    numpy.savez(DIRS["cosmos"] / f"{SAMPLE}_dict_sersic.npz", **dict_sersic)

    # Check loads for future LHS
    inv_cdf_mag, inv_cdfs_hlr, inv_cdfs_sersic = load_data_for_inverse_functions(SAMPLE)

    # example with random lhs
    lhs = draw_lhs(seed=22, n_samples=200_000, dim=8)
    mag = get_projected_magnitude(lhs[:, 0], inv_cdf_mag)
    which_mag_bin = compute_which_bin(mag, mag_bin_edges, bin_centers)
    hlr = get_projected(lhs[:, 1], which_mag_bin, inv_cdfs_hlr)
    sersic = get_projected(lhs[:, 2], which_mag_bin, inv_cdfs_sersic)

    # Check plots for future LHS
    check_distrib_cosmos(mag, hlr, sersic, DIRS["cosmos_plots"], SAMPLE)
    check_lhs_projections(
        DIRS["cosmos_plots"],
        SAMPLE,
        bin_centers,
        mag_bin_edges,
        mag,
        hlr,
        sersic,
        inv_cdfs_hlr,
        inv_cdfs_sersic,
        nb_pdf=30,
    )

    print("FINITO")
