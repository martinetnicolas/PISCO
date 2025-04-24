import numpy
from scipy import interpolate
from utils import CDF_MAG, CDF_NS_WRT_MAG, CONT_CDF_HLR_WRT_MAG, DIRS


def inverse_from_data(data):
    sorted_data = numpy.sort(data)
    percentiles = numpy.linspace(0, 1, len(sorted_data), endpoint=True)

    def inv_fn(u):
        return numpy.interp(u, percentiles, sorted_data)

    return inv_fn


def bins_per_mag(mag, hlr, sersic, mag_bin_edges, bin_centers):
    n_bins = len(mag_bin_edges) - 1
    dict_hlr = {}
    dict_sersic = {}

    for i in range(n_bins):
        mask_bin = (mag >= mag_bin_edges[i]) & (mag < mag_bin_edges[i + 1])
        hlr_bin = hlr[mask_bin]
        sersic_bin = sersic[mask_bin]

        bin_key = f"{bin_centers[i]:.2f}"
        if len(hlr_bin) > 0:
            dict_hlr[bin_key] = hlr_bin
        if len(sersic_bin) > 0:
            dict_sersic[bin_key] = sersic_bin

    return dict_hlr, dict_sersic


def get_cosmos_values(non_filtered_param):
    mag = non_filtered_param["mag_auto"]

    sersic = non_filtered_param["sersicfit"][:, 2]

    r = non_filtered_param["sersicfit"][:, 1]
    q = non_filtered_param["sersicfit"][:, 3]
    hlr = 0.03 * numpy.sqrt(q) * r

    return mag, sersic, hlr


def load_data_for_inverse_functions(style):
    mag_data = numpy.load(DIRS["cosmos"] / f"{style}_mag.npz")["mag"]
    inv_cdf_mag = inverse_from_data(mag_data)

    hlr_data = numpy.load(DIRS["cosmos"] / f"{style}_dict_hlr.npz")
    inv_cdfs_hlr = {float(k): inverse_from_data(hlr_data[k]) for k in hlr_data}

    sersic_data = numpy.load(DIRS["cosmos"] / f"{style}_dict_sersic.npz")
    inv_cdfs_sersic = {float(k): inverse_from_data(sersic_data[k]) for k in sersic_data}

    return inv_cdf_mag, inv_cdfs_hlr, inv_cdfs_sersic


def get_projected_magnitude(array, inv_cdf_mag):
    return inv_cdf_mag(array)


def get_projected(array, which_mag_bin, inv_cdfs_fn):
    for i, mag_bin in enumerate(which_mag_bin):
        ongoing_inv_fn = inv_cdfs_fn[mag_bin]

        array[i] = ongoing_inv_fn(array[i])

    return array


def compute_centers_and_edges(sample, bin_space):
    if sample == "23.5":
        mag_bin_edges = numpy.arange(20, 24, bin_space)
    else:
        mag_bin_edges = numpy.arange(20, 25, bin_space)

    bin_centers = (mag_bin_edges[:-1] + mag_bin_edges[1:]) / 2

    return mag_bin_edges, bin_centers


def compute_which_bin(mag, mag_bin_edges, bin_centers):
    bin_indices = numpy.digitize(mag, mag_bin_edges) - 1
    which_mag_bin = bin_centers[bin_indices]
    return which_mag_bin


def old_get_inv_mags():
    bin_size = 0.5
    magnitudes = CDF_MAG[:, 0] + bin_size
    cdf_values = CDF_MAG[:, 1]

    inv_cdf_mag_fn = interpolate.interp1d(cdf_values, magnitudes, fill_value="extrapolate")

    return inv_cdf_mag_fn


def old_compute_which_edge_bin(array):
    bin_size = 0.5
    edge_bin_mags = bin_size * numpy.floor(array / bin_size)
    return edge_bin_mags


def old_get_sersic_index(array, edge_bin_mags):
    for i, mag in enumerate(edge_bin_mags):
        ns_wrt_mag = CDF_NS_WRT_MAG[CDF_NS_WRT_MAG[:, 0] == mag]

        lhs_value = float(array[i])
        choices_ns = ns_wrt_mag[ns_wrt_mag[:, 2] >= lhs_value]

        array[i] = numpy.amin(choices_ns[:, 1])

    array = numpy.clip(array, 0.3, 6.2)

    return array


def old_get_inv_hlr(edge_bin_mags, which_mag_bin):
    inv_cdfs_hlr_fn = {}
    for i, mag in enumerate(edge_bin_mags):
        case_data = CONT_CDF_HLR_WRT_MAG[CONT_CDF_HLR_WRT_MAG[:, 0] == mag]
        hlr = case_data[:, 1]
        cdf = case_data[:, 2]

        interpol_fn = interpolate.interp1d(cdf, hlr, bounds_error=False)
        inv_cdfs_hlr_fn[which_mag_bin[i]] = interpol_fn

    return inv_cdfs_hlr_fn
