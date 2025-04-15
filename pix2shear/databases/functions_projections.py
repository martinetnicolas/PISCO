import numpy
from functions_cdfs import CDF_NS_WRT_MAG, get_inv_cdfs
from scipy import stats
from utils import scale


def apply_uniform_transformation(array):
    max_radius = 0.6
    radii = numpy.sqrt(array[:, 0] * max_radius**2)
    angles = array[:, 1] * 2 * numpy.pi

    array[:, 0] = radii * numpy.cos(angles)
    array[:, 1] = radii * numpy.sin(angles)

    return array


def checky(samples):
    squares_sum = numpy.sqrt(numpy.square(samples[:, 0]) + numpy.square(samples[:, 1]))
    mask = squares_sum > 0.7
    return mask


def new_sample(number):
    loc = 0
    scale = 0.26
    adding_lhs_sampler = stats.qmc.LatinHypercube(d=2, scramble=False, strength=1)
    adding_lhs_samples = adding_lhs_sampler.random(n=number)

    adding_lhs_samples[:, 0] = stats.norm.ppf(adding_lhs_samples[:, 0], loc=loc, scale=scale)
    adding_lhs_samples[:, 1] = stats.norm.ppf(adding_lhs_samples[:, 1], loc=loc, scale=scale)

    return adding_lhs_samples


def apply_gaussian_transformation(lhs_samples_gal):
    loc = 0
    scale = 0.26
    lhs_samples_gal[:, :2] = stats.norm.ppf(lhs_samples_gal[:, :2], loc=loc, scale=scale)

    mask = checky(lhs_samples_gal)
    number_of_outsider = numpy.sum(mask)

    while number_of_outsider != 0:
        print("number_of_outsider", number_of_outsider)
        new_samples = new_sample(number_of_outsider)
        lhs_samples_gal[mask, :2] = new_samples
        mask = checky(lhs_samples_gal)
        number_of_outsider = numpy.sum(mask)

    return lhs_samples_gal


def get_magnitude(lhs_gal, inv_cdf_mag):
    lhs_gal = inv_cdf_mag(lhs_gal)

    closest_half = numpy.around(lhs_gal * 2) / 2  # closest half categories
    closest_half[closest_half == 20] = 20.5  # to avoid 20.0

    return lhs_gal, closest_half


def get_half_light_radius(lhs_gal, closest_half, inv_cdfs_hlr):
    for i, cas in enumerate(closest_half):
        value_lhs = lhs_gal[i]
        lhs_gal[i] = inv_cdfs_hlr[float(cas)](value_lhs)

    min_value = abs(lhs_gal.min())
    lhs_gal = lhs_gal + 2 * min_value + 0.05

    return lhs_gal


def get_sersic_index(lhs_gal, closest_half, inv_cdfs_ns):
    for i, cas in enumerate(closest_half):
        value_lhs = lhs_gal[i]
        lhs_gal[i] = inv_cdfs_ns[float(cas)](value_lhs)

    ns_possible_values = numpy.unique(CDF_NS_WRT_MAG[:, 1])

    lhs_gal = ns_possible_values[numpy.abs(lhs_gal[:, None] - ns_possible_values).argmin(axis=1)]
    lhs_gal = numpy.clip(lhs_gal, 0.3, 6.2)

    return lhs_gal


def generate_positions(array, pixel_out):
    if pixel_out:
        new_min, new_max = 10, 342
    else:
        new_min, new_max = 0, 352

    array = scale(array, low=new_min, high=new_max)

    return array


def project_lhs(hypercube, ellips_style, pixel_out):
    # ellipticities (column 0 & 1)
    if ellips_style == "uniform":
        hypercube[:, :2] = apply_uniform_transformation(hypercube[:, :2])
    elif ellips_style == "gaussian":
        hypercube[:, :2] = apply_gaussian_transformation(hypercube[:, :2])

    # (column 2 & 3 for offsets)

    # inverse CDFs for physical mg/hlr/sersic
    inv_cdf_mag, inv_cdfs_ns, inv_cdfs_hlr, _ = get_inv_cdfs()

    # mag
    hypercube[:, 4], closest_half = get_magnitude(hypercube[:, 4], inv_cdf_mag)

    # hlr
    hypercube[:, 5] = get_half_light_radius(hypercube[:, 5], closest_half, inv_cdfs_hlr)

    # sersic
    hypercube[:, 6] = get_sersic_index(hypercube[:, 6], closest_half, inv_cdfs_ns)

    # positions
    hypercube[:, 7] = generate_positions(hypercube[:, 7], pixel_out)
    hypercube[:, 8] = generate_positions(hypercube[:, 8], pixel_out)

    return hypercube
