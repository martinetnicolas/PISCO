import numpy
from pandas import read_csv
from scipy import stats
from utils import scale, search_folder


def draw_lhs(dim, seed, n_samples):
    lhs_sampler = stats.qmc.LatinHypercube(
        d=dim, scramble=False, optimization=None, strength=1, seed=seed
    )
    return lhs_sampler.random(n=n_samples)


def generate_shear_scaled(
    g1_init, g2_init, shunsheng_fn, z_grid, z_pdf, g_magnitude, sampler="photoz_dist"
):
    n_samples = g1_init.shape[0]
    if sampler == "uniform":
        z_samples = numpy.random.uniform(0.2, 2.5, size=n_samples)

    elif sampler == "photoz_dist":
        z_samples = numpy.random.choice(z_grid, size=n_samples, p=z_pdf / numpy.sum(z_pdf))

    g_target = shunsheng_fn(z_samples)

    sorted_idx = numpy.argsort(g_magnitude)
    g_target_sorted = numpy.sort(g_target)

    scale = g_target_sorted / (g_magnitude[sorted_idx] + 1e-8)

    g1_scaled = g1_init[sorted_idx] * scale
    g2_scaled = g2_init[sorted_idx] * scale
    z_sorted = z_samples[sorted_idx]

    return z_sorted, g1_scaled, g2_scaled


def draw_redshift(g1_init, g2_init):
    z_grid = numpy.linspace(0.2, 2.5, 10000)
    z_pdf = z_grid**2 * numpy.exp(-z_grid / 0.8)

    # fit function g vs z
    z_data, g_data = redshift_vs_shear_data()
    shunsheng_fn = fit_g_vs_z(z_data, g_data, deg=1)

    # compute magnitudes from the initial g1, g2 vectors
    g_magnitude = numpy.sqrt(g1_init**2 + g2_init**2)

    # generate shear with target magnitudes
    z, g1_scaled, g2_scaled = generate_shear_scaled(
        g1_init, g2_init, shunsheng_fn, z_grid, z_pdf, g_magnitude
    )
    return z, g1_scaled, g2_scaled


def draw_lhs_img(seed, n_samples, with_redshift=False, decimals=3):
    hypercube = draw_lhs(2, seed, n_samples)

    if decimals == 2:
        max_shears = 0.065
    elif decimals == 3:
        max_shears = 0.0605
    hypercube = scale(hypercube, low=-max_shears, high=max_shears)

    if with_redshift:
        z, hypercube[:, 0], hypercube[:, 1] = draw_redshift(hypercube[:, 0], hypercube[:, 1])
    else:
        z, _, _ = draw_redshift(hypercube[:, 0], hypercube[:, 1])

    hypercube = numpy.round(hypercube, decimals=decimals)
    hypercube = numpy.hstack([hypercube, z[:, None]])

    return hypercube


def draw_lhs_gal(lhs_pos, seed, nb_total_galaxies):
    if lhs_pos:
        hypercube = draw_lhs(9, seed, n_samples=nb_total_galaxies)

    else:
        hypercube = draw_lhs(7, seed, n_samples=nb_total_galaxies)
        pos_xy = draw_lhs(2, seed * 2, n_samples=nb_total_galaxies)
        hypercube = numpy.hstack([hypercube, pos_xy])

    return hypercube


def redshift_vs_shear_data():
    DIR_DATASETS = search_folder(name="DATASETS")
    file = read_csv(DIR_DATASETS / "data_z_shear.csv")
    return file["redshift"], 2 * file["shear"]


def fit_g_vs_z(z, g, deg=1):
    coeffs_inv = numpy.polyfit(z, g, deg=deg)
    inverse_function = numpy.poly1d(coeffs_inv)
    return inverse_function
