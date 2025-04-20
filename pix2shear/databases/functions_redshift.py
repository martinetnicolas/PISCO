import numpy
from pandas import read_csv
from utils import DIR_DATASETS


def redshift_vs_shear_data():
    file = read_csv(DIR_DATASETS / "data_z_shear.csv")
    return file["redshift"], 2 * file["shear"]


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


def fit_g_vs_z(z, g, deg=1):
    coeffs_inv = numpy.polyfit(z, g, deg=deg)
    inverse_function = numpy.poly1d(coeffs_inv)
    return inverse_function
