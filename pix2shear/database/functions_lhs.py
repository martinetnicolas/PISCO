import numpy
from pandas import read_csv
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import DIR_DATASETS, DIRS, scale


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
    file = read_csv(DIR_DATASETS / "data_z_shear.csv")
    return file["redshift"], 2 * file["shear"]


def fit_g_vs_z(z, g, deg=1):
    coeffs_inv = numpy.polyfit(z, g, deg=deg)
    inverse_function = numpy.poly1d(coeffs_inv)
    return inverse_function


def normalise(catalog_features, test_points, scaling):
    if scaling == "min_max":
        scaler = MinMaxScaler().fit(catalog_features)
    elif scaling == "z_score":
        scaler = StandardScaler().fit(catalog_features)

    catalog_features_norm = scaler.transform(catalog_features)
    test_points_norm = scaler.transform(test_points)

    return catalog_features_norm, test_points_norm


def euclidian_style(diff):
    euclidian_distances = numpy.linalg.norm(diff, axis=2)
    euclidian_closest_idx = numpy.argmin(euclidian_distances, axis=1)
    return euclidian_closest_idx


def maha_style(catalog_features_norm, diff):
    cov = numpy.cov(catalog_features_norm, rowvar=False)
    inv_cov = numpy.linalg.inv(cov)

    mahalanobis_distances = numpy.sqrt(numpy.einsum("...i,ij,...j->...", diff, inv_cov, diff))
    mahalanobis_closest_idx = numpy.argmin(mahalanobis_distances, axis=1)

    return mahalanobis_closest_idx


def adding_cosmos_catalog_galaxies(lhs_gal, replace_indices):
    SCALING = "z_score"  # "min_max" "z_score"
    NORMA = "euclid"  # "euclid" "mahalanobis"

    if replace_indices.shape[0] != 0:
        changing_samples = lhs_gal[:, 4:7][replace_indices]
        filtered_params = read_csv(DIRS["cosmos"] / "filtered_params.csv")

        catalog_features = filtered_params[["mag_auto", "hlr", "sersic"]].values
        catalog_selection = filtered_params[["indices", "sample"]].values

        catalog_features_norm, test_points_norm = normalise(
            catalog_features, changing_samples, SCALING
        )
        diff = test_points_norm[:, numpy.newaxis, :] - catalog_features_norm[numpy.newaxis, :, :]

        if NORMA == "euclid":
            matched_idx = euclidian_style(diff)
        elif NORMA == "mahalanobis":
            matched_idx = maha_style(catalog_features_norm, diff)

        matched_selection = catalog_selection[matched_idx]
        matched_features = catalog_features[matched_idx]

    new_columns = numpy.zeros((lhs_gal.shape[0], 2), dtype=lhs_gal.dtype) - 1
    lhs_gal = numpy.hstack((lhs_gal, new_columns))

    if replace_indices.shape[0] != 0:
        lhs_gal[replace_indices, 4:7] = matched_features
        lhs_gal[replace_indices, 9:11] = matched_selection

    return lhs_gal
