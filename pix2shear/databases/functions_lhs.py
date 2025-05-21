from collections import defaultdict
from pathlib import Path

import numpy
from functions_cdfs import (
    compute_centers_and_edges,
    compute_which_bin,
    get_projected,
    get_projected_magnitude,
    load_data_for_inverse_functions,
)
from pandas import read_csv
from scipy import spatial, stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import DIRS, RED, RESET, scale

# --------------------------------------------------------------------------------------------------


def draw_lhs(seed, n_samples, dim):
    lhs_sampler = stats.qmc.LatinHypercube(
        d=dim, scramble=False, optimization=None, strength=1, seed=seed
    )
    return lhs_sampler.random(n=n_samples)


def mask_big_ellipticites(samples):
    squares_sum = numpy.sqrt(numpy.square(samples[:, 0]) + numpy.square(samples[:, 1]))
    mask = squares_sum > 0.7
    return mask


def new_sample(number_of_outsider):
    loc = 0
    scale = 0.26

    # seed = numpy.random.randint(0, 2**32 - 1)
    seed = 22  # reproducibility for test parametric / real from cosmos

    adding_lhs_samples = draw_lhs(seed, n_samples=number_of_outsider, dim=2)

    adding_lhs_samples[:, 0] = stats.norm.ppf(adding_lhs_samples[:, 0], loc=loc, scale=scale)
    adding_lhs_samples[:, 1] = stats.norm.ppf(adding_lhs_samples[:, 1], loc=loc, scale=scale)

    return adding_lhs_samples


def ellips_gaussian_transformation(array):
    loc = 0
    scale = 0.26
    array = stats.norm.ppf(array, loc=loc, scale=scale)

    mask = mask_big_ellipticites(array)
    number_of_outsider = numpy.sum(mask)

    while number_of_outsider != 0:
        print("number_of_outsider", number_of_outsider)
        new_samples = new_sample(number_of_outsider)
        array[mask] = new_samples

        mask = mask_big_ellipticites(array)
        number_of_outsider = numpy.sum(mask)

    return array


def ellips_uniform_transformation(array):
    max_radius = 0.6
    radii = numpy.sqrt(array[:, 0] * max_radius**2)
    angles = array[:, 1] * 2 * numpy.pi

    array[:, 0] = radii * numpy.cos(angles)
    array[:, 1] = radii * numpy.sin(angles)

    return array


# create 10 pixels band-free galaxy if wanted
def generate_positions(array, pixel_out, band=10):
    if pixel_out:
        new_min, new_max = int(band), int(352 - band)
    else:
        new_min, new_max = 0, 352

    array = scale(array, low=new_min, high=new_max)

    return array


def normalise(catalog_features, catalog_targets, scaling):
    if scaling == "min_max":
        scaler = MinMaxScaler().fit(catalog_features)
    elif scaling == "z_score":
        scaler = StandardScaler().fit(catalog_features)

    catalog_features_norm = scaler.transform(catalog_features)
    catalog_targets_norm = scaler.transform(catalog_targets)

    return catalog_features_norm, catalog_targets_norm


# duplicates from cosmos catalogs
def duplicates(arr, version):
    row_groups = defaultdict(list)

    for idx, row in enumerate(arr):
        row_key = tuple(row)
        row_groups[row_key].append(idx)

    duplicates = {row: indices for row, indices in row_groups.items() if len(indices) > 1}

    if len(duplicates) > 0:
        percentage = 100 * len(duplicates) / arr.shape[0]
        print(f"{RED} {percentage}% duplicates over {arr.shape[0]} {RESET}")
        path_save = DIRS["duplicates"] / f"{Path(str(version))}_duplicates_cosmos.npy"
        numpy.save(file=path_save, arr=duplicates, allow_pickle=True)


# --------------------------------------------------------------------------------------------------


# LHS for image parameters & projection into real space
def draw_and_project_lhs_img(seed, n_samples, decimals=3):
    hypercube = draw_lhs(seed, n_samples=n_samples, dim=2)

    # keep lhs uniform at the edge when rounded
    max_shears = {2: 0.065, 3: 0.0605}[decimals]
    hypercube = scale(hypercube, low=-max_shears, high=max_shears)

    hypercube = numpy.round(hypercube, decimals=decimals)

    return hypercube


# LHS for galaxies parameters
def draw_lhs_gal(lhs_pos, seed, nb_total_galaxies):
    if lhs_pos:
        hypercube = draw_lhs(seed, n_samples=nb_total_galaxies, dim=9)

    else:
        # case galaxies position not included in lhs init (prob useless?)
        hypercube = draw_lhs(seed, n_samples=nb_total_galaxies, dim=7)
        pos_xy = draw_lhs(seed * 2, n_samples=nb_total_galaxies, dim=2)
        hypercube = numpy.hstack([hypercube, pos_xy])

    # addings for galaxies from cosmos catalog
    new_columns = numpy.zeros((hypercube.shape[0], 2), dtype=hypercube.dtype) - 1
    hypercube = numpy.hstack((hypercube, new_columns))

    return hypercube


# project galaxies parameters into real space
def project_lhs_gal(hypercube, ellips_style, pixel_out):
    # ellipticities (column 0 & 1)
    if ellips_style == "uniform":
        hypercube[:, :2] = ellips_uniform_transformation(hypercube[:, :2])
    elif ellips_style == "gaussian":
        hypercube[:, :2] = ellips_gaussian_transformation(hypercube[:, :2])

    # (column 2 & 3 for offsets)

    # inverse CDFs for physical mag/hlr/sersic
    style = "25.2"
    inv_cdf_mag, inv_cdfs_hlr, inv_cdfs_sersic = load_data_for_inverse_functions(style)
    mag_bin_edges, bin_centers = compute_centers_and_edges(style, bin_space=0.5)

    # mag
    hypercube[:, 4] = get_projected_magnitude(hypercube[:, 4], inv_cdf_mag)
    which_mag_bin = compute_which_bin(hypercube[:, 4], mag_bin_edges, bin_centers)

    # hlr
    hypercube[:, 5] = get_projected(hypercube[:, 5], which_mag_bin, inv_cdfs_hlr)

    # sersic
    hypercube[:, 6] = get_projected(hypercube[:, 6], which_mag_bin, inv_cdfs_sersic)

    # positions
    hypercube[:, 7] = generate_positions(hypercube[:, 7], pixel_out)
    hypercube[:, 8] = generate_positions(hypercube[:, 8], pixel_out)

    return hypercube


# replace random galaxies into real cosmos-like galaxies
def adding_cosmos_catalog_galaxies(
    lhs_gal,
    replace_indices,
    version,
    scaling="z_score",  # "min_max" "z_score"
    metric="euclidean",  # "euclidean" "mahalanobis" "cityblock"
):
    # galaxies we want to replace
    check_columns = [0, 1, 4, 5, 6]  # e1 e2 mag hlr sersic
    check_columns_cosmos = ["e1", "e2", "mag", "hlr", "sersic"]

    # galaxies from lhs
    catalog_targets = lhs_gal[:, check_columns][replace_indices]

    # galaxies from cosmos catalogs (already filtered)
    filtered_params = read_csv(DIRS["cosmos"] / "filtered_params.csv")
    catalog_features = filtered_params[check_columns_cosmos].values
    catalog_index = filtered_params[["indices", "sample"]].values

    # compute distances
    catalog_features_norm, catalog_targets_norm = normalise(
        catalog_features, catalog_targets, scaling
    )
    distances = spatial.distance.cdist(catalog_targets_norm, catalog_features_norm, metric=metric)
    matched_idx = numpy.argmin(distances, axis=1)

    # results and switch
    matched_features = catalog_features[matched_idx]
    matched_index = catalog_index[matched_idx]
    lhs_gal[numpy.ix_(replace_indices, check_columns)] = matched_features
    lhs_gal[replace_indices, 9:11] = matched_index

    # .npy save if any duplicates
    duplicates(matched_features, version)

    return lhs_gal
