import math
import multiprocessing
import random
import time
from collections import defaultdict
from pathlib import Path

import galsim
import numpy
from astropy import io
from joblib import Parallel, delayed, parallel_backend
from pandas import DataFrame, read_csv

DTYPE = numpy.float32


def find_directory(name, init_directory):
    directory_path = Path(init_directory)
    matching_folders = list(directory_path.glob(f"**/{name}"))
    return matching_folders[0]


def search_folder(name):
    if Path.cwd().parts[1] == "home":
        search_dir = "/home"
    else:
        search_dir = "/net/GECO/pisco/PIX2SHEAR"
    return find_directory(name, search_dir)


DIR_DATASETS = search_folder(name="DATASETS")
DIRS = {
    "images_cosmos": DIR_DATASETS / "images" / "cosmos",
    "cosmos": DIR_DATASETS / "cosmos",
    "pisco_euclid": DIR_DATASETS / "pisco_euclid",
    "psf": DIR_DATASETS / "psf",
    "images": DIR_DATASETS / "images",
    "duplicates": DIR_DATASETS / "duplicates",
}


def write_fits(data, path, sampling):
    header = io.fits.Header()
    header["SAMPLING"] = sampling
    hdu = io.fits.PrimaryHDU(data=data, header=header)
    hdulist = io.fits.HDUList([hdu])
    hdulist.writeto(path, overwrite=True)
    hdulist.close()


def load_metrics(arg):
    return numpy.mean(arg), numpy.amax(arg)


def _parallel_with_one_arg(func, data):
    nb_cpus = multiprocessing.cpu_count()
    print(f"using {nb_cpus} cpus")
    with parallel_backend("threading", n_jobs=nb_cpus):
        return numpy.array(
            Parallel(prefer="threads", verbose=0)(delayed(func)(arg=arg) for arg in data)
        )


def getting_norms(data):
    results = _parallel_with_one_arg(func=load_metrics, data=data)

    measures_norms = defaultdict(list)
    for result in results:
        measures_norms["mean"].append(result[0])
        measures_norms["amax"].append(result[1])

    return DataFrame(measures_norms)


def save_psf(direc, style, psf, image_size, pixel_scale, with_noise=False):
    image = galsim.ImageF(image_size, image_size, scale=pixel_scale)
    psf_image = psf.drawImage(image)

    if with_noise:
        gain = 3.1  # e-/ADU
        readoutnoise = 4.2  # e-
        sky_bkg = 22.35  # mag/arcsec2
        ZP = 24.0  # mag
        t_exp = 3 * 565  # s
        F_sky = pixel_scale ** (2) * t_exp * 10 ** (-(sky_bkg - ZP) / 2.5)  # e-/pixel
        noise_variance = (numpy.sqrt(((readoutnoise) ** 2 + F_sky)) * 1 / gain) ** 2
        rng = galsim.BaseDeviate(seed=22)
        noise = galsim.GaussianNoise(rng, sigma=math.sqrt(noise_variance) / 1000.0)
        psf_image.addNoise(noise)

    hdu = io.fits.PrimaryHDU(psf_image.array)
    hdul = io.fits.HDUList([hdu])

    name = direc / f"psf_style-{style}_size-{image_size}_pixelscale-{pixel_scale}"
    hdul.writeto(fileobj=f"{name}.fits", overwrite=True)


def get_indices(hypercube_gal_params):
    cols_to_groupby = [
        col
        for col in hypercube_gal_params.columns
        if col not in ["int_e1", "int_e2", "obs_e1", "obs_e2"]
    ]
    duplicates = hypercube_gal_params[
        hypercube_gal_params.duplicated(subset=cols_to_groupby, keep=False)
    ]

    grouped_duplicates = {
        name: group.index.tolist() for name, group in duplicates.groupby(cols_to_groupby)
    }
    grouped_row_values = list(grouped_duplicates.keys())
    all_duplicate_pairs_divided = [
        [idx // 10 for idx in indices] for indices in grouped_duplicates.values()
    ]

    unique_pairs = set(map(tuple, all_duplicate_pairs_divided))
    convert_to_list = list(map(list, unique_pairs))

    return grouped_row_values, convert_to_list


def check_unique(df, column):
    unique_vals, counts = numpy.unique(df[column].to_numpy(), return_counts=True)
    non_unique_count = numpy.sum(counts > 1)
    return unique_vals, counts, non_unique_count


def get_pairs(pairs_img, hypercube_img, pairs_gal, hypercube_gal):
    all_pairs = []
    sources = [(pairs_img, hypercube_img), (pairs_gal, hypercube_gal)]
    for pairs, hypercube in sources:
        for pair in pairs:
            all_pairs.append((pair, hypercube))
    return all_pairs


def scale(array, low, high):
    return array * (high - low) + low


def print_csv_images_galaxies(keys_dataset, all_params, dir, columns_img, columns_gal):
    (avg_g, avg_int_e, avg_obs_e, keys, gal_params) = (
        numpy.stack([item[key] for item in all_params]) for key in keys_dataset
    )

    hypercube_img = DataFrame(
        data=numpy.hstack([avg_g, avg_int_e, avg_obs_e, keys]), columns=columns_img
    )

    hypercube_img.to_csv(dir / "images_parameters_true.csv", index=False)

    a, b, c = gal_params.shape
    gal_params_csv = gal_params.transpose(0, 2, 1).reshape(a * c, b)
    hypercube_gal = DataFrame(data=gal_params_csv, columns=columns_gal)

    hypercube_gal.to_csv(dir / "galaxies_parameters_true.csv", index=False)


def extract_keys_fn(keys):
    def _extract_keys(x):
        return {key: x[key] for key in keys}

    return _extract_keys


def extract_image(x):
    return x["image"]


def extract_gal_params(x):
    return x["gal_params"]


def slice_csv(file_path, nb_total):
    return read_csv(file_path, sep=",", skiprows=0)[:nb_total]


def generate_seed():
    return (int(time.time() * 1e6) ^ random.getrandbits(32)) % (2**32 - 1)
