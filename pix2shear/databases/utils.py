import math
import multiprocessing
import random
import time
from collections import defaultdict
from pathlib import Path

import galsim
import numpy
import tensorflow
from astropy import io
from joblib import Parallel, delayed, parallel_backend
from pandas import DataFrame, read_csv

DTYPE_16 = numpy.float16
DTYPE_32 = numpy.float32

RED = "\033[38;2;255;80;80m"
BLUE = "\033[38;2;80;180;255m"
RESET = "\033[0m"


def search_folder(name):
    if Path.cwd().parts[1] == "home":
        search_dir = Path("/home")
    else:
        search_dir = Path("/net/GECO/pisco/PIX2SHEAR")
    matching_folders = list(search_dir.glob(f"**/{name}"))
    return matching_folders[0]


DIR_DATASETS = search_folder(name="DATASETS")
DIRS = {
    "cosmos_plots": DIR_DATASETS / "cosmos_plots",
    "cosmos": DIR_DATASETS / "cosmos",
    "duplicates": DIR_DATASETS / "cosmos" / "duplicates",
    "pisco_euclid": DIR_DATASETS / "pisco_euclid",
    "pisco_euclid_se": DIR_DATASETS / "pisco_euclid_se",
    "pisco_euclid_plots": DIR_DATASETS / "pisco_euclid_plots",
    "psf": DIR_DATASETS / "psf",
    "psf_se": DIR_DATASETS / "psf_se",
    "psf_plots": DIR_DATASETS / "psf_plots",
    "wip": DIR_DATASETS / "wip",
}


def load_metrics(arg):
    return numpy.mean(arg), numpy.amax(arg)


def _parallel_with_one_arg(func, data):
    nb_cpus = multiprocessing.cpu_count()
    print(f"using {nb_cpus} cpus")
    with parallel_backend("threading", n_jobs=nb_cpus):
        return numpy.array(
            Parallel(prefer="threads", verbose=0)(delayed(func)(arg=arg) for arg in data)
        )


def write_fits(path, data, sampling):
    header = io.fits.Header()
    header["SAMPLING"] = sampling
    hdu = io.fits.PrimaryHDU(data=data.astype(numpy.float32), header=header)
    hdulist = io.fits.HDUList([hdu])
    hdulist.writeto(path, overwrite=True)
    hdulist.close()


def getting_norms(data):
    results = _parallel_with_one_arg(func=load_metrics, data=data)

    measures_norms = defaultdict(list)
    for result in results:
        measures_norms["mean"].append(result[0])
        measures_norms["amax"].append(result[1])

    return DataFrame(measures_norms)


def create_se(example, num):
    hashed_key = int(example["keys"][0])
    gal_params = example["gal_params"]

    df_gal = DataFrame(
        gal_params.T,
        columns=["int_e1", "int_e2", "obs_e1", "obs_e2", "x", "y", "mag", "hlr", "sersic"],
    )
    df_gal["hashed_key"] = hashed_key
    df_gal["image"] = num

    return df_gal


def write_values_for_se(DIR_SE, sampling, hashed_key, image, df):
    # save galaxy parameters per image
    path_csv = DIR_SE / f"{hashed_key:06d}_true_galaxies_params.csv"
    df.to_csv(path_or_buf=path_csv, index=False)

    # save image as FITS
    path_fits = DIR_SE / f"{hashed_key:06d}_simu.fits"
    write_fits(path=path_fits, data=image, sampling=sampling)

    # save positions (x, y) only for SE assoc
    path_assoc = DIR_SE / f"{hashed_key:06d}_simu_assoc.txt"
    df[["x", "y"]].to_csv(path_or_buf=path_assoc, sep=" ", index=False, header=False)


def create_df_img_and_gal(example, nb_image):
    gal_params = example["gal_params"]
    keys = example["keys"]
    avg_g = example["avg_g"]
    avg_obs_e = example["avg_obs_e"]
    avg_int_e = example["avg_int_e"]

    # accumulate gal properties
    df_gal = DataFrame(
        gal_params.T,
        columns=["int_e1", "int_e2", "obs_e1", "obs_e2", "x", "y", "mag", "hlr", "sersic"],
    )
    df_gal["hashed_key"] = int(keys[0])
    df_gal["image"] = nb_image

    # accumulate avg_g and avg_obs_e for global CSV
    dict_img = {
        "avg_g1": avg_g[0],
        "avg_g2": avg_g[1],
        "avg_obs_e1": avg_obs_e[0],
        "avg_obs_e2": avg_obs_e[1],
        "avg_int_e1": avg_int_e[0],
        "avg_int_e2": avg_int_e[1],
        "hashed_key": int(keys[0]),
        "snc": int(keys[1]),
        "noise_key": int(keys[2]),
        "image": nb_image,
    }
    df_img = DataFrame([dict_img])

    return df_gal, df_img


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


def get_indices_snc2(hypercube_gal):
    cols_to_ignore = ["int_e1", "int_e2", "obs_e1", "obs_e2", "hashed_key", "image"]
    df_compare = hypercube_gal.drop(columns=cols_to_ignore)
    grouped_indices = df_compare.groupby(list(df_compare.columns)).indices
    common_index_lists = [idx_list for idx_list in grouped_indices.values() if len(idx_list) > 1]
    return common_index_lists


def get_indices_snc(hypercube_gal_params):
    cols_to_groupby = [
        col
        for col in hypercube_gal_params.columns
        if col not in ["int_e1", "int_e2", "obs_e1", "obs_e2", "hashed_key"]
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


@tensorflow.autograph.experimental.do_not_convert
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


def check_sampling(dir):
    sampling = []
    for i in range(1, 1001):
        filename = dir / f"{i:06d}_simu.fits"
        with io.fits.open(filename) as hdul:
            header = hdul[0].header
            sampling.append(header.get("SAMPLING"))

    return numpy.unique(numpy.array(sampling))


def generate_seed():
    return (int(time.time() * 1e6) ^ random.getrandbits(32)) % (2**32 - 1)


def render_simple_psf(psf, size, dtype, scale):
    img = galsim.ImageF(size, size)
    img.setOrigin(0, 0)
    img = psf.drawImage(image=img, scale=scale)
    return img.array.astype(dtype)


# binned random style
def binned_style(number):
    dmag = 0.5
    dhlr = 0.1
    list_mag = []
    list_ns = []
    list_hlr = []
    list_e1 = []
    list_e2 = []
    for seed in range(number):
        # numpy.random.seed(seed)
        uds_gal = galsim.UniformDeviate(seed)

        magi = numpy.min(CDF_MAG[CDF_MAG[:, 1] >= uds_gal()][:, 0])
        mag = magi + dmag * uds_gal()
        list_mag.append(mag)

        PDFhlr_cmagi = CDF_HLR_WRT_MAG[CDF_HLR_WRT_MAG[:, 0] == magi]
        hlr = numpy.min(PDFhlr_cmagi[PDFhlr_cmagi[:, 2] >= uds_gal()][:, 1]) + dhlr * uds_gal()
        list_hlr.append(hlr)

        PDFn_cmagi = CDF_NS_WRT_MAG[CDF_NS_WRT_MAG[:, 0] == magi]
        ns = numpy.min(PDFn_cmagi[PDFn_cmagi[:, 2] >= uds_gal()][:, 1])
        list_ns.append(ns)

        gds_gal = galsim.GaussianDeviate(seed=seed, mean=0.0, sigma=0.26)
        list_e1.append(gds_gal())
        list_e2.append(gds_gal())

    return list_mag, list_hlr, list_ns, list_e1, list_e2


# HUBBLE DEEP FIELD STYLE -----------------------------------------------------------------
CDF_MAG = numpy.array(
    [
        [20.5, 0.04508],
        [21.0, 0.09016],
        [21.5, 0.14754],
        [22.0, 0.20902],
        [22.5, 0.31967],
        [23.0, 0.43033],
        [23.5, 0.67213],
        [24.0, 1.0],
    ]
)
CDF_HLR_WRT_MAG = numpy.array(
    [
        [20.5, 0.0, 0.0],
        [20.5, 0.2, 0.0909],
        [20.5, 0.3, 0.1818],
        [20.5, 0.4, 0.4545],
        [20.5, 0.7, 0.5454],
        [20.5, 0.9, 0.6363],
        [20.5, 1.0, 0.8181],
        [20.5, 1.1, 1.0],
        [21.0, 0.0, 0.0],
        [21.0, 0.2, 0.1818],
        [21.0, 0.3, 0.5454],
        [21.0, 0.5, 0.6363],
        [21.0, 0.6, 0.7272],
        [21.0, 0.8, 1.0],
        [21.5, 0.0, 0.0],
        [21.5, 0.1, 0.0714],
        [21.5, 0.2, 0.3571],
        [21.5, 0.3, 0.5],
        [21.5, 0.4, 0.6428],
        [21.5, 0.5, 0.7857],
        [21.5, 0.6, 0.9285],
        [21.5, 0.7, 1.0],
        [22.0, 0.0, 0.0],
        [22.0, 0.1, 0.0666],
        [22.0, 0.2, 0.1333],
        [22.0, 0.3, 0.4666],
        [22.0, 0.4, 0.6],
        [22.0, 0.5, 0.7333],
        [22.0, 0.6, 0.9333],
        [22.0, 0.8, 1.0],
        [22.5, 0.0, 0.0],
        [22.5, 0.1, 0.0370],
        [22.5, 0.2, 0.1111],
        [22.5, 0.3, 0.3703],
        [22.5, 0.4, 0.7037],
        [22.5, 0.5, 0.7777],
        [22.5, 0.6, 0.8148],
        [22.5, 0.7, 0.9259],
        [22.5, 0.8, 1.0],
        [23.0, 0.0, 0.0],
        [23.0, 0.1, 0.0740],
        [23.0, 0.2, 0.2962],
        [23.0, 0.3, 0.4814],
        [23.0, 0.4, 0.7777],
        [23.0, 0.5, 0.8888],
        [23.0, 0.6, 0.9629],
        [23.0, 0.8, 1.0],
        [23.5, 0.0, 0.0166],
        [23.5, 0.1, 0.1833],
        [23.5, 0.2, 0.45],
        [23.5, 0.3, 0.7166],
        [23.5, 0.4, 0.9166],
        [23.5, 0.5, 1.0],
        [24.0, 0.0, 0.0],
        [24.0, 0.1, 0.3125],
        [24.0, 0.2, 0.6],
        [24.0, 0.3, 0.8625],
        [24.0, 0.4, 0.9625],
        [24.0, 0.5, 1.0],
    ]
)
eps = 1e-8
CONT_CDF_HLR_WRT_MAG = numpy.array(
    [
        [20.5, 0.1, 0.0],
        [20.5, 0.2, eps],
        [20.5, 0.3, 0.0909],
        [20.5, 0.4, 0.1818],
        [20.5, 0.5, 0.4545],
        [20.5, 0.6, 0.4545 + eps],
        [20.5, 0.7, 0.4545 + 2 * eps],
        [20.5, 0.8, 0.5454],
        [20.5, 0.9, 0.5454 + eps],
        [20.5, 1.0, 0.6363],
        [20.5, 1.1, 0.8181],
        [20.5, 1.2, 1.0],
        [21.0, 0.1, 0.0],
        [21.0, 0.2, eps],
        [21.0, 0.3, 0.1818],
        [21.0, 0.4, 0.5454],
        [21.0, 0.5, 0.5454 + eps],
        [21.0, 0.6, 0.6363],
        [21.0, 0.7, 0.7272],
        [21.0, 0.8, 0.7272 + eps],
        [21.0, 0.9, 1.0],
        [21.5, 0.1, 0.0],
        [21.5, 0.2, 0.0714],
        [21.5, 0.3, 0.3571],
        [21.5, 0.4, 0.5],
        [21.5, 0.5, 0.6428],
        [21.5, 0.6, 0.7857],
        [21.5, 0.7, 0.9285],
        [21.5, 0.8, 1.0],
        [22.0, 0.1, 0.0],
        [22.0, 0.2, 0.0666],
        [22.0, 0.3, 0.1333],
        [22.0, 0.4, 0.4666],
        [22.0, 0.5, 0.6],
        [22.0, 0.6, 0.7333],
        [22.0, 0.7, 0.9333],
        [22.0, 0.8, 0.9333 + eps],
        [22.0, 0.9, 1.0],
        [22.5, 0.1, 0.0],
        [22.5, 0.2, 0.0370],
        [22.5, 0.3, 0.1111],
        [22.5, 0.4, 0.3703],
        [22.5, 0.5, 0.7037],
        [22.5, 0.6, 0.7777],
        [22.5, 0.7, 0.8148],
        [22.5, 0.8, 0.9259],
        [22.5, 0.9, 1.0],
        [23.0, 0.1, 0.0],
        [23.0, 0.2, 0.0740],
        [23.0, 0.3, 0.2962],
        [23.0, 0.4, 0.4814],
        [23.0, 0.5, 0.7777],
        [23.0, 0.6, 0.8888],
        [23.0, 0.7, 0.9629],
        [23.0, 0.8, 0.9629 + eps],
        [23.0, 0.9, 1.0],
        [23.5, 0.0, 0.0],
        [23.5, 0.1, 0.0166],
        [23.5, 0.2, 0.1833],
        [23.5, 0.3, 0.45],
        [23.5, 0.4, 0.7166],
        [23.5, 0.5, 0.9166],
        [23.5, 0.6, 1.0],
        [24.0, 0.1, 0.0],
        [24.0, 0.2, 0.3125],
        [24.0, 0.3, 0.6],
        [24.0, 0.4, 0.8625],
        [24.0, 0.5, 0.9625],
        [24.0, 0.6, 1.0],
    ]
)
CDF_NS_WRT_MAG = numpy.array(
    [
        [20.5, 0.0, 0.0],
        [20.5, 0.5, 0.2727],
        [20.5, 0.75, 0.5454],
        [20.5, 1.0, 0.63636],
        [20.5, 1.25, 0.7272],
        [20.5, 1.75, 0.81818],
        [20.5, 3.0, 1.0],
        [21.0, 0.0, 0.0],
        [21.0, 0.75, 0.54545],
        [21.0, 1.25, 0.63636],
        [21.0, 1.75, 0.72727],
        [21.0, 4.0, 0.909090],
        [21.0, 4.5, 1.0],
        [21.5, 0.0, 0.0],
        [21.5, 0.25, 0.14285],
        [21.5, 0.5, 0.357142],
        [21.5, 1.0, 0.428571],
        [21.5, 1.25, 0.57142],
        [21.5, 2.0, 0.642857],
        [21.5, 2.25, 0.71428],
        [21.5, 4.0, 0.785714],
        [21.5, 4.5, 0.857142],
        [21.5, 5.0, 0.928571],
        [21.5, 5.25, 1.0],
        [22.0, 0.0, 0.0],
        [22.0, 0.25, 0.13333],
        [22.0, 0.5, 0.266666],
        [22.0, 0.75, 0.46666],
        [22.0, 1.25, 0.53333],
        [22.0, 1.5, 0.599999],
        [22.0, 1.75, 0.66666],
        [22.0, 2.5, 0.799999],
        [22.0, 3.0, 0.866666],
        [22.0, 3.75, 0.93333],
        [22.0, 4.25, 1.0],
        [22.5, 0.0, 0.0],
        [22.5, 0.25, 0.11111],
        [22.5, 0.50, 0.37037],
        [22.5, 0.750, 0.48140],
        [22.5, 1.0, 0.740740],
        [22.5, 1.25, 0.81481],
        [22.5, 1.75, 0.85185],
        [22.5, 2.0, 0.888888],
        [22.5, 2.25, 0.92592],
        [22.5, 2.75, 0.96296],
        [22.5, 4.0, 1.0],
        [23.0, 0.0, 0.0],
        [23.0, 0.25, 0.22222],
        [23.0, 0.50, 0.44444],
        [23.0, 0.75, 0.70370],
        [23.0, 1.00, 0.88888],
        [23.0, 1.25, 0.92592],
        [23.0, 2.00, 0.96296],
        [23.0, 2.50, 1.0],
        [23.5, 0.0, 0.0],
        [23.5, 0.25, 0.1],
        [23.5, 0.50, 0.28333],
        [23.5, 0.75, 0.44999],
        [23.5, 1.00, 0.61666],
        [23.5, 1.25, 0.76666],
        [23.5, 1.50, 0.83333],
        [23.5, 2.25, 0.9],
        [23.5, 2.50, 0.95],
        [23.5, 2.75, 0.96666],
        [23.5, 6.00, 0.98333],
        [23.5, 6.75, 1.0],
        [24.0, 0.0, 0.0],
        [24.0, 0.25, 0.05000],
        [24.0, 0.50, 0.11250],
        [24.0, 0.75, 0.26250],
        [24.0, 1.00, 0.41249],
        [24.0, 1.25, 0.57499],
        [24.0, 1.50, 0.66249],
        [24.0, 1.75, 0.75000],
        [24.0, 2.00, 0.80000],
        [24.0, 2.25, 0.83750],
        [24.0, 2.50, 0.87500],
        [24.0, 2.75, 0.88749],
        [24.0, 3.00, 0.92499],
        [24.0, 3.25, 0.93749],
        [24.0, 3.50, 0.94999],
        [24.0, 3.75, 0.96249],
        [24.0, 5.00, 0.97499],
        [24.0, 7.50, 0.98749],
        [24.0, 8.75, 1.0],
    ]
)
