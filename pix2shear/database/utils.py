import multiprocessing
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy
import pandas
from astropy.io import fits
from joblib import Parallel, delayed, parallel_backend
from scipy import interpolate, stats


def _find_directory(name, init_directory):
    directory_path = Path(init_directory)
    matching_folders = list(directory_path.glob(f"**/{name}"))
    return matching_folders[0]


def search_folder(name):
    if Path.cwd().parts[1] == "home":
        search_dir = "/home"
    else:
        search_dir = "/net/GECO/pisco/PIX2SHEAR"
    return _find_directory(name, search_dir)


def _create_new_dir(dir_init, name):
    new_db = dir_init / name
    new_db.mkdir(parents=False, exist_ok=False)
    return new_db


def erase_test(name, folder):
    try:
        output_folder = _create_new_dir(dir_init=folder, name=name)
    except FileExistsError:
        print(f"!!! {folder / name} already exists !!!")
        response = input("REMOVE AND RECREATE ? (Y/n):")
        if response.lower() == "y" or response == "":
            shutil.rmtree(folder / name)
            output_folder = _create_new_dir(dir_init=folder, name=name)
        else:
            sys.exit()
    return Path(output_folder)


def write_fits(array, path):
    hdu = fits.PrimaryHDU(array)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(path, overwrite=True)
    hdulist.close()


def _load_metrics(arg):
    return numpy.mean(arg), numpy.amax(arg)


def _parallel_with_one_arg(func, data):
    nb_cpus = multiprocessing.cpu_count()
    print(f"using {nb_cpus} cpus")
    with parallel_backend("threading", n_jobs=nb_cpus):
        return numpy.array(
            Parallel(prefer="threads", verbose=0)(delayed(func)(arg=arg) for arg in data)
        )


def getting_norms(data):
    results = _parallel_with_one_arg(func=_load_metrics, data=data)

    measures_norms = defaultdict(list)
    for result in results:
        measures_norms["mean"].append(result[0])
        measures_norms["amax"].append(result[1])

    return pandas.DataFrame(measures_norms)


def draw_lhs(n_dimensions, seed, n_samples):
    lhs_sampler = stats.qmc.LatinHypercube(
        d=n_dimensions,
        scramble=False,
        optimization=None,  # optimization="random-cd"
        strength=1,
        seed=seed,
    )
    return lhs_sampler.random(n=n_samples)


def checky(samples):
    squares_sum = numpy.sqrt(numpy.square(samples[:, 0]) + numpy.square(samples[:, 1]))
    masque = squares_sum > 0.7
    return masque


def position(x, new_min, new_max):
    x *= 352
    return new_min + x * (new_max - new_min) / 352


def new_sample(number, sigma=0.26):
    adding_lhs_sampler = stats.qmc.LatinHypercube(d=2, scramble=False, strength=1)
    adding_lhs_samples = adding_lhs_sampler.random(n=number)
    adding_lhs_samples[:, 0] = stats.norm.ppf(adding_lhs_samples[:, 0], loc=0, scale=sigma)
    adding_lhs_samples[:, 1] = stats.norm.ppf(adding_lhs_samples[:, 1], loc=0, scale=sigma)

    return adding_lhs_samples


def divide_per_gal_for_multiprocess(nb_of_gal, lhs_samples_loop):
    nb_of_gal_to_divide = nb_of_gal.astype(int).tolist()
    start_idx = 0
    list_samples_loop = []
    for num in nb_of_gal_to_divide:
        list_samples_loop.append(lhs_samples_loop[start_idx : start_idx + num, :])
        start_idx += num
    return list_samples_loop


def get_inv_cdfs(cdf_mag, cdf_ns, cdf_hlr):
    cases = numpy.unique(cdf_mag[:, 0])

    # mag
    magnitudes = cdf_mag[:, 0]
    cdf_values = cdf_mag[:, 1]
    inv_cdf_mag = interpolate.interp1d(
        cdf_values, magnitudes, bounds_error=False, fill_value="extrapolate"
    )

    # ns
    inv_cdfs_ns = {}
    for cas in cases:
        case_data = cdf_ns[cdf_ns[:, 0] == cas]
        x_values = case_data[:, 1]
        cdf_values = case_data[:, 2]
        inv_cdfs_ns[cas] = interpolate.interp1d(
            cdf_values, x_values, bounds_error=False, fill_value="extrapolate"
        )

    # hlr
    inv_cdfs_hlr = {}
    for cas in cases:
        case_data = cdf_hlr[cdf_hlr[:, 0] == cas]
        x_values = case_data[:, 1]
        cdf_values = case_data[:, 2]
        inv_cdfs_hlr[cas] = interpolate.interp1d(
            cdf_values, x_values, bounds_error=False, fill_value="extrapolate"
        )

    return (inv_cdf_mag, inv_cdfs_ns, inv_cdfs_hlr, cases)


cdf_mag = numpy.array(
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
cdf_ns_wrt_mag = numpy.array(
    [
        [20.5, 0, 0],
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
        [23.5, 0.00, 0.0],
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
        [24.0, 0.00, 0.0],
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
cdf_hlr_wrt_mag = numpy.array(
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


def save_jax_array_to_fits(jax_array, folder, prefix="simu_"):
    np_array = numpy.array(jax_array.block_until_ready())

    for i in range(np_array.shape[0]):
        if (i + 1) % 1000 == 0:
            print(i + 1)
        filename = f"{folder}/{prefix}{i + 1:05d}.fits"
        fits.writeto(filename, np_array[i], overwrite=True)
