import shutil
from multiprocessing import Pool, cpu_count
from pathlib import Path
from time import time

import numpy
from image_galsim import TransformTfds, create_images
from scipy import stats
from tqdm import tqdm
from utils import (
    cdf_hlr_wrt_mag,
    cdf_mag,
    cdf_ns_wrt_mag,
    checky,
    divide_per_gal_for_multiprocess,
    draw_lhs,
    erase_test,
    get_inv_cdfs,
    new_sample,
    position,
    search_folder,
)

# USER DATA ----------------------------------------------------------------------------------------
NB_OF_SAMPLES = 40_000
ELLIPS_UNIFORM = False  # gaussian of uniform probability for ellipticities
PIXELS_OUT = True  # use a 10 pixels band-free around the image
SEED_G = 22
SEED_GALAXY = 2222
VERSION = "1.1.1"  # name of database created

MAX_SHEARS = 0.065  # bonds
DECIMALS = 2  # number of decimals for SHEAR
NB_GALAXIES = 10  # per image
NAME_DB = f"samples{NB_OF_SAMPLES}_seed-hc{SEED_G}_seed-gal{SEED_GALAXY}_VERS{VERSION}"
FOLDER_DATASETS = search_folder("DATASETS")
FOLDER_RESULTS = erase_test(NAME_DB, FOLDER_DATASETS)
print(FOLDER_RESULTS)

# inverse CDFs for physical mg/hlr/sersic
inv_cdf_mag, inv_cdfs_ns, inv_cdfs_hlr, cases = get_inv_cdfs(
    cdf_mag, cdf_ns_wrt_mag, cdf_hlr_wrt_mag
)


# DRAW LHS -----------------------------------------------------------------------------------------
print("first LHS... ", end="")  # image properties
lhs_samples = draw_lhs(2, seed=SEED_G, n_samples=NB_OF_SAMPLES)
lhs_samples[:, :2] = numpy.round((lhs_samples[:, :2] - 0.5) / 0.5 * MAX_SHEARS, decimals=DECIMALS)

print("second LHS... ", end="")  # galaxies properties
lhs_samples_gal = draw_lhs(7, seed=SEED_GALAXY, n_samples=NB_OF_SAMPLES * NB_GALAXIES)

print("positions galaxies LHS... ", end="")
position_x = draw_lhs(1, seed=SEED_GALAXY * 1000, n_samples=NB_OF_SAMPLES * NB_GALAXIES)
position_y = draw_lhs(1, seed=SEED_GALAXY * 100000000, n_samples=NB_OF_SAMPLES * NB_GALAXIES)


# PROJECTIONS --------------------------------------------------------------------------------------
# galaxies positions
if PIXELS_OUT:
    new_min, new_max = 10, 342
else:
    new_min, new_max = 0, 352
position_x = position(position_x, new_min=new_min, new_max=new_max)
position_y = position(position_y, new_min=new_min, new_max=new_max)

# ellipticities
if ELLIPS_UNIFORM:
    # uniform
    max_radius = 0.6
    radii = numpy.sqrt(lhs_samples_gal[:, 0] * max_radius**2)
    angles = lhs_samples_gal[:, 1] * 2 * numpy.pi
    lhs_samples_gal[:, 0] = radii * numpy.cos(angles)
    lhs_samples_gal[:, 1] = radii * numpy.sin(angles)
else:
    # gaussian
    lhs_samples_gal[:, :2] = stats.norm.ppf(lhs_samples_gal[:, :2], loc=0, scale=0.26)
    mask = checky(lhs_samples_gal)
    number_of_outsider = numpy.sum(mask)
    while number_of_outsider != 0:
        print("number_of_outsider", number_of_outsider)
        new_samples = new_sample(number_of_outsider)
        lhs_samples_gal[mask, :2] = new_samples
        mask = checky(lhs_samples_gal)
        number_of_outsider = numpy.sum(mask)

# offsets : lhs_samples_gal[:, 2:4]

# mag
lhs_samples_gal[:, 4] = inv_cdf_mag(lhs_samples_gal[:, 4])
closest_half = numpy.around(lhs_samples_gal[:, 4] * 2) / 2  # closest half categories
closest_half[closest_half == 20] = 20.5

# hlr
for i, cas in enumerate(closest_half):
    value_lhs = lhs_samples_gal[:, 5][i]
    lhs_samples_gal[:, 5][i] = inv_cdfs_hlr[float(cas)](value_lhs)
min_value = abs(lhs_samples_gal[:, 5].min())
lhs_samples_gal[:, 5] = lhs_samples_gal[:, 5] + 2 * min_value + 0.05

# ns
for i, cas in enumerate(closest_half):
    value_lhs = lhs_samples_gal[:, 6][i]
    lhs_samples_gal[:, 6][i] = inv_cdfs_ns[float(cas)](value_lhs)
ns_possible_values = numpy.unique(cdf_ns_wrt_mag[:, 1])
lhs_samples_gal[:, 6] = ns_possible_values[
    numpy.abs(lhs_samples_gal[:, 6][:, None] - ns_possible_values).argmin(axis=1)
]
lhs_samples_gal[:, 6] = numpy.clip(lhs_samples_gal[:, 6], 0.3, 6.2)


# COMPUTING ----------------------------------------------------------------------------------------
# init
lhs_samples_gal = numpy.column_stack((lhs_samples_gal, position_x, position_y))
folder = [FOLDER_RESULTS] * NB_OF_SAMPLES
list_nb_gal = [NB_GALAXIES] * NB_OF_SAMPLES
lhs_samples_gal_div = divide_per_gal_for_multiprocess(
    numpy.array(list_nb_gal), lhs_samples_gal
)  # case nb galaxies not equal to 10
combined_args = list(zip(lhs_samples, lhs_samples_gal_div, folder))
timei = time()
print(f"computing with: {(number_of_cpu_used := cpu_count())} cpus")

# run
with Pool(processes=cpu_count()) as pool:
    _ = list(tqdm(pool.imap(create_images, enumerate(combined_args)), total=NB_OF_SAMPLES))
print(f"{NB_OF_SAMPLES} simulations run in {round((time() - timei) / 60, 1)} minutes")


# TFDS STYLE ---------------------------------------------------------------------------------------
DIR_OUT = FOLDER_DATASETS / "pisco_euclid" / VERSION  # name of database
builder = TransformTfds(
    dataset_dir=FOLDER_RESULTS, version=VERSION
)  # transform in tensorflow generator
builder._data_dir = DIR_OUT
builder.download_and_prepare()
print(builder.info)


# USELESS FOLDERS ----------------------------------------------------------------------------------
print("deleting original folders...")
useless_folder = Path("/home/rbrunet/tensorflow_datasets")
if useless_folder.exists():
    shutil.rmtree(useless_folder)
if FOLDER_RESULTS.exists():
    shutil.rmtree(FOLDER_RESULTS)
print("FINITO")
