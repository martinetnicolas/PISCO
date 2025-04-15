import shutil
from pathlib import Path

import tensorflow_datasets as tfds
from main import PiscoEuclid
from utils import DIR_DATASETS, DIRS, extract_keys_fn, print_csv_images_galaxies

# USER DATA ---------------------------------------------------------------------------------
VERSION = "1.1.1"
NB_OF_SAMPLES = 1000
PERCENTAGE_COSMOS = 0
TYPE_COSMOS = "real"  # "real" or "parametric"
ELLIPS_STYLE = "gaussian"  # gaussian / uniform probability for ellipticities
PIXELS_OUT = False  # use a 10 pixels band-free around the image
SHAPE_NOISE_ADD = False
SHAPE_NOISE_SUB = False
LHS_POS = False  # x,y included in LHS
SEED_DATABASE = 22
PSF = "airy"  # airy / euclid
BIG_PSF = True  # image or patch size
STYLE = "pisco_euclid"
release_notes = {
    "2.*.*": "lhs + g round(3) + 10 gal/patch + g<=|0.06| + e gauss + seeds 22 + airy",
    "2.0.1": "1_000",
    "2.0.2": "2_000 (1000 x 2 : with shape noise cancellation with add noise)",
    "2.0.3": "4_000 (1000 x 4 : with shape noise cancellation with add noise + sub noise)",
    "2.0.4": "1_000 with lhs_pos",
    "2.0.5": "1_000 with pixel out",
    "2.1.1": "25_000",
    "2.1.2": "50_000 (25_000 x 2 : with shape noise cancellation with add noise)",
    "2.1.3": "100_000 (25_000 x 4 : with shape noise cancellation with add noise + sub noise)",
    "2.1.4": "25_000 with lhs_pos",
    "2.1.5": "25_000 with pixel out",
    "2.2.0": "100_000",
}


# if __name__ == "__main__":
# CREATE DB ---------------------------------------------------------------------------------
PiscoEuclid.BUILDER_CONFIGS = [
    tfds.core.BuilderConfig(
        name=str(DIRS[STYLE]), version=tfds.core.Version(VERSION), release_notes=release_notes
    )
]
PiscoEuclid(
    nb_of_samples=NB_OF_SAMPLES,
    ellips_style=ELLIPS_STYLE,
    pixel_out=PIXELS_OUT,
    seed_database=SEED_DATABASE,
    shape_noise_add=SHAPE_NOISE_ADD,
    shape_noise_sub=SHAPE_NOISE_SUB,
    style_psf=PSF,
    lhs_pos=LHS_POS,
    percentage_cosmos=PERCENTAGE_COSMOS,
    type_cosmos=TYPE_COSMOS,
    big_psf=BIG_PSF,
).download_and_prepare()

# allow permissions to pisco group
DIR_VERSION = DIRS[STYLE] / VERSION
DIR_VERSION.chmod(0o1770)

# deleting tdfs default folder
shutil.rmtree(Path.home() / "tensorflow_datasets")

# CREATE CSV ---------------------------------------------------------------------------------
SPLIT = "dataset"
dataset = tfds.load(f"{STYLE}:{VERSION}", split=f"{SPLIT}[:100%]", data_dir=DIR_DATASETS)
print("number of samples", TOTAL_ELEMENTS := dataset.cardinality().numpy())

keys_dataset = ["avg_g", "avg_int_e", "avg_obs_e", "keys", "gal_params"]
all_params_dataset = dataset.map(extract_keys_fn(keys_dataset))

print("csv creation")
# fmt: off
columns_img = ["avg_g1", "avg_g2", "avg_int_e1", "avg_int_e2", "avg_obs_e1", "avg_obs_e2", "seeds", "hashed_key", "k"] # noqa
columns_gal = ["int_e1", "int_e2", "obs_e1", "obs_e2", "x", "y", "mag", "hlr", "sersic"]
all_params = list(all_params_dataset.as_numpy_iterator())
print_csv_images_galaxies(keys_dataset, all_params, DIR_VERSION, columns_img, columns_gal)
