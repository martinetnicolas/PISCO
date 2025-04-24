import json
import shutil
from pathlib import Path

import tensorflow_datasets as tfds
from classes import PiscoEuclidImage
from utils import DIR_DATASETS, DIRS, RED, RESET, extract_keys_fn, print_csv_images_galaxies

# USER DATA ---------------------------------------------------------------------------------
VERSION = "1.1.1"
NB_OF_SAMPLES = 5_000
PERCENTAGE_COSMOS = 0
GAL_TYPE = "real"  # "real" or "parametric"
NB_GALAXIES = 10
ELLIPS_STYLE = "gaussian"  # gaussian / uniform probability for ellipticities
PIXELS_OUT = False  # use a 10 pixels band-free around the image
SHAPE_NOISE_ADD = False
SHAPE_NOISE_SUB = False
LHS_POS = False  # x,y included in LHS
SEED_DATABASE = 22
STYLE_PSF = "airy"  # airy / euclid
STYLE_DB = "pisco_euclid"


# CREATE DB ---------------------------------------------------------------------------------
if __name__ == "__main__":
    with open("all_release_notes.json") as f:
        release_notes_databases = json.load(f)["release_notes_databases"]

    PiscoEuclidImage.BUILDER_CONFIGS = [
        tfds.core.BuilderConfig(
            name=str(DIRS[STYLE_DB]),
            version=tfds.core.Version(VERSION),
            release_notes=release_notes_databases,
        )
    ]

    PiscoEuclidImage(
        nb_of_samples=NB_OF_SAMPLES,
        ellips_style=ELLIPS_STYLE,
        pixel_out=PIXELS_OUT,
        seed_database=SEED_DATABASE,
        shape_noise_add=SHAPE_NOISE_ADD,
        shape_noise_sub=SHAPE_NOISE_SUB,
        style_psf=STYLE_PSF,
        lhs_pos=LHS_POS,
        percentage_cosmos=PERCENTAGE_COSMOS,
        gal_type=GAL_TYPE,
        nb_galaxies=NB_GALAXIES,
    ).download_and_prepare()

    # allow permissions to pisco group
    DIR_VERSION = DIRS[STYLE_DB] / VERSION
    DIR_VERSION.chmod(0o1770)

    # deleting tdfs default folder
    if (path := Path.home() / "tensorflow_datasets").exists():
        shutil.rmtree(path)
    if (path := DIR_DATASETS / Path(".config") / "metadata.json").exists():
        path.unlink()

    # CREATE CSV ---------------------------------------------------------------------------------
    SPLIT = "dataset"
    dataset = tfds.load(f"{STYLE_DB}:{VERSION}", split=f"{SPLIT}[:100%]", data_dir=DIR_DATASETS)
    keys_dataset = ["avg_g", "avg_int_e", "avg_obs_e", "keys", "gal_params"]
    all_params_dataset = dataset.map(extract_keys_fn(keys_dataset))

    print(f"{RED}CREATE CSV...{RESET}")
    # fmt: off
    columns_img = ["avg_g1", "avg_g2", "avg_int_e1", "avg_int_e2", "avg_obs_e1", "avg_obs_e2", "seeds", "hashed_key", "k"] # noqa
    columns_gal = ["int_e1", "int_e2", "obs_e1", "obs_e2", "x", "y", "mag", "hlr", "sersic"]
    all_params = list(all_params_dataset.as_numpy_iterator())
    print_csv_images_galaxies(keys_dataset, all_params, DIR_VERSION, columns_img, columns_gal)

    print(f"{RED}FINITO{RESET}")
