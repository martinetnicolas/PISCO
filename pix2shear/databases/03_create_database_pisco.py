import json
import shutil
from pathlib import Path

import tensorflow_datasets as tfds
from classes import PiscoEuclidImage
from tqdm import tqdm
from utils import DIR_DATASETS, DIRS, RED, RESET, create_se, write_values_for_se

# USER DATA ---------------------------------------------------------------------------------
VERSION = "0.0.0"
NB_OF_SAMPLES = 100
SE_VERSION = True
SAMPLING = 1
PERCENTAGE_COSMOS = 100
GAL_TYPE = "real"  # "real" or "parametric"
NB_GALAXIES_PER_IMAGE = 10
ELLIPS_STYLE = "gaussian"  # gaussian / uniform probability for ellipticities
PIXELS_OUT = False  # use a 10 pixels band-free around the image
NOISE_CANCEL = True  # x4 number of images
LHS_POS = False  # x,y included in LHS
SEED_DATABASE = 2222
STYLE_PSF = "airy"  # airy / euclid
STYLE_DB = "pisco_euclid"
SPLIT = "dataset"
DIR_VERSION = DIRS["pisco_euclid"] / VERSION
if SE_VERSION:
    DIR_SE = DIRS["pisco_euclid_se"] / VERSION
    DIR_SE.mkdir(parents=True, exist_ok=True)


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
        noise_cancel=NOISE_CANCEL,
        style_psf=STYLE_PSF,
        lhs_pos=LHS_POS,
        percentage_cosmos=PERCENTAGE_COSMOS,
        gal_type=GAL_TYPE,
        nb_galaxies_per_image=NB_GALAXIES_PER_IMAGE,
    ).download_and_prepare()

    # allow permissions to pisco group
    if SE_VERSION:
        DIR_SE.chmod(0o1770)
    DIR_VERSION.chmod(0o1770)

    # deleting tdfs default folder
    if (path := Path.home() / "tensorflow_datasets").exists():
        shutil.rmtree(path)
    if (path := DIR_DATASETS / Path(".config") / "metadata.json").exists():
        path.unlink()

    # If SE version needed
    if SE_VERSION:
        print(f"{RED}SE++{RESET}")

        # load dataset
        dataset = tfds.load(
            f"{STYLE_DB}:{VERSION}",
            split=f"{SPLIT}[:100%]",
            data_dir=DIR_DATASETS,
            shuffle_files=False,
        )
        NB_TOTAL = dataset.cardinality().numpy()
        keys_dataset = ["keys", "gal_params", "image"]
        all_params_dataset = dataset.map(lambda x: {key: x[key] for key in keys_dataset})

        for num, example in tqdm(enumerate(tfds.as_numpy(all_params_dataset)), total=NB_TOTAL):
            df_gal = create_se(example, num)
            write_values_for_se(DIR_SE, SAMPLING, num, example["image"], df_gal)

    print(f"{RED}FINITO{RESET}")
