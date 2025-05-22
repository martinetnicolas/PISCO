import json
import shutil
from pathlib import Path

import tensorflow_datasets as tfds
from classes import PiscoEuclidImage
from pandas import DataFrame, concat, read_csv
from tqdm import tqdm
from utils import DIR_DATASETS, DIRS, RED, RESET, write_fits

# USER DATA ---------------------------------------------------------------------------------
VERSION = "0.0.8"
NB_OF_SAMPLES = 10_000
SE_VERSION = True
SAMPLING = 1
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
SPLIT = "dataset"
DIR_VERSION = DIRS["pisco_euclid"] / VERSION
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
        shape_noise_add=SHAPE_NOISE_ADD,
        shape_noise_sub=SHAPE_NOISE_SUB,
        style_psf=STYLE_PSF,
        lhs_pos=LHS_POS,
        percentage_cosmos=PERCENTAGE_COSMOS,
        gal_type=GAL_TYPE,
        nb_galaxies=NB_GALAXIES,
    ).download_and_prepare()

    # allow permissions to pisco group
    DIR_SE.chmod(0o1770)
    DIR_VERSION.chmod(0o1770)

    # deleting tdfs default folder
    if (path := Path.home() / "tensorflow_datasets").exists():
        shutil.rmtree(path)
    if (path := DIR_DATASETS / Path(".config") / "metadata.json").exists():
        path.unlink()

    # CREATE CSV & SE VERSION-----------------------------------------------------------------------
    print(f"{RED}CREATE CSV...{RESET}")

    # load dataset
    dataset = tfds.load(
        f"{STYLE_DB}:{VERSION}", split=f"{SPLIT}[:100%]", data_dir=DIR_DATASETS, shuffle_files=False
    )
    NB_TOTAL = dataset.cardinality().numpy()
    keys_dataset = ["avg_g", "avg_obs_e", "avg_int_e", "keys", "gal_params", "image"]
    all_params_dataset = dataset.map(lambda x: {key: x[key] for key in keys_dataset})

    # loop for csv & se
    all_img_params, all_gal_params = [], []
    for global_index, example in tqdm(enumerate(tfds.as_numpy(all_params_dataset)), total=NB_TOTAL):
        gal_params = example["gal_params"]
        keys = example["keys"]
        avg_g = example["avg_g"]
        avg_obs_e = example["avg_obs_e"]
        avg_int_e = example["avg_int_e"]
        image = example["image"]

        df = DataFrame(
            gal_params.T,
            columns=["int_e1", "int_e2", "obs_e1", "obs_e2", "x", "y", "mag", "hlr", "sersic"],
        )
        path_csv = DIR_SE / f"{global_index:06d}_true_galaxies_params.csv"
        df["image"] = global_index
        all_gal_params.append(df)

        # accumulate avg_g and avg_obs_e for global CSV
        all_img_params.append(
            {
                "index": global_index,
                "avg_g1": avg_g[0],
                "avg_g2": avg_g[1],
                "avg_obs_e1": avg_obs_e[0],
                "avg_obs_e2": avg_obs_e[1],
                "avg_int_e1": avg_int_e[0],
                "avg_int_e2": avg_int_e[1],
                "seeds": keys[0],
                "hashed_key": keys[1],
                "k": keys[2],
            }
        )

        # If SE version needed
        if SE_VERSION:
            # save galaxy parameters per image
            df.to_csv(path_or_buf=path_csv, index=False)

            # save image as FITS
            path_fits = DIR_SE / f"{global_index:06d}_simu.fits"
            write_fits(path=path_fits, data=image, sampling=SAMPLING)

            # save positions (x, y) only for SE assoc
            path_assoc = DIR_SE / f"{global_index:06d}_simu_assoc.txt"
            df[["x", "y"]].to_csv(path_or_buf=path_assoc, sep=" ", index=False, header=False)

    # Write CSVs
    DataFrame(all_img_params).to_csv(DIR_VERSION / "images_parameters_true.csv", index=False)
    df_all_galaxies = concat(all_gal_params, ignore_index=True)
    df_all_galaxies.to_csv(DIR_VERSION / "galaxies_parameters_true.csv", index=False)

    # Check
    NUMBER = 10
    df_img = read_csv(DIR_VERSION / "images_parameters_true.csv")
    df_gal_se = read_csv(DIR_SE / f"{NUMBER:06d}_true_galaxies_params.csv")
    df_gal = read_csv(DIR_VERSION / "galaxies_parameters_true.csv")
    print(df_img.iloc[NUMBER][["avg_obs_e1", "avg_obs_e2"]])
    print(df_gal_se[["obs_e1", "obs_e2"]].mean())
    print(df_gal[df_gal["image"] == NUMBER][["obs_e1", "obs_e2"]].mean())

    print(f"{RED}FINITO{RESET}")
