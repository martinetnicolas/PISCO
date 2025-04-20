import json
import shutil
from pathlib import Path

import tensorflow_datasets as tfds
from classes import PiscoEuclidPsf
from utils import DIR_DATASETS, DIRS, RED, RESET

# USER DATA ---------------------------------------------------------------------------------
VERSION = "1.0.4"
NB_OF_SAMPLES = 50_000
STYLE_PSF = "airy"  # airy / euclid
BIG_PSF = True  # image or patch size psf
NOISY_PSF = False  # raw or noisy psf
GRID = False  # grid of psf by patch
UNDER_FLOAT = False  # DTYPE_32 or DTYPE_16


# CREATE DB ---------------------------------------------------------------------------------
assert not (GRID and not BIG_PSF), f"{RED}GRID=True requires BIG_PSF=True{RESET}"
with open("all_release_notes.json") as f:
    release_notes_psf = json.load(f)["release_notes_psf"]

PiscoEuclidPsf.BUILDER_CONFIGS = [
    tfds.core.BuilderConfig(
        name=str(DIRS["psf"]), version=tfds.core.Version(VERSION), release_notes=release_notes_psf
    )
]

PiscoEuclidPsf(
    nb_of_samples=NB_OF_SAMPLES,
    style_psf=STYLE_PSF,
    big_psf=BIG_PSF,
    noisy_psf=NOISY_PSF,
    under_float=UNDER_FLOAT,
    grid=GRID,
).download_and_prepare()

# allow permissions to pisco group
DIR_VERSION = DIRS["psf"] / VERSION
DIR_VERSION.chmod(0o1770)

# deleting tdfs default folder
if (path := Path.home() / "tensorflow_datasets").exists():
    shutil.rmtree(path)
if (path := DIR_DATASETS / Path(".config") / "metadata.json").exists():
    path.unlink()
