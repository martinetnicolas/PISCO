from functools import partial
from multiprocessing import Pool, cpu_count

import galsim
import numpy
import tensorflow_datasets as tfds
from functions_images import create_images, create_psf
from functions_lhs import (
    adding_cosmos_catalog_galaxies,
    divide_for_multiprocess,
    draw_and_project_lhs_img,
    draw_lhs_gal,
    noise_cancel,
    project_lhs_gal,
)
from tqdm import tqdm
from utils import DIRS, DTYPE_16, DTYPE_32

# --------------------------------------------------------------------------------------------------


# create euclid like pisco database
class PiscoEuclidImage(tfds.core.GeneratorBasedBuilder):
    def __init__(
        self,
        nb_of_samples,
        ellips_style,
        pixel_out,
        seed_database,
        noise_cancel,
        style_psf,
        lhs_pos,
        percentage_cosmos,
        gal_type,
        nb_galaxies_per_image,
    ):
        self.nb_galaxies_per_image = nb_galaxies_per_image
        self.nb_of_samples = nb_of_samples
        self.nb_of_galaxies = self.nb_of_samples * self.nb_galaxies_per_image
        self.seed_database = seed_database
        self.percentage_cosmos = percentage_cosmos
        self.gal_type = gal_type
        self.noise_cancel = noise_cancel
        self.style_psf = style_psf
        self.ellips_style = ellips_style
        self.pixel_out = pixel_out
        self.lhs_pos = lhs_pos

        self.images_size = (352, 352)
        self.patch_size = (64, 64)
        self.nb_galaxies_parameters = 9

        self.total_nb_of_samples = (
            self.nb_of_samples * 4 if self.noise_cancel else self.nb_of_samples
        )
        self.total_nb_of_galaxies = (
            self.nb_of_galaxies * 4 if self.noise_cancel else self.nb_of_galaxies
        )

        super().__init__()

    def _info(self) -> tfds.core.DatasetInfo:
        features = {
            "image": tfds.features.Tensor(shape=self.images_size, dtype=DTYPE_16),
            "avg_g": tfds.features.Tensor(shape=(2,), dtype=DTYPE_32),
            "avg_obs_e": tfds.features.Tensor(shape=(2,), dtype=DTYPE_32),
            "avg_int_e": tfds.features.Tensor(shape=(2,), dtype=DTYPE_32),
            "keys": tfds.features.Tensor(shape=(3,), dtype=DTYPE_32),
            "gal_params": tfds.features.Tensor(
                shape=(self.nb_galaxies_parameters, self.nb_galaxies_per_image), dtype=DTYPE_32
            ),
        }

        metadata = {
            "nb_of_samples": self.nb_of_samples,
            "nb_galaxies_per_image": self.nb_galaxies_per_image,
            "nb_of_galaxies": self.nb_of_galaxies,
            "noise_cancel": self.noise_cancel,
            "total_nb_of_samples": self.total_nb_of_samples,
            "total_nb_of_galaxies": self.total_nb_of_galaxies,
            "nb_galaxies_parameters": self.nb_galaxies_parameters,
            "style_psf": self.style_psf,
            "ellips_style": self.ellips_style,
            "pixel_out": self.pixel_out,
            "lhs_pos": self.lhs_pos,
            "images_size": self.images_size,
            "patch_size": self.patch_size,
            "seed_database": self.seed_database,
        }

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(features), metadata=tfds.core.MetadataDict(metadata)
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {"dataset": self._generate_examples()}

    def _generate_examples(self):
        # create lhs image
        lhs_img = draw_and_project_lhs_img(
            seed=self.seed_database * 1_000, n_samples=self.nb_of_samples
        )

        # create lhs galaxies
        lhs_gal = draw_lhs_gal(self.lhs_pos, self.seed_database, self.nb_of_galaxies)

        # lhs galaxies projection
        lhs_gal = project_lhs_gal(
            hypercube=lhs_gal, ellips_style=self.ellips_style, pixel_out=self.pixel_out
        )

        # taking real/parametric galaxies from cosmos catalogs
        if self.percentage_cosmos:
            nb_galaxies_cosmos = int(self.percentage_cosmos * self.nb_of_galaxies / 100)
            rng_local = numpy.random.default_rng(seed=self.seed_database)
            indices = rng_local.choice(self.nb_of_galaxies, nb_galaxies_cosmos, replace=False)
            lhs_gal = adding_cosmos_catalog_galaxies(lhs_gal, numpy.sort(indices), self.version)

        # adding shape_noise_cancel
        if self.noise_cancel:
            lhs_img, lhs_gal = noise_cancel(lhs_img, lhs_gal)

        # prepocess for multiprocess
        lhs_gal = divide_for_multiprocess(
            lhs_gal, self.total_nb_of_samples, self.nb_galaxies_per_image
        )

        # combined for multiprocess
        combined_lhs = list(zip(lhs_img, lhs_gal))

        # use_real of not (save RAM)
        if self.gal_type == "parametric" or self.percentage_cosmos == 0:
            use_real = False
        elif self.gal_type == "real":
            use_real = True

        # pre loaded cosmos catalogs
        EXPTIME = 4 * 566
        cosmos_cat_23 = galsim.COSMOSCatalog(
            sample="23.5",
            dir=DIRS["cosmos"],
            exclusion_level="none",
            exptime=EXPTIME,
            preload=False,
            use_real=use_real,
        )
        cosmos_cat_25 = galsim.COSMOSCatalog(
            sample="25.2",
            dir=DIRS["cosmos"],
            exclusion_level="none",
            exptime=EXPTIME,
            preload=False,
            use_real=use_real,
        )

        # fix function parameters
        create_images_partial = partial(
            create_images,
            style_psf=self.style_psf,
            gal_type=self.gal_type,
            cosmos_cat_23=cosmos_cat_23,
            cosmos_cat_25=cosmos_cat_25,
        )

        # compute images
        with Pool(processes=cpu_count()) as pool:
            for trial, result in tqdm(
                pool.imap(create_images_partial, enumerate(combined_lhs), chunksize=8),
                total=self.total_nb_of_samples,
            ):
                yield int(trial), result


# --------------------------------------------------------------------------------------------------


# create PSF database
class PiscoEuclidPsf(tfds.core.GeneratorBasedBuilder):
    def __init__(self, nb_of_samples, style_psf, big_psf, noisy_psf, under_float, grid):
        self.nb_of_samples = nb_of_samples
        self.style_psf = style_psf
        self.big_psf = big_psf
        self.noisy_psf = noisy_psf
        self.under_float = under_float
        self.grid = grid

        super().__init__()

    def _info(self) -> tfds.core.DatasetInfo:
        features = {
            "image_psf": tfds.features.Tensor(
                shape=(352, 352) if self.big_psf else (64, 64),
                dtype=DTYPE_16 if self.under_float else DTYPE_32,
            )
        }

        metadata = {
            "nb_of_samples": self.nb_of_samples,
            "style_psf": self.style_psf,
            "under_float": self.under_float,
            "noisy_psf": self.noisy_psf,
            "big_psf": self.big_psf,
        }

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(features), metadata=tfds.core.MetadataDict(metadata)
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {"dataset": self._generate_examples()}

    def _generate_examples(self):
        create_psf_partial = partial(
            create_psf,
            under_float=self.under_float,
            noisy_psf=self.noisy_psf,
            style_psf=self.style_psf,
            big_psf=self.big_psf,
            grid=self.grid,
        )

        with Pool(processes=cpu_count()) as pool:
            for trial, result in tqdm(
                pool.imap(create_psf_partial, range(self.nb_of_samples), chunksize=8),
                total=self.nb_of_samples,
            ):
                yield int(trial), result
