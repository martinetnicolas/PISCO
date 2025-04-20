from functools import partial
from multiprocessing import Pool, cpu_count

import numpy
import tensorflow_datasets as tfds
from functions_images import create_images, create_psf
from functions_lhs import (
    adding_cosmos_catalog_galaxies,
    draw_and_project_lhs_img,
    draw_lhs_gal,
    project_lhs_gal,
)
from tqdm import tqdm
from utils import BLUE, DTYPE_16, DTYPE_32, RESET

# --------------------------------------------------------------------------------------------------


# create euclid like pisco database
class PiscoEuclidImage(tfds.core.GeneratorBasedBuilder):
    def __init__(
        self,
        nb_of_samples,
        ellips_style,
        pixel_out,
        seed_database,
        shape_noise_add,
        shape_noise_sub,
        style_psf,
        lhs_pos,
        percentage_cosmos,
        gal_type,
        nb_galaxies,
    ):
        self.nb_galaxies = nb_galaxies
        self.nb_of_samples = nb_of_samples
        self.seed_database = seed_database
        self.percentage_cosmos = percentage_cosmos
        self.gal_type = gal_type

        self.shape_noise_add = shape_noise_add
        self.shape_noise_sub = shape_noise_sub
        self.style_psf = style_psf
        self.ellips_style = ellips_style
        self.pixel_out = pixel_out
        self.lhs_pos = lhs_pos

        self.images_size = (352, 352)
        self.patch_size = (64, 64)
        self.nb_galaxies_parameters = 9

        self.slice_shape_noise = (self.shape_noise_add + 1) * (self.shape_noise_sub + 1)
        self.total_nb_of_samples = self.slice_shape_noise * self.nb_of_samples
        self.nb_total_galaxies = self.nb_of_samples * self.nb_galaxies

        super().__init__()

    def _info(self) -> tfds.core.DatasetInfo:
        features = {
            "image": tfds.features.Tensor(shape=self.images_size, dtype=DTYPE_16),
            "avg_g": tfds.features.Tensor(shape=(2,), dtype=DTYPE_32),
            "avg_obs_e": tfds.features.Tensor(shape=(2,), dtype=DTYPE_32),
            "avg_int_e": tfds.features.Tensor(shape=(2,), dtype=DTYPE_32),
            "keys": tfds.features.Tensor(shape=(3,), dtype=DTYPE_32),
            "gal_params": tfds.features.Tensor(
                shape=(self.nb_galaxies_parameters, self.nb_galaxies), dtype=DTYPE_32
            ),
        }

        metadata = {
            "total_nb_of_samples": self.total_nb_of_samples,
            "nb_of_samples": self.nb_of_samples,
            "slice_shape_noise": self.slice_shape_noise,
            "shape_noise_add": self.shape_noise_add,
            "shape_noise_sub": self.shape_noise_sub,
            "nb_galaxies_parameters": self.nb_galaxies_parameters,
            "nb_total_galaxies": self.nb_total_galaxies,
            "style_psf": self.style_psf,
            "ellips_style": self.ellips_style,
            "pixel_out": self.pixel_out,
            "lhs_pos": self.lhs_pos,
            "images_size": self.images_size,
            "patch_size": self.patch_size,
            "nb_galaxies": self.nb_galaxies,
            "seed_database": self.seed_database,
        }

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(features), metadata=tfds.core.MetadataDict(metadata)
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {"dataset": self._generate_examples()}

    def divide_for_multiprocess(self, hc, nb_of_samples):
        # TODO different number of galaxies per image
        list_nb_gal = [self.nb_galaxies] * nb_of_samples  # case nb of galaxies constant per image

        nb_of_gal = numpy.array(list_nb_gal)
        nb_of_gal_to_divide = nb_of_gal.astype(int).tolist()

        start_idx = 0
        list_samples_loop = []
        for num in nb_of_gal_to_divide:
            list_samples_loop.append(hc[start_idx : start_idx + num, :])
            start_idx += num

        return list_samples_loop

    def _generate_examples(self):
        # create lhs image
        lhs_img = draw_and_project_lhs_img(
            seed=self.seed_database * 1_000, n_samples=self.nb_of_samples
        )

        # create lhs galaxies
        lhs_gal = draw_lhs_gal(self.lhs_pos, self.seed_database, self.nb_total_galaxies)

        # lhs galaxies projection
        lhs_gal = project_lhs_gal(
            hypercube=lhs_gal, ellips_style=self.ellips_style, pixel_out=self.pixel_out
        )

        # taking real/parametric galaxies from cosmos catalogs
        if self.percentage_cosmos:
            nb_galaxies_cosmos = int(self.percentage_cosmos * self.nb_total_galaxies / 100)
            rng_local = numpy.random.default_rng(seed=self.seed_database)
            indices = rng_local.choice(self.nb_total_galaxies, nb_galaxies_cosmos, replace=False)
            lhs_gal = adding_cosmos_catalog_galaxies(lhs_gal, numpy.sort(indices), self.version)

        # prepocess for multiprocess
        lhs_gal = self.divide_for_multiprocess(lhs_gal, self.nb_of_samples)

        # combined for multiprocess
        combined_lhs = list(zip(lhs_img, lhs_gal))

        # compute image
        for k in range(self.slice_shape_noise):
            # fix function parameters
            create_images_partial = partial(
                create_images,
                shape_noise_add=self.shape_noise_add,
                shape_noise_sub=self.shape_noise_sub,
                style_psf=self.style_psf,
                gal_type=self.gal_type,
                k=k,
            )

            # run
            with Pool(processes=cpu_count()) as pool:
                for trial, result in tqdm(
                    pool.imap(create_images_partial, enumerate(combined_lhs), chunksize=8),
                    total=self.nb_of_samples,
                    desc=f"{BLUE}SLICE {k + 1}{RESET}",
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
