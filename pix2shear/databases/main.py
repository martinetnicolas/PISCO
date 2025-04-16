from functools import partial
from multiprocessing import Pool, cpu_count

import numpy
import tensorflow_datasets as tfds
from functions_images import create_images
from functions_lhs import adding_cosmos_catalog_galaxies, draw_lhs_gal, draw_lhs_img
from functions_projections import project_lhs
from tqdm import tqdm
from utils import DTYPE


class PiscoEuclid(tfds.core.GeneratorBasedBuilder):
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
        type_cosmos,
        big_psf,
        nb_galaxies,
    ):
        self.nb_of_samples = nb_of_samples
        self.seed_database = seed_database
        self.percentage_cosmos = percentage_cosmos
        self.type_cosmos = type_cosmos

        self.shape_noise_add = shape_noise_add
        self.shape_noise_sub = shape_noise_sub
        self.style_psf = style_psf
        self.ellips_style = ellips_style
        self.pixel_out = pixel_out
        self.lhs_pos = lhs_pos
        self.big_psf = big_psf

        self.images_size = (352, 352)
        self.patch_size = (64, 64)
        self.nb_galaxies = nb_galaxies
        self.nb_galaxies_parameters = 9

        self.slice_shape_noise = (self.shape_noise_add + 1) * (self.shape_noise_sub + 1)
        self.total_nb_of_samples = self.slice_shape_noise * self.nb_of_samples
        self.nb_total_galaxies = self.nb_of_samples * self.nb_galaxies

        super().__init__()

    def _info(self) -> tfds.core.DatasetInfo:
        features = {
            "image": tfds.features.Tensor(shape=self.images_size, dtype=DTYPE),
            "avg_g": tfds.features.Tensor(shape=(2,), dtype=DTYPE),
            "avg_obs_e": tfds.features.Tensor(shape=(2,), dtype=DTYPE),
            "avg_int_e": tfds.features.Tensor(shape=(2,), dtype=DTYPE),
            "keys": tfds.features.Tensor(shape=(3,), dtype=DTYPE),
            "gal_params": tfds.features.Tensor(
                shape=(self.nb_galaxies_parameters, self.nb_galaxies), dtype=DTYPE
            ),
        }
        if self.big_psf:
            features["image_psf"] = tfds.features.Tensor(shape=self.images_size, dtype=DTYPE)
        else:
            features["image_psf"] = tfds.features.Tensor(shape=self.patch_size, dtype=DTYPE)

        metadata = {
            "nb_of_samples": self.nb_of_samples,
            "seed_database": self.seed_database,
            "shape_noise_add": self.shape_noise_add,
            "shape_noise_sub": self.shape_noise_sub,
            "style_psf": self.style_psf,
            "ellips_style": self.ellips_style,
            "pixel_out": self.pixel_out,
            "lhs_pos": self.lhs_pos,
            "images_size": self.images_size,
            "patch_size": self.patch_size,
            "nb_galaxies": self.nb_galaxies,
            "nb_galaxies_parameters": self.nb_galaxies_parameters,
            "nb_total_galaxies": self.nb_total_galaxies,
            "total_nb_of_samples": self.total_nb_of_samples,
            "slice_shape_noise": self.slice_shape_noise,
            "big_psf": self.big_psf,
        }

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(features), metadata=tfds.core.MetadataDict(metadata)
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {"dataset": self._generate_examples()}

    def divide_for_multiprocess(self, hc, nb_of_samples):
        list_nb_gal = [self.nb_galaxies] * nb_of_samples

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
        lhs_img = draw_lhs_img(seed=self.seed_database * 1_000, n_samples=self.nb_of_samples)

        # create lhs galaxies
        lhs_gal = draw_lhs_gal(self.lhs_pos, self.seed_database, self.nb_total_galaxies)

        # lhs galaxies projection
        lhs_gal = project_lhs(
            hypercube=lhs_gal, ellips_style=self.ellips_style, pixel_out=self.pixel_out
        )

        # taking real/parametric galaxies from cosmos catalogs
        nb_galaxies_cosmos = int(self.percentage_cosmos * self.nb_total_galaxies / 100)
        replace_indices = numpy.random.choice(
            a=self.nb_total_galaxies, size=nb_galaxies_cosmos, replace=False
        )
        lhs_gal = adding_cosmos_catalog_galaxies(lhs_gal, numpy.sort(replace_indices), self.version)

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
                type_cosmos=self.type_cosmos,
                big_psf=self.big_psf,
                k=k,
            )

            # run
            with Pool(processes=cpu_count()) as pool:
                for trial, result in tqdm(
                    pool.imap(create_images_partial, enumerate(combined_lhs), chunksize=8),
                    total=self.nb_of_samples,
                    desc=f"Slice {k + 1}",
                ):
                    yield int(trial), result
