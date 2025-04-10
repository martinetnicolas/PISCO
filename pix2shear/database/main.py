from functools import partial
from multiprocessing import Pool, cpu_count

import numpy
import tensorflow_datasets as tfds
from functions_images import create_images
from functions_lhs import draw_lhs_gal, draw_lhs_img
from functions_projections import project_lhs
from tqdm import tqdm


class PiscoEuclid(tfds.core.GeneratorBasedBuilder):
    def __init__(
        self,
        nb_of_samples,
        ellips_style,
        pixel_out,
        seed_database,
        shape_noise_cancel_add_noise,
        shape_noise_cancel_sub_noise,
        style_psf,
        lhs_pos,
    ):
        self.nb_of_samples = nb_of_samples
        self.seed_database = seed_database

        self.shape_noise_cancel_add_noise = shape_noise_cancel_add_noise
        self.shape_noise_cancel_sub_noise = shape_noise_cancel_sub_noise
        self.style_psf = style_psf
        self.ellips_style = ellips_style
        self.pixel_out = pixel_out
        self.lhs_pos = lhs_pos

        self.images_size = (352, 352)
        self.nb_galaxies = 10
        self.nb_galaxies_parameters = 9
        self.nb_total_galaxies = self.nb_of_samples * self.nb_galaxies

        super().__init__()

    def _info(self) -> tfds.core.DatasetInfo:
        dtype = numpy.float32
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Tensor(shape=self.images_size, dtype=dtype),
                    # "image_psf": tfds.features.Tensor(shape=self.images_size, dtype=numpy.float32),  # needs to be taken out for dbb > 100k #noqa
                    "avg_g": tfds.features.Tensor(shape=(2,), dtype=dtype),
                    "avg_obs_e": tfds.features.Tensor(shape=(2,), dtype=dtype),
                    "avg_int_e": tfds.features.Tensor(shape=(2,), dtype=dtype),
                    "keys": tfds.features.Tensor(shape=(3,), dtype=dtype),
                    "gal_params": tfds.features.Tensor(
                        shape=(self.nb_galaxies_parameters, self.nb_galaxies), dtype=dtype
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {"dataset": self._generate_examples()}

    @staticmethod
    def divide_for_multiprocess(hc, nb_of_samples):
        list_nb_gal = [10] * nb_of_samples

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

        # prepocess for multiprocess
        lhs_gal = self.divide_for_multiprocess(lhs_gal, self.nb_of_samples)

        # combined for multiprocess
        combined_lhs = list(zip(lhs_img, lhs_gal))

        # shape noise cancellation
        TOTAL = 2 if self.shape_noise_cancel_add_noise else 1
        TOTAL *= 2 if self.shape_noise_cancel_sub_noise else 1

        # compute image
        for k in range(TOTAL):
            # fix function parameters
            create_images_partial = partial(
                create_images,
                shape_noise_cancel_add_noise=self.shape_noise_cancel_add_noise,
                shape_noise_cancel_sub_noise=self.shape_noise_cancel_sub_noise,
                style_psf=self.style_psf,
                k=k,
            )

            # run
            with Pool(processes=cpu_count()) as pool:
                results = tqdm(
                    pool.map(func=create_images_partial, iterable=enumerate(combined_lhs)),
                    total=self.nb_of_samples,
                )

            # yield results
            for trial, result in results:
                yield int(trial), result
