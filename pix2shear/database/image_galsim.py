import math
import random
from pathlib import Path
from time import time

import galsim
import numpy
import tensorflow_datasets as tfds
from astropy.io import fits
from utils import write_fits


def create_images(index_args_tuple):
    number, arguments = index_args_tuple

    # Latin hypercube parameters
    g1 = arguments[0][0]
    g2 = arguments[0][1]
    ells1 = arguments[1][:, 0]
    ells2 = arguments[1][:, 1]
    offset_x = arguments[1][:, 2]
    offset_y = arguments[1][:, 3]
    mag = arguments[1][:, 4]
    half_light_radius = arguments[1][:, 5]
    nsersic = arguments[1][:, 6]
    x = arguments[1][:, 7]
    y = arguments[1][:, 8]
    folder_results = arguments[2]

    number += 1
    seed = (int(time() * 20000000) + random.randint(0, 10_000_000)) % (2**32 - 1)

    pixel_scale = 0.1  # pix size in arcsec (sizes in input catalog in pixels)
    xsize = 64  # gal patch size in pixels (6.4'')
    ysize = 64  # gal patch size in pixels (6.4'')
    image_size = 352  # image size in pixels (0.59')
    nobj = x.shape[0]
    t_exp = 3 * 565  # s
    gain = 3.1  # e-/ADU
    readoutnoise = 4.2  # e-
    sky_bkg = 22.35  # mag/arcsec2
    ZP = 24.0  # mag
    F_sky = pixel_scale ** (2) * t_exp * 10 ** (-(sky_bkg - ZP) / 2.5)  # e-/pixel
    noise_variance = (
        numpy.sqrt(((readoutnoise) ** 2 + F_sky)) * 1 / gain
    ) ** 2  # e- -> ADU by dividing sigma by gain ; sigma = 4.9ADU

    # CREATE OUTPUT IMAGES
    full_image = galsim.ImageF(image_size, image_size)
    full_image.setOrigin(1, 1)

    # MAKE THE WCS COORDINATES (test11)
    # Make a slightly non-trivial WCS.  We'll use a slightly rotated coordinate system
    # and center it at the image center.
    theta = 0.17 * galsim.degrees
    dudx = numpy.cos(theta) * pixel_scale
    dudy = -numpy.sin(theta) * pixel_scale
    dvdx = numpy.sin(theta) * pixel_scale
    dvdy = numpy.cos(theta) * pixel_scale
    image_center = full_image.true_center  # noqa
    affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=full_image.true_center)

    # We can also put it on the celestial sphere to give it a bit more realism.
    # The TAN projection takes a (u,v) coordinate system on a tangent plane and projects
    # that plane onto the sky using a given point as the tangent point.  The tangent
    # point should be given as a CelestialCoord.
    sky_center = galsim.CelestialCoord(ra=3.544151 * galsim.hours, dec=-27.791371 * galsim.degrees)
    # The third parameter, units, defaults to arcsec, but we make it explicit here.
    # It sets the angular units of the (u,v) intermediate coordinate system.

    wcs = galsim.TanWCS(affine, sky_center, units=galsim.arcsec)
    full_image.wcs = wcs
    # logger.info('Image %r and %r created',file_name,file_name_noise)

    # TUNE THE SPEED OF FFT
    # slightly decrease the precision on fourrier and convolution to speed up.
    # Taken from Jarvis discussion https://github.com/GalSim-developers/GalSim/issues/566
    gsparams = galsim.GSParams(
        xvalue_accuracy=2.0e-4,
        kvalue_accuracy=2.0e-4,
        maxk_threshold=5.0e-3,
        folding_threshold=1.0e-2,
    )

    # BUILD PSF
    psf = (
        galsim.Airy(lam=800, diam=1.2, obscuration=0.3, scale_unit=galsim.arcsec, flux=1.0 / 3)
        + galsim.Airy(lam=700, diam=1.2, obscuration=0.3, scale_unit=galsim.arcsec, flux=1.0 / 3)
        + galsim.Airy(lam=600, diam=1.2, obscuration=0.3, scale_unit=galsim.arcsec, flux=1.0 / 3)
    )

    # initiate mean ell and g
    ave_ell1 = 0
    ave_ell2 = 0
    ave_g1 = 0
    ave_g2 = 0

    # loop on gals
    list_x, list_y, list_ell_1, list_ell_2, list_off_x, list_off_y = [], [], [], [], [], []
    list_mag, list_half_light_radius, list_n_sersic = [], [], []
    for i in range(nobj):
        # generate galaxy parameters
        list_x.append(x[i])
        list_y.append(y[i])
        list_ell_1.append(ells1[i])
        list_ell_2.append(ells2[i])
        list_off_x.append(offset_x[i])
        list_off_y.append(offset_y[i])
        list_mag.append(mag[i])
        list_half_light_radius.append(half_light_radius[i])
        list_n_sersic.append(nsersic[i])

        # Get position on sky
        image_pos = galsim.PositionD(x[i], y[i])
        world_pos = affine.toWorld(image_pos)  # noqa

        # Calculate gal flux
        fluxflux = t_exp / gain * 10 ** (-(mag[i] - ZP) / 2.5)

        # Compute Sersic profile
        gal = galsim.Sersic(
            n=nsersic[i],
            half_light_radius=half_light_radius[i],
            flux=fluxflux,
            gsparams=gsparams,
            trunc=half_light_radius[i] * 4.5,
        )
        gal = gal.shear(e1=ells1[i], e2=ells2[i])

        # Apply shear
        gal = gal.shear(g1=g1, g2=g2)

        # Calculate mean gal ellipticity
        ave_ell1 = ave_ell1 + (ells1[i] + g1) / nobj  # refine calculation
        ave_ell2 = ave_ell2 + (ells2[i] + g2) / nobj
        ave_g1 = ave_g1 + g1 / nobj
        ave_g2 = ave_g2 + g2 / nobj

        # convolve galaxy with PSF
        final = galsim.Convolve([psf, gal])

        # offset the center for pixelization (of random fraction of half a pixel)
        x_nominal = image_pos.x + 0.5
        y_nominal = image_pos.y + 0.5
        ix_nominal = int(math.floor(x_nominal + 0.5))
        iy_nominal = int(math.floor(y_nominal + 0.5))
        dx = (x_nominal - ix_nominal) * (2 * offset_x[i] - 1)
        dy = (y_nominal - iy_nominal) * (2 * offset_y[i] - 1)
        offset = galsim.PositionD(dx, dy)

        # draw galaxy
        image = galsim.ImageF(xsize, ysize, scale=pixel_scale)
        final.drawImage(image=image, wcs=wcs.local(image_pos), offset=offset)
        image.setCenter(ix_nominal, iy_nominal)

        # add stamps to single image
        bounds = image.bounds & full_image.bounds
        full_image[bounds] += image[bounds]

    # add Gaussian noise
    rng = galsim.BaseDeviate(seed)
    noise = galsim.GaussianNoise(rng, sigma=math.sqrt(noise_variance))
    full_image.addNoise(noise)

    # Savings
    write_fits(full_image.array, folder_results / f"simu_{number:05d}.fits")
    write_fits(numpy.array([ave_g1, ave_g2]), folder_results / f"avg_shear_{number:05d}.fits")
    write_fits(numpy.array([ave_ell1, ave_ell2]), folder_results / f"avg_ellips_{number:05d}.fits")
    write_fits(numpy.array([seed]), folder_results / f"seeds_{number:05d}.fits")
    write_fits(
        numpy.array(
            [
                list_ell_1,
                list_ell_2,
                list_x,
                list_y,
                list_off_x,
                list_off_y,
                list_mag,
                list_half_light_radius,
                list_n_sersic,
            ]
        ),
        folder_results / f"galaxies_parameters_{number:05d}.fits",
    )


class TransformTfds(tfds.core.GeneratorBasedBuilder):
    def __init__(self, dataset_dir: Path, version: str):
        self.VERSION = tfds.core.Version(version)
        super().__init__()
        self.dataset_dir = dataset_dir

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Tensor(shape=(352, 352), dtype=numpy.float32),
                    "avg_g": tfds.features.Tensor(shape=(2,), dtype=numpy.float32),
                    "avg_e": tfds.features.Tensor(shape=(2,), dtype=numpy.float32),
                    "gal_params": tfds.features.Tensor(shape=(9, 10), dtype=numpy.float32),
                    "seeds": tfds.features.Tensor(shape=(1,), dtype=numpy.float32),
                }
            ),
            supervised_keys=("image", "avg_g", "avg_e"),
        )

    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples(self.dataset_dir)}

    def _generate_examples(self, data_dir):
        simu_files = sorted(data_dir.glob("simu_*.fits"))
        shear_files = sorted(data_dir.glob("avg_shear_*.fits"))
        ellips_files = sorted(data_dir.glob("avg_ellips_*.fits"))
        gal_params_files = sorted(data_dir.glob("galaxies_parameters_*.fits"))
        seed_files = sorted(data_dir.glob("seeds_*.fits"))

        for simu_file, ellips_file, shear_file, gal_params_file, seed_file in zip(
            simu_files, ellips_files, shear_files, gal_params_files, seed_files
        ):
            simu_data = fits.getdata(simu_file).astype(numpy.float32)
            ellips_data = fits.getdata(ellips_file).astype(numpy.float32)
            shear_data = fits.getdata(shear_file).astype(numpy.float32)
            gal_params_data = fits.getdata(gal_params_file).astype(numpy.float32)
            seed_data = fits.getdata(seed_file).astype(numpy.float32)

            yield (
                simu_file.stem,
                {
                    "image": simu_data,
                    "avg_e": ellips_data,
                    "avg_g": shear_data,
                    "gal_params": gal_params_data,
                    "seeds": seed_data,
                },
            )
