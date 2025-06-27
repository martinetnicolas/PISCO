import math

import galsim
import numpy
from utils import DTYPE_16, DTYPE_32

# --------------------------------------------------------------------------------------------------


def unpack_lhs(arguments):
    g1, g2, snc, noise_key = arguments[0]
    lhs = arguments[1]

    return {
        "g1": g1,
        "g2": g2,
        "snc": int(snc),
        "noise_key": int(noise_key),
        "int_e1": lhs[:, 0],
        "int_e2": lhs[:, 1],
        "off_x": lhs[:, 2],
        "off_y": lhs[:, 3],
        "mag": lhs[:, 4],
        "hlr": lhs[:, 5],
        "sersic": lhs[:, 6],
        "x": lhs[:, 7],
        "y": lhs[:, 8],
        "index_cosmos": lhs[:, 9],
        "sample_cosmos": lhs[:, 10],
    }


def get_constants():
    pixel_scale = 0.1  # pix size in arcsec (sizes in input catalog in pixels)
    patch_size = 64  # gal patch size in pixels (6.4'')
    image_size = 352  # image size in pixels (0.59')
    t_exp = 4 * 566  # s
    gain = 3.1  # e-/ADU
    readoutnoise = 4.2  # e-
    sky_bkg = 22.35  # mag/arcsec2
    zp = 23.68  # mag

    return pixel_scale, patch_size, image_size, t_exp, gain, readoutnoise, sky_bkg, zp


def build_psf(style):
    flux = 1.0 / 3

    if style == "airy":
        psf = (
            galsim.Airy(lam=800, diam=1.2, obscuration=0.3, scale_unit=galsim.arcsec, flux=flux)
            + galsim.Airy(lam=700, diam=1.2, obscuration=0.3, scale_unit=galsim.arcsec, flux=flux)
            + galsim.Airy(lam=600, diam=1.2, obscuration=0.3, scale_unit=galsim.arcsec, flux=flux)
        )
    elif style == "euclid":
        psf = (
            galsim.OpticalPSF(
                lam=800,
                diam=1.2,
                obscuration=0.3,
                scale_unit=galsim.arcsec,
                flux=flux,
                nstruts=3,
            )
            + galsim.OpticalPSF(
                lam=700,
                diam=1.2,
                obscuration=0.3,
                scale_unit=galsim.arcsec,
                flux=flux,
                nstruts=3,
            )
            + galsim.OpticalPSF(
                lam=600,
                diam=1.2,
                obscuration=0.3,
                scale_unit=galsim.arcsec,
                flux=flux,
                nstruts=3,
            )
        )

    return psf


def build_full_image(image_size, pixel_scale):
    # create galsim image
    full_image = galsim.ImageF(image_size, image_size)
    full_image.setOrigin(1, 1)

    # slightly non-trivial WCS and center it at the image center
    theta = 0.17 * galsim.degrees
    dudx = numpy.cos(theta) * pixel_scale
    dudy = -numpy.sin(theta) * pixel_scale
    dvdx = numpy.sin(theta) * pixel_scale
    dvdy = numpy.cos(theta) * pixel_scale

    # projects that plane onto the sky using a given point as the tangent point
    affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=full_image.true_center)

    # We can also put it on the celestial sphere to give it a bit more realism.
    sky_center = galsim.CelestialCoord(ra=3.544151 * galsim.hours, dec=-27.791371 * galsim.degrees)

    # sets the angular units of the (u,v) intermediate coordinate system.
    wcs = galsim.TanWCS(affine, sky_center, units=galsim.arcsec)
    full_image.wcs = wcs

    return full_image


# Tune the speed of FFT (taken from Jarvis discussion https://github.com/GalSim-developers/GalSim/issues/566)
GSPARAMS = galsim.GSParams(
    xvalue_accuracy=2.0e-4,
    kvalue_accuracy=2.0e-4,
    maxk_threshold=5.0e-3,
    folding_threshold=1.0e-2,
)


def get_offset(off_x, off_y, x, y):
    # get position on sky
    image_pos = galsim.PositionD(x, y)

    # offset the center for pixelization (of random fraction of half a pixel)
    x_nominal = image_pos.x + 0.5
    y_nominal = image_pos.y + 0.5
    ix_nominal = int(math.floor(x_nominal + 0.5))
    iy_nominal = int(math.floor(y_nominal + 0.5))

    dx = (x_nominal - ix_nominal) * (2 * off_x - 1)
    dy = (y_nominal - iy_nominal) * (2 * off_y - 1)
    offset = galsim.PositionD(dx, dy)

    return image_pos, ix_nominal, iy_nominal, offset


def create_mask_wrt_hlr(pixel_scale, hlr, ny, nx):
    y, x = numpy.mgrid[0:ny, 0:nx]

    x_center = (nx - 1) / 2
    y_center = (ny - 1) / 2

    r = numpy.sqrt((x - x_center) ** 2 + (y - y_center) ** 2) * pixel_scale
    mask = r <= hlr

    return mask


def apply_bin_indices_area(image, label_stamp, bounds_pixel):
    for y in range(bounds_pixel.ymin, bounds_pixel.ymax + 1):
        for x in range(bounds_pixel.xmin, bounds_pixel.xmax + 1):
            val = label_stamp(x, y)
            if val > 0:
                image.setValue(x, y, val)

    return image


# --------------------------------------------------------------------------------------------------


# create pisco euclid-like image
def create_images(index_args_tuple, style_psf, gal_type, cosmos_cat_23, cosmos_cat_25):
    # get seed for noise and args for galsim
    key, arguments = index_args_tuple
    key = int(key)

    # params simu
    params = unpack_lhs(arguments)
    NB_OBJET = len(params["x"])

    # constant arguments
    PIXEL_SCALE, PATCH_SIZE, IMG_SIZE, EXPTIME, GAIN, readoutnoise, SKY_BKG, ZP = get_constants()

    # psf
    psf = build_psf(style_psf)

    # observed ellipticities
    obs_e1 = params["int_e1"] + params["g1"]
    obs_e2 = params["int_e2"] + params["g2"]

    # calculate gal flux
    flux = EXPTIME / GAIN * 10 ** (-(params["mag"] - ZP) / 2.5)

    # galsim images
    full_image = build_full_image(IMG_SIZE, PIXEL_SCALE)
    noise_image = build_full_image(IMG_SIZE, PIXEL_SCALE)

    # print each object
    for i in range(NB_OBJET):
        if params["index_cosmos"][i] > 0:
            # compute Sersic profile with cosmos catalog galaxies
            SAMPLE = str(params["sample_cosmos"][i])
            if SAMPLE == "23.5":
                cosmos_cat = cosmos_cat_23
            elif SAMPLE == "25.2":
                cosmos_cat = cosmos_cat_25

            INDEX = int(params["index_cosmos"][i])
            gal = cosmos_cat.makeGalaxy(
                index=INDEX,
                gal_type=gal_type,
                noise_pad_size=0,
                sersic_prec=0.05,
                gsparams=GSPARAMS,
            )

            # adding our own flux
            gal = gal.withFlux(flux[i])

            # rotate galaxy
            ang = 0 if params["snc"] in [1, 3] else 90
            gal = gal.rotate(theta=ang * galsim.degrees)

        else:
            # compute Sersic profile with galsim
            gal = galsim.Sersic(
                n=params["sersic"][i],
                half_light_radius=params["hlr"][i],
                flux=flux[i],
                gsparams=GSPARAMS,
                trunc=params["hlr"][i] * 4.5,
            )

            # apply ellips
            gal = gal.shear(e1=params["int_e1"][i], e2=params["int_e2"][i])

        # apply shear
        gal = gal.shear(g1=params["g1"], g2=params["g2"])

        # convolve galaxy with PSF
        final = galsim.Convolve([psf, gal])

        # offset
        image_pos, ix_nominal, iy_nominal, offset = get_offset(
            params["off_x"][i], params["off_y"][i], params["x"][i], params["y"][i]
        )

        # draw galaxy
        stamp = galsim.ImageF(PATCH_SIZE, PATCH_SIZE, scale=PIXEL_SCALE)
        final.drawImage(image=stamp, wcs=full_image.wcs.local(image_pos), offset=offset)
        stamp.setCenter(ix_nominal, iy_nominal)

        # add stamp to the main image
        bounds = stamp.bounds & full_image.bounds
        full_image[bounds] += stamp[bounds]

    # gaussian noise image
    F_SKY = PIXEL_SCALE ** (2) * EXPTIME * 10 ** (-(SKY_BKG - ZP) / 2.5)  # e-/pixel
    variance = (numpy.sqrt(readoutnoise**2 + F_SKY) / GAIN) ** 2  # e- -> ADU
    noise = galsim.GaussianNoise(
        rng=galsim.BaseDeviate(params["noise_key"]), sigma=math.sqrt(variance)
    )
    noise_image.addNoise(noise)

    # add noise
    sign = 1 if params["snc"] in [1, 2] else -1
    full_image += sign * noise_image

    # return arguments
    avg_obs_e1 = numpy.mean(obs_e1)
    avg_obs_e2 = numpy.mean(obs_e2)
    avg_int_e1 = numpy.mean(params["int_e1"])
    avg_int_e2 = numpy.mean(params["int_e2"])
    avg_g1 = params["g1"]
    avg_g2 = params["g2"]

    dict_results = {
        "image": full_image.array.astype(DTYPE_16),
        "avg_g": numpy.array([avg_g1, avg_g2]).astype(DTYPE_32),
        "avg_obs_e": numpy.array([avg_obs_e1, avg_obs_e2]).astype(DTYPE_32),
        "avg_int_e": numpy.array([avg_int_e1, avg_int_e2]).astype(DTYPE_32),
        "keys": numpy.array([key, params["snc"], params["noise_key"]]).astype(DTYPE_32),
        "gal_params": numpy.stack(
            [
                params["int_e1"],
                params["int_e2"],
                obs_e1,
                obs_e2,
                params["x"],
                params["y"],
                params["mag"],
                params["hlr"],
                params["sersic"],
            ],
        ).astype(DTYPE_32),
    }

    return key, dict_results


# --------------------------------------------------------------------------------------------------


# create pisco psf images
def create_psf(seed, under_float, noisy_psf, style_psf, big_psf, grid):
    # constant arguments
    PIXEL_SCALE, PATCH_SIZE, IMG_SIZE, EXPTIME, GAIN, readoutnoise, SKY_BKG, ZP = get_constants()

    # gaussian noise
    F_SKY = PIXEL_SCALE ** (2) * EXPTIME * 10 ** (-(SKY_BKG - ZP) / 2.5)  # e-/pixel
    variance = (numpy.sqrt(readoutnoise**2 + F_SKY) / GAIN) ** 2  # e- -> ADU
    noise = galsim.GaussianNoise(rng=galsim.BaseDeviate(seed), sigma=math.sqrt(variance))

    # style psf
    psf = build_psf(style_psf)

    # noise image
    if noisy_psf:
        SIZE = PATCH_SIZE if grid or not big_psf else IMG_SIZE
        noise_image = galsim.ImageF(SIZE, SIZE)
        noise_image.setOrigin(0, 0)
        noise_image.addNoise(noise)

    # grid psf image
    if grid:
        full_image_psf = galsim.ImageF(IMG_SIZE, IMG_SIZE)
        full_image_psf.setOrigin(0, 0)
        patch_image = psf.drawImage(nx=PATCH_SIZE, ny=PATCH_SIZE, scale=PIXEL_SCALE)

        NB_PATCHES = IMG_SIZE // PATCH_SIZE
        for i in range(NB_PATCHES):
            for j in range(NB_PATCHES):
                x_start = i * PATCH_SIZE
                y_start = j * PATCH_SIZE
                bounds = galsim.BoundsI(
                    x_start + 1, x_start + PATCH_SIZE, y_start + 1, y_start + PATCH_SIZE
                )

                sub_image = full_image_psf[bounds]
                sub_image.array[:, :] = patch_image.array

                if noisy_psf:
                    sub_image.array[:, :] += noise_image.array

    # normal psf image
    else:
        SIZE = PATCH_SIZE if grid or not big_psf else IMG_SIZE

        # psf
        full_image_psf = galsim.ImageF(SIZE, SIZE)
        full_image_psf.setOrigin(0, 0)
        full_image_psf = psf.drawImage(image=full_image_psf, scale=PIXEL_SCALE)

        if noisy_psf:
            full_image_psf += noise_image

    # return image & key
    dtype = DTYPE_16 if under_float else DTYPE_32
    return seed, {"image_psf": full_image_psf.array.astype(dtype)}
