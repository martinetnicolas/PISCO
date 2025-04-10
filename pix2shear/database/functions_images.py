import math

import galsim
import numpy


def unpack_lhs(arguments):
    g1, g2, z = arguments[0]
    lhs = arguments[1]

    return {
        "g1": g1,
        "g2": g2,
        "z": z,
        "int_e1": lhs[:, 0],
        "int_e2": lhs[:, 1],
        "off_x": lhs[:, 2],
        "off_y": lhs[:, 3],
        "mag": lhs[:, 4],
        "hlr": lhs[:, 5],
        "sersic": lhs[:, 6],
        "x": lhs[:, 7],
        "y": lhs[:, 8],
    }


def get_constants():
    pixel_scale = 0.1  # pix size in arcsec (sizes in input catalog in pixels)
    xsize = 64  # gal patch size in pixels (6.4'')
    ysize = 64  # gal patch size in pixels (6.4'')
    image_size = 352  # image size in pixels (0.59')
    t_exp = 3 * 565  # s
    gain = 3.1  # e-/ADU
    readoutnoise = 4.2  # e-
    sky_bkg = 22.35  # mag/arcsec2
    ZP = 24.0  # mag

    return pixel_scale, xsize, ysize, image_size, t_exp, gain, readoutnoise, sky_bkg, ZP


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
                lam=800, diam=1.2, obscuration=0.3, scale_unit=galsim.arcsec, flux=flux, nstruts=3
            )
            + galsim.OpticalPSF(
                lam=700, diam=1.2, obscuration=0.3, scale_unit=galsim.arcsec, flux=flux, nstruts=3
            )
            + galsim.OpticalPSF(
                lam=600, diam=1.2, obscuration=0.3, scale_unit=galsim.arcsec, flux=flux, nstruts=3
            )
        )

    return psf


def rotation(k):
    theta_rad = numpy.radians(k * 90)
    theta_rot = k * 2 * 90 * galsim.degrees

    cos2t = 0 if abs(numpy.cos(2 * theta_rad)) < 1e-10 else numpy.cos(2 * theta_rad)
    sin2t = 0 if abs(numpy.sin(2 * theta_rad)) < 1e-10 else numpy.sin(2 * theta_rad)

    return cos2t, sin2t, theta_rot


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
    xvalue_accuracy=2.0e-4, kvalue_accuracy=2.0e-4, maxk_threshold=5.0e-3, folding_threshold=1.0e-2
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


def bin_edgeds_bounds(max_shears=0.065, deg_poly_fit_g_vs_z=1):
    if max_shears == 0.055:
        max_bin_edges = 2.6
    elif max_shears == 0.065:
        max_bin_edges = 3.2
    if deg_poly_fit_g_vs_z == 1:
        min_bin_edges = 0.019
    elif deg_poly_fit_g_vs_z == 2:
        min_bin_edges = 0.07

    return min_bin_edges, max_bin_edges


def binned_shear_and_redshift(redshift, nb_z_bins):
    min_bin_edges, max_bin_edges = bin_edgeds_bounds()
    bin_edges = numpy.linspace(min_bin_edges, max_bin_edges, nb_z_bins + 1)
    bin_indices = numpy.digitize(redshift, bins=bin_edges, right=False)
    return bin_indices


def create_mask_over_flux(pixel_scale, hlr, ny, nx):
    y, x = numpy.mgrid[0:ny, 0:nx]
    x_center = (nx - 1) / 2
    y_center = (ny - 1) / 2
    r = numpy.sqrt((x - x_center) ** 2 + (y - y_center) ** 2) * pixel_scale
    mask = r <= hlr
    return mask


def print_flux_binned_redshift(full_image_z_flux, label_stamp, bounds_pixel):
    for y in range(bounds_pixel.ymin, bounds_pixel.ymax + 1):
        for x in range(bounds_pixel.xmin, bounds_pixel.xmax + 1):
            val = label_stamp(x, y)
            if val > 0:
                full_image_z_flux.setValue(x, y, val)

    return full_image_z_flux


def create_images(
    index_args_tuple, shape_noise_cancel_add_noise, shape_noise_cancel_sub_noise, style_psf, k
):
    # get seed for noise and args for galsim
    seed, arguments = index_args_tuple
    seed = int(seed)

    # key for tfds
    hashed_key = seed + 1
    hashed_key *= 2 if shape_noise_cancel_add_noise else 1
    hashed_key *= 2 if shape_noise_cancel_sub_noise else 1
    hashed_key -= k

    # params simu
    params = unpack_lhs(arguments)

    # constant arguments
    pixel_scale, xsize, ysize, image_size, t_exp, gain, readoutnoise, sky_bkg, ZP = get_constants()
    F_sky = pixel_scale ** (2) * t_exp * 10 ** (-(sky_bkg - ZP) / 2.5)  # e-/pixel

    # psf
    psf = build_psf(style_psf)

    # rotation
    cos2t, sin2t, theta_rot = rotation(k)

    # picked z & g w.r.t. bin
    # bin_indices = binned_shear_and_redshift(params["z"], nb_z_bins)

    # intrinsic ellipticities
    int_e1 = params["int_e1"] * cos2t + params["int_e2"] * sin2t
    int_e2 = -params["int_e1"] * sin2t + params["int_e2"] * cos2t
    avg_int_e1 = numpy.mean(int_e1)
    avg_int_e2 = numpy.mean(int_e2)

    # observed ellipticities
    obs_e1 = int_e1 + params["g1"]
    obs_e2 = int_e2 + params["g2"]
    avg_obs_e1 = numpy.mean(obs_e1)
    avg_obs_e2 = numpy.mean(obs_e2)

    # shear components
    # avg_g1 = numpy.mean(params["g1"])
    # avg_g2 = numpy.mean(params["g2"])
    avg_g1 = params["g1"]
    avg_g2 = params["g2"]

    # calculate gal flux
    flux = t_exp / gain * 10 ** (-(params["mag"] - ZP) / 2.5)

    # galsim images
    full_image = build_full_image(image_size, pixel_scale)
    noise_image = build_full_image(image_size, pixel_scale)
    # full_image_z = build_full_image(image_size, pixel_scale)

    # print each object
    NB_OBJET = len(params["x"])
    for i in range(NB_OBJET):
        # compute Sersic profile
        gal = galsim.Sersic(
            n=params["sersic"][i],
            half_light_radius=params["hlr"][i],
            flux=flux[i],
            gsparams=GSPARAMS,
            trunc=params["hlr"][i] * 4.5,
        )

        # apply ellips
        gal = gal.shear(e1=int_e1[i], e2=int_e2[i])

        # rotate galaxy
        gal = gal.rotate(theta=theta_rot)

        # apply shear
        gal = gal.shear(g1=avg_g1, g2=avg_g2)  # simple case
        # gal = gal.shear(g1=params["g1"][i], g2=params["g2"][i])  # normal case

        # convolve galaxy with PSF
        final = galsim.Convolve([psf, gal])

        # offset
        image_pos, ix_nominal, iy_nominal, offset = get_offset(
            params["off_x"][i], params["off_y"][i], params["x"][i], params["y"][i]
        )

        # draw galaxy
        stamp = galsim.ImageF(xsize, ysize, scale=pixel_scale)
        final.drawImage(image=stamp, wcs=full_image.wcs.local(image_pos), offset=offset)
        stamp.setCenter(ix_nominal, iy_nominal)

        # add stamp to the main image
        bounds = stamp.bounds & full_image.bounds
        full_image[bounds] += stamp[bounds]

        # # flux-binned redshift
        # label_stamp = galsim.ImageI(xsize, ysize, scale=pixel_scale)
        # label_stamp.setCenter(ix_nominal, iy_nominal)
        # mask = create_mask_over_flux(pixel_scale, params["hlr"][i], xsize, ysize)
        # label_stamp.array[mask] = bin_indices

        # # pixeled-binned redshift image
        # pos_gal = galsim.PositionI(int(round(ix_nominal)), int(round(iy_nominal)))
        # full_image_z[pos_gal.x, pos_gal.y] = bin_indices

        # # flux-binned redshift image
        # bounds_pixel = label_stamp.bounds & full_image_z.bounds
        # full_image_z = print_flux_binned_redshift(full_image_z, label_stamp, bounds_pixel)

    # gaussian noise image
    variance = (numpy.sqrt(readoutnoise**2 + F_sky) / gain) ** 2  # e- -> ADU
    noise = galsim.GaussianNoise(rng=galsim.BaseDeviate(seed), sigma=math.sqrt(variance))
    noise_image.addNoise(noise)

    # add noise
    if shape_noise_cancel_add_noise and shape_noise_cancel_sub_noise:
        full_image += noise_image if k < 2 else -noise_image
    elif shape_noise_cancel_add_noise:
        full_image += noise_image
    elif shape_noise_cancel_sub_noise:
        full_image -= noise_image
    else:
        full_image += noise_image

    # # psf image
    # full_image_psf = galsim.ImageF(image_size, image_size)
    # full_image_psf.setOrigin(1, 1)
    # full_image_psf = psf.drawImage(image=full_image_psf, scale=pixel_scale)
    # # full_image_psf.addNoise(noise)

    # image arguments
    return hashed_key, {
        "image": full_image.array,
        # "image_psf": full_image_psf.array, # needs to be taken out for dbb > 100k
        "avg_g": numpy.array([avg_g1, avg_g2]).astype(numpy.float32),
        "avg_obs_e": numpy.array([avg_obs_e1, avg_obs_e2]).astype(numpy.float32),
        "avg_int_e": numpy.array([avg_int_e1, avg_int_e2]).astype(numpy.float32),
        "keys": numpy.array([seed, hashed_key, k]).astype(numpy.float32),
        "gal_params": numpy.stack(
            [
                int_e1,
                int_e2,
                obs_e1,
                obs_e2,
                params["x"],
                params["y"],
                params["mag"],
                params["hlr"],
                params["sersic"],
            ],
        ).astype(numpy.float32),
    }
