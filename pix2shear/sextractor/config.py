from pathlib import Path
from sys import argv

import numpy
from sourcextractor.config import (
    DependentParameter,
    FreeParameter,
    ImageGroup,
    MeasurementGroup,
    MeasurementImage,
    Range,
    RangeType,
    SersicModel,
    add_model,
    add_output_column,
    add_prior,
    get_flux_parameter,
    get_pos_parameters,
    set_deblend_factor,
    set_meta_iteration_stop,
    set_meta_iterations,
    use_iterative_fitting,
)

## S++ PROCESSING UNIT
PSF_DIR = Path(argv[1]) / "airy.psf.fits"
FILE_NAME = str(argv[2])
measurement_images = [MeasurementImage(fits_file=FILE_NAME, psf_file=PSF_DIR)]
measurement_group = MeasurementGroup(ImageGroup(images=measurement_images))

## PARAMETERS
x, y = get_pos_parameters()  # center of the galaxy
rad = FreeParameter(
    lambda o: o.radius, Range(lambda v, o: (0.01, 2 * v), RangeType.EXPONENTIAL)
)  # effective radius: encloses 50% of the total light
e_range = Range((-0.999, 0.999), RangeType.LINEAR)
e1 = FreeParameter(0.0, e_range)
e2 = FreeParameter(0.0, e_range)
add_prior(e1, 0.0, 0.26)
add_prior(e2, 0.0, 0.26)
emod = DependentParameter(lambda x, y: numpy.sqrt(x * x + y * y), e1, e2)  # mag of ellip vector
ratio = DependentParameter(lambda e: numpy.abs(1 - e) / (1 + e), emod)  # axis ratio for ell measure
angle = DependentParameter(lambda e1, e2: 0.5 * numpy.arctan2(e2, e1), e1, e2)  # angle major axis
X_ser = FreeParameter(-2.3, Range((-20, 20), RangeType.LINEAR))
BOUND_X = 5
MAX_SERSIC = 6.3
MIN_SERSIC = 0.3
add_prior(X_ser, -BOUND_X, BOUND_X)
sersic = DependentParameter(
    lambda x: (MAX_SERSIC * numpy.exp(x) + MIN_SERSIC) / (1 + numpy.exp(x)), X_ser
)

## MODEL
use_iterative_fitting(True)
set_meta_iterations(3)
set_deblend_factor(0.9)  # Usually should be close to 1, but not exactly
set_meta_iteration_stop(0.0001)  # increase -> faster!
ZP = 24
flux = {}
flux = get_flux_parameter()  # total flux (brightness) of the galaxy
mag = {}
mag = DependentParameter(lambda f, zp=ZP: zp - 2.5 * numpy.log10(f), flux)
add_model(measurement_group, SersicModel(x, y, flux, rad, ratio, angle, sersic))

## ADDITIONAL OUTPUTS
add_output_column("mag_EUCLID_LIKE", mag)
add_output_column("ratio", ratio)
add_output_column("angle", angle)
add_output_column("e1", e1)
add_output_column("e2", e2)
add_output_column("sersic", sersic)
