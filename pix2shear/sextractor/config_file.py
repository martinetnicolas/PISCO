import sys
from pathlib import Path

import numpy
from sourcextractor.config import (
    ByKeyword,
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
    get_world_position_parameters,
    set_deblend_factor,
    set_max_iterations,
    set_meta_iteration_stop,
    set_meta_iterations,
    use_iterative_fitting,
)

# Ongoing loop bash args
PSF_DIR = Path(sys.argv[1])
FILE_NAME = str(sys.argv[2])


#  MeasurementImage & Group for S++ algo
measurement_images = [
    MeasurementImage(fits_file=FILE_NAME, psf_file=PSF_DIR / "airy.psf.fits", gain=3.1)
]
top = ImageGroup(images=measurement_images)
top.split(ByKeyword("IMAGE_FILENAME"))
measurement_group = MeasurementGroup(top)


# Parameters
x, y = get_pos_parameters()
ra, dec = get_world_position_parameters(x, y)
rad = FreeParameter(lambda o: o.radius, Range(lambda v, o: (0.01, 2 * v), RangeType.EXPONENTIAL))
e1 = FreeParameter(0.0, Range((-0.9999, 0.9999), RangeType.LINEAR))
e2 = FreeParameter(0.0, Range((-0.9999, 0.9999), RangeType.LINEAR))
add_prior(e1, 0.0, 0.26)
add_prior(e2, 0.0, 0.26)
emod = DependentParameter(lambda x, y: numpy.sqrt(x * x + y * y), e1, e2)
angle = DependentParameter(lambda e1, e2: 0.5 * numpy.arctan2(e2, e1), e1, e2)
ratio = DependentParameter(lambda e: numpy.abs(1 - e) / (1 + e), emod)
sersic = FreeParameter(2.0, Range((0.3, 6.2), RangeType.LINEAR))
X_sersic = DependentParameter(lambda n: numpy.log((n - 0.25) / (10 - n)), sersic)
add_prior(X_sersic, -0.8, 1.5)


# Define model
use_iterative_fitting(True)
set_meta_iterations(3)
set_max_iterations(150)
set_deblend_factor(0.9)  # Usually should be close to 1, but not exactly
set_meta_iteration_stop(0.0001)  # increase -> faster!
flux = {}
mag = {}
ZP = 24
flux = get_flux_parameter()
mag = DependentParameter(lambda f, zp=ZP: zp - 2.5 * numpy.log10(f), flux)
add_model(measurement_group, SersicModel(x, y, flux, rad, ratio, angle, sersic))


# Define outputs
add_output_column("mag_EUCLID_LIKE", mag)
add_output_column("x", x)
add_output_column("y", y)
add_output_column("ra", ra)
add_output_column("dec", dec)
add_output_column("radius", rad)
add_output_column("ratio", ratio)
add_output_column("angle", angle)
add_output_column("e1", e1)
add_output_column("e2", e2)
add_output_column("sersic", sersic)
