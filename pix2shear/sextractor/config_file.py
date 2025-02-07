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
    get_flux_parameter,
    get_pos_parameters,
    set_deblend_factor,
    set_max_iterations,
    set_meta_iteration_stop,
    set_meta_iterations,
    use_iterative_fitting,
)

# ONGOING LOOP ARGS
PSF_DIR = Path(sys.argv[1])
FILE_NAME = str(sys.argv[2])
psf_file = PSF_DIR / "airy.psf.fits"
GAIN = 3.1


## S++ PROCESSING UNIT
measurement_images = [MeasurementImage(fits_file=FILE_NAME, psf_file=psf_file, gain=GAIN)]
top = ImageGroup(images=measurement_images)
top.split(ByKeyword("IMAGE_FILENAME"))
measurement_group = MeasurementGroup(top)


# PARAMETERS
x, y = get_pos_parameters()
rad = FreeParameter(lambda o: o.radius, Range(lambda v, o: (0.01, 2 * v), RangeType.EXPONENTIAL))
e1 = FreeParameter(0.0, Range((-0.9999, 0.9999), RangeType.LINEAR))
e2 = FreeParameter(0.0, Range((-0.9999, 0.9999), RangeType.LINEAR))
emod = DependentParameter(lambda x, y: numpy.sqrt(x * x + y * y), e1, e2)
angle = DependentParameter(lambda e1, e2: 0.5 * numpy.arctan2(e2, e1), e1, e2)
ratio = DependentParameter(lambda e: numpy.abs(1 - e) / (1 + e), emod)
sersic = FreeParameter(2.0, Range((0.1, 7.0), RangeType.LINEAR))


## MODEL
use_iterative_fitting(True)
set_meta_iterations(3)
set_max_iterations(150)
set_deblend_factor(0.9)  # Usually should be close to 1, but not exactly
set_meta_iteration_stop(0.0001)  # increase -> faster!
ZP = 24
flux = {}
flux = get_flux_parameter()
mag = {}
mag = DependentParameter(lambda f, zp=ZP: zp - 2.5 * numpy.log10(f), flux)
add_model(measurement_group, SersicModel(x, y, flux, rad, ratio, angle, sersic))


## OUTPUTS
add_output_column("mag_EUCLID_LIKE", mag)
add_output_column("x", x)
add_output_column("y", y)
add_output_column("ratio", ratio)
add_output_column("angle", angle)
add_output_column("e1", e1)
add_output_column("e2", e2)
add_output_column("sersic", sersic)
