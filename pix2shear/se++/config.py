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
    get_flux_parameter,
    get_pos_parameters,
    set_deblend_factor,
    set_meta_iteration_stop,
    set_meta_iterations,
    use_iterative_fitting,
)

## S++ PROCESSING UNIT
measurement_images = MeasurementImage(
    fits_file=argv[1], psf_file=argv[2], gain=3.1, saturation=10_000
)
measurement_group = MeasurementGroup(ImageGroup(images=[measurement_images]))

## PARAMETERS
x, y = get_pos_parameters()  # center of the galaxy
rad = FreeParameter(lambda o: o.radius, Range(lambda v, o: (0.01, 10 * v), RangeType.EXPONENTIAL))
e_range = Range((-1.0, 1.0), RangeType.LINEAR)
e1 = FreeParameter(0.0, e_range)
e2 = FreeParameter(0.0, e_range)
emod = DependentParameter(lambda x, y: numpy.sqrt(x * x + y * y), e1, e2)  # mag of ellip vector
ratio = DependentParameter(lambda e: numpy.abs(1 - e) / (1 + e), emod)  # axis ratio for ell measure
angle = DependentParameter(lambda e1, e2: 0.5 * numpy.arctan2(e2, e1), e1, e2)  # angle major axis
sersic = FreeParameter(0.8, Range((0.3, 10), RangeType.LINEAR))

## MODEL
use_iterative_fitting(True)
set_meta_iterations(3)
set_deblend_factor(0.9)  # Usually should be close to 1, but not exactly
set_meta_iteration_stop(0.0001)  # increase -> faster!
ZP = 30.84
flux = {}
flux = get_flux_parameter()  # total flux (brightness) of the galaxy
mag = {}
mag = DependentParameter(lambda f, zp=ZP: zp - 2.5 * numpy.log10(f), flux)
add_model(measurement_group, SersicModel(x, y, flux, rad, ratio, angle, sersic))

## USER OUTPUTS
add_output_column("x_model", x)
add_output_column("y_model", y)
add_output_column("sersic_model", sersic)
add_output_column("mag_model", mag)
add_output_column("flux_model", flux)
add_output_column("e1_model", e1)
add_output_column("e2_model", e2)
