import numpy
from polars import read_csv


def load_and_slice_csv(file_path, nb_total):
    return numpy.array(read_csv(file_path, separator=",", skip_rows=0)[:nb_total])


def X_sersic(n):
    return numpy.log((n - 0.25) / (10 - n))


def sersic_from_X(X):
    return (10 * numpy.exp(X) + 0.25) / (1 + numpy.exp(X))


def ratio_angle_to_e1e2(ratio, angle):
    e = (1 - ratio) / (1 + ratio)
    return e * numpy.cos(2 * angle), e * numpy.sin(2 * angle)
