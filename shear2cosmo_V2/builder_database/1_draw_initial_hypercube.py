from pathlib import Path

import numpy
import pandas
from scipy.stats import qmc
from utils import create_dir

# --------------------------------------------------------------------------------------------------


NUMBER_OF_COSMO = 0
DBB = "DBB_OPT_0"


# --------------------------------------------------------------------------------------------------


def create_lhs(n):
    parameters = ["Om", "S8", "h", "w0"]

    # parameters_intervals = numpy.array(
    #     [[0.1, 0.55], [0.6, 0.9], [0.6, 0.82], [-2.0, -0.5]]
    # )  # cosmoslics
    # parameters_intervals = numpy.array(
    #     [[0.2, 0.55], [0.6, 0.9], [0.6, 0.82], [-2.0, -0.5]]
    # )  # for cam to not crash
    parameters_intervals = numpy.array(
        [[0.2, 0.45], [0.65, 0.85], [0.65, 0.77], [-1.25, -0.75]]
    )  # fore test data

    hypercube = qmc.LatinHypercube(
        d=len(parameters),
        strength=1,
        scramble=False,
        optimization="lloyd",
        # seed=2,
    )

    samples = hypercube.random(n=n)
    scaled_samples = qmc.scale(samples, parameters_intervals[:, 0], parameters_intervals[:, 1])
    scaled_samples = pandas.DataFrame(data=scaled_samples, columns=parameters)

    scaled_samples["sig8"] = scaled_samples["S8"] * numpy.sqrt(0.3 / scaled_samples["Om"])
    Ob = 0.0473
    scaled_samples["Oc"] = scaled_samples["Om"] - Ob

    return scaled_samples


FOLDER_BASE = Path("/net/GECO/pisco/SHEAR2COSMO")
FOLDER_RESULTS = create_dir(FOLDER_BASE / DBB)

# build lhs
cosmo_slics = create_lhs(n=NUMBER_OF_COSMO)

# add future values
cosmo_slics["zmax"] = None
cosmo_slics["lmax"] = None
cosmo_slics["final_As"] = None
cosmo_slics["final_sig8"] = None

# save
cosmo_slics.to_csv(FOLDER_RESULTS / f"lhs_{NUMBER_OF_COSMO}.csv", index=False)
