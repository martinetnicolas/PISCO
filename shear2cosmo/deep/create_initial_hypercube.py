from pathlib import Path

import numpy
import pandas
from scipy.stats import qmc
from pyDOE2 import lhs

# --------------------------------------------------------------------------------------------------


NUMBER_OF_COSMO = 25
DBB = "DBB_OPT_COSMOSLICS"
lhs_vanilla = 1
lhs_maximin = 0
seed_list = [0,1,2,3]

# --------------------------------------------------------------------------------------------------

for seed in seed_list:
    def create_lhs(n,seed=seed):
        parameters = ["Om", "S8", "h", "w0"]
        parameters_intervals = numpy.array([[0.1, 0.55], [0.6, 0.9], [0.6, 0.82], [-2.0, -0.5]])
        #parameters_intervals = numpy.array([[0.2, 0.45], [0.65, 0.85], [0.65, 0.77], [-1.25, -0.75]])
        hypercube = qmc.LatinHypercube(
            d=len(parameters),
            scramble=True,
            optimization=None,
            seed=seed,
        )
        samples = hypercube.random(n=n)
        scaled_samples = qmc.scale(samples, parameters_intervals[:, 0], parameters_intervals[:, 1])
        scaled_samples = pandas.DataFrame(data=scaled_samples, columns=parameters)

        scaled_samples["sig8"] = scaled_samples["S8"] * numpy.sqrt(0.3 / scaled_samples["Om"])
        Ob = 0.0473
        scaled_samples["Oc"] = scaled_samples["Om"] - Ob

        return scaled_samples

    def create_lhs_maximin(n, seed=seed):
        parameters = ["Om", "S8", "h", "w0"]
        parameters_intervals = numpy.array([[0.1, 0.55], [0.6, 0.9], [0.6, 0.82], [-2.0, -0.5]])
        hypercube = lhs(len(parameters), samples=n, criterion='maximin', random_state=seed)
        scaled_samples = qmc.scale(hypercube, parameters_intervals[:, 0], parameters_intervals[:, 1])
        scaled_samples_df = pandas.DataFrame(data=scaled_samples, columns=parameters)

        scaled_samples_df["sig8"] = scaled_samples_df["S8"] * numpy.sqrt(0.3 / scaled_samples_df["Om"])
        Ob = 0.0473
        scaled_samples_df["Oc"] = scaled_samples_df["Om"] - Ob

        return scaled_samples_df


    def create_dir(directory):
        directory.mkdir(parents=False, exist_ok=True)
        return directory


    FOLDER_BASE = Path("/net/GECO/pisco/SHEAR2COSMO")
    #FOLDER_RESULTS = create_dir(FOLDER_BASE / DBB)
    FOLDER_RESULTS = Path('/net/GECO/pisco/SHEAR2COSMO/Leo_COSMO-SLICS/LHS_comparison/lhs_25')
    # build lhs
    if lhs_vanilla :
        cosmo_slics = create_lhs(n=NUMBER_OF_COSMO)
    if lhs_maximin :
        cosmo_slics = create_lhs_maximin(n=NUMBER_OF_COSMO)

    # add future values
    cosmo_slics["zmax"] = None
    cosmo_slics["lmax"] = None
    cosmo_slics["final_As"] = None
    cosmo_slics["final_sig8"] = None

    # save
    cosmo_slics.to_csv(FOLDER_RESULTS / f"lhs_{NUMBER_OF_COSMO}_{seed}.csv", index=False)