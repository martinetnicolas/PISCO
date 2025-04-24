import numpy
from functions_filters import (
    filter_bad_fit,
    filter_noise,
    # filter_overlap_ident,
    filter_overlap_ra_dec,
    filter_pisco,
)
from pandas import DataFrame, concat
from plots import histos_filters
from tqdm import tqdm
from utils import DIRS, RED, RESET

# USER DATA ----------------------------------------------------------------------------------------
EXPTIME = 3 * 565
MAX_VAR = 1.8e-5
MAX_MEAN = 14.0e-4
# SAMPLES = ["25.2"]
SAMPLES = ["23.5", "25.2"]


# FILTER INDICES -----------------------------------------------------------------------------------
if __name__ == "__main__":
    all_filtered_df = []
    for SAMPLE in tqdm(SAMPLES, desc="Processing samples"):
        # so-called bad fits fro cosmos cat itself
        mask_bad_fit, parameters = filter_bad_fit(SAMPLE, EXPTIME)

        # filter to match pisco work
        mask_pisco, parameters = filter_pisco(parameters)

        # take out too noisy galaxy to match better with pisco noise database
        mask_noise, parameters = filter_noise(SAMPLE, parameters, MAX_VAR, MAX_MEAN)

        # all masks together
        mask_final = mask_bad_fit & mask_pisco & mask_noise
        if SAMPLE == "23.5":
            # filter identical galaxy from cosmos 23.5 and 25.2 samples
            mask_non_overlapping_23 = filter_overlap_ra_dec()  # filter_overlap_ident()
            mask_final &= mask_non_overlapping_23

        # filtering
        filtered_parameters = parameters[mask_final]
        filtered_parameters = filtered_parameters.byteswap().newbyteorder()
        indices = numpy.where(mask_final)[0]

        # plot per sample
        filtered_df = DataFrame(
            {
                "indices": indices,
                "sample": SAMPLE,
                "mag": filtered_parameters["mag_auto"],
                "hlr": filtered_parameters["hlr_pisco"],
                "sersic": filtered_parameters["sersicfit"][:, 2],
                "e1": filtered_parameters["e1"],
                "e2": filtered_parameters["e2"],
                "variance": filtered_parameters["variance"],
                "noise_mean": filtered_parameters["noise_mean"],
            }
        )
        all_filtered_df.append(filtered_df)
        histos_filters(
            data=filtered_df.drop(columns=["indices"], inplace=False),
            dir=DIRS["cosmos_plots"] / f"{SAMPLE}_filtered_histos",
        )

    # both samples
    final_df = concat(all_filtered_df, ignore_index=True)
    final_df.to_csv(DIRS["cosmos"] / "filtered_params.csv", index=False)
    histos_filters(
        final_df.drop(columns=["indices"], inplace=False),
        DIRS["cosmos_plots"] / "filtered_histos_all",
    )

    print(f"{RED}FINITO{RESET}")
