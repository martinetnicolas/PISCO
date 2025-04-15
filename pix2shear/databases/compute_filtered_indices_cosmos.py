import galsim
import numpy
from astropy import coordinates, table, units
from matplotlib import pyplot
from pandas import DataFrame, concat
from tqdm import tqdm
from utils import DIRS

EXPTIME = 3 * 565
MAX_VAR = 1.2e-5
MAX_MEAN = 1.0e-3


def get_field(sample):
    return table.Table.read(DIRS["cosmos"] / f"real_galaxy_catalog_{sample}.fits")


def angular_separation(fields_23, fields_25):
    coords23 = coordinates.SkyCoord(
        ra=fields_23["RA"] * units.deg, dec=fields_23["DEC"] * units.deg
    )
    coords25 = coordinates.SkyCoord(
        ra=fields_25["RA"] * units.deg, dec=fields_25["DEC"] * units.deg
    )

    _, sep2d, _ = coords23.match_to_catalog_sky(coords25, nthneighbor=1)

    return sep2d


def filter_overlap(radius):
    fields_23 = get_field(sample="23.5")
    fields_25 = get_field(sample="25.2")

    sep2d = angular_separation(fields_23, fields_25)

    max_sep = radius * units.arcsec
    matches_mask = sep2d < max_sep

    is_unmatched = ~matches_mask
    return is_unmatched


def filter_bad_fit(sample, exptime, exclusion_level="marginal"):
    cosmos_cat = galsim.COSMOSCatalog(
        sample=sample, dir=DIRS["cosmos"], exclusion_level=exclusion_level, exptime=exptime
    )

    non_filtered_param = cosmos_cat.param_cat
    filtered_indices = cosmos_cat.orig_index

    mask_viable_sersic = non_filtered_param["viable_sersic"] != 0
    mask_fit_status = non_filtered_param["fit_status"][:, 4] == 1

    TOTAL_NB_ELEMENTS = non_filtered_param.shape[0]
    mask_exclusion_level = numpy.zeros(TOTAL_NB_ELEMENTS, dtype=bool)
    mask_exclusion_level[filtered_indices] = True

    mask_bad_fit = mask_viable_sersic & mask_fit_status & mask_exclusion_level
    return mask_bad_fit, non_filtered_param


def add_entry(params, array, new_entry):
    new_dtype = params.dtype.descr + [(new_entry, ">f8")]

    extended = numpy.empty(params.shape, dtype=new_dtype)

    for name in params.dtype.names:
        extended[name] = params[name]

    extended[new_entry] = array

    return extended


def filter_noise(sample, non_filtered_param, max_var, max_mean):
    field = get_field(sample)

    variance = field["NOISE_VARIANCE"]
    mask_variance = variance < max_var
    non_filtered_param = add_entry(non_filtered_param, array=variance, new_entry="variance")

    noise_mean = field["NOISE_MEAN"]
    mask_mean = noise_mean < max_mean
    non_filtered_param = add_entry(non_filtered_param, array=noise_mean, new_entry="noise_mean")

    mask_noise = mask_variance & mask_mean

    return mask_noise, non_filtered_param


def filter_pisco(non_filtered_param):
    sersic = non_filtered_param["sersicfit"][:, 2]
    mask_sersic = (sersic >= 0.3) & (sersic <= 6.2)

    mag = non_filtered_param["mag_auto"]
    mask_mag = (mag >= 20.5) & (mag <= 24.5)

    r = non_filtered_param["sersicfit"][:, 1]
    q = non_filtered_param["sersicfit"][:, 3]
    ellipticity = (1 - q) / (1 + q)
    theta = non_filtered_param["sersicfit"][:, 7]

    hlr = 0.03 * numpy.sqrt(q) * r
    mask_hlr = (hlr >= 0.05) & (hlr <= 1.2)
    non_filtered_param = add_entry(non_filtered_param, array=hlr, new_entry="hlr_pisco")

    max_e = 0.7
    e1 = ellipticity * numpy.cos(2 * theta)
    mask_e1 = (e1 >= -max_e) & (e1 <= max_e)
    non_filtered_param = add_entry(non_filtered_param, array=e1, new_entry="e1")

    e2 = ellipticity * numpy.sin(2 * theta)
    mask_e2 = (e2 >= -max_e) & (e2 <= max_e)
    non_filtered_param = add_entry(non_filtered_param, array=e2, new_entry="e2")

    mask_pisco = mask_sersic & mask_mag & mask_hlr & mask_e1 & mask_e2

    return mask_pisco, non_filtered_param


def histos(hypercube, name, bins=50):
    font_style = {"fontsize": 10, "fontweight": "bold", "family": "serif", "color": "darkgreen"}

    fig, axs = pyplot.subplots(2, 4, figsize=(12, 6))
    axs = axs.ravel()

    for i, (key, values) in enumerate(hypercube.items()):
        color = "skyblue" if i % 2 == 0 else "orange"
        axs[i].hist(values, bins=bins, alpha=0.7, color=color)
        axs[i].set_title(key, fontdict=font_style)
        axs[i].set_ylabel("Count")

    fig.suptitle(f"SAMPLE {name} => {hypercube.shape[0]}")
    fig.tight_layout()
    fig.savefig(DIRS["images_cosmos"] / f"{name}.pdf", dpi=300)


if __name__ == "__main__":
    mask_non_overlapping_23 = filter_overlap(radius=0.1)

    all_filtered_df = []
    for i, SAMPLE in enumerate(tqdm(["23.5", "25.2"], desc="Processing samples"), start=1):
        mask_bad_fit, non_filtered_param = filter_bad_fit(SAMPLE, EXPTIME)
        mask_pisco, non_filtered_param = filter_pisco(non_filtered_param)
        mask_noise, non_filtered_param = filter_noise(SAMPLE, non_filtered_param, MAX_VAR, MAX_MEAN)

        mask_final = mask_noise & mask_bad_fit & mask_pisco
        if SAMPLE == "23.5":
            mask_final &= mask_non_overlapping_23

        filtered_param = non_filtered_param[mask_final].byteswap().newbyteorder()
        indices = numpy.where(mask_final)[0]

        df = DataFrame(
            {
                "indices": indices,
                "sample": SAMPLE,
                "mag": filtered_param["mag_auto"],
                "hlr": filtered_param["hlr_pisco"],
                "sersic": filtered_param["sersicfit"][:, 2],
                "e1": filtered_param["e1"],
                "e2": filtered_param["e2"],
                "variance": filtered_param["variance"],
                "noise_mean": filtered_param["noise_mean"],
            }
        )
        histos(df.drop(columns=["indices"], inplace=False), f"histos_{SAMPLE}")

        all_filtered_df.append(df)

    final_df = concat(all_filtered_df, ignore_index=True)
    histos(final_df.drop(columns=["indices"], inplace=False), "histos_ALL")
    final_df.to_csv(DIRS["cosmos"] / "filtered_params.csv", index=False)
    print("FINITO")
