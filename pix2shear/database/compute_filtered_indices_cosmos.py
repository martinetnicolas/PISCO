import galsim
import numpy
from astropy import coordinates, table, units
from matplotlib import pyplot
from pandas import DataFrame, concat
from tqdm import tqdm
from utils import search_folder

USE_REAL = True
EXPTIME = 3 * 565


def sky_coord(fields_23, fields_25):
    coords23 = coordinates.SkyCoord(
        ra=fields_23["RA"] * units.deg, dec=fields_23["DEC"] * units.deg
    )
    coords25 = coordinates.SkyCoord(
        ra=fields_25["RA"] * units.deg, dec=fields_25["DEC"] * units.deg
    )

    _, sep2d, _ = coords23.match_to_catalog_sky(coords25, nthneighbor=1)

    return sep2d


def compute_indices(sep2d, radius):
    max_sep = radius * units.arcsec
    matches_mask = sep2d < max_sep

    is_unmatched = ~matches_mask
    indices = numpy.where(is_unmatched)[0]

    return indices


def get_fields(dir, sample):
    return table.Table.read(dir / f"real_galaxy_catalog_{sample}.fits")


def filter_fit(sample, dir, use_real, exptime):
    filtered_cosmos_cat = galsim.COSMOSCatalog(
        sample=sample,
        dir=dir,
        use_real=use_real,
        exclusion_level="marginal",
        exptime=exptime,
    )

    filtered_indices = filtered_cosmos_cat.orig_index
    filtered_params_cat = filtered_cosmos_cat.param_cat[filtered_indices]

    viable_sersic_mask = filtered_params_cat["viable_sersic"] != 0
    fit_status_mask = filtered_params_cat["fit_status"][:, 4] == 1

    final_mask = viable_sersic_mask & fit_status_mask

    return filtered_indices[final_mask], filtered_params_cat[final_mask]


def add_entry_structured_array(params_cat, hlr):
    new_dtype = params_cat.dtype.descr + [("hlr_pisco", ">f8")]

    extended = numpy.empty(params_cat.shape, dtype=new_dtype)

    for name in params_cat.dtype.names:
        extended[name] = params_cat[name]

    extended["hlr_pisco"] = hlr

    return extended


def filter_pisco(indices, params_cat, mask_mag=True):
    sersic = params_cat["sersicfit"][:, 2]
    mask = (sersic >= 0.3) & (sersic <= 6.2)

    if mask_mag:
        mag = params_cat["mag_auto"]
        mask = mask & (mag >= 20) & (mag <= 24)

    r = params_cat["sersicfit"][:, 1]
    q = params_cat["sersicfit"][:, 3]
    hlr = 0.03 * numpy.sqrt(q) * r
    mask = mask & (hlr >= 0.05) & (hlr <= 1.2)

    params_cat = add_entry_structured_array(params_cat, hlr)

    return indices[mask], params_cat[mask]


def histos(hypercube, dir_plots, name, bins=50):
    font_style = {"fontsize": 10, "fontweight": "bold", "family": "serif", "color": "darkgreen"}

    fig, axs = pyplot.subplots(1, 3, figsize=(12, 4))
    axs = axs.ravel()

    for i, (key, values) in enumerate(hypercube.items()):
        color = "skyblue" if i % 2 == 0 else "orange"
        axs[i].hist(values, bins=bins, alpha=0.7, color=color)
        axs[i].set_title(key, fontdict=font_style)
        axs[i].set_xlabel(key)
        axs[i].set_ylabel("Count")

    fig.suptitle(f"SAMPLE {name} => {hypercube.shape[0]}")
    fig.tight_layout()
    fig.savefig(dir_plots / f"{name}.pdf", dpi=300)


if __name__ == "__main__":
    DIR_DATASETS = search_folder(name="DATASETS")
    DIRS = {"images": DIR_DATASETS / "images" / "cosmos", "cosmos": DIR_DATASETS / "cosmos"}
    for path in DIRS.values():
        path.mkdir(parents=True, exist_ok=True)

    fields_23 = get_fields(DIRS["cosmos"], sample="23.5")
    fields_25 = get_fields(DIRS["cosmos"], sample="25.2")

    sep2d = sky_coord(fields_23, fields_25)
    non_overlapping_indices_23 = compute_indices(sep2d, radius=0.1)

    all_filtered_df = []
    for i, SAMPLE in enumerate(tqdm(["23.5", "25.2"], desc="Processing samples"), start=1):
        filtered_indices, filtered_params = filter_fit(SAMPLE, DIRS["cosmos"], USE_REAL, EXPTIME)

        filtered_indices, filtered_params = filter_pisco(filtered_indices, filtered_params)

        if SAMPLE == "23.5":
            common_indices = numpy.intersect1d(non_overlapping_indices_23, filtered_indices)
            keep_mask = numpy.isin(filtered_indices, common_indices)

            filtered_indices = filtered_indices[keep_mask]
            filtered_params = filtered_params[keep_mask]

        filtered_params = filtered_params.byteswap().newbyteorder()
        df = DataFrame(
            {
                "indices": filtered_indices,
                "sample": SAMPLE,
                "mag_auto": filtered_params["mag_auto"],
                "hlr": filtered_params["hlr_pisco"],
                "sersic": filtered_params["sersicfit"][:, 2],
            }
        )
        all_filtered_df.append(df)

        histos(df[["mag_auto", "hlr", "sersic"]], DIRS["images"], f"histos_{SAMPLE}")

    final_df = concat(all_filtered_df, ignore_index=True)
    final_df.to_csv(DIRS["cosmos"] / "filtered_params.csv", index=False)

    print("FINITO")
