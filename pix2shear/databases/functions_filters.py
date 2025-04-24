import galsim
import numpy
from astropy import coordinates, table, units
from utils import DIRS


def get_field_cosmos(sample):
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


# check overlapping with RA & DEC
def filter_overlap_ra_dec(radius=0.1):
    fields_23 = get_field_cosmos(sample="23.5")
    fields_25 = get_field_cosmos(sample="25.2")

    sep2d = angular_separation(fields_23, fields_25)

    max_sep = radius * units.arcsec
    matches_mask = sep2d < max_sep

    is_unmatched = ~matches_mask

    return is_unmatched


# check overlapping with IDENT
def filter_overlap_ident():
    fields_23 = get_field_cosmos(sample="23.5")
    fields_25 = get_field_cosmos(sample="25.2")

    IDENT_23 = numpy.array(fields_23["IDENT"]).astype(numpy.int32)
    IDENT_25 = numpy.array(fields_25["IDENT"]).astype(numpy.int32)

    mask = ~numpy.isin(IDENT_23, IDENT_25)
    non_overlap_indices_ident = numpy.where(mask)[0]

    return non_overlap_indices_ident.astype(numpy.int32)


# so-called bad fits fro cosmos cat itself
def filter_bad_fit(sample, exptime, exclusion_level="marginal"):
    cosmos_cat = galsim.COSMOSCatalog(
        sample=sample, dir=DIRS["cosmos"], exclusion_level=exclusion_level, exptime=exptime
    )

    non_filtered_param = cosmos_cat.param_cat
    filtered_indices = cosmos_cat.orig_index

    mask_viable_sersic = non_filtered_param["viable_sersic"] != 0
    mask_fit_status = non_filtered_param["fit_status"][:, 4] == 1

    total_nb_elements = non_filtered_param.shape[0]
    mask_exclusion_level = numpy.zeros(total_nb_elements, dtype=bool)
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


# take out too noisy galaxy to match better with pisco noise database
def filter_noise(sample, non_filtered_param, max_var, max_mean):
    field = get_field_cosmos(sample)

    variance = field["NOISE_VARIANCE"]
    mask_variance = variance < max_var
    non_filtered_param = add_entry(non_filtered_param, array=variance, new_entry="variance")

    noise_mean = field["NOISE_MEAN"]
    mask_mean = noise_mean < max_mean
    non_filtered_param = add_entry(non_filtered_param, array=noise_mean, new_entry="noise_mean")

    mask_noise = mask_variance & mask_mean

    return mask_noise, non_filtered_param


# filter to match pisco work
def filter_pisco(non_filtered_param):
    sersic_cosmos = non_filtered_param["sersicfit"][:, 2]
    mask_pisco = (sersic_cosmos >= 0.3) & (sersic_cosmos < 6.0)

    mag = non_filtered_param["mag_auto"]
    mask_pisco &= (mag >= 20.0) & (mag <= 24.5)

    r = non_filtered_param["sersicfit"][:, 1]
    q = non_filtered_param["sersicfit"][:, 3]
    ellipticity = (1 - q) / (1 + q)
    theta = non_filtered_param["sersicfit"][:, 7]

    hlr = 0.03 * numpy.sqrt(q) * r
    mask_pisco &= (hlr >= 0.05) & (hlr <= 1.5)
    non_filtered_param = add_entry(non_filtered_param, array=hlr, new_entry="hlr_pisco")

    max_e = 0.7
    e1 = ellipticity * numpy.cos(2 * theta)
    mask_pisco &= (e1 >= -max_e) & (e1 <= max_e)
    non_filtered_param = add_entry(non_filtered_param, array=e1, new_entry="e1")

    e2 = ellipticity * numpy.sin(2 * theta)
    mask_pisco &= (e2 >= -max_e) & (e2 <= max_e)
    non_filtered_param = add_entry(non_filtered_param, array=e2, new_entry="e2")

    return mask_pisco, non_filtered_param
