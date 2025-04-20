import tensorflow_datasets as tfds
from astropy import io
from jax import numpy
from pandas import DataFrame, read_csv
from tqdm import tqdm
from utils import DIR_DATASETS, DIRS, RED, RESET, extract_gal_params, extract_image

# USER DATA ---------------------------------------------------------------------------------
VERSION = "1.0.1"
SAMPLING = 0.6
NB_CHUNKS = 8

SPLIT = "dataset"
DIR_VERSION = DIRS["pisco_euclid"] / VERSION
STYLE_DB = "pisco_euclid"
DIR_SE = DIRS["pisco_euclid_se"] / VERSION
DIR_SE.mkdir(parents=True, exist_ok=True)
DIR_SE.chmod(0o1770)


def chunk_data(version, nb_chunks, style, split, nb_total_elements, chunk_size, chunk_idx):
    # chunk data
    start = chunk_idx * chunk_size
    end = min((chunk_idx + 1) * chunk_size, nb_total_elements)
    global_indices = range(start, end)

    # load only the slice of data for this chunk
    chunk_split = f"{split}[{start}:{end}]"
    dataset_chunk = tfds.load(f"{style}:{version}", split=chunk_split, data_dir=DIR_DATASETS)
    print(f"{RED}chunk {chunk_idx + 1}/{nb_chunks} from {start} to {end - 1}{RESET}")

    # extract image data and galaxy parameters
    images = numpy.stack(list(dataset_chunk.map(extract_image).as_numpy_iterator()))
    gal_params = numpy.stack(list(dataset_chunk.map(extract_gal_params).as_numpy_iterator()))

    return global_indices, images, gal_params


def write_fits(path, data, sampling):
    header = io.fits.Header()
    header["SAMPLING"] = sampling
    hdu = io.fits.PrimaryHDU(data=data.astype(numpy.float32), header=header)
    hdulist = io.fits.HDUList([hdu])
    hdulist.writeto(path, overwrite=True)
    hdulist.close()


def fits_and_csv(sampling, dir_se, columns, images, gal_params, local_index, global_index):
    # save image as FITS
    path_fits = dir_se / f"{global_index + 1:06d}_simu.fits"
    write_fits(path=path_fits, data=images[local_index], sampling=sampling)

    # save galaxy parameters per image
    df = DataFrame(gal_params[local_index].T, columns=columns)
    path_csv = dir_se / f"{global_index + 1:06d}_true_galaxies_params.csv"
    df.to_csv(path_or_buf=path_csv, index=False)

    # save positions (x, y) only for SE assoc
    path_assoc = dir_se / f"{global_index + 1:06d}_simu_assoc.txt"
    df[["x", "y"]].to_csv(path_or_buf=path_assoc, sep=" ", index=False, header=False)


# CREATE DB ---------------------------------------------------------------------------------
if __name__ == "__main__":
    # Load the full dataset metadata
    dataset = tfds.load(f"{STYLE_DB}:{VERSION}", split=f"{SPLIT}[:100%]", data_dir=DIR_DATASETS)
    NB_TOTAL_ELEMENTS = dataset.cardinality().numpy()

    # Compute the number of elements per chunk
    CHUNK_SIZE = int(NB_TOTAL_ELEMENTS // NB_CHUNKS + int(NB_TOTAL_ELEMENTS % NB_CHUNKS != 0))

    # Read galaxy parameter column names
    columns = read_csv(DIR_VERSION / "galaxies_parameters_true.csv").columns

    # Loop through each chunk
    for chunk_idx in range(NB_CHUNKS):
        global_indices, images, gal_params = chunk_data(
            VERSION, NB_CHUNKS, STYLE_DB, SPLIT, NB_TOTAL_ELEMENTS, CHUNK_SIZE, chunk_idx
        )

        # Loop over all samples in the chunk
        for local_index, global_index in enumerate(tqdm(global_indices, desc=f"ch{chunk_idx + 1}")):
            fits_and_csv(SAMPLING, DIR_SE, columns, images, gal_params, local_index, global_index)

        del global_indices, images, gal_params

    print(f"{RED}FINITO{RESET}")

# TODO save one by one instead of chunks!
