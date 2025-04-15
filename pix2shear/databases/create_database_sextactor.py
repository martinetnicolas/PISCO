import tensorflow_datasets as tfds
from jax import numpy
from pandas import DataFrame, read_csv
from tqdm import tqdm
from utils import extract_gal_params, extract_image, search_folder, write_fits

VERSION = "2.2.0"
N_CHUNKS = 8

SAMPLING = 0.6
STYLE = "pisco_euclid"
SPLIT = "dataset"
DIR_DATASETS = search_folder(name="DATASETS")
DIRS = {
    "version": DIR_DATASETS / "pisco_euclid" / VERSION,
    "sextractor": DIR_DATASETS / "pisco_euclid" / f"{VERSION}_sextractor",
}
DIRS["sextractor"].mkdir(parents=True, exist_ok=True)

dataset = tfds.load(f"{STYLE}:{VERSION}", split=f"{SPLIT}[:100%]", data_dir=DIR_DATASETS)
TOTAL_ELEMENTS = dataset.cardinality().numpy()
print("\033[38;5;196mTOTAL_ELEMENTS\033[0m", TOTAL_ELEMENTS)

CHUNK_SIZE = int(TOTAL_ELEMENTS // N_CHUNKS + int(TOTAL_ELEMENTS % N_CHUNKS != 0))
print("CHUNK_SIZE", CHUNK_SIZE)

columns = read_csv(DIRS["version"] / "galaxies_parameters_true.csv").columns

for chunk_id in range(N_CHUNKS):
    start = chunk_id * CHUNK_SIZE
    end = min((chunk_id + 1) * CHUNK_SIZE, TOTAL_ELEMENTS)

    chunk_split = f"{SPLIT}[{start}:{end}]"
    dataset_chunk = tfds.load(f"{STYLE}:{VERSION}", split=chunk_split, data_dir=DIR_DATASETS)
    print(f"\033[38;5;33mchunk {chunk_id + 1}/{N_CHUNKS} from {start} to {end - 1}\033[0m")

    images = numpy.stack(list(dataset_chunk.map(extract_image).as_numpy_iterator()))
    gal_params = numpy.stack(list(dataset_chunk.map(extract_gal_params).as_numpy_iterator()))

    for local_idx, sample in enumerate(tqdm(range(start, end), desc=f"Chunk {chunk_id + 1}")):
        path_fits = DIRS["sextractor"] / f"{sample + 1:06d}_simu.fits"
        write_fits(path=path_fits, data=images[local_idx], sampling=SAMPLING)

        df = DataFrame(gal_params[local_idx].T, columns=columns)

        path_csv = DIRS["sextractor"] / f"{sample + 1:06d}_true_galaxies_params.csv"
        df.to_csv(path_or_buf=path_csv, index=False)

        path_assoc = DIRS["sextractor"] / f"{sample + 1:06d}_simu_assoc.txt"
        df[["x", "y"]].to_csv(path_or_buf=path_assoc, sep=" ", index=False, header=False)

# lock permissions
DIRS["sextractor"].chmod(0o1770)
