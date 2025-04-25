import warnings
from pathlib import Path
from sys import argv

import numpy
import pandas
from astropy import units
from astropy.io import fits

warnings.simplefilter("ignore", category=units.UnitsWarning)


# Read parameters
CATALOG_DIR = Path(argv[1])
simu_name = Path(argv[2])

# Read sextractor catalog results
fits_file = CATALOG_DIR / f"{simu_name}.cat"
with fits.open(fits_file) as hdul:
    objects_data = hdul[2].data

excluded_column = "VECTOR_ASSOC"
columns_to_keep = [col for col in objects_data.columns.names if col != excluded_column]

catalog = pandas.DataFrame(
    {col: (objects_data[col]).byteswap().newbyteorder() for col in columns_to_keep}
)

if excluded_column in objects_data.columns.names:
    # Add 2D vectors after
    catalog["assoc1"] = objects_data[excluded_column][:, 0].astype(numpy.float32)
    catalog["assoc2"] = objects_data[excluded_column][:, 1].astype(numpy.float32)

# Save final catalog
catalog.to_csv(CATALOG_DIR / f"{simu_name}.csv", index=False)
fits_file.unlink()
