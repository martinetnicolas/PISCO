import sys
import warnings

import astropy.units as u
from astropy.table import Table

warnings.simplefilter("ignore", category=u.UnitsWarning)

# read parameters
catalog_file = sys.argv[1]
output_csv = sys.argv[2]
catalog = Table.read(catalog_file, format="fits")

# keep wanted parameters
columns = [
    "x",
    "y",
    "ra",
    "dec",
    "radius",
    "ratio",
    "angle",
    "e1",
    "e2",
    "sersic",
    "mag_EUCLID_LIKE",
]
wanted_columns = [col for col in columns if col in catalog.colnames]

# save as csv
catalog[wanted_columns].to_pandas().to_csv(output_csv, index=False)
