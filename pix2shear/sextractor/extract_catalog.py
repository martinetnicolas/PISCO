import warnings
from pathlib import Path
from sys import argv

from astropy import io, table, units

warnings.simplefilter("ignore", category=units.UnitsWarning)


# read parameters
CATALOG_DIR = Path(argv[1])
IMAGES_DIR = Path(argv[2])
simu_name = argv[3]
counter = argv[4]
nb_max = argv[5]

# read sextractor catalog results
catalog = table.Table.read(CATALOG_DIR / f"catalog_{simu_name}.fits", format="fits")

# save vignettes
if counter < nb_max:
    vignet_data = catalog["vignet"]
    hdu = io.fits.PrimaryHDU(vignet_data)
    hdul = io.fits.HDUList([hdu])
    hdul.writeto(IMAGES_DIR / f"vignettes_{simu_name}.fits", overwrite=True)

# save scalar outputs
multidim_columns = [col for col in catalog.colnames if len(catalog[col].shape) > 1]
for col in multidim_columns:
    catalog.remove_column(col)  # remove non scalar outputs
catalog.to_pandas().to_csv(CATALOG_DIR / f"catalog_{simu_name}.csv", index=False)
