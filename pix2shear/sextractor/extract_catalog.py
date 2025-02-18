import warnings
from pathlib import Path
from sys import argv

from astropy import io, table, units

warnings.simplefilter("ignore", category=units.UnitsWarning)


# Read parameters
CATALOG_DIR = Path(argv[1])
IMAGES_DIR = Path(argv[2])
simu_name = str(argv[3])
counter = int(argv[4])
nb_max = int(argv[5])

# Read sextractor catalog results
fits_file = CATALOG_DIR / f"catalog_{simu_name}.fits"
catalog = table.Table.read(fits_file, format="fits")
fits_file.unlink()  # delete fits file

# Save vignet for the first images
if counter < nb_max:
    vignet_data = catalog["vignet"]
    hdu = io.fits.PrimaryHDU(vignet_data)
    hdul = io.fits.HDUList([hdu])
    hdul.writeto(IMAGES_DIR / f"vignettes_{simu_name}.fits", overwrite=True)

# Save scalar outputs
multidim_columns = [col for col in catalog.colnames if len(catalog[col].shape) > 1]
for col in multidim_columns:
    catalog.remove_column(col)  # remove non scalar outputs
catalog.to_pandas().to_csv(CATALOG_DIR / f"catalog_{simu_name}.csv", index=False)
