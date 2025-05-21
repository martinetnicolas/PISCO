import warnings
from pathlib import Path
from sys import argv

from astropy import io, table, units

warnings.simplefilter("ignore", category=units.UnitsWarning)


# Read parameters
CATALOG_DIR = Path(argv[1])
IMAGES_DIR = Path(argv[2])
simu_name = Path(argv[3])
index = int(argv[4])
nb_max = int(argv[5])
assoc = int(argv[6])

# Read sextractor catalog results
fits_file = CATALOG_DIR / simu_name
catalog = table.Table.read(fits_file, format="fits")

# Save vignet for the first images
if index < nb_max:
    vignet_data = catalog["vignet"]
    hdu = io.fits.PrimaryHDU(vignet_data)
    hdul = io.fits.HDUList([hdu])
    hdul.writeto(IMAGES_DIR / f"vignettes_{simu_name}", overwrite=True)

# Delete multidimensional columns
multidim_columns = [col for col in catalog.colnames if len(catalog[col].shape) > 1]
for col in multidim_columns:
    catalog.remove_column(col)

catalog.to_pandas().to_csv(CATALOG_DIR / f"{simu_name.stem}.csv", index=False)
fits_file.unlink()  # delete fits file
