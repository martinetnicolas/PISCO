import warnings
from pathlib import Path
from sys import argv

from astropy import io, table, units

warnings.simplefilter("ignore", category=units.UnitsWarning)

# read parameters
CATALOG_DIR = Path(argv[1])
catalog_file = argv[2]
output_csv = argv[3]

# read sextractor catalog results
catalog = table.Table.read(CATALOG_DIR / catalog_file, format="fits")

# save vignettes
vignet_data = catalog["vignet"]
hdu = io.fits.PrimaryHDU(vignet_data)
hdul = io.fits.HDUList([hdu])
hdul.writeto(CATALOG_DIR / "vignettes.fits", overwrite=True)

# save scalar outputs
catalog.remove_column("vignet")
catalog.to_pandas().to_csv(CATALOG_DIR / output_csv, index=False)
