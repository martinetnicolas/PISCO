import sys
from pathlib import Path

from astropy.io import fits

# Ongoing bash args
CATALOG_DIR = Path(sys.argv[1])
VIGNET_DIR = Path(sys.argv[2])
name_simu = Path(sys.argv[3])

# Directory
SIMU_VIGNET_DIR = VIGNET_DIR / name_simu
SIMU_VIGNET_DIR.mkdir(parents=True, exist_ok=True)

# Get 3D vignet data
all_results = fits.open(CATALOG_DIR / f"catalog_{name_simu}.fits")
bin_table_hdu = all_results[1]
all_vignettes = bin_table_hdu.data["Vignet"]

# Write results
for i in range(all_vignettes.shape[0]):
    hdu = fits.PrimaryHDU(all_vignettes[i, :, :])
    hdul = fits.HDUList([hdu])
    hdul.writeto(SIMU_VIGNET_DIR / f"vignette{i + 1:02d}.fits", overwrite=True)
