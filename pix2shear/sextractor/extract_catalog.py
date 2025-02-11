import sys
import warnings

from astropy import io, table, units

warnings.simplefilter("ignore", category=units.UnitsWarning)

# read parameters
catalog_file = sys.argv[1]
output_csv = sys.argv[2]
catalog = table.Table.read(catalog_file, format="fits")

# save as csv
# print(catalog["vignet"])
if "vignet" in catalog.colnames:
    vignet_data = catalog["vignet"]

    # Print the type of "vignet" column
    print(f"Type of 'vignetCC' column: {type(vignet_data)}")

    # Print the shape of the entire vignet column
    print(f"Shape of 'vignetCC' column: {vignet_data.shape}")

    print(f"Type of 'vignetCC' column[0]: {(vignet_data[0])}")


# # Directory
# SIMU_VIGNET_DIR = VIGNET_DIR / name_simu
# SIMU_VIGNET_DIR.mkdir(parents=True, exist_ok=True)

# # Get 3D vignet data
# all_results = io.fits.open(CATALOG_DIR / f"catalog_{name_simu}.fits")
# bin_table_hdu = all_results[1]
# all_vignettes = bin_table_hdu.data["vignet"]

# print(f"Type of all_vignettes: {type(all_vignettes)}")
# print(f"Shape of all_vignettes: {all_vignettes.shape}")
# print(f"Type of all_vignettes[0]: {(all_vignettes[0])}")

# # Write results
# for i in range(all_vignettes.shape[0]):
#     hdu = io.fits.PrimaryHDU(all_vignettes[i, :, :])
#     hdul = io.fits.HDUList([hdu])
#     hdul.writeto(SIMU_VIGNET_DIR / f"vignette{i + 1:02d}.fits", overwrite=True)


# catalog.remove_column("vignet")
# catalog.to_pandas().to_csv(output_csv, index=False)
