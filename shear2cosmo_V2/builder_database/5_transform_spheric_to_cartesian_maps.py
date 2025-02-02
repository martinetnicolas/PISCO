from pathlib import Path

import healpy
import numpy
from matplotlib import pyplot
from matplotlib.colors import TwoSlopeNorm
from utils import create_dir, generate_subpatch_central_points, get_flat_projection

# --------------------------------------------------------------------------------------------------


DBB = "DBB_KMEANS_25"
# DBB = "DBB_DENSITY_25"

NUMBER_OF_COSMOS = 25
NB_OF_SEEDS = 4
PLOT_STEP = 500

FOLDER_BASE = Path("/net/GECO/pisco/SHEAR2COSMO")
FOLDER_RESULTS = FOLDER_BASE / DBB
FOLDER_SPHERICAL = FOLDER_RESULTS / "spherical"
FOLDER_CARTESIAN = create_dir(FOLDER_RESULTS / "cartesian")
FOLDER_PLOTS = create_dir(FOLDER_RESULTS / "plots")


# --------------------------------------------------------------------------------------------------


for cosmo in range(NUMBER_OF_COSMOS):
    print("cosmo", cosmo)
    for seed in range(NB_OF_SEEDS):
        print("seed", seed)
        # Load GLASS spherical map [nested ordering, NOT ring]
        sph_map = numpy.load(FOLDER_SPHERICAL / f"id{cosmo:02}_seed{seed}.npy")

        # Define properties of the final flat map computing the effective flat map resolution given
        # side_deg and map_res_sph
        """ Note: Since the number of total pixels per side in the flat map is an integer,
        the effective flat map resolution will be slightly different from the spherical one! """
        sph_nside = healpy.pixelfunc.get_nside(sph_map)
        map_res_sph = healpy.pixelfunc.nside2resol(sph_nside, arcmin=True)
        side_deg = 10  # map side in deg
        nside = 128  # int(side_deg / (map_res_sph / 60))  # number of pixels per map side
        flat_nside = nside
        map_res = 60 * side_deg / nside  # map side in arcmins

        """
        Note: The spherical splitting into flat sub-patches is defined only by side_deg
        and not by sph_nside. Of course, higher the spherical map resolution (sph_nside)
        and higher the accuracy of the projected flat maps!
        """

        # Divide full-sky into N flat sub-patches where projection effects
        # are limited (lat +/- 30 deg)
        central_points = generate_subpatch_central_points(side_deg)  # central points
        map_cuts = numpy.empty((len(central_points), flat_nside, flat_nside))

        # centre sub-patch coords as lon (x-axis), lat (y-axis) in degrees
        # Rebinning directly during the projection (best)
        for ind, square_subpatch_center_coords in enumerate(central_points):
            map_cuts[ind] = get_flat_projection(
                "Cart",
                sph_map,
                [square_subpatch_center_coords[0], square_subpatch_center_coords[1]],
                side_deg,
                flat_nside,
            )

            # Remove decimals from pointing-coords
            square_subpatch_center_coords = (
                int(square_subpatch_center_coords[0]),
                int(square_subpatch_center_coords[1]),
            )

            # Save projected map
            sim_fname = f"id{cosmo:02}_seed{seed}_part{ind:03}.npy"
            sim_fname_plot = f"id{cosmo:02}_seed{seed}_part{ind:03}.png"
            numpy.save(FOLDER_CARTESIAN / sim_fname, map_cuts[ind])

            # Save projected map plot
            if ind % PLOT_STEP == 0:
                fig, ax = pyplot.subplots(1, 1, figsize=(5, 5), squeeze=False)
                ax[0, 0].set_title(
                    f"Projected GLASS map \n [center {square_subpatch_center_coords} deg {map_cuts[ind].shape[0]}x{map_cuts[ind].shape[1]} pxs {side_deg}x{side_deg} sq.deg]"
                )
                norm = TwoSlopeNorm(
                    vmin=(map_cuts[ind]).min(), vcenter=0, vmax=(map_cuts[ind]).max()
                )
                mappable = ax[0, 0].imshow(map_cuts[ind], cmap="bwr", origin="lower", norm=norm)
                cbar = fig.colorbar(mappable, ax=ax[0, 0], orientation="vertical", label="$\kappa$")
                norm = TwoSlopeNorm(
                    vmin=(map_cuts[ind] / numpy.std(map_cuts[ind])).min(),
                    vcenter=0,
                    vmax=(map_cuts[ind] / numpy.std(map_cuts[ind])).max(),
                )
                mappable = ax[0, 0].imshow(
                    map_cuts[ind] / numpy.std(map_cuts[ind]), cmap="bwr", origin="lower", norm=norm
                )
                cbar = fig.colorbar(
                    mappable, ax=ax[0, 0], orientation="horizontal", label="$\kappa / \sigma$"
                )
                fig.savefig(FOLDER_PLOTS / sim_fname_plot, dpi=300)
                pyplot.close(fig)
