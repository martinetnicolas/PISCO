from functools import partial

import glass.ext.camb
import healpy as hp
import matplotlib.pyplot as plt
import numpy


def create_dir(folder):
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
    return folder


def compute_matter_cls_with_timeout(sim_pars, lmax, ws, timeout, result_queue):
    try:
        matter_cls = glass.ext.camb.matter_cls(sim_pars, lmax, ws)
        result_queue.put(matter_cls)
    except Exception as e:
        result_queue.put(e)


# Cartesian/gnomonic projection from spherical map (about equivalent)
def get_flat_projection(proj_method, sph_map, center, side_deg, flat_nside):
    """

    Important reminder: lat:[-90, +90]  long:[0, 360] deg

    0th. Note: healpix pixels are qaudrilateral of SAME AREA but with varying elongation (https://iopscience.iop.org/article/10.1086/427976/pdf).
               The area of a general quadrilateral on a unitary sphere in spherical lat/long coords
                    is https://www.johndcook.com/blog/2023/02/21/sphere-grid-area/ (area given in sterads)
               For a fixed delta in lat/long, the area is consistent with a flat square close to the
                    equator (lat=0), while it is will be severely affected by the curvature thus reducing
                    towards the poles (lat=+/-90).
               Pixels towards the poles are more elongated in order to embed the same area.
               That is why, when fixing the same delta on lat and long, the number of pixels
                    contained into the given coord range reduces towards the poles.

    1st. Note: flat map has a pixel resolution approximated by side_deg / flat_nside.
               Depending on the projection method, the effective side_deg can slightly change thus
                changing the effective pixel resolution.
               'Cart' has a more accurate effective side_deg than 'Gnom'.


    2nd. Note: the cut from spherical to cartesian cannot be perfect since:
        1) the spherical resolution is an approximation of the flat resolution,
        2) the spherical pixel geometry (polygonal) is different wrt the flat pixel geometry (square)

    3rd. Note: The projection algorithm takes as inputs side_deg and flat_nside defining a physical
                    binning (arcmins) from which projecting the spherical pixels.
               side_deg defines the proper angular size of the final flat map and not the range of
                    spherical coordinates from which operate the projection!
               Given side_deg, the algorithm will spot the quadrilateral on the sphere centered on
                    the provided central point such that its area is equal to side_deg^2, and then
                    projecting it into a cartesian grid.
               If the ratio side_deg/flat_nside is lower wrt the spherical pixel size, the projected
                    map will contain replicated pixels!
               It is safer to select side_deg, flat_nside for which side_deg/flat_nside is equal to
                    the spherical pixel size.

    4th. Note: proj_Cart.projmap returns a Cartesian map in matricial convention:
                    (i, j) = (row, col) = (long, lat) = (ra, dec) where (0,0) element is the
                    lower-left-corner (e.g. projecting a 10x10 sq.deg map w/ centre in
                    long, lat = 5, 5 deg, map[0,0] will correspond to a 45 degrees rotation of
                    (long=10, lat=0)
               To match the visual convention (long-ra on y-axis, lat-dec on x-axis) one should
                    transpose the map

    """

    """ Cartesian grid must be clockwise/counterclockwise rotated by 45deg (in this case, we opted
    for clockwise rotation) in order to be aligned with the Healpix grid """
    psi = +45  # positive psi = clockwise
    center = (center[0], center[1], psi)

    if proj_method == "Cart":
        angle = side_deg / 2
        angular_size = [-angle, angle]  # effective FoV of projected map
        proj_Cart = hp.projector.CartesianProj(
            rot=center,
            coord=["G"],
            xsize=flat_nside,
            ysize=flat_nside,
            lonra=angular_size,
            latra=angular_size,
        )
        map_flat = proj_Cart.projmap(
            sph_map, vec2pix_func=partial(hp.vec2pix, hp.get_nside(sph_map))
        )

    if proj_method == "Gnom":
        fig_gnom, ax = plt.subplots(1, 1)
        map_flat = hp.gnomview(
            sph_map,
            rot=center,
            coord=["G"],
            reso=hp.nside2resol(hp.get_nside(sph_map), arcmin=True),
            xsize=flat_nside,
            return_projected_map=True,
            cmap="bwr",
            nest=False,
            fig=fig_gnom,
        )
        plt.close(fig_gnom)

    return map_flat


# Defines central points on sphere from which projecting a flat map
def generate_subpatch_central_points(side_deg):
    """

    Note: the following central points depend only on the angular aperture of the final projected flat map (flat_nside) and not on the spherical nside (sph_nside)!
          Of course, higher the spherical nside and higher the accuracy of the projected flat maps!

    Note: The following algorithm is based on a flat geometry approximation. So it is accurate for small enough latitudes.
          It means that we are assuming to produce flat square patches from  spherical coords of central points +/- deg_side/2.
          As we know, moving at higher latitudes, the area embedded into central points +/- deg_side/2 will be lower than expected in a flat case because of the curvature.
          The proper spherical coords ranges corresponding to the projected flat patches -with area side_deg^2- will be wider because of the curvature.
          For the same reason, the assumed separation (3 deg) between the flat patches reduces at higher latitudes since we are not taking into account the widening of the spherical coords ranges of the projected flat maps.

    Note: the projection effects are minimized towards the equator (lat=0).
          It is worth to reduce the latitude domain close to the equator (lat = +/- 20 deg).
          Infact, the area of a spherical sub-patch for a given delta_lat, delta_lon reduces towards the poles!

    Note: the sub-patches are separated by 3 deg in both lat and lon to avoid overlappings in the final flat patches (it seems to be a good value for several low nside's, found by hand)

    Note: longitude domain is narrowed down by a constant factor 0.5*side_deg to avoid extremal points transitions

    """

    longitudes = []
    latitudes = []
    patches_separation = 3  # deg

    # Generate central points
    for lon in numpy.linspace(
        0 + 0.5 * side_deg + 0.5 * patches_separation,
        360 - 0.5 * side_deg - 0.5 * patches_separation,
        num=int((360 - side_deg) / (side_deg + patches_separation)),
        endpoint=True,
    ):
        for lat in numpy.arange(-20, 20, side_deg + patches_separation):
            longitudes.append(lon)
            latitudes.append(lat)

    print(f"Total selected independent flat sub-patches: {len(longitudes)}")

    return list(zip(longitudes, latitudes))


# Rotation matrix function
def rotator_2D(x, y, angle, x_shift=0, y_shift=0, units="DEGREES"):
    import math

    """
    
    FUNDAMENTAL NOTE: rotator_2D() and rot() in get_flat_projection() follow same rotational convention (clockwise rotation if angle is positive)
    
    Rotates a point in the xy-plane counterclockwise through an angle wrt a central point [x_shift, y_shift] (aka origin if [0,0])
    https://en.wikipedia.org/wiki/Rotation_matrix
    :param x: x coordinate
    :param y: y coordinate
    :param x_shift: x-axis shift from origin (0, 0)
    :param y_shift: y-axis shift from origin (0, 0)
    :param angle: The rotation angle in degrees between [angle >/< 0 clockwise/counterclockwise]
    :param units: DEGREES (default) or RADIANS
    :return: Tuple of rotated x and y
    """

    # Shift to origin (0,0)
    x = x - x_shift
    y = y - y_shift

    # Convert degrees to radians
    if units == "DEGREES":
        angle = math.radians(angle)

    # Rotation matrix multiplication to get rotated x & y
    xr = (x * math.cos(angle)) - (y * math.sin(angle)) + x_shift
    yr = (x * math.sin(angle)) + (y * math.cos(angle)) + y_shift

    return xr, yr
