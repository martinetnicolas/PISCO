from pathlib import Path

import tensorflow_datasets as tfds
from astropy.io import fits
from jax import numpy
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import loadtxt
from pandas import read_csv
from scipy.spatial import cKDTree

# def weighted_ellips_model(e1_err, e2_err, e1, e2):
#     e1_err = e1_err.to_numpy()
#     e2_err = e2_err.to_numpy()
#     e1 = e1.to_numpy()
#     e2 = e2.to_numpy()

#     weight = 1 / numpy.sqrt(((e1_err + e2_err) / 2) ** 2 + 0.26**2)
#     e1_weighted_mean = numpy.sum(weight * e1) / numpy.sum(weight)
#     e2_weighted_mean = numpy.sum(weight * e2) / numpy.sum(weight)

#     return float(e1_weighted_mean), float(e2_weighted_mean)


# def ellips_et(e, theta):
#     e = e.to_numpy()
#     theta = theta.to_numpy()

#     rad = numpy.deg2rad(theta)
#     e1 = e * numpy.cos(2 * rad)
#     e2 = e * numpy.sin(2 * rad)
#     return float(e1.mean()), float(e2.mean())


# def weighted_results(e1_err, e2_err, e1, e2):
#     weight = 1 / numpy.sqrt(((e1_err + e2_err) / 2) ** 2 + 0.26**2)
#     e1_weighted_mean = numpy.sum(weight * e1) / numpy.sum(weight)
#     e2_weighted_mean = numpy.sum(weight * e2) / numpy.sum(weight)
#     return e1_weighted_mean, e2_weighted_mean


def tfds_info(style, version, data_dir, split=True):
    if split:
        version = version.split("_", 1)[0]

    builder = tfds.builder(
        name=f"{style}:{version}",
        data_dir=data_dir,
    )

    return builder.info


def slice_csv(file_path, nb_total):
    return read_csv(file_path, sep=",", skiprows=0)[:nb_total]


def _find_directory(name, init_directory):
    directory_path = Path(init_directory)
    matching_folders = list(directory_path.glob(f"**/{name}"))
    return matching_folders[0]


def search_folder(name):
    if Path.cwd().parts[1] == "home":
        search_dir = "/home"
    else:
        search_dir = "/net/GECO/pisco/PIX2SHEAR"
    return _find_directory(name, search_dir)


def write_fits(array, path):
    hdu = fits.PrimaryHDU(array)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(path, overwrite=True)
    hdulist.close()


# --------------------------------------------------------------------------------------------------


def _identity(x):
    return x


def image_fitting(dataset_dir, images_dir, choice, plots_dir, transformed=False):
    if transformed:
        func = numpy.arcsinh
        title = "arcsinh"
        name_extra = images_dir / f"residual_{choice}_1"
        title_extra = "residual"
    else:
        func = _identity
        title = ""
        name_extra = images_dir / f"segmentation_{choice}"
        title_extra = "segmentation"

    name_image = dataset_dir / choice
    name_fitting = images_dir / f"fitting_{choice}_1"

    assoc = loadtxt(f"{name_image}_assoc.txt")
    image = func(fits.open(f"{name_image}.fits")[0].data.astype(numpy.float32))
    fitting = func(fits.open(f"{name_fitting}.fits")[0].data.astype(numpy.float32))
    extra = fits.open(f"{name_extra}.fits")[0].data.astype(numpy.float32)

    fig, axes = pyplot.subplots(1, 3, figsize=(15, 5))

    minimum = min(numpy.amin(image), numpy.amin(fitting))
    maximum = max(numpy.amax(image), numpy.amax(fitting))

    im1 = axes[0].imshow(image, vmin=minimum, vmax=maximum)
    axes[0].set_title(f"image {title}")

    im2 = axes[1].imshow(fitting, vmin=minimum, vmax=maximum)
    axes[1].set_title(f"fitting {title}")

    im3 = axes[2].imshow(extra, cmap="gray")
    axes[2].set_title(title_extra)
    if not transformed:
        axes[2].scatter(assoc[:, 0], assoc[:, 1], color="red", marker="o", s=2, label="assoc")

    for ax, img in zip(axes.flat, [im1, im2, im3]):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        pyplot.colorbar(img, cax=cax)
        ax.invert_yaxis()

    fig.tight_layout()
    fig.show()
    fig.savefig(plots_dir / f"fitting{title}.pdf", dpi=300)


def residuals(datasets_dir, images_dir, choice, plot_dir, zoom_size=20, num_ticks=5):
    name_assoc = datasets_dir / choice
    assoc = loadtxt(f"{name_assoc}_assoc.txt")
    assoc = assoc[numpy.argsort(assoc[:, 0])]

    name_residual = images_dir / f"residual_{choice}_1"
    image = fits.getdata(f"{name_residual}.fits").astype(numpy.float32)

    fig, axes = pyplot.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()

    for i, (x, y) in enumerate(assoc):
        x, y = int(x), int(y)

        x_min, x_max = max(0, x - zoom_size), min(352, x + zoom_size)
        y_min, y_max = max(0, y - zoom_size), min(352, y + zoom_size)

        zoomed_img = image[y_min:y_max, x_min:x_max]

        img = axes[i].imshow(zoomed_img, cmap="gray", origin="lower")
        axes[i].set_title(f"center = ({x}, {y})", fontsize=10)

        xticks = numpy.linspace(0, zoomed_img.shape[1] - 1, num_ticks, dtype=numpy.float32)
        yticks = numpy.linspace(0, zoomed_img.shape[0] - 1, num_ticks, dtype=numpy.float32)
        x_labels = numpy.linspace(x_min, x_max, num_ticks, dtype=numpy.float32)
        y_labels = numpy.linspace(y_min, y_max, num_ticks, dtype=numpy.float32)
        axes[i].set_xticks(xticks)
        axes[i].set_yticks(yticks)
        axes[i].set_xticklabels(numpy.round(x_labels).astype(int), fontsize=8)
        axes[i].set_yticklabels(numpy.round(y_labels).astype(int), fontsize=8)

        divider = make_axes_locatable(axes[i])
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        pyplot.colorbar(img, cax=cax1)

    fig.suptitle("residuals", fontsize=16)
    fig.tight_layout()
    fig.savefig(plot_dir / "residuals.pdf", dpi=300)
    fig.show()


def vignettes(catalog_dir, images_dir, choice, plots_dir):
    csv_file = catalog_dir / f"{choice}.csv"
    bounds = read_csv(csv_file)[
        "pixel_boundaries_min_x",
        "pixel_boundaries_max_x",
        "pixel_boundaries_min_y",
        "pixel_boundaries_max_y",
    ].to_numpy()
    approx_x_center = numpy.mean(bounds[:, :2], axis=1)
    approx_y_center = numpy.mean(bounds[:, 2:], axis=1)

    sorted_indices = numpy.argsort(approx_x_center)
    approx_x_center = approx_x_center[sorted_indices]
    approx_y_center = approx_y_center[sorted_indices]

    vignettes_file = images_dir / f"vignettes_{choice}.fits"
    vignettes = fits.open(vignettes_file)[0].data
    vignettes = vignettes[sorted_indices, :, :]
    vignettes = vignettes.swapaxes(1, 2)

    number = vignettes.shape[0]
    if number > 10:
        fig, axes = pyplot.subplots(3, 5, figsize=(12, 9))
    else:
        fig, axes = pyplot.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()

    for i, (vignette, x, y) in enumerate(zip(vignettes, approx_x_center, approx_y_center)):
        img = axes[i].imshow(vignette)
        axes[i].set_title(f"center = ({x}, {y})", fontsize=10)
        divider = make_axes_locatable(axes[i])
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        pyplot.colorbar(img, cax=cax1)
        axes[i].axis("off")
        axes[i].invert_yaxis()

    for j in range(number, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"vignettes shape {vignettes.shape[1]}, nb galaxies {vignettes.shape[0]}, ", fontsize=16
    )
    fig.tight_layout()
    fig.show()
    fig.savefig(plots_dir / "vignettes.pdf", dpi=300)


def case_result(avg_obs_e, true_avg_g, directory, title, symbol="gamma"):
    s = 10
    alpha = 0.2
    c1, c2 = "C0", "C1"
    t = numpy.linspace(true_avg_g.min(), true_avg_g.max())

    diff_pred_g1 = avg_obs_e[:, 0] - true_avg_g[:, 0]
    diff_pred_g2 = avg_obs_e[:, 1] - true_avg_g[:, 1]
    poly_g1 = numpy.polyfit(true_avg_g[:, 0], diff_pred_g1, 1)
    poly_g2 = numpy.polyfit(true_avg_g[:, 1], diff_pred_g2, 1)

    fig, axes = pyplot.subplots(1, 2, figsize=(12, 5))
    axes = axes.flatten()

    lab_1 = f"m= {poly_g1[0]:.3f}, c={poly_g1[1]:.3f}"
    axes[0].scatter(
        true_avg_g[:, 0], diff_pred_g1, alpha=alpha, color=c1, marker="x", s=s, label=lab_1
    )
    lab_2 = f"m= {poly_g2[0]:.3f}, c={poly_g2[1]:.3f}"
    axes[0].scatter(
        true_avg_g[:, 1], diff_pred_g2, alpha=alpha, color=c2, marker="o", s=s, label=lab_2
    )
    axes[0].plot(t, poly_g1[1] + poly_g1[0] * t, color=c1)
    axes[0].plot(t, poly_g2[1] + poly_g2[0] * t, color=c2)

    axes[1].scatter(true_avg_g[:, 0], avg_obs_e[:, 0], alpha=alpha, color=c1, marker="x", s=s)
    axes[1].scatter(true_avg_g[:, 1], avg_obs_e[:, 1], alpha=alpha, color=c2, marker="o", s=s)
    lims = axes[1].get_xlim()
    axes[1].plot(lims, lims, color="black", linestyle="--", linewidth=1, label="y=x")

    axes[0].axhline(0, color="black", linestyle="--", linewidth=1, label="y=0")

    axes[0].set_ylabel(rf"$\epsilon^{{pred}}$ - $\{symbol}^{{true}}$", rotation=0, labelpad=20)
    axes[1].set_ylabel(r"$\epsilon^{pred}$", rotation=0, labelpad=20)

    for ax in axes:
        if symbol == "gamma":
            ax.set_xlim(-0.055, 0.055)
        else:
            ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        ax.set_xlabel(rf"$\{symbol}^{{true}}$")
        ax.legend()

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.tight_layout()
    fig.savefig(directory, dpi=300)


def case_result_param(pred, truth, loss_comp, loss_gal, directory, symbol, err=0, pixel_scale=0.1):
    if symbol == "hlr":
        pred *= pixel_scale
    s = 1
    color_id = "#00FF00"
    color_fit = "red"
    mini, maxi = truth.min(), truth.max()
    t = numpy.linspace(mini, maxi)

    diff = pred - truth
    poly_g1 = numpy.polyfit(truth, diff, 1)

    fig, axes = pyplot.subplots(1, 2, figsize=(12, 5))
    axes = axes.flatten()

    axes[0].axhline(0, c=color_id, ls="--", lw=1, label="y=0")

    label = f"m= {poly_g1[0]:.3f}, c={poly_g1[1]:.3f}"
    axes[0].plot(t, poly_g1[1] + poly_g1[0] * t, color=color_fit, label=label, zorder=2)

    if symbol == "hlr":
        axes[0].scatter(truth, diff, marker="x", s=s)
    else:
        axes[0].errorbar(truth, diff, err, ms=3, fmt="o", capsize=2, label="err", zorder=1)
        # axes[0].scatter(truth, diff, marker="x", s=s)

    axes[0].set_ylabel(rf"${symbol}^{{obs}}$ - ${symbol}^{{true}}$", rotation=45, labelpad=20)

    axes[1].scatter(truth, pred, marker="x", s=s)
    lims = axes[1].get_xlim()
    axes[1].plot(lims, lims, c=color_id, ls="--", lw=1, label="y=x")
    axes[1].set_ylabel(rf"${symbol}^{{obs}}$", rotation=0, labelpad=20)
    axes[1].set_ylim(mini, maxi)

    for ax in axes:
        ax.set_xlim(mini, maxi)
        ax.set_xlabel(rf"${symbol}^{{true}}$")
        ax.legend()

    fig.suptitle(f"{symbol} with {loss_comp}% loss over {loss_gal}% detected", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.tight_layout()
    fig.savefig(directory / f"{symbol}.pdf", dpi=300)


def histos(
    detections,
    sersic,
    sersic_err,
    true_sersic,
    mag,
    mag_err,
    true_mag,
    total_hlr,
    total_true_hlr,
    sex_pred_ellips,
    model_pred_obs_e,
    true_ellps,
    directory,
    pixel_scale,
    bins=50,
):
    fig, axes = pyplot.subplots(3, 4, figsize=(16, 12))
    alpha = 0.3

    axes[0, 0].hist(model_pred_obs_e[:, 0], alpha=alpha, bins=bins, label="predictions")
    axes[0, 0].hist(true_ellps[:, 0], alpha=alpha, bins=bins, label="truth")
    axes[0, 0].set_title("e1 model (weighted errors)")

    axes[0, 1].hist(model_pred_obs_e[:, 1], alpha=alpha, bins=bins, label="predictions")
    axes[0, 1].hist(true_ellps[:, 1], alpha=alpha, bins=bins, label="truth")
    axes[0, 1].set_title("e2 model (weighted errors)")

    axes[0, 2].hist(sex_pred_ellips[:, 0], alpha=alpha, bins=bins, label="predictions")
    axes[0, 2].hist(true_ellps[:, 0], alpha=alpha, bins=bins, label="truth")
    axes[0, 2].set_title("e1 internal (NO weighted errors)")

    axes[0, 3].hist(sex_pred_ellips[:, 1], alpha=alpha, bins=bins, label="predictions")
    axes[0, 3].hist(true_ellps[:, 1], alpha=alpha, bins=bins, label="truth")
    axes[0, 3].set_title("e2 internal (NO weighted errors)")

    axes[1, 0].hist(sersic, alpha=alpha, bins=bins, label=f"predictions {sersic.shape[0]}")
    axes[1, 0].hist(true_sersic, alpha=alpha, bins=bins, label=f"truth {true_sersic.shape[0]}")
    axes[1, 0].set_title("sersic")

    axes[1, 1].errorbar(
        sersic,
        sersic,
        yerr=sersic_err,
        alpha=alpha,
        ms=3,
        fmt="o",
        color="blue",
        label="Predicted ± Error",
        capsize=4,
        zorder=1,
    )
    axes[1, 1].scatter(sersic, sersic, color="green", marker="o", s=20, zorder=2, label="Predicted")
    axes[1, 1].plot(true_sersic, true_sersic, color="red", lw=2, label="truth", zorder=3)
    axes[1, 1].set_title("sersic")

    axes[1, 2].hist(mag, alpha=alpha, bins=bins, label=f"predictions {mag.shape[0]}")
    axes[1, 2].hist(true_mag, alpha=alpha, bins=bins, label=f"truth {true_mag.shape[0]}")
    axes[1, 2].set_title("mag")

    axes[1, 3].errorbar(
        mag,
        mag,
        yerr=mag_err,
        alpha=alpha,
        ms=3,
        fmt="o",
        color="blue",
        label="Predicted ± Error",
        capsize=4,
        zorder=1,
    )
    axes[1, 3].scatter(mag, mag, color="green", marker="o", s=20, zorder=2, label="Predicted")
    axes[1, 3].plot(true_mag, true_mag, color="red", lw=2, label="truth", zorder=3)
    axes[1, 3].set_title("mag")

    axes[2, 0].hist(
        total_hlr * pixel_scale, alpha=alpha, bins=bins, label=f"predictions {total_hlr.shape[0]}"
    )
    axes[2, 0].hist(
        total_true_hlr, alpha=alpha, bins=bins, label=f"truth {total_true_hlr.shape[0]}"
    )
    axes[2, 0].set_title(f"hlr with pixel scale = {pixel_scale}")

    axes[2, 1].hist(detections, bins=bins, label=f"detections {numpy.sum(detections)}")
    axes[2, 1].set_yscale("log")
    axes[2, 1].set_title("galaxies")

    for idx, ax in enumerate(axes.ravel()):
        if idx < len(axes.ravel()) - 2:
            ax.legend()
    fig.delaxes(axes[2, 2])
    fig.delaxes(axes[2, 3])
    fig.tight_layout()
    fig.savefig(directory / "errors.pdf", dpi=300)
    fig.show()


# def background(names):
#     img = fits.open(f"{names}.fits")[0].data.astype(numpy.float32)
#     pyplot.title(f"background => Min: {numpy.min(img):.3f}, Max: {numpy.max(img):.3f}")
#     im = pyplot.imshow(img, cmap="gray", vmax=numpy.max(img))
#     pyplot.gca().invert_yaxis()
#     pyplot.colorbar(im)
