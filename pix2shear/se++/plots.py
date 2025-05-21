from pathlib import Path

import numpy
import tensorflow_datasets as tfds
from astropy.io import fits
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import loadtxt
from pandas import concat, read_csv
from scipy import spatial, stats


def search_folder(name):
    if Path.cwd().parts[1] == "home":
        search_dir = Path("/home")
    else:
        search_dir = Path("/net/GECO/pisco/PIX2SHEAR")
    matching_folders = list(search_dir.glob(f"**/{name}"))
    return matching_folders[0]


DIR_DATASETS = search_folder(name="DATASETS")
DIR_RESULTS = search_folder(name="RESULTS_SEXTRACTOR")
DIRS = {
    "DATASETS": DIR_DATASETS,
    "RESULTS_SEXTRACTOR": DIR_RESULTS,
    "psf_se": DIR_DATASETS / "psf_se",
    "pisco_euclid_se": DIR_DATASETS / "pisco_euclid_se",
    "pisco_euclid": DIR_DATASETS / "pisco_euclid",
}


def ellips_model(e1_err, e2_err, e1, e2):
    weight_e1 = 1 / (e1_err**2 + 0.26**2)
    e1_weighted_mean = numpy.sum(weight_e1 * e1) / numpy.sum(weight_e1)

    weight_e2 = 1 / (e2_err**2 + 0.26**2)
    e2_weighted_mean = numpy.sum(weight_e2 * e2) / numpy.sum(weight_e2)

    return float(e1_weighted_mean), float(e2_weighted_mean)


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


def write_fits(array, path):
    hdu = fits.PrimaryHDU(array)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(path, overwrite=True)
    hdulist.close()


def get_match(df_pred, df_true, radius):
    tree_preds = spatial.cKDTree(df_pred[["x_model", "y_model"]].values)
    truth_to_preds = tree_preds.query_ball_point(df_true[["x", "y"]].values, r=radius)

    indices_truth = [i for i, lst in enumerate(truth_to_preds) if len(lst) == 1]
    indices_pred = [truth_to_preds[i][0] for i in indices_truth]

    filter_truth = df_true.iloc[indices_truth].copy()
    filter_pred = df_pred.iloc[indices_pred].copy()

    return filter_pred, filter_truth


# --------------------------------------------------------------------------------------------------


def _identity(x):
    return x


def make_table(ax, data):
    table_data = data.round(4).astype(str).values.tolist()
    table = ax.table(cellText=table_data, colLabels=data.columns, loc="center")
    table.scale(1.5, 2)
    ax.axis("off")


def comparaison(name_image, vectors_preds, vectors_true, detec, g1, g2, plot_dir):
    fig, axs = pyplot.subplots(1, 2, figsize=(18, 6), gridspec_kw={"width_ratios": [3, 1]})

    result = concat([vectors_preds, vectors_true], axis=1)
    make_table(axs[0], result)

    image = numpy.arcsinh(fits.open(f"{name_image}.fits")[0].data.astype(numpy.float32))
    axs[1].imshow(image)
    axs[1].set_title(f"nb detec {detec} // shear {g1:.3f} // {g2:.3f}", fontsize=10)
    axs[1].invert_yaxis()

    fig.tight_layout()
    fig.show()
    fig.savefig(plot_dir / "test_comp.pdf", dpi=300)


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

    fig, axes = pyplot.subplots(1, 3, figsize=(9, 3))

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
    fig.savefig(plots_dir / f"test_fitting{title}.pdf", dpi=300)


def residuals(datasets_dir, images_dir, choice, plot_dir, zoom_size=20, num_ticks=5):
    name_assoc = datasets_dir / choice
    assoc = loadtxt(f"{name_assoc}_assoc.txt")
    assoc = assoc[numpy.argsort(assoc[:, 0])]

    name_residual = images_dir / f"residual_{choice}_1"
    image = fits.getdata(f"{name_residual}.fits").astype(numpy.float32)

    fig, axes = pyplot.subplots(2, 5, figsize=(10, 4))
    axes = axes.flatten()

    for i, (x, y) in enumerate(assoc):
        x, y = int(x), int(y)

        x_min, x_max = max(0, x - zoom_size), min(352, x + zoom_size)
        y_min, y_max = max(0, y - zoom_size), min(352, y + zoom_size)

        zoomed_img = image[y_min:y_max, x_min:x_max]

        img = axes[i].imshow(zoomed_img, origin="lower")
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
    fig.savefig(plot_dir / "test_residuals.pdf", dpi=300)
    fig.show()


def vignettes(catalog_dir, images_dir, choice, plots_dir):
    csv_file = catalog_dir / f"{choice}.csv"
    bounds = read_csv(csv_file)[
        [
            "pixel_boundaries_min_x",
            "pixel_boundaries_max_x",
            "pixel_boundaries_min_y",
            "pixel_boundaries_max_y",
        ]
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
        fig, axes = pyplot.subplots(3, 5, figsize=(10, 6))
    else:
        fig, axes = pyplot.subplots(2, 5, figsize=(10, 4))
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
        f"vignettes shape {vignettes.shape[1]}, nb galaxies {vignettes.shape[0]}", fontsize=16
    )
    fig.tight_layout()
    fig.show()
    fig.savefig(plots_dir / "test_vignettes.pdf", dpi=300)


def x_n_seric(MIN_SERSIC, MAX_SERSIC, mu_x, sigma_x, plot_dir):
    def sersic_from_x(x):
        return (MAX_SERSIC * numpy.exp(x) + MIN_SERSIC) / (1 + numpy.exp(x))

    x_vals = numpy.linspace(-10, 10, 1000)
    sersic_vals = sersic_from_x(x_vals)
    prior_density = stats.norm.pdf(x_vals, loc=mu_x, scale=sigma_x)
    # prior_density_scaled = (
    #     prior_density * (max(sersic_vals) - min(sersic_vals)) / max(prior_density)
    # )

    fig, ax1 = pyplot.subplots(figsize=(10, 6))

    pyplot.text(
        10.2, MIN_SERSIC, f"{MIN_SERSIC}", va="center", ha="left", fontsize=10, color="gray"
    )
    pyplot.text(
        10.2, MAX_SERSIC, f"{MAX_SERSIC}", va="center", ha="left", fontsize=10, color="gray"
    )

    pyplot.grid(True)

    ax1.plot(x_vals, sersic_vals, color="blue", label="Sérsic index vs X_ser")
    ax1.set_xlabel("X_ser (latent parameter)")
    ax1.set_ylabel("Sérsic index", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(x_vals, prior_density, color="red", linestyle="--", label="Gaussian prior on X_ser")
    ax2.set_ylabel("Prior density", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    x1 = mu_x - sigma_x
    x2 = mu_x + sigma_x
    s1 = sersic_from_x(x1)
    s2 = sersic_from_x(x2)

    ax1.axvspan(x1, x2, color="red", alpha=0.2, label=r"1$\sigma$ region (X_ser)")
    ax1.axhspan(s1, s2, color="blue", alpha=0.1, label="Covered Sérsic range")

    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    pyplot.title(f"Mapping Gaussian Prior on X_ser to Sérsic Index Range : {sersic_from_x(mu_x)}")
    fig.tight_layout()
    fig.savefig(plot_dir / "x_n_sersic.pdf", dpi=300)
    fig.show()


def case_result(pred_e1, pred_e2, shears1, shears2, directory, symbol="gamma"):
    s = 5
    alpha = 0.25
    c1, c2 = "C0", "C1"
    t = numpy.linspace(min(shears1.min(), shears2.min()), max(shears1.max(), shears2.max()))

    diff_pred_g1 = pred_e1 - shears1
    diff_pred_g2 = pred_e2 - shears2
    poly_g1, cov1 = numpy.polyfit(shears1, diff_pred_g1, 1, cov=True)
    poly_g2, cov2 = numpy.polyfit(shears2, diff_pred_g2, 1, cov=True)

    fig, axes = pyplot.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes = axes.flatten()

    lab_1 = (
        f"m = {poly_g1[0]:.2e} ± {numpy.sqrt(cov1[0, 0]):.2e}",
        f"c = {poly_g1[1]:.2e} ± {numpy.sqrt(cov1[1, 1]):.2e}",
    )
    lab_2 = (
        f"m = {poly_g2[0]:.2e} ± {numpy.sqrt(cov2[0, 0]):.2e}",
        f"c = {poly_g2[1]:.2e} ± {numpy.sqrt(cov2[1, 1]):.2e}",
    )

    axes[0].scatter(shears1, diff_pred_g1, alpha=alpha, color=c1, marker="x", s=s, label=lab_1)
    axes[0].scatter(shears2, diff_pred_g2, alpha=alpha, color=c2, marker="x", s=s, label=lab_2)
    axes[0].plot(t, poly_g1[1] + poly_g1[0] * t, color=c1, lw=5)
    axes[0].plot(t, poly_g2[1] + poly_g2[0] * t, color=c2, lw=5, ls="--")

    axes[1].scatter(shears1, pred_e1, alpha=alpha, color=c1, marker="x", s=s)
    axes[1].scatter(shears2, pred_e2, alpha=alpha, color=c2, marker="o", s=s)
    lims = axes[1].get_xlim()
    axes[1].plot(lims, lims, color="black", linestyle="--", linewidth=1, label="y=x")

    axes[0].axhline(0, color="black", linestyle="--", linewidth=1, label="y=0")

    axes[0].set_ylabel(rf"$\epsilon^{{obs}}$ - $\{symbol}^{{true}}$", rotation=0, labelpad=20)
    axes[1].set_ylabel(r"$\epsilon^{obs}$", rotation=0, labelpad=20)

    for ax in axes:
        if symbol == "gamma":
            ax.set_xlim(-0.055, 0.055)
        else:
            ax.set_xlim(-0.35, 0.35)
        ax.set_ylim(-0.35, 0.35)
        ax.set_xlabel(rf"$\{symbol}^{{true}}$")
        ax.legend()

    fig.suptitle(f"{directory.name} : {pred_e1.shape[0]} images", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.tight_layout()
    fig.savefig(directory, dpi=300)


def logged_values(bins, preds, trues, pixel_scale, plot_dir):
    fig, axes = pyplot.subplots(1, 3, figsize=(9, 3), sharey=True)
    log = True
    alpha = 0.2

    axes[0].hist(
        preds["sersic_model"],
        alpha=alpha,
        bins=bins,
        label=f"predictions {preds['sersic_model'].shape[0]}",
        log=log,
    )
    axes[0].hist(
        trues["sersic"], alpha=alpha, bins=bins, label=f"truth {trues['sersic'].shape[0]}", log=log
    )
    axes[0].set_title("Sersic")
    axes[0].legend()

    axes[1].hist(
        preds["mag_model"],
        alpha=alpha,
        bins=bins,
        label=f"predictions {preds['mag_model'].shape[0]}",
        log=log,
    )
    axes[1].hist(
        trues["mag"], alpha=alpha, bins=bins, label=f"truth {trues['mag'].shape[0]}", log=log
    )
    axes[1].set_title("Magnitude")
    axes[1].legend()

    axes[2].hist(
        preds["flux_radius"] * pixel_scale,
        alpha=alpha,
        bins=bins,
        label=f"predictions {preds['flux_radius'].shape[0]}",
        log=log,
    )
    axes[2].hist(
        trues["hlr"], alpha=alpha, bins=bins, label=f"truth {trues['hlr'].shape[0]}", log=log
    )
    axes[2].set_title("Half-Light Radius (HLR)")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(plot_dir / "logged_sersic_mag_hlr.pdf", dpi=300)
    fig.show()


def fmf_values(bins, preds, plot_dir, log=False):
    fig, axes = pyplot.subplots(1, 3, figsize=(9, 3), sharey=True)
    alpha = 0.2

    axes[0].hist(
        preds["fmf_reduced_chi_2"], alpha=alpha, bins=bins, label="fmf_reduced_chi_2", log=log
    )
    axes[1].hist(preds["fmf_stop_reason"], alpha=alpha, bins=bins, label="fmf_stop_reason", log=log)
    axes[2].hist(preds["fmf_flags"], alpha=alpha, bins=bins, label="fmf_flags", log=log)
    for ax in axes.flat:
        ax.legend()

    fig.tight_layout()
    fig.savefig(plot_dir / "fmf_values.pdf", dpi=300)
    fig.show()


def histos(
    detections,
    pred_galaxies_params,
    true_galaxies_params,
    pred_avg_e1,
    pred_avg_e2,
    true_images_params,
    directory,
    pixel_scale,
    bins,
    density=True,
):
    fig, axes = pyplot.subplots(2, 4, figsize=(12, 6))
    alpha = 0.3

    axes[0, 0].hist(
        pred_avg_e1,
        alpha=alpha,
        bins=bins,
        density=density,
        label=f"predictions {pred_avg_e1.shape[0]}",
    )
    axes[0, 0].hist(
        true_images_params["avg_obs_e1"],
        alpha=alpha,
        bins=bins,
        density=density,
        label=f"truth {true_images_params['avg_obs_e1'].shape[0]}",
    )
    axes[0, 0].set_title("e1")

    axes[0, 1].hist(
        pred_avg_e2,
        alpha=alpha,
        bins=bins,
        density=density,
        label=f"predictions {pred_avg_e2.shape[0]}",
    )
    axes[0, 1].hist(
        true_images_params["avg_obs_e2"],
        alpha=alpha,
        bins=bins,
        density=density,
        label=f"truth {true_images_params['avg_obs_e2'].shape[0]}",
    )
    axes[0, 1].set_title("e2")

    axes[0, 2].hist(
        pred_galaxies_params["e1_model"],
        alpha=alpha,
        bins=bins,
        density=density,
        label=f"predictions {pred_galaxies_params['e1_model'].shape[0]}",
    )
    axes[0, 2].hist(
        true_galaxies_params["obs_e1"],
        alpha=alpha,
        bins=bins,
        density=density,
        label=f"truth {true_galaxies_params['obs_e1'].shape[0]}",
    )
    axes[0, 2].set_title("e1")

    axes[0, 3].hist(
        pred_galaxies_params["e2_model"],
        alpha=alpha,
        bins=bins,
        density=density,
        label=f"predictions {pred_galaxies_params['e2_model'].shape[0]}",
    )
    axes[0, 3].hist(
        true_galaxies_params["obs_e2"],
        alpha=alpha,
        bins=bins,
        density=density,
        label=f"truth {true_galaxies_params['obs_e2'].shape[0]}",
    )
    axes[0, 3].set_title("e2")

    axes[1, 0].hist(
        pred_galaxies_params["flux_radius"] * pixel_scale,
        alpha=alpha,
        bins=bins,
        density=density,
        label=f"predictions {pred_galaxies_params['flux_radius'].shape[0]}",
    )
    axes[1, 0].hist(
        true_galaxies_params["hlr"],
        alpha=alpha,
        bins=bins,
        density=density,
        label=f"truth {true_galaxies_params['hlr'].shape[0]}",
    )
    axes[1, 0].set_title(f"hlr with pixel scale = {pixel_scale}")

    axes[1, 1].hist(
        pred_galaxies_params["sersic_model"],
        alpha=alpha,
        bins=bins,
        density=density,
        label=f"predictions {pred_galaxies_params['sersic_model'].shape[0]}",
    )
    axes[1, 1].hist(
        true_galaxies_params["sersic"],
        alpha=alpha,
        bins=bins,
        density=density,
        label=f"truth {true_galaxies_params['sersic'].shape[0]}",
    )
    axes[1, 1].set_title("sersic")

    axes[1, 2].hist(
        pred_galaxies_params["mag_model"],
        alpha=alpha,
        bins=bins,
        density=density,
        label=f"predictions {pred_galaxies_params['mag_model'].shape[0]}",
    )
    axes[1, 2].hist(
        true_galaxies_params["mag"],
        alpha=alpha,
        bins=bins,
        density=density,
        label=f"truth {true_galaxies_params['mag'].shape[0]}",
    )
    axes[1, 2].set_title("mag")

    axes[1, 3].hist(detections, bins=bins, label=f"detections {numpy.sum(detections)}")
    axes[1, 3].set_title("galaxies")

    for ax in axes.ravel():
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "histos.pdf", dpi=300)
    fig.show()


def background(names):
    img = fits.open(f"{names}.fits")[0].data.astype(numpy.float32)
    pyplot.title(f"background => Min: {numpy.min(img):.3f}, Max: {numpy.max(img):.3f}")
    im = pyplot.imshow(img, cmap="gray", vmax=numpy.max(img))
    pyplot.gca().invert_yaxis()
    pyplot.colorbar(im)


def case_result_param(pred, true, symbol, err=0):
    pred = numpy.array(pred)
    true = numpy.array(true)
    err = numpy.array(err)
    s = 1

    fig, axs = pyplot.subplots(1, 2, figsize=(8, 4))

    axs[0].scatter(true, pred - true, s=2, alpha=0.5)
    axs[0].axhline(0, color="black", linestyle="--")
    axs[0].set_xlabel("True")
    axs[0].set_ylabel("Pred - True")
    axs[0].set_title(f"Residuals of {symbol}")

    lim_min = min(pred.min(), true.min())
    lim_max = max(pred.max(), true.max())

    if symbol == "hlr":
        axs[1].scatter(true, pred, marker="x", s=s)
    else:
        axs[1].errorbar(true, pred, yerr=err, fmt="x", capsize=2, markersize=3, alpha=0.5)

    axs[1].plot([lim_min, lim_max], [lim_min, lim_max], "k--", zorder=10)
    axs[1].set_xlim(lim_min, lim_max)
    axs[1].set_ylim(lim_min, lim_max)
    axs[1].set_xlabel("True")
    axs[1].set_ylabel("Pred")
    axs[1].set_title(f"Predicted vs True {symbol}")

    fig.tight_layout()
    fig.show()
