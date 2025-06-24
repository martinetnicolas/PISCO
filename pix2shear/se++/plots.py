from pathlib import Path

import numpy
import tensorflow_datasets as tfds
from astropy.io import fits
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import loadtxt
from pandas import DataFrame, concat, read_csv
from scipy import spatial, stats  # , optimize
from tqdm import tqdm

DPI = 200


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

TO_DROP = [
    "pixel_boundaries_min_x",
    "pixel_boundaries_min_y",
    "pixel_boundaries_max_x",
    "pixel_boundaries_max_y",
    "fmf_iterations",
    "fmf_duration",
    "fmf_meta_iterations",
    "auto_mag",
    "auto_mag_err",
    "auto_flags",
    "auto_flux",
    "auto_flux_err",
    "x_model_err",
    "y_model_err",
    "flux_model",
    "flux_model_err",
]


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


def compute_res(IMAGES_DIR):
    residual_files = sorted(IMAGES_DIR.glob("residual_*_simu_1.fits"))
    all_pixels = []
    image_ids = []
    for idx, file in tqdm(enumerate(residual_files), total=len(residual_files)):
        image = fits.getdata(file).astype(numpy.float32)
        all_pixels.append(image.max())
        image_ids.append(idx)
    all_pixels = numpy.array(all_pixels)
    image_ids = numpy.array(image_ids)
    df = DataFrame({"pixel_value": all_pixels, "image_id": image_ids})
    return df, all_pixels


def filtering(pred_avg, true_avg, detections, pred_gal, true_gal, mask):
    images_mask = mask[mask].index.to_numpy()

    filter_pred_avg = pred_avg[mask]
    filter_true_avg = true_avg[mask]
    filter_detections = detections[mask]

    filter_pred_gal = pred_gal[pred_gal["image"].isin(images_mask)]
    filter_true_gal = true_gal[true_gal["image"].isin(images_mask)]

    return filter_pred_avg, filter_true_avg, filter_detections, filter_pred_gal, filter_true_gal


def load_truth(dataset):
    NB_TOTAL = dataset.cardinality().numpy()
    all_params_dataset = dataset.map(
        lambda x: {key: x[key] for key in ["avg_g", "avg_obs_e", "gal_params", "keys"]}
    )

    all_img_params, all_gal_params = [], []
    for nb_image, example in tqdm(enumerate(tfds.as_numpy(all_params_dataset)), total=NB_TOTAL):
        df_gal, df_img = create_dfs(example, nb_image)
        all_img_params.append(df_img)
        all_gal_params.append(df_gal)

    true_avg = concat(all_img_params, ignore_index=True)
    true_gal = concat(all_gal_params, ignore_index=True)

    return true_avg, true_gal


def create_dfs(example, nb_image):
    gal_params = example["gal_params"]
    avg_g = example["avg_g"]
    avg_obs_e = example["avg_obs_e"]
    snc = example["keys"][1]
    df_gal = DataFrame(
        gal_params.T,
        columns=["int_e1", "int_e2", "obs_e1", "obs_e2", "x", "y", "mag", "hlr", "sersic"],
    )
    df_gal["image"] = nb_image
    df_gal["snc"] = snc

    dict_img = {
        "avg_g1": avg_g[0],
        "avg_g2": avg_g[1],
        "avg_obs_e1": avg_obs_e[0],
        "avg_obs_e2": avg_obs_e[1],
        "snc": snc,
        "image": nb_image,
    }
    df_img = DataFrame([dict_img])

    return df_gal, df_img


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


def decode_flags(flag):
    meanings = {
        1: "has neighbors (was deblended)",
        2: "was blended",
        4: "saturated pixel",
        8: "close to image border",
        16: "aperture data corrupted",
        32: "isophotal data corrupted",
        64: "memory overflow (deblending)",
        128: "memory overflow (extraction)",
    }
    return [meaning for bit, meaning in meanings.items() if flag & bit]


def compute_flags(pred_gal_mask):
    pyplot.hist(pred_gal_mask["source_flags"], bins=17, log=True)

    # images_with_flags = pred_gal_mask.loc[pred_gal_mask["source_flags"] != 0, "image"].unique()
    nonzero_flags = pred_gal_mask.loc[pred_gal_mask["source_flags"] != 0, "source_flags"].unique()
    return nonzero_flags


def _identity(x):
    return x


# --------------------------------------------------------------------------------------------------


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
    fig.savefig(plots_dir / f"test_fitting{title}.pdf", dpi=DPI)


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
    fig.savefig(plot_dir / "test_residuals.pdf", dpi=DPI)
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
    fig.savefig(plots_dir / "test_vignettes.pdf", dpi=DPI)


def x_n_seric(MIN_SERSIC, MAX_SERSIC, mu_x, sigma_x, plot_dir):
    def sersic_from_x(x):
        return (MAX_SERSIC * numpy.exp(x) + MIN_SERSIC) / (1 + numpy.exp(x))

    x_vals = numpy.linspace(-10, 10, 1000)
    sersic_vals = sersic_from_x(x_vals)
    prior_density = stats.norm.pdf(x_vals, loc=mu_x, scale=sigma_x)
    # prior_density_scaled = (
    #     prior_density * (max(sersic_vals) - min(sersic_vals)) / max(prior_density)
    # )

    fig, ax1 = pyplot.subplots(figsize=(8, 4))

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


def case_result(pred_e1, pred_e2, shears1, shears2, number, directory, symbol="gamma"):
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
        f"m = {poly_g1[0]:.2e} ± {numpy.sqrt(cov1[0, 0]):.2e} ; c = {poly_g1[1]:.2e} ± {numpy.sqrt(cov1[1, 1]):.2e}",  # noqa
    )
    lab_2 = (
        f"m = {poly_g2[0]:.2e} ± {numpy.sqrt(cov2[0, 0]):.2e} ; c = {poly_g2[1]:.2e} ± {numpy.sqrt(cov2[1, 1]):.2e}",  # noqa
    )

    axes[0].scatter(shears1, diff_pred_g1, alpha=alpha, color=c1, marker="x", s=s, label="e1")
    axes[0].scatter(shears2, diff_pred_g2, alpha=alpha, color=c2, marker="x", s=s, label="e2")
    axes[0].plot(t, poly_g1[1] + poly_g1[0] * t, color=c1, lw=5, label=lab_1)
    axes[0].plot(t, poly_g2[1] + poly_g2[0] * t, color=c2, lw=5, ls="--", label=lab_2)

    axes[1].scatter(shears1, pred_e1, alpha=alpha, color=c1, marker="x", s=s, label="e1")
    axes[1].scatter(shears2, pred_e2, alpha=alpha, color=c2, marker="o", s=s, label="e2")
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
        ax.legend(fontsize=9)

    fig.suptitle(
        f"{directory.name} : {pred_e1.shape[0]} images with {number} galaxies", fontsize=16
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(directory, dpi=300)


def histos(
    detections,
    preds,
    trues,
    pred_avg,
    true_avg,
    directory,
    pixel_scale,
    bins,
    density=True,
):
    fig, axes = pyplot.subplots(4, 4, figsize=(12, 12))
    alpha = 0.3

    nb_preds = preds["e1_model"].shape[0]
    nb_trues = trues["obs_e1"].shape[0]
    label_pred = f"preds : {nb_preds}"
    label_true = f"trues : {nb_trues}"

    axes[0, 0].hist(pred_avg["avg_obs_e1"], alpha=alpha, bins=bins, density=density, label="preds")
    axes[0, 0].hist(true_avg["avg_obs_e1"], alpha=alpha, bins=bins, density=density, label="true")
    axes[0, 0].set_title(f"avg e1 : {pred_avg['avg_obs_e1'].shape[0]}")

    axes[0, 1].hist(pred_avg["avg_obs_e2"], alpha=alpha, bins=bins, density=density, label="preds")
    axes[0, 1].hist(true_avg["avg_obs_e2"], alpha=alpha, bins=bins, density=density, label="true")
    axes[0, 1].set_title(f"avg e2 : {pred_avg['avg_obs_e2'].shape[0]}")

    axes[0, 2].hist(pred_avg["avg_obs_e1_err"], alpha=alpha, bins=10, label="e1", log=False)
    axes[0, 2].hist(pred_avg["avg_obs_e2_err"], alpha=alpha, bins=10, label="e2", log=False)
    axes[0, 2].set_title("avg e model err")

    axes[1, 0].hist(preds["e1_model"], alpha=alpha, bins=bins, density=density, label=label_pred)
    axes[1, 0].hist(trues["obs_e1"], alpha=alpha, bins=bins, density=density, label=label_true)
    axes[1, 0].set_title("e1")

    axes[1, 1].hist(preds["e2_model"], alpha=alpha, bins=bins, density=density, label=label_pred)
    axes[1, 1].hist(trues["obs_e2"], alpha=alpha, bins=bins, density=density, label=label_true)
    axes[1, 1].set_title("e2")

    axes[1, 2].hist(preds["e1_model_err"], alpha=alpha, bins=10, label="e1", log=True)
    axes[1, 2].hist(preds["e2_model_err"], alpha=alpha, bins=10, label="e2", log=True)
    axes[1, 2].set_title("e model err")

    new_flux = preds["flux_radius"] * pixel_scale
    axes[2, 0].hist(new_flux, alpha=alpha, bins=bins, density=density, label=label_pred)
    axes[2, 0].hist(trues["hlr"], alpha=alpha, bins=bins, density=density, label=label_true)
    axes[2, 0].set_title("hlr")

    axes[2, 1].hist(
        preds["sersic_model"], alpha=alpha, bins=bins, density=density, label=label_pred
    )
    axes[2, 1].hist(trues["sersic"], alpha=alpha, bins=bins, density=density, label=label_true)
    axes[2, 1].set_title("sersic")

    axes[2, 2].hist(preds["mag_model"], alpha=alpha, bins=bins, density=density, label=label_pred)
    axes[2, 2].hist(trues["mag"], alpha=alpha, bins=bins, density=density, label=label_true)
    axes[2, 2].set_title("mag")

    axes[3, 0].hist(
        preds["flux_radius"] * pixel_scale, alpha=alpha, bins=bins, label=label_pred, log=True
    )
    axes[3, 0].hist(trues["hlr"], alpha=alpha, bins=bins, label=label_true, log=True)
    axes[3, 0].set_title("logged hlr")

    axes[3, 1].hist(preds["sersic_model"], alpha=alpha, bins=bins, label=label_pred, log=True)
    axes[3, 1].hist(trues["sersic"], alpha=alpha, bins=bins, label=label_true, log=True)
    axes[3, 1].set_title("logged sersic")

    axes[3, 2].hist(preds["mag_model"], alpha=alpha, bins=bins, label=label_pred, log=True)
    axes[3, 2].hist(trues["mag"], alpha=alpha, bins=bins, label=label_true, log=True)
    axes[3, 2].set_title("logged magnitude")

    axes[0, 3].hist(detections, bins=bins, label=f"detections {numpy.sum(detections)}")
    axes[0, 3].set_title("galaxies")

    axes[1, 3].hist(
        preds["fmf_reduced_chi_2"], alpha=alpha, bins=bins, label="fmf_reduced_chi_2", log=True
    )
    axes[2, 3].hist(
        preds["fmf_stop_reason"], alpha=alpha, bins=bins, label="fmf_stop_reason", log=True
    )
    axes[3, 3].hist(preds["fmf_flags"], alpha=alpha, bins=bins, label="fmf_flags", log=True)

    for ax in axes.ravel():
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "histos.pdf", dpi=DPI)
    fig.show()


def background(names):
    img = fits.open(f"{names}.fits")[0].data.astype(numpy.float32)
    pyplot.title(f"background => Min: {numpy.min(img):.3f}, Max: {numpy.max(img):.3f}")
    im = pyplot.imshow(img, cmap="gray", vmax=numpy.max(img))
    pyplot.gca().invert_yaxis()
    pyplot.colorbar(im)


def result_e(directory, slice_pred, slice_truth, err=0):
    lim_min = min(slice_truth["e1_model"].min(), slice_pred["e1_model"].min())
    lim_max = max(slice_truth["e1_model"].max(), slice_pred["e1_model"].max())
    lim_1 = max(abs(lim_min), abs(lim_max))

    lim_min = min(slice_truth["e2_model"].min(), slice_pred["e2_model"].min())
    lim_max = max(slice_truth["e2_model"].max(), slice_pred["e2_model"].max())
    lim_2 = max(abs(lim_min), abs(lim_max))

    fig, axs = pyplot.subplots(2, 2, figsize=(10, 10))

    t_1 = numpy.linspace(slice_truth["e1_model"].min(), slice_truth["e1_model"].max())
    t_2 = numpy.linspace(slice_truth["e2_model"].min(), slice_truth["e2_model"].max())

    nb_mags = numpy.unique(slice_truth["mag_bin"]).shape[0]
    cmap = pyplot.get_cmap("tab20")
    colors = [cmap(i / nb_mags) for i in range(nb_mags)]

    for idx, mag_bin in enumerate(numpy.unique(slice_truth["mag_bin"])):
        MASK_BIN_MAG = slice_truth["mag_bin"] == mag_bin

        pred_mag_1 = slice_pred[MASK_BIN_MAG]["e1_model"].values
        true_mag_1 = slice_truth[MASK_BIN_MAG]["e1_model"].values

        diff_mag_1 = pred_mag_1 - true_mag_1
        poly_g, _ = numpy.polyfit(true_mag_1, diff_mag_1, 1, cov=True)
        axs[0, 0].plot(
            t_1,
            poly_g[1] + poly_g[0] * t_1,
            color=colors[idx],
            lw=5,
            ls="--",
            label=f"mag {idx + 1}",
        )
        axs[0, 0].scatter(true_mag_1, diff_mag_1, s=5, alpha=0.1, color=colors[idx])
        axs[0, 0].axhline(0, color="black", linestyle="--")
        axs[0, 0].set_xlabel("True")
        axs[0, 0].set_ylabel("Pred - True")
        axs[0, 0].set_title("Residuals of e1")
        axs[0, 0].legend(loc="upper right")

        axs[0, 1].errorbar(true_mag_1, pred_mag_1, yerr=err, fmt="x", alpha=0.25, color=colors[idx])
        axs[0, 1].plot([-lim_1, lim_1], [-lim_1, lim_1], "k--", zorder=10)
        axs[0, 1].set_xlim(-lim_1, lim_1)
        axs[0, 1].set_ylim(-lim_1, lim_1)
        axs[0, 1].set_xlabel("True")
        axs[0, 1].set_ylabel("Pred")
        axs[0, 1].set_title("Predicted vs True e1")

        pred_mag_2 = slice_pred[MASK_BIN_MAG]["e2_model"].values
        true_mag_2 = slice_truth[MASK_BIN_MAG]["e2_model"].values

        diff_mag_2 = pred_mag_2 - true_mag_2
        poly_g_2, _ = numpy.polyfit(true_mag_2, diff_mag_2, 1, cov=True)
        axs[1, 0].plot(t_2, poly_g_2[1] + poly_g_2[0] * t_2, color=colors[idx], lw=5, ls="--")
        axs[1, 0].scatter(true_mag_2, diff_mag_2, s=5, alpha=0.1, color=colors[idx])
        axs[1, 0].axhline(0, color="black", linestyle="--")
        axs[1, 0].set_xlabel("True")
        axs[1, 0].set_ylabel("Pred - True")
        axs[1, 0].set_title("Residuals of e2")

        axs[1, 1].errorbar(true_mag_2, pred_mag_2, yerr=err, fmt="x", alpha=0.25, color=colors[idx])
        axs[1, 1].plot([-lim_2, lim_2], [-lim_2, lim_2], "k--", zorder=10)
        axs[1, 1].set_xlim(-lim_2, lim_2)
        axs[1, 1].set_ylim(-lim_2, lim_2)
        axs[1, 1].set_xlabel("True")
        axs[1, 1].set_ylabel("Pred")
        axs[1, 1].set_title("Predicted vs True e2")

    fig.tight_layout()
    fig.savefig(directory / "res_ellips.pdf", dpi=DPI)
    fig.show()


def case_result_param(directory, pred, truth, PIXEL_SCALE):
    fig, axs = pyplot.subplots(2, 5, figsize=(20, 8))

    params = [
        ("sersic", "sersic_model"),
        ("mag", "mag_model"),
        ("hlr", "flux_radius"),
        ("x", "x_model"),
        ("y", "y_model"),
    ]

    for i, (true_key, pred_key) in enumerate(params):
        y_pred = pred[pred_key]
        if pred_key == "flux_radius":
            y_pred *= PIXEL_SCALE
        y_true = truth[true_key]

        axs[0, i].scatter(y_true, y_pred - y_true, s=2, alpha=0.5)
        axs[0, i].axhline(0, color="black", linestyle="--")
        axs[0, i].set_xlabel("True")
        axs[0, i].set_ylabel("Pred - True")
        axs[0, i].set_title(f"Residuals of {true_key}")

        # axs[1, i].errorbar(y_true, y_pred, yerr=err, fmt="x", capsize=2, markersize=3, alpha=0.5)
        axs[1, i].errorbar(y_true, y_pred, fmt="x", capsize=2, markersize=3, alpha=0.5)

        lim_min = min(y_pred.min(), y_true.min())
        lim_max = max(y_pred.max(), y_true.max())
        axs[1, i].plot([lim_min, lim_max], [lim_min, lim_max], "k--", zorder=10)
        axs[1, i].set_xlim(lim_min, lim_max)
        axs[1, i].set_ylim(lim_min, lim_max)
        axs[1, i].set_xlabel("True")
        axs[1, i].set_ylabel("Pred")

    fig.tight_layout()
    fig.savefig(directory / "res_params_galaxies.pdf", dpi=DPI)
    fig.show()


def results_by_image(directory, pred_avg, true_avg, alpha, name):
    fig, axes = pyplot.subplots(1, 2, figsize=(12, 5), sharey=False)

    for i in range(2):
        x_avg_g = true_avg[f"avg_g{i + 1}"]
        y_avg_g = pred_avg[f"avg_obs_e{i + 1}"] - x_avg_g
        if name == "pred":
            y_err = pred_avg[f"avg_obs_e{i + 1}_err"]
            w = 1.0 / y_err
            (m, c), cov = numpy.polyfit(x_avg_g, y_avg_g, deg=1, w=w, cov=True)
        else:
            y_err = 0
            (m, c), cov = numpy.polyfit(x_avg_g, y_avg_g, deg=1, cov=True)

        sigma_m, sigma_c = numpy.sqrt(numpy.diag(cov))

        t = numpy.linspace(-0.06, 0.06)
        y_fit = m * t + c
        sigma_fit = numpy.sqrt(t**2 * cov[0, 0] + cov[1, 1] + 2 * t * cov[0, 1])

        axes[0].errorbar(
            x_avg_g, y_avg_g, yerr=y_err, fmt="o", ms=2, capsize=3, label=f"e{i + 1}", alpha=alpha
        )

        label = f"m = {m:.2e} ± {sigma_m:.2e} / c = {c:.2e} ± {sigma_c:.2e}"
        axes[1].plot(t, y_fit, label=label, zorder=10)
        if name == "pred":
            axes[1].fill_between(
                t, y_fit - sigma_fit, y_fit + sigma_fit, alpha=alpha, label="uncertainty"
            )

        axes[i].set_xlabel("True avg shear")
        axes[i].set_ylabel("predicted avg ellips - true avg shear")
        axes[i].axhline(0, color="black", linestyle="--", linewidth=1, label="y=0")
        axes[0].legend(fontsize=9)
        axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(directory / f"results_avg_per_image_{name}.pdf", dpi=DPI)
    fig.show()


def build_single_column_groups(df, column, possible_values):
    groups = {}
    for x in possible_values:
        mask = df[column] == x
        groups[float(x)] = df.index[mask]

    return groups


def compute_ellips(ellips, ellips_err):
    all_zero = (ellips_err == 0).all()
    if all_zero:
        mu = numpy.mean(ellips)
        err = 0
    else:
        weight = 1.0 / (ellips_err**2 + 0.26**2)
        mu = numpy.sum(weight * ellips) / numpy.sum(weight)

        # mu2 = numpy.sum(weight * ellips**2) / numpy.sum(weight)
        # variance = mu2 - mu**2
        # err = numpy.sqrt(variance / numpy.sum(weight) ** 2 / numpy.sum(weight**2))
        # err = numpy.sqrt(variance) / numpy.sqrt(ellips.shape[0])

        err = numpy.std(ellips) / numpy.sqrt(ellips.shape[0])

    return mu, err


def compute_by_shear_bin(true_avg, pred_gal, group, model):
    res_pred_gal, res_pred_gal_err, res_true_gal = {}, {}, {}
    for bin_label, idx_list in group.items():
        mu_e_true = true_avg.loc[idx_list][f"avg_obs_{model}"].mean()

        filtered_pred = pred_gal[pred_gal["image"].isin(idx_list)]

        ellips = filtered_pred[f"{model}_model"]
        ellips_err = filtered_pred[f"{model}_model_err"]
        mu, err = compute_ellips(ellips, ellips_err)

        res_pred_gal[bin_label] = mu
        res_pred_gal_err[bin_label] = err
        res_true_gal[bin_label] = mu_e_true

    df = DataFrame({"pred_e": res_pred_gal, "pred_e_err": res_pred_gal_err, "true_e": res_true_gal})
    df.index.name = "true_g"
    df = df.reset_index()

    return df


def compute_by_bin(true_avg, pred_gal):
    possible = numpy.sort(true_avg["avg_g1"].unique())

    groups_g1 = build_single_column_groups(true_avg, "avg_g1", possible)
    groups_g2 = build_single_column_groups(true_avg, "avg_g2", possible)

    df_1 = compute_by_shear_bin(true_avg, pred_gal, groups_g1, "e1")
    df_2 = compute_by_shear_bin(true_avg, pred_gal, groups_g2, "e2")

    return df_1, df_2


def result_by_bin(directory, df_e1, df_e2, style, name):
    # def function(x, m, c):
    #     return m * x + c

    fig, axes = pyplot.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes = axes.flatten()

    for i, df in enumerate([df_e1, df_e2]):
        x = df[style]
        y = df["pred_e"] - x
        if name == "pred":
            y_err = df["pred_e_err"]
            w = 1 / y_err
            (m, c), cov = numpy.polyfit(x, y, deg=1, w=w, cov=True)
        else:
            y_err = 0
            (m, c), cov = numpy.polyfit(x, y, deg=1, cov=True)

        sigma_m, sigma_c = numpy.sqrt(numpy.diag(cov))
        # t = numpy.linspace(-0.06, 0.06)
        # y_fit = m * t + c

        # (m, c), cov = optimize.curve_fit(function, x, y, sigma=y_err)
        y_fit = m * x + c
        # sigma_m, sigma_c = numpy.sqrt(numpy.diag(cov))

        # y_fit = m * t + c
        # sigma_fit = numpy.sqrt(t**2 * cov[0, 0] + cov[1, 1] + 2 * t * cov[0, 1])

        label = f"m = {m:.2e} ± {sigma_m:.2e} / c = {c:.2e} ± {sigma_c:.2e}"

        axes[i].errorbar(x, y, yerr=y_err, fmt="o", capsize=3, label="Data")
        # axes[i].plot(t, y_fit, label=label)
        axes[i].plot(x, y_fit, label=label)
        # axes[i].plot(t, y_fit, label=label)
        # axes[i].fill_between(
        #     t, y_fit - sigma_fit, y_fit + sigma_fit, alpha=0.3, label="uncertainty"
        # )

        axes[i].set_xlabel(f"Shear bin center {style}")
        axes[i].set_ylabel("predicted - true")
        axes[i].set_title(f"Residuals of total mean e{i + 1}")
        # axes[i].set_ylim(-0.005, 0.005)
        axes[i].axhline(0, color="black", linestyle="--", linewidth=1, label="y=0")
        axes[i].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(directory / f"res_{style}_{name}.pdf", dpi=DPI)
    fig.show()
