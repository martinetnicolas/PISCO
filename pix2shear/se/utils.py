from pathlib import Path

import tensorflow_datasets as tfds
from astropy.io import fits
from jax import numpy
from matplotlib import pyplot
from numpy import loadtxt
from pandas import read_csv


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


def tfds_info(style, version, data_dir, split=True):
    if split:
        version = version.split("_", 1)[0]

    builder = tfds.builder(
        name=f"{style}:{version}",
        data_dir=data_dir,
    )

    return builder.info


def ellips_model(e1_err, e2_err, e1, e2):
    e1_err = e1_err.to_numpy()
    e2_err = e2_err.to_numpy()
    e1 = e1.to_numpy()
    e2 = e2.to_numpy()

    weight_e1 = 1 / numpy.sqrt(e1_err**2 + 0.26**2)
    e1_weighted_mean = numpy.sum(weight_e1 * e1) / numpy.sum(weight_e1)

    weight_e2 = 1 / numpy.sqrt(e2_err**2 + 0.26**2)
    e2_weighted_mean = numpy.sum(weight_e2 * e2) / numpy.sum(weight_e2)

    return float(e1_weighted_mean.mean()), float(e2_weighted_mean.mean())


def ellips_se(e, theta):
    e = e.to_numpy()
    theta = theta.to_numpy()

    rad = numpy.deg2rad(theta)
    e1 = e * numpy.cos(2 * rad)
    e2 = e * numpy.sin(2 * rad)
    return float(e1.mean()), float(e2.mean())


def ldac(psf_dir, psf_name):
    with fits.open(psf_dir / f"{psf_name}.ldac") as hdul:
        hdul.info()
        LDAC_IMHEAD = hdul[1].data
        objects_data = hdul[2].data

    print("\n=== LDAC_IMHEAD (HDU[1]) ===")
    for i, row in enumerate(LDAC_IMHEAD):
        print(f"Row {i + 1}: {row}")

    return objects_data


def gauss_conv(file_path, plots_dir):
    with open(file_path, "r") as f:
        lines = f.readlines()

    title = lines[1].strip()

    data = numpy.array([list(map(float, line.split())) for line in lines[2:]])

    fig, ax = pyplot.subplots(figsize=(3, 3))

    im = ax.imshow(data, cmap="viridis")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Weight")

    fig.savefig(plots_dir / "gauss_conv.pdf", dpi=300)
    fig.suptitle(title, fontsize=10)
    fig.show()


def hdul_object_ldac(objects_data, plots_dir):
    for i, row in enumerate(objects_data.names):
        if row != "VIGNET":
            print(f"{row} {objects_data[row][0]}")

    fig, ax = pyplot.subplots(figsize=(3, 3))

    im = ax.imshow(objects_data["VIGNET"][0][5:35, 5:35], cmap="viridis")

    fig.suptitle("VIGNET")
    ax.invert_yaxis()

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Weight")

    fig.tight_layout()
    fig.savefig(plots_dir / "vignet.pdf", dpi=300)
    fig.show()

    return objects_data["VIGNET"][0].shape[0]


def extra_psf_images(datasets_psf_dir, psf_name, psf_dir, plots_dir, loggy=True):
    func = numpy.log if loggy else identity
    sup_title = "logged" if loggy else "normal"

    snap = func(
        fits.open(psf_dir / f"psfex.snapshots_{psf_name}.fits")[0].data.astype(numpy.float32)
    )
    proto = func(
        fits.open(psf_dir / f"psfex.prototypes_{psf_name}.fits")[0].data.astype(numpy.float32)
    )
    psfex = func(
        fits.open(psf_dir / f"{psf_name}.psf")[1].data["PSF_MASK"][0][0].astype(numpy.float32)
    )
    psf = func(fits.open(datasets_psf_dir / f"{psf_name}.fits")[0].data.astype(numpy.float32))
    objects = func(fits.open(psf_dir / f"{psf_name}.obj.fits")[0].data.astype(numpy.float32))

    titles = ["snapshots", "prototypes", "psf_psfex", "psf", "objects"]
    images = [snap, proto, psfex, psf, objects]

    fig, axes = pyplot.subplots(1, 5, figsize=(15, 3))

    for i, (ax, img, title) in enumerate(zip(axes.flat, images, titles)):
        im = ax.imshow(img, cmap="viridis", origin="lower")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes.flat[len(images) :]:
        ax.axis("off")

    fig.suptitle(sup_title, fontsize=16)
    fig.tight_layout()
    fig.savefig(plots_dir / f"psf_extra_{sup_title}.pdf", dpi=300)
    fig.show()


def extra_psf_images_reshaped(psf_dir, psf_name, shape_vignet, plots_dir, style):
    resi = fits.open(psf_dir / f"psfex.{style}_{psf_name}.fits")[0].data.astype(numpy.float32)

    num_vignet = resi.shape[1] // shape_vignet

    resi_vignettes = resi.reshape(num_vignet, shape_vignet, shape_vignet)

    fig, axes = pyplot.subplots(
        2, num_vignet, figsize=(20, 8), gridspec_kw={"height_ratios": [2, 1]}
    )

    ax = fig.add_subplot(2, 1, 1)
    im = ax.imshow(resi, cmap="viridis", origin="lower")
    ax.set_xlabel("X pixels")
    ax.set_ylabel("Y pixels")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Intensity")
    ax.axis("on")

    for i in range(num_vignet):
        pyplot.delaxes(axes[0, i])

    for i in range(num_vignet):
        ax = axes[1, i]
        ax.imshow(resi_vignettes[i], cmap="viridis", origin="lower")
        ax.set_title(f"V {i + 1}", fontsize=10)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Intensity")

    fig.suptitle(f"{style} / {num_vignet} vignet {resi_vignettes.shape}", fontsize=16)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{style}_reshaped.pdf", dpi=300)
    fig.show()


def background(names):
    img = fits.open(f"{names}.fits")[0].data.astype(numpy.float32)
    pyplot.title(f"background => Min: {numpy.min(img):.3f}, Max: {numpy.max(img):.3f}")
    im = pyplot.imshow(img, cmap="gray", vmax=numpy.max(img))
    pyplot.gca().invert_yaxis()
    pyplot.colorbar(im)


def identity(x):
    return x


def image_fitting(dataset_dir, images_dir, choice, plots_dir, transformed=False):
    if transformed:
        func = numpy.arcsinh
        title = "arcsinh"
    else:
        func = identity
        title = "normal"

    file_keys = ["model", "obj", "seg", "submodel", "subobj"]
    file_paths = {key: images_dir / f"{key}_{choice}" for key in file_keys}
    file_paths["image"] = dataset_dir / choice

    assoc = loadtxt(f"{file_paths['image']}_assoc.txt")

    data = {
        key: (
            func(fits.open(f"{path}.fits")[0].data.astype(numpy.float32))
            if transformed
            else fits.open(f"{path}.fits")[0].data.astype(numpy.float32)
        )
        for key, path in file_paths.items()
    }

    fig, axes = pyplot.subplots(2, 3, figsize=(12, 8))
    axes = axes.flat

    for i, key in enumerate(data.keys()):
        im = axes[i].imshow(data[key])
        axes[i].set_title(key)
        axes[i].invert_yaxis()
        axes[i].scatter(assoc[:, 0], assoc[:, 1], color="red", marker="o", s=2, label="assoc")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.show()
    fig.savefig(plots_dir / f"image_fitting_{title}_{choice}.pdf", dpi=300)


def case_result(pred_e, shears, directory, symbol="gamma"):
    s = 10
    alpha = 0.2
    c1, c2 = "C0", "C1"
    t = numpy.linspace(shears.min(), shears.max())

    diff_pred_g1 = pred_e[:, 0] - shears[:, 0]
    diff_pred_g2 = pred_e[:, 1] - shears[:, 1]
    poly_g1 = numpy.polyfit(shears[:, 0], diff_pred_g1, 1)
    poly_g2 = numpy.polyfit(shears[:, 1], diff_pred_g2, 1)

    fig, axes = pyplot.subplots(1, 2, figsize=(12, 5))
    axes = axes.flatten()

    lab_1 = f"m= {poly_g1[0]:.3f}, c={poly_g1[1]:.3f}"
    axes[0].scatter(shears[:, 0], diff_pred_g1, alpha=alpha, color=c1, marker="x", s=s, label=lab_1)
    lab_2 = f"m= {poly_g2[0]:.3f}, c={poly_g2[1]:.3f}"
    axes[0].scatter(shears[:, 1], diff_pred_g2, alpha=alpha, color=c2, marker="o", s=s, label=lab_2)
    axes[0].plot(t, poly_g1[1] + poly_g1[0] * t, color=c1)
    axes[0].plot(t, poly_g2[1] + poly_g2[0] * t, color=c2)

    axes[1].scatter(shears[:, 0], pred_e[:, 0], alpha=alpha, color=c1, marker="x", s=s)
    axes[1].scatter(shears[:, 1], pred_e[:, 1], alpha=alpha, color=c2, marker="o", s=s)
    lims = axes[1].get_xlim()
    axes[1].plot(lims, lims, color="black", linestyle="--", linewidth=1, label="y=x")

    axes[0].axhline(0, color="black", linestyle="--", linewidth=1, label="y=0")

    axes[0].set_ylabel(rf"$\epsilon^{{obs}}$ - $\{symbol}^{{true}}$", rotation=0, labelpad=20)
    axes[1].set_ylabel(r"$\epsilon^{obs}$", rotation=0, labelpad=20)

    for ax in axes:
        if symbol == "gamma":
            ax.set_xlim(-0.055, 0.055)
        else:
            ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        ax.set_xlabel(rf"$\{symbol}^{{true}}$")
        ax.legend()

    fig.suptitle(f"{directory.name}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.tight_layout()
    fig.savefig(directory, dpi=300)


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
    preds_ellps,
    true_ellps,
    directory,
    pixel_scale,
    bins=50,
    loggy=False,
):
    fig, axes = pyplot.subplots(3, 4, figsize=(16, 12))
    alpha = 0.3

    axes[0, 0].hist(preds_ellps[:, 0], alpha=alpha, bins=bins, label="predictions")
    axes[0, 0].hist(true_ellps[:, 0], alpha=alpha, bins=bins, label="truth")
    axes[0, 0].set_title("e1 model (weighted errors)")

    axes[0, 1].hist(preds_ellps[:, 1], alpha=alpha, bins=bins, label="predictions")
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
        if loggy:
            ax.set_yscale("log")
        if idx < len(axes.ravel()) - 2:
            ax.legend()
    fig.delaxes(axes[2, 2])
    fig.delaxes(axes[2, 3])
    fig.tight_layout()
    fig.savefig(directory / "histos.pdf", dpi=300)
    fig.show()


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
    fig.savefig(directory / f"results_{symbol}.pdf", dpi=300)


def delete_duplicate(df_pred):
    df_pred = df_pred.filter(df_pred["NUMBER_ASSOC"] == 1)

    duplicated_mask = df_pred["assoc1"].is_duplicated()

    df_duplicates = df_pred.filter(duplicated_mask)

    distance = (
        (df_duplicates["X_IMAGE"] - df_duplicates["assoc1"]) ** 2
        + (df_duplicates["Y_IMAGE"] - df_duplicates["assoc2"]) ** 2
    ).sqrt()

    df_duplicates = df_duplicates.with_columns(distance.alias("distance"))

    df_best = (
        df_duplicates.sort("distance").unique(subset=["assoc1"], keep="first").drop("distance")
    )

    df_pred = df_pred.filter(~duplicated_mask).vstack(df_best).sort("assoc1")

    return df_pred
