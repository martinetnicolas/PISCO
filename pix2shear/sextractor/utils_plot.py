from astropy.io import fits
from jax import numpy
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import X_sersic


def plot_vignettes(vignettes_images, pos_x, pos_y, directory):
    fig, axes = pyplot.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()

    for i, (fits_file, x, y) in enumerate(zip(vignettes_images[:12], pos_x, pos_y)):
        image = fits.open(fits_file)[0].data
        img = axes[i].imshow(image, cmap="gray")
        axes[i].set_title(f"{fits_file.stem} {x:.2f} {y:.2f}")
        divider = make_axes_locatable(axes[i])
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        pyplot.colorbar(img, cax=cax1)
        axes[i].axis("off")

    for j in range(len(vignettes_images), 12):
        axes[j].axis("off")

    fig.tight_layout()
    fig.show()
    fig.savefig(directory / "vignettes.pdf", dpi=300)


def check_psf(file, directory):
    img_psf = fits.open(file)[0].data
    fig, axes = pyplot.subplots(1, 2, figsize=(12, 6))
    im1 = axes[0].imshow(numpy.log(img_psf.astype(numpy.float32)), cmap="gray")
    axes[0].set_title("Airy Logarithmic Scale")
    divider = make_axes_locatable(axes[0])
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    pyplot.colorbar(im1, cax=cax1)
    im2 = axes[1].imshow(img_psf, cmap="gray")
    axes[1].set_title("Airy Linear Scale")
    divider = make_axes_locatable(axes[1])
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    pyplot.colorbar(im2, cax=cax2)
    pyplot.show()
    fig.savefig(directory / "psf.pdf", dpi=300)


def print_fits(names, titles, directory, func=numpy.arcsinh):
    fig, axes = pyplot.subplots(2, 4, figsize=(20, 8))
    fig.suptitle("FITS Image Grid", fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < len(names):
            img = fits.open(f"{names[i]}.fits")[0].data
            img = img.astype(numpy.float32)

            if i in {0, 1}:
                img = func(img) / 3

            ax.set_title(f"{titles[i]} => Min: {numpy.min(img):.3f}, Max: {numpy.max(img):.3f}")
            im = ax.imshow(img, cmap="gray", vmax=numpy.max(img))
            fig.colorbar(im, ax=ax)
        else:
            ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.show()
    fig.savefig(directory / "output_sextractor.pdf", dpi=300)


def plot_sersic(X_min, X_max, directory):
    sersic_values = numpy.linspace(0.1, 7, 500)
    X_values = X_sersic(sersic_values)
    fig, ax1 = pyplot.subplots(figsize=(8, 5))
    ax1.plot(sersic_values, X_values, label=r"$X_{\mathrm{sersic}}$", color="b")
    ax1.axhline(X_min, color="r", linestyle="--", label="Prior lower bound")
    ax1.axhline(X_max, color="g", linestyle="--", label="Prior upper bound")
    ax1.fill_between(sersic_values, X_min, X_max, color="gray", alpha=0.2, label="Allowed range")
    ax1.set_xlabel("Sersic Index (n)")
    ax1.set_ylabel(r"$X_{\mathrm{sersic}}$")
    ax1.set_title("Mapping of Sersic Index to X_sersic with Prior for numerical stability?")
    ax1.legend()
    ax1.grid(True)
    fig.show()
    fig.savefig(directory / "sersic.pdf", dpi=300)


def histo_err(diff_e1, diff_e2, directory):
    _, ax = pyplot.subplots(figsize=(5, 5))
    pyplot.hist(diff_e1, bins=50, alpha=0.7, label="Diff e1", color="blue")
    pyplot.hist(diff_e2, bins=50, alpha=0.7, label="Diff e2", color="red")
    temp = 0.05
    ax.set_xlim(-temp, temp)
    pyplot.legend()
    pyplot.savefig(directory / "sersic.pdf", dpi=300)
    pyplot.show()


def hist_gal(data, nb_not_detected, directory):
    pyplot.hist(data, bins=5, edgecolor="black", label="galaxies detected")
    pyplot.legend()
    pyplot.title(f"{nb_not_detected} not correct")
    pyplot.savefig(directory / "detected_gal.pdf", dpi=300)
    pyplot.show()


def final_results(true_e, pred_e, shears, directory):
    s = 10
    alpha = 0.2
    c1, c2 = "C0", "C1"
    t = numpy.linspace(shears.min(), shears.max())
    max_plot1 = max(numpy.max(true_e - shears), numpy.max(pred_e - shears))
    max_plot2 = max(numpy.max(pred_e), numpy.max(true_e))

    diff_pred_g1 = pred_e[:, 0] - shears[:, 0]
    diff_pred_g2 = pred_e[:, 1] - shears[:, 1]
    poly_g1 = numpy.polyfit(shears[:, 0], diff_pred_g1, 1)
    poly_g2 = numpy.polyfit(shears[:, 1], diff_pred_g2, 1)

    res_true_g1 = true_e[:, 0] - shears[:, 0]
    res_true_g2 = true_e[:, 1] - shears[:, 1]
    poly_g1_true = numpy.polyfit(shears[:, 0], res_true_g1, 1)
    poly_g2_true = numpy.polyfit(shears[:, 1], res_true_g2, 1)

    fig, axes = pyplot.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    lab_1 = f"m= {poly_g1[0]:.3f}, c={poly_g1[1]:.3f}"
    axes[0].scatter(shears[:, 0], diff_pred_g1, alpha=alpha, color=c1, marker="x", s=s, label=lab_1)
    lab_2 = f"m= {poly_g2[0]:.3f}, c={poly_g2[1]:.3f}"
    axes[0].scatter(shears[:, 1], diff_pred_g2, alpha=alpha, color=c2, marker="o", s=s, label=lab_2)
    axes[0].plot(t, poly_g1[1] + poly_g1[0] * t, color=c1)
    axes[0].plot(t, poly_g2[1] + poly_g2[0] * t, color=c2)
    axes[0].set_ylim(-max_plot1, max_plot1)

    axes[1].scatter(shears[:, 0], pred_e[:, 0], alpha=alpha, color=c1, marker="x", s=s)
    axes[1].scatter(shears[:, 1], pred_e[:, 1], alpha=alpha, color=c2, marker="o", s=s)
    lims = axes[1].get_xlim()
    axes[1].plot(lims, lims, color="black", linestyle="--", linewidth=1, label="y=x")
    axes[1].set_ylim(-max_plot2, max_plot2)

    lab_3 = f"m= {poly_g1_true[0]:.3f}, c={poly_g1_true[1]:.3f}"
    axes[2].scatter(shears[:, 0], res_true_g1, alpha=alpha, color=c1, marker="x", s=s, label=lab_3)
    lab_4 = f"m= {poly_g2_true[0]:.3f}, c={poly_g2_true[1]:.3f}"
    axes[2].scatter(shears[:, 1], res_true_g2, alpha=alpha, color=c2, marker="o", s=s, label=lab_4)
    axes[2].plot(t, poly_g1_true[1] + poly_g1_true[0] * t, color=c1)
    axes[2].plot(t, poly_g2_true[1] + poly_g2_true[0] * t, color=c2)
    axes[2].set_ylim(-max_plot1, max_plot1)

    axes[3].scatter(shears[:, 0], true_e[:, 0], alpha=alpha, color=c1, marker="x", s=s)
    axes[3].scatter(shears[:, 1], true_e[:, 1], alpha=alpha, color=c2, marker="o", s=s)
    lims = axes[3].get_xlim()
    axes[3].plot(lims, lims, color="black", linestyle="--", linewidth=1, label="y=x")
    axes[3].set_ylim(-max_plot2, max_plot2)

    axes[0].axhline(0, color="black", linestyle="--", linewidth=1, label="y=0")
    axes[2].axhline(0, color="black", linestyle="--", linewidth=1, label="y=0")

    axes[0].set_ylabel(r"$\epsilon^{{obs}}$ - $\gamma^{{true}}$", rotation=0, labelpad=20)
    axes[1].set_ylabel(r"$\epsilon^{{obs}}$", rotation=0, labelpad=20)
    axes[2].set_ylabel(r"$\epsilon^{{true}}$ - $\gamma^{{true}}$", rotation=0, labelpad=20)
    axes[3].set_ylabel(r"$\epsilon^{{true}}$", rotation=0, labelpad=20)

    for ax in axes:
        ax.set_xlim(-0.055, 0.055)
        ax.set_xlabel(r"$\gamma^{true}$")
        ax.legend()

    fig.tight_layout()
    fig.savefig(directory / "results_sextractor.pdf", dpi=300)
