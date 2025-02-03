import re
from time import time

import scipy
import seaborn
from blocks import EncoderPool
from jax import numpy, random
from matplotlib import gridspec, pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable


def loss_optimiser(
    train_losses, val_losses, directory, temps, steps, first_step=1, last_step=10000, ylims=None
):
    epochs = numpy.arange(1, len(train_losses) + 1)

    _, ax1 = pyplot.subplots(figsize=(6, 5))

    ax1.set_xlabel("steps")
    ax1.set_ylabel("Loss")
    ax1.plot(
        epochs[first_step:last_step],
        train_losses[first_step:last_step],
        "b-",
        label="Train",
    )
    ax1.plot(
        epochs[first_step:last_step],
        val_losses[first_step:last_step],
        "g-",
        label="Val",
    )
    ax1.tick_params(axis="y", labelcolor="black")
    if ylims is not None:
        ax1.set_ylim(ylims[0], ylims[1])
    ax1.legend(loc="lower left")

    if steps is not None:
        ax2 = ax1.twinx()
        ax2.plot(
            epochs[first_step:last_step],
            steps[first_step:last_step],
            "r--",
            label="step value",
        )
        ax2.set_ylim(0, numpy.max(steps) * 3 / 2)
        ax2.tick_params(axis="y", labelcolor="r")
        ax2.legend(loc="upper right")

    pyplot.title(f"Training & Validation Losses with {temps} h")
    pyplot.savefig(directory / "losses.pdf", dpi=300)


def histos(weights, means, maxies_1, maxies_2, bins, style, directory):
    fig, ax = pyplot.subplots(1, 3, figsize=(18, 6))

    if style == "uni":
        ax[0].hist(weights[:, 0].flatten(), color="r", bins=bins, alpha=0.7, label="g1")
        ax[0].hist(weights[:, 1].flatten(), color="b", bins=bins, alpha=0.5, label="g2")
        ax[0].legend()
    elif style == "multi":
        ax[0].hist(weights.flatten(), color="g", bins=bins, alpha=0.5)
    ax[0].set_title(f" WEIGHTS for {weights.shape[-1]} components")
    ax[0].set_ylabel("Frequency")

    ax[1].hist(means[:, 0].flatten(), color="r", bins=bins, alpha=0.7, label="g1")
    ax[1].hist(means[:, 1].flatten(), color="b", bins=bins, alpha=0.5, label="g2")
    ax[1].set_title(f" MEANS for {means.shape[-1]} components")
    ax[1].legend()

    ax[2].hist(maxies_1.flatten(), color="r", bins=bins, alpha=0.7, label="g1")
    ax[2].hist(maxies_2.flatten(), color="b", bins=bins, alpha=0.5, label="g2")
    ax[2].set_title(" means wrt max weights")
    ax[2].legend()

    fig.tight_layout()
    fig.show()
    fig.savefig(directory / "histos.pdf", dpi=300)


def error_bar(truths, modes, stddevs_gmm, directory):
    list_percentage = [0, 68, 95, 99.7]
    list_percentage_g1, list_percentage_g2 = [], []
    total = modes.shape[0]
    for nb in range(len(list_percentage)):
        percentage_low = modes - (nb) * stddevs_gmm / 2
        percentage_up = modes + (nb) * stddevs_gmm / 2
        inside_bounds = (truths >= percentage_low) & (truths <= percentage_up)
        list_percentage_g1.append(numpy.sum(inside_bounds[:, 0]) / total * 100)
        list_percentage_g2.append(numpy.sum(inside_bounds[:, 1]) / total * 100)

    pyplot.figure(figsize=(6, 5))
    pyplot.plot(numpy.array(list_percentage), numpy.array(list_percentage_g1), "b-*", label="g1")
    pyplot.plot(numpy.array(list_percentage), numpy.array(list_percentage_g2), "r-*", label="g2")
    pyplot.plot(numpy.array(list_percentage), numpy.array(list_percentage), "k--", label="x = y")
    pyplot.xlabel("percentage nominal")
    pyplot.ylabel("percentage empirical")
    _ = pyplot.legend()
    pyplot.savefig(directory / "error_bars_test.pdf", dpi=300)


def plot_examples(predictions, img, inputs_vars, compress, name, directory):
    if compress:
        func = numpy.arcsinh
    else:
        func = lambda x: x  # noqa: E731

    pyplot.figure(figsize=(20, 7))
    gs = gridspec.GridSpec(1, 4, width_ratios=[2, 1, 2, 1])

    for i in range(predictions.shape[0]):
        # picture
        ax_image = pyplot.subplot(gs[2 * i])
        ax_image.imshow(func(img[i]), cmap="viridis", interpolation=None)
        formatted_arr = [f"{x:.3f}" for x in predictions[i]]
        ax_image.set_title(f"predicted shear {formatted_arr}")
        ax_image.axis("off")

        # table
        ax_table = pyplot.subplot(gs[2 * i + 1])
        ax_table.axis("off")
        variables = inputs_vars.columns
        values = inputs_vars.to_numpy()[i]
        formatted_values = [f"{x:.3f}" if isinstance(x, (int, float)) else x for x in values]
        table_data = [[variables[j], formatted_values[j]] for j in range(len(variables))]
        table = ax_table.table(
            cellText=table_data,
            colLabels=["VARIABLES", "VALUES"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

    pyplot.suptitle(f"{name}", fontsize="25", y=1.0)
    pyplot.tight_layout()
    pyplot.savefig(directory / f"examples{name}.pdf", dpi=300)


def combined_plots(mode, truth, bins, directory):
    erreurs = mode - truth
    fig, axs = pyplot.subplots(1, 2, figsize=(10 * 2, 5))

    grid_x, grid_y = numpy.mgrid[-0.05:0.05:300j, -0.05:0.05:300j]
    extent = (-0.05, 0.05, -0.05, 0.05)
    vmax = 0.1

    errors = numpy.linalg.norm(erreurs, axis=1)

    grid_z = scipy.interpolate.griddata(
        points=(truth[:, 1], truth[:, 0]),
        values=errors,
        xi=(grid_x, grid_y),
        method="linear",
    )

    im = axs[0].imshow(grid_z, extent=extent, origin="lower", cmap="viridis_r", vmax=vmax)

    axs[0].set_xlabel("true g1")
    axs[0].set_ylabel("true g2")
    axs[0].set_xlim(-0.05, 0.05)
    axs[0].set_ylim(-0.05, 0.05)
    axs[0].set_title("Modes")
    pyplot.colorbar(im, ax=axs[0], orientation="vertical")

    data = [(truth[:, i] - mode[:, i]) for i in range(2)]
    for dat, label in zip(data, ["1", "2"]):
        seaborn.histplot(dat, bins=bins, ax=axs[1], stat="density", label=f"Diff {label}")
    axs[1].legend()
    axs[1].set_title("Histogram of (True - Predicted)")
    axs[1].set_xlabel("Difference")
    axs[1].set_ylabel("Density")

    fig.tight_layout()
    fig.suptitle("Combined Plot: Euclidean Error & Histogram", fontsize=14, y=1.05)
    fig.savefig(directory / "combined_euclidean_histogram.pdf", dpi=300)


def pdf_multi(pdf, mode, truth, xy, choix, directory):
    fig = pyplot.figure(figsize=(5, 5))
    pyplot.suptitle(
        f"{choix} => mode ({mode[0]:.2f}, {mode[1]:.2f}) & truth ({truth[0]:.2f}, {truth[1]:.2f})",
        fontsize=10,
    )
    col_mode = "magenta"
    col_truth = "red"
    cmap = "viridis"
    s = 50
    extent = (min(xy[0, :]), max(xy[0, :]), min(xy[1, :]), max(xy[1, :]))
    img = pyplot.imshow(pdf, extent=extent, origin="lower", cmap=cmap)
    cbar = pyplot.colorbar(img, ax=pyplot.gca(), orientation="vertical")
    cbar.set_label("PDF Value")  # Add a label for the colorbar
    pyplot.tick_params(axis="x", labelrotation=25)
    pyplot.scatter(mode[0], mode[1], c=col_mode, marker="x", s=s, label="mode")
    pyplot.scatter(truth[0], truth[1], c=col_truth, marker="x", s=s, label="truth")
    pyplot.legend(fontsize="12")
    pyplot.xlabel("g1")
    pyplot.ylabel("g2")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(directory / f"multi_pdf_{choix}.pdf", dpi=300)


def pdf_uni(x_values, pdf, mean, mode, truth, choix, zoom, directory):
    fig = pyplot.figure(figsize=(25, 8))
    ax1 = pyplot.subplot2grid(shape=(1, 6), loc=(0, 0), colspan=2, fig=fig)
    ax2 = pyplot.subplot2grid(shape=(1, 6), loc=(0, 2), colspan=2, fig=fig)
    ax3 = pyplot.subplot2grid(shape=(1, 6), loc=(0, 4), colspan=2, fig=fig)

    c1 = "red"
    c2 = "blue"
    c3 = "#ff9999"
    c4 = "skyblue"

    for ax in [ax1, ax2, ax3]:
        ax.plot(x_values, pdf[0], ls="-", c=c2, label="pdf g1")
        for i, avg in enumerate(mean[0]):
            ax.axvline(avg, c=c2, ls=":", linewidth=2, label="means g1" if i == 0 else "")
        if mean.shape[-1] != 1:
            ax.axvline(mode[0], c=c2, ls="--", linewidth=2, label="mode g1")
        ax.axvline(truth[0], c=c4, ls="-", linewidth=3, label="truth g1")

        ax.plot(x_values, pdf[1], ls="-", c=c1, label="pdf g2")
        for i, avg in enumerate(mean[1]):
            ax.axvline(avg, c=c1, ls=":", label="means g2" if i == 0 else "", alpha=1)
        if mean.shape[-1] != 1:
            ax.axvline(mode[1], c=c1, ls="--", linewidth=1, label="mode g2")
        ax.axvline(truth[1], c=c3, ls="-", linewidth=3, label="truth g2")

        ax.set_xlabel("shear values")
        ax.set_ylabel("density")
        ax.legend(fontsize=12)

    ax2.set_xlim(mode[0] - zoom, mode[0] + zoom)
    ax3.set_xlim(mode[1] - zoom, mode[1] + zoom)
    ax2.set_title("ZOOM g1")
    ax3.set_title("ZOOM g2")

    fig.subplots_adjust(wspace=0.25)
    fig.suptitle("Predictions for a single image test", size="13", x=0.42)
    fig.tight_layout()
    fig.savefig(directory / f"pdf_{choix}.pdf", dpi=300)


def channels_layers(layers, number, which, directory, compress=True):
    nb = 2
    size = 3 * nb
    layer = layers[which]
    index = random.permutation(random.PRNGKey(int(time())), layer[0].shape[-1])[:size]

    fig, axs = pyplot.subplots(nb, 3, figsize=(15, 10))
    for i, ax in enumerate(axs.flatten()):
        image = layer[..., index[i]]
        if compress:
            im = ax.imshow(numpy.arcsinh(image), cmap="gray", interpolation="nearest")
        else:
            im = ax.imshow(image, cmap="gray", interpolation="nearest")
        fig.colorbar(im, ax=axs[i // 3, i % 3])
        ax.set_title(f"Channel {index[i]}")

    fig.savefig(directory / f"conv_{which}_{number}.pdf", dpi=300)
    fig.show()


def add_colorbar(im, ax, fig):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)


def plot_cams(image_test, number, cam, directory, compress=True, name="cam"):
    if compress:
        func = numpy.arcsinh
        max_val = numpy.max(cam)
        min_val = numpy.min(cam)
        cam = (cam - min_val) / (max_val - min_val)
    else:
        func = lambda x: x  # noqa: E731

    fig, axs = pyplot.subplots(2, 2, figsize=(10, 10))

    cmap = "viridis"
    alpha = 0.6

    im00 = axs[0, 0].imshow(cam, cmap=cmap, interpolation="nearest")
    axs[0, 0].set_title("normalised cam")
    add_colorbar(im00, axs[0, 0], fig)

    order = 3
    zoom = 352 / cam.shape[0]

    im01 = axs[0, 1].imshow(func(image_test), cmap="gray", interpolation="nearest")
    axs[0, 1].set_title("image test")
    add_colorbar(im01, axs[0, 1], fig)
    axs[0, 1].set_axis_off()

    upscaled_g = scipy.ndimage.zoom(input=cam, zoom=zoom, order=order)
    axs[1, 0].set_title("upscaled cam")
    im10 = axs[1, 0].imshow(upscaled_g, alpha=alpha, cmap=cmap)
    add_colorbar(im10, axs[1, 0], fig)
    axs[1, 0].set_axis_off()

    axs[1, 1].set_title("both")
    axs[1, 1].imshow(func(image_test), cmap="gray", interpolation="nearest")
    im11 = axs[1, 1].imshow(upscaled_g, alpha=alpha, cmap=cmap)
    add_colorbar(im11, axs[1, 1], fig)
    axs[1, 1].set_axis_off()

    fig.suptitle(f"result for {name}", y=1.0, size=20)
    fig.tight_layout()
    fig.savefig(directory / f"{name}_{number}.pdf", dpi=300)
    fig.show()


def predictions(pred, shear, directory, symbol, bornes_res=None):
    res_pred_g1 = pred[:, 0] - shear[:, 0]
    res_pred_g2 = pred[:, 1] - shear[:, 1]
    poly_g1 = numpy.polyfit(shear[:, 0], res_pred_g1, 1)
    poly_g2 = numpy.polyfit(shear[:, 1], res_pred_g2, 1)
    t = numpy.linspace(shear.min(), shear.max())
    t0 = numpy.linspace(shear[:, 0].min(), shear[:, 0].max())
    t1 = numpy.linspace(shear[:, 1].min(), shear[:, 1].max())

    pyplot.figure(figsize=(20, 16))

    pyplot.subplot(221)
    s = 25
    alpha = 0.25
    c1, c2 = "C0", "C1"

    pyplot.hlines(y=0, xmin=-0.051, xmax=0.051, color="k", label="y=0")

    label_1 = rf"$\{symbol}_1$ : m= {poly_g1[0]:.3f}, c={poly_g1[1]:.3f}"
    pyplot.scatter(shear[:, 0], res_pred_g1, alpha=alpha, label=label_1, color=c1, marker="o", s=s)
    label_2 = rf"$\{symbol}_2$ : m= {poly_g2[0]:.3f}, c={poly_g2[1]:.3f}"
    pyplot.scatter(shear[:, 1], res_pred_g2, alpha=alpha, label=label_2, color=c2, marker="x", s=s)
    pyplot.plot(t0, poly_g1[1] + poly_g1[0] * t0, color=c1)
    pyplot.plot(t1, poly_g2[1] + poly_g2[0] * t1, color=c2)

    pyplot.xlabel(r"$\gamma^{true}$")
    pyplot.ylabel(rf"$\{symbol}^{{obs}}$ - $\gamma^{{true}}$", rotation=0, labelpad=20)
    pyplot.legend()
    if bornes_res is not None:
        pyplot.ylim(-bornes_res, bornes_res)

    overall_min = numpy.minimum(res_pred_g1.min(), res_pred_g2.min())
    overall_max = numpy.maximum(res_pred_g1.max(), res_pred_g2.max())
    pyplot.title(label=f"min = {overall_min:.3f} // max = {overall_max:.3f}", fontsize=10)

    pyplot.subplot(222)
    mean_pred_g1, median_pred_g1, std_pred_g1 = [], [], []
    mean_pred_g2, median_pred_g2, std_pred_g2 = [], [], []
    unique_gamma = []
    for gamma in numpy.unique(shear[:, 0]):
        unique_gamma.append(gamma)

        mask1 = shear[:, 0] == gamma
        preds_mask1 = pred[mask1, 0]
        mean_pred_g1.append(numpy.mean(preds_mask1))
        median_pred_g1.append(numpy.median(preds_mask1))
        std_pred_g1.append(numpy.std(preds_mask1))

        mask2 = shear[:, 1] == gamma
        preds_mask2 = pred[mask2, 1]
        mean_pred_g2.append(numpy.mean(preds_mask2))
        median_pred_g2.append(numpy.median(preds_mask2))
        std_pred_g2.append(numpy.std(preds_mask2))

    pyplot.errorbar(
        unique_gamma,
        mean_pred_g1,
        yerr=std_pred_g1,
        fmt="o",
        capsize=3,
        color=c1,
        elinewidth=1,
        markeredgewidth=1,
        alpha=0.5,
        label=rf"Mean & Std $\{symbol}_1$",
    )
    pyplot.scatter(
        unique_gamma,
        median_pred_g1,
        color=c1,
        alpha=0.7,
        marker="x",
        s=50,
        label=rf"Median $\{symbol}_1$",
    )
    pyplot.errorbar(
        unique_gamma,
        mean_pred_g2,
        yerr=std_pred_g2,
        fmt="o",
        capsize=3,
        color=c2,
        elinewidth=1,
        markeredgewidth=1,
        alpha=0.5,
        label=rf"Mean & Std $\{symbol}_1$",
    )
    pyplot.scatter(
        unique_gamma,
        median_pred_g2,
        color=c2,
        alpha=0.7,
        marker="x",
        s=50,
        label=rf"Median $\{symbol}_2$",
    )
    pyplot.xlabel(r"$\gamma^{true}$")
    pyplot.ylabel(rf"$\{symbol}^{{obs}}$", rotation=0, labelpad=20)
    pyplot.legend()

    pyplot.subplot(223)
    pyplot.scatter(shear[:, 0], pred[:, 0], alpha=alpha, color=c1, marker="o", s=s)
    pyplot.plot(t, t, color="k", label="x=y")
    pyplot.xlabel(r"$\gamma^{true}$")
    pyplot.ylabel(rf"$\{symbol}^{{obs}}_1$", rotation=0, labelpad=20)
    pyplot.legend()

    pyplot.subplot(224)
    pyplot.scatter(shear[:, 1], pred[:, 1], alpha=alpha, color=c2, marker="o", s=s)
    pyplot.plot(t, t, color="k", label="x=y")
    pyplot.xlabel(r"$\gamma^{true}$")
    pyplot.ylabel(rf"$\{symbol}^{{obs}}_2$", rotation=0, labelpad=20)
    pyplot.legend()

    pyplot.savefig(f"{directory}.pdf")


def remove_ansi_escape_sequences(text):
    ansi_escape = re.compile(
        r"""
        \x1B  # ESC
        (?:   # 7-bit C1 Fe (except CSI)
            [@-Z\\-_]
        |     # or [ for CSI, followed by a control sequence
            \[
            [0-?]*  # Parameter bytes
            [ -/]*  # Intermediate bytes
            [@-~]   # Final byte
        )
    """,
        re.VERBOSE,
    )
    return ansi_escape.sub("", text)


def shape_network(model_config, folder):
    pad = model_config["pad"]
    kernel = model_config["kernel_size"]
    encoding_network = EncoderPool(
        mode_pad=model_config["mode_pad"],
        pad_width=[(0, 0), (pad, pad), (pad, pad), (0, 0)],
        kernel_size=(kernel, kernel),
        pre_layer=model_config["pre_layer"],
        layer=model_config["layer"],
    )

    # tabulate conv shapes
    shapes = encoding_network.tabulate(
        rngs=random.PRNGKey(0),
        x=numpy.ones((1, 352, 352, 1)),
        console_kwargs={"width": 280},
    )

    # remove ansi sequences
    cleaned_output = remove_ansi_escape_sequences(shapes)
    with open(folder / "encoding_network_shape.txt", "w") as file:
        file.write(cleaned_output)

    print(cleaned_output)
