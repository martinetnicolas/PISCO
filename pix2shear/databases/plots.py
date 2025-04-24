import galsim
import holoviews
import numpy
import plotly
from astropy import coordinates, io, units
from functions_images import build_psf
from matplotlib import colors, gridspec, pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from panel import Column, Row, bind, depends, widgets
from plotly import graph_objects, subplots
from seaborn import heatmap
from utils import check_unique


def plot_dist(selection, bandwidth, data):
    ongoing_values = data[selection]
    holo_distri = holoviews.Distribution(ongoing_values)
    return holo_distri.opts(width=1500, height=400, bandwidth=bandwidth)


def interactive_dist(data):
    slider = widgets.FloatSlider(name="Bandwidth", start=1e-2, end=1.0, step=1e-2, value=1e-1)
    selector = widgets.Select(name="Data", options=list(data.keys()), value=list(data.keys())[0])
    interactive_plot = bind(plot_dist, selection=selector, bandwidth=slider, data=data)
    return Column(selector, slider, interactive_plot)


def interactive_pairplot(data, size, directory, width, height):
    colored_values = data.columns.to_list()
    color_selector = widgets.Select(name="Colored", options=colored_values, value=colored_values[0])
    colors = ["Viridis", "Plasma", "Cividis", "Jet"]
    color_scale_selector = widgets.Select(name="Color", options=colors, value=colors[0])

    @depends(color=color_selector, color_scale=color_scale_selector)
    def plot_scatter_matrix(color, color_scale):
        fig = plotly.express.scatter_matrix(
            data,
            dimensions=colored_values,
            color=color,
            color_continuous_scale=color_scale,
            width=width,
            height=height,
        )
        fig.update_traces(marker=dict(size=size))
        plotly.io.write_image(fig=fig, file=f"{directory}.pdf", format="pdf")
        return fig

    return Column(Row(color_selector, color_scale_selector), plot_scatter_matrix)


def plot_example(selected_images, directory):
    shape = selected_images.shape[0]
    fig, axes = pyplot.subplots(nrows=1, ncols=shape, figsize=(6 * shape, 10))
    for ax, img in zip(axes, selected_images):
        ax.imshow(numpy.arcsinh(img), cmap="viridis")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(directory / "examples_images.pdf", dpi=300)


def plot_histo(selection, bins, data):
    ongoing_values = data[selection]
    hist = numpy.histogram(ongoing_values.to_numpy(), bins=bins)
    holo_hist = holoviews.Histogram(hist)
    return holo_hist.opts(width=1500, height=400)


def interactive_hist(data):
    slider = widgets.IntSlider(name="Bins", start=50, end=300, step=1, value=100)
    selector = widgets.Select(name="Data", options=list(data.columns), value=list(data.columns)[0])
    interactive_plot_hist = bind(plot_histo, selection=selector, bins=slider, data=data)
    return Column(selector, slider, interactive_plot_hist)


def large_ellps(selected_images, elps1, elps2, number, directory, name, size=3):
    fig, axes = pyplot.subplots(nrows=1, ncols=size, figsize=(20, 10))
    for i, (ax, img) in enumerate(zip(axes, selected_images)):
        ax.imshow(numpy.arcsinh(img), cmap="viridis")
        ax.set_title(f"{number[i]} => {[elps1.iloc[i], elps2.iloc[i]]}", fontsize=12)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(directory / f"large_{name}_ellps_images.pdf", dpi=300)


def heatmaps_inputs(galaxy_parameters, concatenated_inputs, directory):
    fig, axes = pyplot.subplots(1, 2, figsize=(20, 10))
    for i, (data, title) in enumerate(
        zip([galaxy_parameters, concatenated_inputs], ["GalParam", "HC+Norms"])
    ):
        final_data = data.corr()
        mask = numpy.triu(numpy.ones_like(final_data, dtype=bool), k=1)
        heatmap(final_data, annot=True, mask=mask, cbar=False, ax=axes[i])
        axes[i].set_title(title)
    fig.savefig(directory / "heatmaps.pdf", dpi=300)


def style_norma(data, style, directory):
    pyplot.hist(data, 50)
    y_value = 10
    pyplot.axhline(y_value, color="r", linewidth=2)
    x_min, x_max = pyplot.xlim()
    pyplot.axvline((x_min + x_max) / 2, color="r", linestyle="--", linewidth=2)
    pyplot.yscale("log")
    pyplot.savefig(directory / f"normalisation_{style}.pdf", dpi=300)


def before_after(img, directory):
    pyplot.subplot(121)
    pyplot.imshow(img)
    pyplot.title("Before")
    pyplot.subplot(122)
    pyplot.imshow(numpy.arcsinh(img / 3.14 / 3))
    pyplot.title("After")
    pyplot.savefig(directory / "normalisation_before_after.pdf", dpi=300)


def check_psf(file):
    img_psf = io.fits.open(file)[0].data.astype(numpy.float32)

    fig, axes = pyplot.subplots(1, 2, figsize=(12, 6))

    im1 = axes[0].imshow(numpy.log(img_psf), cmap="inferno")
    axes[0].set_title("Logarithmic Scale")
    divider = make_axes_locatable(axes[0])
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    pyplot.colorbar(im1, cax=cax1)

    im2 = axes[1].imshow(img_psf, cmap="gray")
    axes[1].set_title("Linear Scale")
    divider = make_axes_locatable(axes[1])
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    pyplot.colorbar(im2, cax=cax2)

    fig.savefig(file.parent / f"{file.stem}.pdf", dpi=300)
    fig.show()


def visu_inputs_ouputs(dbb, loggy=False):
    if loggy:
        func = numpy.arcsinh
    else:
        func = numpy.array

    df_input = dbb.pop("inputs")

    sim_index_slider = widgets.IntSlider(name="Index", start=0, end=len(df_input) - 1, value=0)

    mini = func(dbb["simu"]).min()
    maxi = func(dbb["simu"]).max()
    clim_slider = widgets.RangeSlider(name="Range", start=mini, end=maxi, step=10)

    options_color = ["copper", "viridis", "magma", "plasma", "jet", "gray"]
    colormap_selector = widgets.Select(name="Color Map", options=options_color, value="copper")

    frame_inputs = widgets.DataFrame(
        df_input,
        height=300,
        width=550,
        disabled=True,
        selection=[sim_index_slider.value],
    )

    bind(
        lambda sel: setattr(sim_index_slider, "value", sel[0]),
        frame_inputs.param.selection,
        watch=True,
    )
    bind(
        lambda idx: setattr(frame_inputs, "selection", [idx]),
        sim_index_slider.param.value_throttled,
        watch=True,
    )

    @depends(index=sim_index_slider, cmap=colormap_selector, clim=clim_slider)
    def plot(index, cmap, clim):
        return holoviews.Image(func(dbb["simu"][index]), vdims="simu").opts(
            cmap=cmap,
            colorbar=True,
            frame_height=425,
            frame_width=425,
            clim=clim,
            xaxis=None,
            yaxis=None,
        )

    return Row(Column(sim_index_slider, colormap_selector, clim_slider), frame_inputs, plot)


def noise_shape(dir_plots, images, corresponding_csv, apply_arcsinh):
    number = images.shape[0]

    indices = corresponding_csv.index.to_list()
    if number == 2:
        colLabels = [indices[0], indices[1]]
    elif number == 4:
        colLabels = [indices[0], indices[1], indices[2], indices[3]]

    fig, axes = pyplot.subplots(2, number, figsize=((number + 1) * 5, 15))

    def process_image(image, apply_arcsinh):
        if apply_arcsinh:
            return numpy.arcsinh(image)
        return image

    for i in range(number):
        im = axes[0, i].imshow(process_image(images[i], apply_arcsinh))
        axes[0, i].set_title(f"images: {indices[i]}", fontsize=12)
        fig.colorbar(im, ax=axes[0, i], orientation="vertical", fraction=0.046, pad=0.04)

    table_data = corresponding_csv.T
    table = axes[1, 1].table(
        cellText=table_data.values,
        colLabels=colLabels,
        rowLabels=table_data.index,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    for _, cell in table.get_celld().items():
        cell.set_width(0.4)
        cell.set_height(0.05)

    for ax in axes.flatten():
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(dir_plots / "noise_shape_cancel.pdf", dpi=300)
    fig.show()


def lhs_visu(directory, all_pairs, alpha, size, bins=11):
    n_plots = len(all_pairs)
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axs = pyplot.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs = axs.flatten()

    for i, ((key_x, key_y), hypercube) in enumerate(all_pairs):
        x_vals = hypercube[key_x]
        y_vals = hypercube[key_y]

        if key_x == "x":
            x_range = (0, 352)
            y_range = (0, 352)
        else:
            x_range = (x_vals.min(), x_vals.max())
            y_range = (y_vals.min(), y_vals.max())

        h2d = axs[i].hist2d(x_vals, y_vals, bins=bins, range=[x_range, y_range], cmap="copper")
        axs[i].scatter(x_vals, y_vals, s=size, alpha=alpha, marker="x", color="cyan")

        axs[i].set_xlabel(key_x)
        axs[i].set_ylabel(key_y)
        fig.colorbar(h2d[3], ax=axs[i])

        if key_x == "avg_g1":
            square_x = [-0.05, 0.05, 0.05, -0.05, -0.05]
            square_y = [-0.05, -0.05, 0.05, 0.05, -0.05]
            axs[i].plot(square_x, square_y, color="white", linestyle="--", linewidth=1)

    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    fig.tight_layout()
    fig.savefig(directory / "lhs_all_galaxies.pdf", dpi=300)
    pyplot.show()


def lhs_visu_single(directory, all_pairs, alpha, bins=11, s=1):
    pair, hypercube = all_pairs
    val_1, val_2 = hypercube[pair[0]], hypercube[pair[1]]

    x_range = (val_1.min(), val_1.max())
    y_range = (val_2.min(), val_2.max())

    xlabel = str(val_1.name)
    ylabel = str(val_2.name)
    if xlabel == "x":
        x_range = (0, 352)
        y_range = (0, 352)

    fig = pyplot.figure(figsize=(9, 6))
    gs = fig.add_gridspec(
        5, 6, wspace=0.2, hspace=0.2, height_ratios=[2, 4, 4, 4, 4], width_ratios=[4, 4, 4, 4, 4, 2]
    )

    ax_main = fig.add_subplot(gs[1:, :-1])
    h2d = ax_main.hist2d(val_1, val_2, bins=bins, range=[x_range, y_range], cmap="copper")
    fig.colorbar(h2d[3], ax=ax_main)
    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    ax_main.scatter(val_1, val_2, s=s, alpha=alpha, marker="x", color="cyan")

    if xlabel == "avg_g1":
        square_x = [-0.05, 0.05, 0.05, -0.05, -0.05]
        square_y = [-0.05, -0.05, 0.05, 0.05, -0.05]
        ax_main.plot(square_x, square_y, color="white", linestyle="--", linewidth=1)

    ax_top = fig.add_subplot(gs[0, :4], sharex=ax_main)
    ax_top.hist(val_1, bins=bins, range=x_range, color="blue", alpha=0.7)
    ax_top.set_ylabel("Count")
    ax_top.tick_params(axis="x", labelbottom=False)

    ax_right = fig.add_subplot(gs[1:, -1:], sharey=ax_main)
    ax_right.hist(
        val_2, bins=bins, range=y_range, orientation="horizontal", color="green", alpha=0.7
    )
    ax_right.set_xlabel("Count")
    ax_right.tick_params(axis="y", labelleft=False)
    pyplot.savefig(directory / f"lhs_{xlabel}_{ylabel}.pdf", dpi=300)


def images_redshift(dir_plots, images, images_z_pixel, images_z_flux):
    fig, axs = pyplot.subplots(1, 3, figsize=(15, 5))
    solid_red_cmap = colors.ListedColormap(["red"])

    im1 = axs[0].imshow(numpy.arcsinh(images), origin="lower")
    masked_img = numpy.ma.masked_equal(images_z_pixel, 0)
    axs[0].imshow(masked_img, origin="lower", cmap=solid_red_cmap)

    im2 = axs[1].imshow(masked_img, origin="lower", cmap="tab10", vmin=0.5, vmax=10.5)

    masked_img_full = numpy.ma.masked_equal(images_z_flux, 0)
    im3 = axs[2].imshow(masked_img_full, origin="lower", cmap="tab10", vmin=0.5, vmax=10.5)

    for ax, im in zip(axs.flat, [im1, im2, im3]):
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(dir_plots / "redshift.pdf", dpi=300)


def histo_galaxies(hypercube_gal, dir_plots, bins=50):
    fontdict = {"fontsize": 10, "fontweight": "bold", "family": "serif", "color": "darkgreen"}
    alpha = 0.3

    fig, axes = pyplot.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    axes[0].hist(hypercube_gal["int_e1"], bins=bins, alpha=alpha, label="int_e1")
    axes[0].hist(hypercube_gal["int_e2"], bins=bins, alpha=alpha, label="int_e2")
    axes[0].legend()

    axes[1].hist(hypercube_gal["obs_e1"], bins=bins, alpha=alpha, label="obs_e1")
    axes[1].hist(hypercube_gal["obs_e2"], bins=bins, alpha=alpha, label="obs_e2")
    axes[1].legend()

    axes[2].hist(hypercube_gal["x"], bins=bins, alpha=alpha, label="x")
    axes[2].hist(hypercube_gal["y"], bins=bins, alpha=alpha, label="y")
    axes[2].legend()

    axes[3].hist(hypercube_gal["mag"], bins=bins, alpha=alpha, color="orange")
    axes[3].set_title("mag", fontdict)

    axes[4].hist(hypercube_gal["hlr"], bins=bins, alpha=alpha)
    axes[4].set_title("hlr", fontdict)

    axes[5].hist(hypercube_gal["sersic"], bins=bins, alpha=alpha, color="orange")
    axes[5].set_title("sersic", fontdict)

    fig.suptitle(f"{hypercube_gal.shape[0]} galaxies")
    fig.tight_layout()
    fig.savefig(dir_plots / "histo_hc_gal.pdf", dpi=300)


def histo_images(hypercube_img, dir_plots):
    unique_seeds, _, non_unique_count_seeds = check_unique(hypercube_img, "seeds")
    unique_key, _, non_unique_count_key = check_unique(hypercube_img, "hashed_key")
    unique_k, _, non_unique_count_k = check_unique(hypercube_img, "k")

    fontdict = {"fontsize": 10, "fontweight": "bold", "family": "serif", "color": "darkgreen"}

    max_shears = 0.055
    alpha = 0.2
    bins = 13

    fig, axes = pyplot.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].hist(hypercube_img["avg_g1"], bins=bins, alpha=alpha, label="avg_g1")
    axes[0, 0].hist(hypercube_img["avg_g2"], bins=bins, alpha=alpha, label="avg_g2")
    axes[0, 0].set_xlim(-max_shears, max_shears)
    axes[0, 0].legend()

    axes[0, 1].hist(hypercube_img["avg_int_e1"], bins=bins, alpha=alpha, label="avg_int_e1")
    axes[0, 1].hist(hypercube_img["avg_int_e2"], bins=bins, alpha=alpha, label="avg_int_e2")
    axes[0, 1].legend()

    axes[0, 2].hist(hypercube_img["avg_obs_e1"], bins=bins, alpha=alpha, label="avg_obs_e1")
    axes[0, 2].hist(hypercube_img["avg_obs_e2"], bins=bins, alpha=alpha, label="avg_obs_e2")
    axes[0, 2].legend()

    axes[1, 0].hist(hypercube_img["seeds"], bins=50, label="seeds")
    axes[1, 0].set_title(
        f"seeds : {unique_seeds.shape[0]} w. {non_unique_count_seeds} duplicates", fontdict
    )

    axes[1, 1].hist(hypercube_img["hashed_key"], bins=50, label="key")
    axes[1, 1].set_title(
        f"key : {unique_key.shape[0]} w. {non_unique_count_key} duplicates", fontdict
    )

    axes[1, 2].hist(hypercube_img["k"], bins=8, label="k")
    axes[1, 2].set_title(f"k : {unique_k.shape[0]} w. {non_unique_count_k} duplicates", fontdict)

    fig.suptitle(f"{hypercube_img.shape[0]} samples")
    fig.tight_layout()
    fig.savefig(dir_plots / "histo_hc_img.pdf", dpi=300)


def combined_plot(
    dir, z, z_grid, z_pdf, z_data, g_data, shunsheng_fn, hypercube, g1_scaled, g2_scaled
):
    fig, axes = pyplot.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].hist(z, bins=50, density=True, alpha=0.5, label="Sampled z")
    axes[0, 0].plot(z_grid, z_pdf / numpy.trapz(z_pdf, z_grid), label="Target PDF", color="red")
    axes[0, 0].set_xlabel("Redshift z")
    axes[0, 0].set_ylabel("Probability Density")
    axes[0, 0].set_title("Redshift Sampling from $z^2 e^{-z/0.8}$")
    axes[0, 0].legend()

    x_fit = numpy.linspace(0, 0.08, 200)
    axes[0, 1].plot(g_data, z_data, "o", label="Data")
    axes[0, 1].plot(x_fit, shunsheng_fn(x_fit), label="Fit")
    axes[0, 1].set_xlabel("Shear")
    axes[0, 1].set_ylabel("Redshift")
    axes[0, 1].grid(True)
    axes[0, 1].set_title(f"en 0 : {shunsheng_fn(0):.2f} en 0.07 : {shunsheng_fn(0.07):.2f}")
    axes[0, 1].legend()

    axes[1, 0].hist(hypercube[:, 0], bins=20, alpha=0.3, density=True, label="$g_1$")
    axes[1, 0].hist(hypercube[:, 1], bins=20, alpha=0.3, density=True, label="$g_2$")
    axes[1, 0].set_title("Distribution of $g_1$ and $g_2$")
    axes[1, 0].legend()

    axes[1, 1].hist(g1_scaled, bins=20, alpha=0.3, density=True, label="$g_1$")
    axes[1, 1].hist(g2_scaled, bins=20, alpha=0.3, density=True, label="$g_2$")
    axes[1, 1].set_title("Distribution of $g_1$ and $g_2$ rescaled ")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(dir / "z_vs_g.pdf", dpi=300)
    fig.show()


def images_psf(dir_plots, image_psf):
    fig, axs = pyplot.subplots(1, 2, figsize=(12, 4))

    im1 = axs[0].imshow(numpy.log(image_psf), origin="lower", cmap="inferno")
    axs[0].set_title("log")

    im2 = axs[1].imshow(image_psf, origin="lower", cmap="inferno")

    for ax, im in zip(axs.flat, [im1, im2]):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(f"{dir_plots}.pdf", dpi=300)


def diff_real_param_cosmos(choice, scale, patch_size, real_gal, param_gal, directory):
    psf = build_psf("airy")

    real_gal = galsim.Convolve(real_gal, psf)
    im_real = galsim.Image(patch_size, patch_size)
    real_gal.drawImage(image=im_real, scale=scale)

    param_gal = galsim.Convolve(param_gal, psf)
    im_param = galsim.Image(patch_size, patch_size)
    param_gal.drawImage(image=im_param, scale=scale)

    fig, axs = pyplot.subplots(1, 2, figsize=(8, 4))

    im0 = axs[0].imshow(im_real.array, origin="lower")
    axs[0].set_title(f"Real Galaxy {choice}")
    fig.colorbar(im0, ax=axs[0], orientation="vertical", fraction=0.05, pad=0.05)
    im1 = axs[1].imshow(im_param.array, origin="lower")
    axs[1].set_title(f"Parametric Model {choice}")
    fig.colorbar(im1, ax=axs[1], orientation="vertical", fraction=0.05, pad=0.05)

    fig.savefig(directory / "diff_param_real_cosmos.pdf", dpi=300)
    fig.tight_layout()
    fig.show()


def sky_coverage(ra_23, dec_23, ra_25, dec_25, directory):
    ra_23 = ra_23 * units.deg
    dec_23 = dec_23 * units.deg
    ra_25 = ra_25 * units.deg
    dec_25 = dec_25 * units.deg

    fig = pyplot.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(ra_23, dec_23, s=1, alpha=0.5)
    ax1.set_title("COSMOS 23.5 Sky Coverage")
    ax1.set_xlabel("Right Ascension (deg)")
    ax1.set_ylabel("Declination (deg)")
    ax1.invert_xaxis()
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax2.scatter(ra_25, dec_25, s=1, alpha=0.5, color="orange")
    ax2.set_title("COSMOS 25.2 Sky Coverage")
    ax2.set_xlabel("Right Ascension (deg)")
    ax2.invert_xaxis()
    ax2.grid(True)

    coords = coordinates.SkyCoord(ra=ra_23, dec=dec_23)
    ra_rad = coords.ra.wrap_at(180 * units.deg).radian
    dec_rad = coords.dec.radian

    ax3 = fig.add_subplot(gs[1, :], projection="mollweide")
    ax3.scatter(ra_rad, dec_rad, s=2, alpha=0.5)
    ax3.grid(True)

    ax3.xaxis.set_ticks_position("top")
    ax3.xaxis.set_label_position("top")

    fig.savefig(directory / "mollmoll.pdf", dpi=300)
    fig.tight_layout()
    fig.show()


def check_image(path, plot_dir, compress=False):
    assoc_data = numpy.loadtxt(f"{path}_assoc.txt")
    if assoc_data.ndim == 1:
        assoc_data = numpy.expand_dims(assoc_data, axis=0)

    with io.fits.open(f"{path}.fits") as hdul:
        img = hdul[0].data
        header = hdul[0].header

    fig, ax = pyplot.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [3, 1]})

    if compress:
        ax[0].imshow(numpy.arcsinh(img.astype(float)), origin="lower")
    else:
        ax[0].imshow(img.astype(float), origin="lower")
    ax[0].scatter(assoc_data[:, 0], assoc_data[:, 1], color="red", marker="o", s=5, label="center")
    ax[0].legend()

    header_text = "\n".join([f"{key}: {header[key]}" for key in list(header.keys())[:10]])
    ax[1].text(
        0.05, 0.95, header_text, fontsize=10, verticalalignment="top", transform=ax[1].transAxes
    )
    ax[1].axis("off")

    fig.savefig(plot_dir / "assoc_s++.pdf", dpi=300)
    fig.show()


def histos_filters(data, dir, bins=50):
    font_style = {"fontsize": 10, "fontweight": "bold", "family": "serif", "color": "darkgreen"}

    fig, axs = pyplot.subplots(2, 4, figsize=(12, 6))
    axs = axs.ravel()

    for i, (key, values) in enumerate(data.items()):
        color = "skyblue" if i % 2 == 0 else "orange"
        axs[i].hist(values, bins=bins, alpha=0.7, color=color)
        axs[i].set_title(key, fontdict=font_style)
        axs[i].set_ylabel("Count")

    fig.suptitle(f"SAMPLE {dir.stem} => {data.shape[0]}")
    fig.tight_layout()
    fig.savefig(f"{dir}.pdf", dpi=300)


def lhs_visu_single_test(directory, hypercube, alpha, bins=11, s=1):
    fig = pyplot.figure(figsize=(9, 6))
    gs = fig.add_gridspec(
        5, 6, wspace=0.2, hspace=0.2, height_ratios=[2, 4, 4, 4, 4], width_ratios=[4, 4, 4, 4, 4, 2]
    )

    val_1 = hypercube[:, 0]
    val_2 = hypercube[:, 1]

    ax_main = fig.add_subplot(gs[1:, :-1])
    h2d = ax_main.hist2d(val_1, val_2, bins=bins, cmap="copper")
    fig.colorbar(h2d[3], ax=ax_main)
    ax_main.scatter(val_1, val_2, s=s, alpha=alpha, marker="x", color="cyan")

    ax_top = fig.add_subplot(gs[0, :4], sharex=ax_main)
    ax_top.hist(val_1, bins=bins, color="blue", alpha=0.7)
    ax_top.set_ylabel("Count")
    ax_top.tick_params(axis="x", labelbottom=False)

    ax_right = fig.add_subplot(gs[1:, -1:], sharey=ax_main)
    ax_right.hist(val_2, bins=bins, orientation="horizontal", color="green", alpha=0.7)
    ax_right.set_xlabel("Count")
    ax_right.tick_params(axis="y", labelleft=False)
    pyplot.savefig(directory / "lhs_test.pdf", dpi=300)


def distrib_cosmos(mag, hlr, sersic, dir, sample, bins):
    pyplot.figure(figsize=(12, 5))
    pyplot.title(f"{dir.stem} {mag.shape[0]}")
    pyplot.subplot(2, 3, 1)
    pyplot.hist(mag, bins=bins)
    pyplot.subplot(2, 3, 2)
    pyplot.hist(hlr, bins=bins)
    pyplot.subplot(2, 3, 3)
    pyplot.hist(sersic, bins=bins)
    pyplot.subplot(2, 3, 4)
    pyplot.hist(mag, bins=bins)
    pyplot.yscale("log")
    pyplot.subplot(2, 3, 5)
    pyplot.hist(hlr, bins=bins)
    pyplot.yscale("log")
    pyplot.subplot(2, 3, 6)
    pyplot.hist(sersic, bins=bins)
    pyplot.yscale("log")
    pyplot.savefig(dir / f"{sample}_total_distributions.pdf", dpi=300)


def check_distrib_cosmos(mag, hlr, sersic, dir, sample, bins=9):
    data = [mag, hlr, sersic]
    titles = ["Magnitude", "Half-light Radius", "Sersic Index"]

    pyplot.figure(figsize=(12, 4))
    pyplot.suptitle(f"{dir.stem} {mag.shape[0]}")

    for i, (d, title) in enumerate(zip(data, titles), 1):
        pyplot.subplot(1, 3, i)
        pyplot.hist(d, bins=bins)
        pyplot.title(title)

    pyplot.tight_layout(rect=[0, 0, 1, 0.95])
    pyplot.savefig(dir / f"{sample}_check_total_distributions.pdf", dpi=300)
    pyplot.show()


def histos_cosmos(sample, mag, hlr, sersic, mag_bin_edges, density, dir, bins):
    n_bins = len(mag_bin_edges) - 1
    colors = pyplot.cm.viridis(numpy.linspace(0, 1, n_bins))

    pyplot.figure(figsize=(10, 4))
    for i, (data, xlabel) in enumerate(zip([hlr, sersic], ["hlr", "sersic"])):
        pyplot.subplot(1, 2, i + 1)
        for i in range(n_bins):
            mask_bin = (mag >= mag_bin_edges[i]) & (mag < mag_bin_edges[i + 1])
            data_bin = data[mask_bin]
            if len(data_bin) == 0:
                continue

            pyplot.hist(
                data_bin,
                bins=bins,
                density=density,
                histtype="step",
                linewidth=1.5,
                color=colors[i],
                label=f"{mag_bin_edges[i]:.1f}-{mag_bin_edges[i + 1]:.1f}",
            )

        pyplot.xlabel(xlabel)
        pyplot.legend(fontsize=7, title="Mag bin", loc="upper right")
        pyplot.grid(True)
    pyplot.savefig(dir / f"{sample}_histo_per_mag_density{density}.pdf", dpi=300)
    pyplot.tight_layout()
    pyplot.show()


def cdfs_cosmos(sample, mag, hlr, sersic, mag_bin_edges, dir):
    n_bins = len(mag_bin_edges) - 1
    colors = pyplot.cm.viridis(numpy.linspace(0, 1, n_bins))

    pyplot.figure(figsize=(10, 4))

    for i, (data, xlabel) in enumerate(zip([hlr, sersic], ["hlr", "sersic"])):
        pyplot.subplot(1, 2, i + 1)
        for i in range(n_bins):
            mask_bin = (mag >= mag_bin_edges[i]) & (mag < mag_bin_edges[i + 1])
            data_bin = data[mask_bin]
            if len(data_bin) == 0:
                continue

            sorted_vals = numpy.sort(data_bin)
            cdf_vals = numpy.linspace(0, 1, len(sorted_vals))

            pyplot.plot(
                sorted_vals,
                cdf_vals,
                color=colors[i],
                label=f"{mag_bin_edges[i]:.1f}-{mag_bin_edges[i + 1]:.1f}",
            )

        pyplot.xlabel(xlabel)
        pyplot.legend(fontsize=7, title="Mag bin")
        pyplot.grid(True)

    pyplot.savefig(dir / f"{sample}_cfds_per_mag.pdf", dpi=300)
    pyplot.tight_layout()
    pyplot.show()


def comparaison_3d(feat, target, results, indices, directory, with_all_catalog):
    target_text = [f"{i + 1}" for i in range(len(target))]

    target = target[indices]

    colors = {
        ("min_max", "euclidian"): "blue",
        ("min_max", "mahalanobis"): "green",
        ("z_score", "euclidian"): "orange",
        ("z_score", "mahalanobis"): "purple",
    }
    sizes = {
        ("min_max", "euclidian"): 8,
        ("min_max", "mahalanobis"): 10,
        ("z_score", "euclidian"): 12,
        ("z_score", "mahalanobis"): 14,
    }
    symbols = {
        ("min_max", "euclidian"): "circle",
        ("min_max", "mahalanobis"): "square",
        ("z_score", "euclidian"): "diamond",
        ("z_score", "mahalanobis"): "cross",
    }
    opacity = 0.3

    fig = subplots.make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]], column_widths=[0.5, 0.5]
    )

    if with_all_catalog:
        fig.add_trace(
            graph_objects.Scatter3d(
                x=feat[:, 2],
                y=feat[:, 3],
                z=feat[:, 4],
                mode="markers",
                marker=dict(size=1, color="lightgray", opacity=opacity),
                name="feat",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        graph_objects.Scatter3d(
            x=target[:, 2],
            y=target[:, 3],
            z=target[:, 4],
            mode="markers+text",
            marker=dict(size=3, color="red", opacity=opacity),
            text=target_text,
            textposition="middle left",
            name="target",
        ),
        row=1,
        col=1,
    )

    for (SCALING, NORMA), matched in results.items():
        fig.add_trace(
            graph_objects.Scatter3d(
                x=matched[:, 2][indices],
                y=matched[:, 3][indices],
                z=matched[:, 4][indices],
                text=target_text,
                mode="markers+text",
                textposition="middle right",
                marker=dict(size=4, color=colors[(SCALING, NORMA)], opacity=opacity),
                name=f"{SCALING.upper()} + {NORMA.title()}",
            ),
            row=1,
            col=1,
        )

    if with_all_catalog:
        fig.add_trace(
            graph_objects.Scatter(
                x=feat[:, 0],
                y=feat[:, 1],
                mode="markers",
                marker=dict(size=3, color="lightgray", opacity=0.3),
                name="feat (2D)",
            ),
            row=1,
            col=2,
        )

    fig.add_trace(
        graph_objects.Scatter(
            x=target[:, 0],
            y=target[:, 1],
            mode="markers+text",
            text=target_text,
            textposition="middle left",
            marker=dict(color="red", size=6),
            name="Test points",
        ),
        row=1,
        col=2,
    )

    for (SCALING, NORMA), matched in results.items():
        fig.add_trace(
            graph_objects.Scatter(
                x=matched[:, 0][indices],
                y=matched[:, 1][indices],
                mode="markers+text",
                text=target_text,
                textposition="middle right",
                marker=dict(
                    size=sizes[(SCALING, NORMA)],
                    color=colors[(SCALING, NORMA)],
                    symbol=symbols[(SCALING, NORMA)],
                    line=dict(width=1, color="black"),
                ),
                name=f"{SCALING.upper()} + {NORMA.title()} (2D)",
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="e1", row=1, col=2)
    fig.update_yaxes(title_text="e2", row=1, col=2)
    fig.update_layout(
        scene=dict(xaxis_title="mag", yaxis_title="hlr", zaxis_title="sersic"),
        width=1600,
        height=800,
        showlegend=True,
    )
    fig.write_image(directory / "replace_comparaison_3d.pdf", width=1200, height=600)
    fig.show()


def diff_binned_continuous(
    list_mag, list_hlr, list_ns, list_e1, list_e2, hypercube, directory, color
):
    c1 = hypercube[:, 3] if color else "#1f77b4"
    c2 = numpy.array(list_e1) if color else "#1f77b4"
    c3 = hypercube[:, 4] if color else "#1f77b4"
    c4 = numpy.array(list_e2) if color else "#1f77b4"

    size_fig = (12, 10) if color else (8, 8)
    fig, axs = pyplot.subplots(2, 2, figsize=(size_fig), sharey=False)

    sc1 = axs[0, 0].scatter(hypercube[:, 0], hypercube[:, 1], s=2, alpha=0.3, c=c1)
    sc2 = axs[0, 1].scatter(numpy.array(list_mag), numpy.array(list_hlr), s=2, alpha=0.3, c=c2)
    sc3 = axs[1, 0].scatter(hypercube[:, 0], hypercube[:, 2], s=2, alpha=0.3, c=c3)
    sc4 = axs[1, 1].scatter(numpy.array(list_mag), numpy.array(list_ns), s=2, alpha=0.3, c=c4)
    if color:
        fig.colorbar(sc1, ax=axs[0, 0], orientation="vertical", shrink=1, aspect=20, pad=0.02)
        fig.colorbar(sc2, ax=axs[0, 1], orientation="vertical", shrink=1, aspect=20, pad=0.02)
        fig.colorbar(sc3, ax=axs[1, 0], orientation="vertical", shrink=1, aspect=20, pad=0.02)
        fig.colorbar(sc4, ax=axs[1, 1], orientation="vertical", shrink=1, aspect=20, pad=0.02)
    else:
        fig.tight_layout()

    axs[0, 0].set_ylim(0.05, 1.4)
    axs[0, 1].set_ylim(0.05, 1.4)
    axs[1, 0].set_ylim(0.3, 6)
    axs[1, 1].set_ylim(0.3, 6)

    fig.savefig(directory / "binned_vs_cont.pdf", dpi=300)
    fig.show()


def check_lhs_projections(
    dir,
    sample,
    bin_centers,
    mag_bin_edges,
    mag,
    hlr,
    sersic,
    inv_cdfs_hlr,
    inv_cdfs_sersic,
    nb_pdf,
    sersic_continue,
    alpha=0.4,
    bins=18,
):
    fig, axs = pyplot.subplots(1, 3, figsize=(15, 5))
    u = numpy.linspace(0.0, 1, nb_pdf)
    density = True

    for i in range(bin_centers.shape[0]):
        mask_mag_bin = (mag >= mag_bin_edges[i]) & (mag < mag_bin_edges[i + 1])

        mag_binned = mag[mask_mag_bin]
        axs[0].hist(mag_binned, bins=9, label=f"bin {i}")

        hlr_binned = hlr[mask_mag_bin]
        inv_hlr_fn = inv_cdfs_hlr[bin_centers[i]](u)
        inv_hlr_fn = inv_cdfs_hlr[bin_centers[i]](u)
        pdf_hlr = 1.0 / numpy.gradient(inv_hlr_fn, u)

        _, _, rectangles = axs[1].hist(
            hlr_binned, bins=bins, alpha=alpha, label=f"bin {i}", density=density
        )
        axs[1].plot(inv_hlr_fn, pdf_hlr, alpha=1, color=rectangles[0].get_facecolor(), lw=2)

        sersic_binned = sersic[mask_mag_bin]
        if sersic_continue:
            inv_sersic_fn = inv_cdfs_sersic[bin_centers[i]](u)
            inv_sersic_fn = inv_cdfs_sersic[bin_centers[i]](u)
            pdf_sersic = 1.0 / numpy.gradient(inv_sersic_fn, u)

            _, _, rectangles = axs[2].hist(
                sersic_binned, bins=bins, alpha=alpha, label=f"bin {i}", density=density
            )
            axs[2].plot(
                inv_sersic_fn, pdf_sersic, alpha=1, color=rectangles[0].get_facecolor(), lw=2
            )
        else:
            axs[2].hist(sersic_binned, bins=20, alpha=alpha, label=f"bin {i}")

    axs[0].set_title("Mag")
    axs[1].set_title("HLR + PDF")
    axs[2].set_title("Sersic + PDF")

    for ax in axs:
        ax.legend()
        ax.grid(True)

    fig.tight_layout()
    fig.savefig(dir / f"{sample}_check_projection_lhs_density{density}.pdf", dpi=300)
    fig.show()
