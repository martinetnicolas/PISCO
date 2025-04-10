import holoviews
import numpy
import plotly
from astropy import io
from matplotlib import colors, pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from panel import Column, Row, bind, depends, widgets
from scipy import integrate
from seaborn import heatmap
from utils import check_unique


def cdf_inv_cdf(inv_cdf_wrt_mag, cdf_wrt_mag, cases, directory, bins=50):
    probabilities = numpy.linspace(0, 1, 10000)
    _, axes = pyplot.subplots(1, 2, figsize=(10, 4))

    for cas in cases:
        values = inv_cdf_wrt_mag[cas](probabilities)
        axes[0].hist(values, bins=bins, alpha=0.5, label=f"Case {cas}", density=True)

    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Histograms of Sampled Inverse CDF Values for All Cases")
    axes[0].legend()
    axes[0].grid()

    for cas in cases:
        case_data = cdf_wrt_mag[cdf_wrt_mag[:, 0] == cas]
        x_values = case_data[:, 1]
        cdf_values = case_data[:, 2]
        axes[1].plot(x_values, cdf_values, marker="o", label=f"Case {cas}")

    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Cumulative Probability")
    axes[1].set_title("CDFs for All Cases")
    axes[1].legend()
    axes[1].grid()

    pyplot.suptitle(directory.stem)
    pyplot.tight_layout()
    pyplot.savefig(f"{directory}.pdf", dpi=300)


def cdf_inv_cdf_case(plot_case, inv_cdf_wrt_mag, cdf_wrt_mag, directory, deg, bins=50):
    probabilities = numpy.linspace(0, 1, 10000)
    pyplot.figure(figsize=(5, 4))

    values = inv_cdf_wrt_mag(probabilities)

    case_data = cdf_wrt_mag[cdf_wrt_mag[:, 0] == plot_case]
    x_values = case_data[:, 1]
    cdf_values = case_data[:, 2]

    poly_coeffs_np = numpy.polyfit(x_values, cdf_values, deg=deg)
    poly_fit_np = numpy.poly1d(poly_coeffs_np)
    pdf_poly_derivative = numpy.polyder(poly_fit_np)
    xdata = numpy.linspace(values.min(), values.max(), 100)
    y_cdf_fit = poly_fit_np(xdata)
    y_pdf_fit = pdf_poly_derivative(xdata)
    total_probability, _ = integrate.quad(
        lambda x: pdf_poly_derivative(x), values.min(), values.max()
    )

    pyplot.hist(
        values,
        bins=bins,
        alpha=0.2,
        density=True,
        color="yellow",
        label="Histo inv CDF",
        edgecolor="black",
    )
    pyplot.plot(x_values, cdf_values, marker="o", label=f"CDF for Case {plot_case}")
    pyplot.plot(xdata, y_cdf_fit, marker="x", label=f"CDF fit deg {deg}")
    pyplot.plot(xdata, y_pdf_fit, marker="x", label=f"PDF {total_probability:.2e}")
    pyplot.title(f"Case mag = {plot_case}")
    pyplot.xlabel("Value")
    pyplot.legend()
    pyplot.grid()

    pyplot.tight_layout()
    pyplot.savefig(f"{directory}.pdf", dpi=300)


def cdf_inv_cdf_mag_case(inv_cdf, cdf, directory, deg=4, bins=50):
    probabilities = numpy.linspace(0, 1, 1000)
    pyplot.figure(figsize=(5, 4))

    values = inv_cdf(probabilities)
    pyplot.hist(values, bins=bins, alpha=0.7, density=True, color="blue", edgecolor="black")
    pyplot.xlabel("Value")
    pyplot.ylabel("Density")
    pyplot.title("Histogram of Sampled Inverse CDF Values (Case )")
    pyplot.grid()

    x_values = cdf[:, 0]
    cdf_values = cdf[:, 1]

    poly_coeffs_np = numpy.polyfit(x_values, cdf_values, deg=deg)
    poly_fit_np = numpy.poly1d(poly_coeffs_np)
    pdf_poly_derivative = numpy.polyder(poly_fit_np)
    xdata = numpy.linspace(values.min(), values.max(), 100)
    y_cdf_fit = poly_fit_np(xdata)
    y_pdf_fit = pdf_poly_derivative(xdata)
    total_probability, _ = integrate.quad(
        lambda x: pdf_poly_derivative(x), values.min(), values.max()
    )

    pyplot.plot(x_values, cdf_values, marker="o", color="r", label="CDF")
    pyplot.plot(xdata, y_cdf_fit, marker="x", ls="-", label=f"CDF fit deg {deg}")
    pyplot.plot(xdata, y_pdf_fit, marker="x", label=f"PDF {total_probability:.2e}")
    pyplot.legend()
    pyplot.grid()

    pyplot.tight_layout()
    pyplot.savefig(f"{directory}.pdf", dpi=300)


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


def check_image(path, plot_dir, compress=False):
    assoc_data = numpy.loadtxt(f"{path}_assoc.txt")

    with io.fits.open(f"{path}.fits") as hdul:
        img = hdul[0].data
        header = hdul[0].header

    fig, ax = pyplot.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [3, 1]})

    if compress:
        im = ax[0].imshow(numpy.arcsinh(img.astype(float)), origin="lower")
    else:
        im = ax[0].imshow(img.astype(float), origin="lower")
    ax[0].scatter(assoc_data[:, 0], assoc_data[:, 1], color="red", marker="o", s=5, label="center")
    ax[0].legend()
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    # cbar.set_label("Pixel Intensity")

    header_text = "\n".join([f"{key}: {header[key]}" for key in list(header.keys())[:10]])
    ax[1].text(
        0.05, 0.95, header_text, fontsize=10, verticalalignment="top", transform=ax[1].transAxes
    )
    ax[1].axis("off")

    fig.savefig(plot_dir / "assoc_s++.pdf", dpi=300)
    pyplot.show()


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
    fig.savefig(directory / "lhs_gal.pdf", dpi=300)
    pyplot.show()


def lhs_visu_single(directory, all_pairs, alpha, bins=11, s=1):
    pair, hypercube = all_pairs
    val_1, val_2 = hypercube[pair[0]], hypercube[pair[1]]

    coeff = 1.1
    x_range = (val_1.min() * coeff, val_1.max() * coeff)
    y_range = (val_2.min() * coeff, val_2.max() * coeff)

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
    axs[0].set_title("log without noise")

    im2 = axs[1].imshow(image_psf, origin="lower", cmap="inferno")
    axs[1].set_title("without noise")

    for ax, im in zip(axs.flat, [im1, im2]):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(dir_plots / "psfs.pdf", dpi=300)
