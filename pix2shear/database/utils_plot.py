import holoviews
import numpy
from matplotlib import pyplot
from panel import Column, Row, bind, depends, widgets
from plotly import express, io
from scipy import integrate
from seaborn import heatmap

BINS = 50


def cdf_inv_cdf(inv_cdf_wrt_mag, cdf_wrt_mag, cases, directory):
    probabilities = numpy.linspace(0, 1, 10000)
    _, axes = pyplot.subplots(1, 2, figsize=(16, 6))

    for cas in cases:
        values = inv_cdf_wrt_mag[cas](probabilities)
        axes[0].hist(values, bins=BINS, alpha=0.5, label=f"Case {cas}", density=True)

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

    pyplot.tight_layout()
    pyplot.savefig(f"{directory}.pdf", dpi=300)


def cdf_inv_cdf_case(plot_case, inv_cdf_wrt_mag, cdf_wrt_mag, directory, deg=3):
    probabilities = numpy.linspace(0, 1, 10000)
    pyplot.figure(figsize=(8, 6))

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
        bins=BINS,
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


def cdf_inv_cdf_mag_case(inv_cdf, cdf, directory, deg=4):
    probabilities = numpy.linspace(0, 1, 1000)
    pyplot.figure(figsize=(8, 5))

    values = inv_cdf(probabilities)
    pyplot.hist(values, bins=BINS, alpha=0.7, density=True, color="blue", edgecolor="black")
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


def final_case(lhs_samples_gal, custom_titles, directory):
    test = lhs_samples_gal.shape[1]
    _, axes = pyplot.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()
    for i in range(9):
        if i == test:
            break

        axes[i].hist(
            lhs_samples_gal[:, i], bins=BINS, alpha=0.7, color="blue" if i % 2 == 0 else "orange"
        )
        axes[i].set_title(custom_titles[i])
    pyplot.tight_layout()
    pyplot.savefig(f"{directory}.pdf", dpi=300)


def lhs_visu(val_1, val_2, alpha, bins, directory, s=2):
    coeff = 1.1
    x_range = (val_1.min() * coeff, val_1.max() * coeff)
    y_range = (val_2.min() * coeff, val_2.max() * coeff)

    xlabel = str(val_1.name)
    ylabel = str(val_2.name)
    if xlabel == "x":
        x_range = (0, 352)
        y_range = (0, 352)

    fig = pyplot.figure(figsize=(12, 8))
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
    pyplot.savefig(directory / f"sampling_{xlabel}_{ylabel}_bin{bins}.pdf", dpi=300)


def visu_inputs_ouputs(dbb):
    df_input = dbb.pop("inputs")
    sim_index_slider = widgets.IntSlider(name="Index", start=0, end=len(df_input) - 1, value=0)
    mini = numpy.arcsinh(dbb["simu"]).min()
    maxi = numpy.arcsinh(dbb["simu"].max())
    clim_slider = widgets.RangeSlider(name="Range", start=mini, end=maxi, step=10)
    options_color = ["copper", "viridis", "magma", "plasma", "jet", "gray"]
    colormap_selector = widgets.Select(name="Color Map", options=options_color, value="copper")
    frame_inputs = widgets.DataFrame(
        df_input,
        height=300,
        width=800,
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
        return holoviews.Image(numpy.arcsinh(dbb["simu"][index]), vdims="simu").opts(
            cmap=cmap,
            colorbar=True,
            frame_height=425,
            frame_width=425,
            clim=clim,
            xaxis=None,
            yaxis=None,
        )

    return Row(Column(sim_index_slider, colormap_selector, clim_slider), frame_inputs, plot)


def plot_dist(selection, bandwidth, data):
    ongoing_values = data[selection]
    holo_distri = holoviews.Distribution(ongoing_values)
    return holo_distri.opts(width=1500, height=400, bandwidth=bandwidth)


def interactive_dist(data):
    slider = widgets.FloatSlider(name="Bandwidth", start=1e-2, end=1.0, step=1e-2, value=1e-1)
    selector = widgets.Select(name="Data", options=list(data.keys()), value=list(data.keys())[0])
    interactive_plot = bind(plot_dist, selection=selector, bandwidth=slider, data=data)
    return Column(selector, slider, interactive_plot)


def interactive_pairplot(data, size, name, directory):
    colored_values = data.columns.to_list()
    color_selector = widgets.Select(name="Colored", options=colored_values, value=colored_values[0])
    colors = ["Viridis", "Plasma", "Cividis", "Jet"]
    color_scale_selector = widgets.Select(name="Color", options=colors, value=colors[0])

    @depends(color=color_selector, color_scale=color_scale_selector)
    def plot_scatter_matrix(color, color_scale):
        fig = express.scatter_matrix(
            data,
            dimensions=colored_values,
            color=color,
            color_continuous_scale=color_scale,
            width=1500,
            height=1500,
        )
        fig.update_traces(marker=dict(size=size))
        io.write_image(fig=fig, file=directory / f"{name}.pdf", format="pdf")
        return fig

    return Column(Row(color_selector, color_scale_selector), plot_scatter_matrix)


def plot_example(selected_images, directory):
    fig, axes = pyplot.subplots(nrows=1, ncols=3, figsize=(20, 10))
    for ax, img in zip(axes, selected_images):
        ax.imshow(numpy.arcsinh(img), cmap="viridis")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(directory / "examples_images.pdf", dpi=300)


def plot_histo(selection, bins, data):
    ongoing_values = data[selection]
    hist = numpy.histogram(ongoing_values, bins=bins)
    holo_hist = holoviews.Histogram(hist)
    return holo_hist.opts(width=1500, height=400)


def interactive_hist(data):
    slider = widgets.IntSlider(name="Bins", start=50, end=300, step=1, value=100)
    selector = widgets.Select(name="Data", options=list(data.keys()), value=list(data.keys())[0])
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


def histo_seed(seedy, bins, directory):
    pyplot.figure(figsize=(10, 6))

    try:
        for i, alpha in enumerate([0.8, 0.6, 0.4, 0.2, 0.1]):
            unique_values, counts = numpy.unique(seedy[:, i], return_counts=True)
            non_unique_count = numpy.sum(counts > 1)
            pyplot.hist(
                unique_values,
                bins=bins,
                alpha=alpha,
                label=f"seed {i + 1} {unique_values.shape[0]} vs {non_unique_count} w. bin = {bins}",  # noqa
            )
        rows_with_duplicates = [i for i, row in enumerate(seedy) if len(set(row)) < len(row)]
        if len(rows_with_duplicates) < 10:
            pyplot.title(f"rows_with_duplicates => {rows_with_duplicates}")
        else:
            pyplot.title("rows_with_duplicates => !!! MORE THAN 10")

    except IndexError:
        unique_values, counts = numpy.unique(seedy, return_counts=True)
        non_unique_count = numpy.sum(counts > 1)
        pyplot.hist(
            unique_values,
            bins=bins,
            label=f"seed {unique_values.shape[0]} vs {non_unique_count} w. bin = {bins}",
        )

    pyplot.xlabel("Unique Values")
    pyplot.ylabel("Frequency")
    pyplot.legend()
    pyplot.savefig(directory / "seeds_histogram_plot.pdf", dpi=300, bbox_inches="tight")


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
    pyplot.savefig(directory / "before_after.pdf", dpi=300)
