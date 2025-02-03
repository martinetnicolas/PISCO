import shutil
import sys
from pathlib import Path

import commentjson
from jax import nn, numpy, scipy


def _find_file_cwd(name):
    return next((f for f in Path.cwd().rglob(name)))


def _get_json_file_parameters():
    with open(_find_file_cwd("parameters.jsonc"), "r", encoding="utf-8") as file:
        user_parameters = commentjson.load(file)
    return user_parameters


def general_params():
    db = _get_json_file_parameters()["db"]
    folder = search_folder(db)
    return folder


def _find_directory(name, init_directory):
    directory_path = Path(init_directory)
    matching_folders = list(directory_path.glob(f"**/{name}"))
    return matching_folders[0]


def search_folder(name):
    if Path.cwd().parts[1] == "home":
        search_dir = "/home/rbrunet/pisco/PIX2SHEAR"
    else:
        search_dir = "/net/GECO/pisco/PIX2SHEAR"

    return _find_directory(name, search_dir)


def network_params():
    params = _get_json_file_parameters()["network"]
    return (
        params["styles"],
        params["batches"],
        params["steps"],
        params["versions"],
        params["predict"],
    )


def architecture_params():
    params = _get_json_file_parameters()["architecture"]
    return (
        params["kernel_sizes_and_pads"],
        params["pre_layers"],
        params["layers"],
        params["nb_cpts"],
        params["nb_feats"],
    )


def get_name(model_config, version, predict):
    style = model_config["style"]
    batch = model_config["batch"]
    steps = model_config["steps"]
    nb_cpt = model_config["nb_cpt"]
    nb_feat = model_config["nb_feat"]
    name_case = f"Net-{style:02}_batch-{batch}_steps-{steps}_db-{version}_pred-{predict}_"
    if style in (1,):
        return name_case + "q-multiscalar"
    if style in (2, 3):
        name_case += f"cpt-{nb_cpt}_feat-{nb_feat}_"
        return name_case + "q-multi"
    elif style in (4, 5):
        name_case += f"cpt-{nb_cpt}_feat-{nb_feat}_"
        return name_case + "q-uni"
    else:
        name_case += f"cpt-{nb_cpt}_"
        kernel_size = model_config["kernel_size"]
        pre_layer = model_config["pre_layer"]
        pre_layer = "".join(str(pre) for pre in pre_layer)
        layer = model_config["layer"]
        layer = "".join(str(res) for res in layer)
        name_case += f"prelayer-{pre_layer}_layer-{layer}_kernel-{kernel_size}_"
        if style in (10,):
            return name_case + "q-uni"
        elif style in (11,):
            name_case += f"cpt-{nb_cpt}_feat-{nb_feat}_"
            return name_case + "q-uni"
        elif style in (12,):
            return name_case + "q-multi"


def slurm_mode():
    import os

    if "SLURM_JOB_ID" in os.environ:
        if os.environ.get("SLURM_JOB_NAME") != "bash":
            print("Running in sbatch mode")
            return True

    print("Running in interactive mode")
    return False


def erase_test(case):
    RED, GREEN = "\033[91m", "\033[92m"
    try:
        case.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        print(f"!!! {RED}{case.name}\033[0m {GREEN}already exists \033[0m !!!")

        response = input("REMOVE AND RECREATE ? (Y/n):")
        if response.lower() == "y" or response == "":
            shutil.rmtree(case)
            case.mkdir(parents=False, exist_ok=False)
        else:
            sys.exit()
    return case


def get_arg(name_bdd, value):
    parts = name_bdd.split("_")
    dist_part = next((part for part in parts if value in part), None)
    return dist_part.split("-")[-1]


def params_for_uni(q1, q2):
    weights_g1 = nn.softmax(q1.distribution.mixture_distribution.logits)
    weights_g2 = nn.softmax(q2.distribution.mixture_distribution.logits)
    weights = numpy.stack([weights_g1, weights_g2], axis=0)

    means_g1 = q1.distribution.components_distribution.loc
    means_g2 = q2.distribution.components_distribution.loc
    means = numpy.stack([means_g1, means_g2], axis=0)

    std_devs_g1 = q1.distribution.components_distribution.scale
    std_devs_g2 = q2.distribution.components_distribution.scale
    stddevs = numpy.stack([std_devs_g1, std_devs_g2], axis=0)

    return (
        numpy.transpose(weights, axes=(1, 0, 2)),
        numpy.transpose(means, axes=(1, 0, 2)),
        numpy.transpose(stddevs, axes=(1, 0, 2)),
    )


def sampling_pdf_uni(weights, means, std_devs, xvalues):
    nb_of_images = weights.shape[0]
    pdf = numpy.tile(numpy.zeros_like(xvalues), (nb_of_images, 1))

    for img in range(nb_of_images):
        temp_pdf = numpy.zeros(pdf.shape[1])

        for poid, mean, std_dev in zip(weights[img, :], means[img, :], std_devs[img, :]):
            pdf_values = scipy.stats.norm.pdf(xvalues, mean, std_dev)
            temp_pdf += poid * pdf_values
        pdf = pdf.at[img].set(temp_pdf)

    return pdf


def compute_mode(pdf, x_values):
    max_index = numpy.argmax(pdf)
    return x_values[max_index]


def params_for_multi(q):
    weights = nn.softmax(q.mixture_distribution.logits)
    means = q.components_distribution.loc
    stddevs = q.components_distribution.scale_tri
    return (
        weights,
        numpy.transpose(means, axes=(0, 2, 1)),
        numpy.transpose(stddevs, axes=(0, 2, 3, 1)),
    )


def adaptive_sampling_for_multi(truth, zoom, bounds, nb_points):
    low_bound = numpy.maximum(truth - zoom, bounds[0])
    up_bound = numpy.minimum(truth + zoom, bounds[1])
    return numpy.linspace(low_bound, up_bound, nb_points)


def _compute_mode_multi(xy, pdf):
    max_index = numpy.argmax(pdf)
    row, col = numpy.unravel_index(max_index, pdf.shape)
    mode_x = xy[0, col]
    mode_y = xy[1, row]
    return numpy.array([mode_x, mode_y])


def sampling_pdf_multi(weights, means, stddevs, xy):
    NB_IMAGES = weights.shape[0]
    NB_COMPONENTS = weights.shape[-1]
    NB_POINTS = xy.shape[-1]
    pdf = numpy.zeros((NB_IMAGES, NB_POINTS, NB_POINTS))
    mode = numpy.zeros((NB_IMAGES, 2))

    for img in range(NB_IMAGES):
        # compute grids
        if xy.ndim == 3:
            X, Y = numpy.meshgrid(xy[img, 0], xy[img, 1])
        else:
            X, Y = numpy.meshgrid(xy[0], xy[1])
        pos = numpy.dstack((X, Y))
        temp_pdf = numpy.zeros(pos.shape[:2])

        # compute pdfs
        for i in range(NB_COMPONENTS):
            mean = means[img, :, i]
            cov = stddevs[img, :, :, i]
            pdf_component = scipy.stats.multivariate_normal.pdf(pos, mean=mean, cov=cov)
            temp_pdf += weights[img, i] * pdf_component
        pdf = pdf.at[img].set(temp_pdf)

        # compute modes
        if xy.ndim == 3:
            temp_mode = _compute_mode_multi(xy[img], pdf[img])
        else:
            temp_mode = _compute_mode_multi(xy, pdf[img])
        mode = mode.at[img].set(temp_mode)

    return pdf, mode


def compute_combine_std(weights, means, stddevs):
    normalised_weights = weights / numpy.sum(weights, axis=-1, keepdims=True)
    combined_mean = numpy.sum(normalised_weights[:, None, :] * means, axis=-1)
    weighted_variance = numpy.sum(normalised_weights[:, None, :] * (stddevs**2), axis=-1)
    mean_variance = numpy.sum(
        normalised_weights[:, None, :] * (means - combined_mean[:, :, None]) ** 2, axis=-1
    )
    combined_variance = weighted_variance + mean_variance
    combined_std = numpy.sqrt(combined_variance)
    return combined_std


def compute_combine_std_uni(weights, means, stddevs):
    normalised_weights = weights / numpy.sum(weights, axis=-1, keepdims=True)
    combined_mean = numpy.sum(normalised_weights * means, axis=-1)
    weighted_variance = numpy.sum(normalised_weights * (stddevs**2), axis=-1)
    mean_variance = numpy.sum(
        normalised_weights * (means - combined_mean[:, :, None]) ** 2, axis=-1
    )
    combined_variance = weighted_variance + mean_variance
    combined_std = numpy.sqrt(combined_variance)
    return combined_std


def best_fit_line(x, y):
    slope, intercept = numpy.polyfit(x, y, 1)
    return slope * x + intercept, slope, intercept


def fitting(preds, true):
    diff = preds - true
    fit_g1, slope_g1, bias_g1 = best_fit_line(true[:, 0], diff[:, 0])
    fit_g2, slope_g2, bias_g2 = best_fit_line(true[:, 1], diff[:, 1])
    return (
        numpy.stack([fit_g1, fit_g2], axis=-1),
        [slope_g1, slope_g2],
        [bias_g1, bias_g2],
        diff,
    )


def get_resnet_blocks(states):
    layer_conv = [numpy.array([]) for _ in range(len(states) - 1)]
    for idx, (_, data) in enumerate((states.items())):
        if isinstance(data, dict):
            name_block = list(data.keys())[0]
            layer_conv[idx] = numpy.array(data[name_block][0]).squeeze()
    return layer_conv


def get_resnet_blocks2(states):
    layer_conv = [numpy.array([]) for _ in range(len(states))]
    for idx, (_, data) in enumerate((states.items())):
        layer_conv[idx] = data[0].squeeze()
    return layer_conv


def simplify(array):
    def modify(value):
        if isinstance(value, numpy.ndarray):
            return int(value)
        elif isinstance(value, list):
            return [int(v) for v in value]
        return value

    return {key: modify(value) for key, value in array.items()}
