from functools import partial
from typing import Dict, Sequence

import optax
import orbax.checkpoint
import tensorflow
from blocks import (
    BivariateNormal,
    BivariateNormalD,
    EncoderPool,
    ResNet18,
    UnivariateNormal,
    UnivariateNormalD,
)
from flax import linen as nn
from flax.training import train_state
from jax import jit, numpy, random, value_and_grad

# --------------------------------------------------------------------------------------------------
# NETWORK SCALAR & DISTRIBUTION
# --------------------------------------------------------------------------------------------------


ResNet_18 = partial(ResNet18)

mean_fn = partial(numpy.mean, axis=(1, 2))
# mean_fn = partial(numpy.max, axis=(1, 2))
# x = x.reshape((x.shape[0], -1))


class ResNet_MuScalar(nn.Module):
    nb_cpt: int  # for general consistancy
    nb_feat: int  # for general consistancy

    @nn.compact
    def __call__(self, x):
        x = ResNet_18()(x)
        x = self.perturb("conv_1", x)
        self.sow("intermediates", "final_conv_layer", x)
        x = mean_fn(x)
        return nn.Dense(2, name="Dense")(x)


class ResNetDistB(nn.Module):
    nb_cpt: int
    nb_feat: int  # for general consistancy

    @nn.compact
    def __call__(self, x):
        x = ResNet_18()(x)
        x = self.perturb("conv_1", x)
        self.sow("intermediates", "final_conv_layer", x)
        x = mean_fn(x)
        return BivariateNormal(nb_cpt=self.nb_cpt, name="Bivariate")(x)


class ResNetDistBD(nn.Module):
    nb_cpt: int
    nb_feat: int

    @nn.compact
    def __call__(self, x):
        x = ResNet_18()(x)
        x = self.perturb("conv_1", x)
        self.sow("intermediates", "final_conv_layer", x)
        x = mean_fn(x)
        return BivariateNormalD(nb_cpt=self.nb_cpt, nb_feat=self.nb_feat, name="Bivariate")(x)


class ResNetDistU(nn.Module):
    nb_cpt: int
    nb_feat: int  # for general consistancy

    @nn.compact
    def __call__(self, x):
        x = ResNet_18()(x)
        x = self.perturb("conv_1", x)
        self.sow("intermediates", "final_conv_layer", x)
        x = mean_fn(x)
        return UnivariateNormal(nb_cpt=self.nb_cpt, name="Univariate")(x)


class ResNetDistUD(nn.Module):
    nb_cpt: int
    nb_feat: int

    @nn.compact
    def __call__(self, x):
        x = ResNet_18()(x)
        x = self.perturb("conv_1", x)
        self.sow("intermediates", "final_conv_layer", x)
        x = mean_fn(x)
        return UnivariateNormalD(nb_cpt=self.nb_cpt, nb_feat=self.nb_feat, name="Univariate")(x)


class OwnResNet(nn.Module):
    pre_layer: Sequence[int]
    layer: Sequence[int]
    kernel_size: int
    pad: int
    nb_cpt: int
    nb_feat: int  # for general consistancy

    @nn.compact
    def __call__(self, x):
        x = EncoderPool(
            pre_layer=self.pre_layer,
            layer=self.layer,
            kernel_size=(self.kernel_size, self.kernel_size),
            pad_width=[(0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)],
            name="Encoder",
        )(x)
        x = self.perturb("conv_1", x)
        self.sow("intermediates", "final_conv_layer", x)
        x = mean_fn(x)
        return UnivariateNormal(nb_cpt=self.nb_cpt, name="Univariate")(x)


class OwnResNetD(nn.Module):
    pre_layer: Sequence[int]
    layer: Sequence[int]
    kernel_size: int
    pad: int
    nb_cpt: int
    nb_feat: int

    @nn.compact
    def __call__(self, x):
        x = EncoderPool(
            pre_layer=self.pre_layer,
            layer=self.layer,
            kernel_size=(self.kernel_size, self.kernel_size),
            pad_width=[(0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)],
            name="Encoder",
        )(x)
        x = self.perturb("conv_1", x)
        self.sow("intermediates", "final_conv_layer", x)
        x = mean_fn(x)
        return UnivariateNormalD(nb_cpt=self.nb_cpt, nb_feat=self.nb_feat, name="Univariate")(x)


class OwnResNetB(nn.Module):
    pre_layer: Sequence[int]
    layer: Sequence[int]
    kernel_size: int
    pad: int
    nb_cpt: int
    nb_feat: int  # for general consistancy

    @nn.compact
    def __call__(self, x):
        x = EncoderPool(
            pre_layer=self.pre_layer,
            layer=self.layer,
            kernel_size=(self.kernel_size, self.kernel_size),
            pad_width=[(0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)],
            name="Encoder",
        )(x)
        x = self.perturb("conv_1", x)
        self.sow("intermediates", "final_conv_layer", x)
        x = mean_fn(x)
        return BivariateNormal(nb_cpt=self.nb_cpt, name="Bivariate")(x)


# --------------------------------------------------------------------------------------------------
# STATE
# --------------------------------------------------------------------------------------------------


class TrainState(train_state.TrainState):
    model_config: Dict
    batch_stats: Dict
    perturbations: Dict


def create_train_state(model_config):
    if model_config["style"] < 10:
        model = MODEL_TYPES[model_config["style"]](
            nb_cpt=model_config["nb_cpt"], nb_feat=model_config["nb_feat"]
        )
    else:
        model = MODEL_TYPES[model_config["style"]](
            kernel_size=model_config["kernel_size"],
            pad=model_config["pad"],
            pre_layer=model_config["pre_layer"],
            layer=model_config["layer"],
            nb_cpt=model_config["nb_cpt"],
            nb_feat=model_config["nb_feat"],
        )
    model_config = {key: numpy.array(value) for key, value in model_config.items()}

    variables = model.init(random.key(seed=22), numpy.ones([1, 352, 352, 1]))

    init_value = 0.001
    steps_and_lrs = [
        (7500, 0.5),
        (15000, 0.5),
        (20000, 0.2),
        (25000, 0.5),
        (30000, 0.5),
        (35000, 0.2),
        (40000, 0.2),
    ]
    learning_rate = optax.schedules.piecewise_interpolate_schedule(
        interpolate_type="cosine",
        init_value=init_value,
        boundaries_and_scales={step: lr for step, lr in steps_and_lrs},
    )
    optimiser = optax.adamw(learning_rate=learning_rate, weight_decay=1e-3)

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        batch_stats=variables["batch_stats"],
        perturbations=variables["perturbations"],
        tx=optimiser,
        model_config=model_config,
    )

    return learning_rate, train_state


def load_state(folder):
    # restore best weights and bias
    restored_data = orbax.checkpoint.PyTreeCheckpointer().restore(folder)
    state = restored_data["model"]

    # model characteristics
    model_config = state["model_config"]
    style = int(model_config["style"])

    # define best model
    if model_config["style"] < 10:
        model = MODEL_TYPES[style](nb_cpt=model_config["nb_cpt"], nb_feat=model_config["nb_feat"])
    else:
        model = MODEL_TYPES[style](
            kernel_size=model_config["kernel_size"],
            pad=model_config["pad"],
            pre_layer=model_config["pre_layer"],
            layer=model_config["layer"],
            nb_cpt=model_config["nb_cpt"],
            nb_feat=model_config["nb_feat"],
        )

    return model, state


# --------------------------------------------------------------------------------------------------
# UPDATE & EVAL FUNCTIONS
# --------------------------------------------------------------------------------------------------


@jit
def update_model_scalar(state, x, y):
    def loss_fn_multi(params, batch_stats, perturbations):
        variables = {"params": params, "batch_stats": batch_stats, "perturbations": perturbations}
        preds = state.apply_fn(variables, x)
        loss_val = numpy.mean(numpy.sum((preds - y) ** 2, axis=-1))
        return loss_val

    gradient_fn = value_and_grad(loss_fn_multi)
    loss, grads = gradient_fn(state.params, state.batch_stats, state.perturbations)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


@jit
def eval_function_scalar(state, x, y):
    variables = {
        "params": state.params,
        "batch_stats": state.batch_stats,
        "perturbations": state.perturbations,
    }
    preds = state.apply_fn(variables, x)
    return numpy.mean(numpy.sum((preds - y) ** 2, axis=-1))


@jit
def update_model_bivariate_normal(state, x, y):
    def loss_fn_multi(params, batch_stats, perturbations):
        variables = {"params": params, "batch_stats": batch_stats, "perturbations": perturbations}
        q = state.apply_fn(variables, x)
        loss_val = numpy.mean(-q.log_prob(y))
        return loss_val

    gradient_fn = value_and_grad(loss_fn_multi)
    loss, grads = gradient_fn(state.params, state.batch_stats, state.perturbations)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


@jit
def eval_function_bivariate_normal(state, x, y):
    variables = {
        "params": state.params,
        "batch_stats": state.batch_stats,
        "perturbations": state.perturbations,
    }
    q = state.apply_fn(variables, x)
    return numpy.mean(-q.log_prob(y))


@jit
def update_model_univariate_normal(state, x, y):
    def loss_fn_uni(params, batch_stats, perturbations):
        variables = {"params": params, "batch_stats": batch_stats, "perturbations": perturbations}
        q1, q2 = state.apply_fn(variables, x)
        loss_val = numpy.mean(-q1.log_prob(y[:, 0]) - q2.log_prob(y[:, 1]))
        return loss_val

    gradient_fn = value_and_grad(loss_fn_uni)
    loss, grads = gradient_fn(state.params, state.batch_stats, state.perturbations)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


@jit
def eval_function_univariate_normal(state, x, y):
    variables = {
        "params": state.params,
        "batch_stats": state.batch_stats,
        "perturbations": state.perturbations,
    }
    q1, q2 = state.apply_fn(variables, x)
    return numpy.mean(-q1.log_prob(y[:, 0]) - q2.log_prob(y[:, 1]))


@jit
def update_model_bivariate_normal_batch_stats(state, x, y):
    def loss_fn_multi(params, batch_stats):
        var = {"params": params, "batch_stats": batch_stats}
        q, updates = state.apply_fn(var, x, mutable=["batch_stats"])
        loss_val = numpy.mean(-q.log_prob(y))
        return loss_val, updates

    gradient_fn = value_and_grad(loss_fn_multi, has_aux=True)
    (loss, updates), grads = gradient_fn(state.params, state.batch_stats)
    new_state = state.apply_gradients(grads=grads, batch_stats=updates["batch_stats"])
    return new_state, loss


@jit
def update_model_univariate_normal_batch_stats(state, x, y):
    def loss_fn_uni(params, batch_stats):
        var = {"params": params, "batch_stats": batch_stats}
        (q1, q2), updates = state.apply_fn(var, x, mutable=["batch_stats"])
        loss_val = numpy.mean(-q1.log_prob(y[:, 0]) - q2.log_prob(y[:, 1]))
        return loss_val, updates

    gradient_fn = value_and_grad(loss_fn_uni, has_aux=True)
    (loss, updates), grads = gradient_fn(state.params, state.batch_stats)
    new_state = state.apply_gradients(grads=grads, batch_stats=updates["batch_stats"])
    return new_state, loss


# --------------------------------------------------------------------------------------------------
# SUMMARY
# --------------------------------------------------------------------------------------------------


MODEL_TYPES = {
    1: ResNet_MuScalar,
    2: ResNetDistB,
    3: ResNetDistBD,
    4: ResNetDistU,
    5: ResNetDistUD,
    10: OwnResNet,
    11: OwnResNetD,
    12: OwnResNetB,
}


LIST_UPDATE_FUNCTION = {
    1: update_model_scalar,
    2: update_model_bivariate_normal,
    3: update_model_bivariate_normal,
    4: update_model_univariate_normal,
    5: update_model_univariate_normal,
    10: update_model_univariate_normal,
    11: update_model_univariate_normal,
    12: update_model_bivariate_normal,
}


LIST_EVAL_FUNCTION = {
    1: eval_function_scalar,
    2: eval_function_bivariate_normal,
    3: eval_function_bivariate_normal,
    4: eval_function_univariate_normal,
    5: eval_function_univariate_normal,
    10: eval_function_univariate_normal,
    11: eval_function_univariate_normal,
    12: eval_function_bivariate_normal,
}


# --------------------------------------------------------------------------------------------------
# PREPROCESSING
# --------------------------------------------------------------------------------------------------


BOUNDS_NORMA = numpy.array([1e-5, 1 - 1e-5])


def rescale(x, old_min, old_max, new_min, new_max):
    old_size = old_max - old_min
    new_size = new_max - new_min
    scale_factor = new_size / old_size
    return new_min + (x - old_min) * scale_factor


def range_compression(img, std_images):
    return tensorflow.clip_by_value(
        tensorflow.math.asinh(
            tensorflow.expand_dims(img, axis=-1) / tensorflow.constant(std_images) / 3.0
        ),
        -10,
        10,
    )


def normalised_preds(x, old_min, old_max):
    return rescale(
        x, old_min=old_min, old_max=old_max, new_min=BOUNDS_NORMA[0], new_max=BOUNDS_NORMA[1]
    )


def preprocessing_shear(example, std_images, old_min, old_max):
    return range_compression(example["image"], std_images), normalised_preds(
        example["avg_g"], old_min, old_max
    )


def preprocessing_ellips(example, std_images, old_min, old_max):
    return range_compression(example["image"], std_images), normalised_preds(
        example["avg_e"], old_min, old_max
    )


def denormalised_preds(x, new_min, new_max):
    return rescale(
        x, old_min=BOUNDS_NORMA[0], old_max=BOUNDS_NORMA[1], new_min=new_min, new_max=new_max
    )
