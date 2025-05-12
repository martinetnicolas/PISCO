from functools import partial
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import distrax
from flax import linen as nn
from jax import numpy

ModuleDef = Callable[..., Callable]
InitFn = Callable[[Any, Iterable[int], Any], Any]


# --------------------------------------------------------------------------------------------------
# PARAMS
# --------------------------------------------------------------------------------------------------


KERNEL_STEM = 7  # 7x7 filter for first layer
KERNEL_INIT = nn.initializers.lecun_normal()  # init weights
MOMENTUM = 0.9  # decay rate for the exponential moving average of the batch statistics

ACTIVATION = nn.gelu
# ACTIVATION = nn.relu


# --------------------------------------------------------------------------------------------------
# BLOCKS
# --------------------------------------------------------------------------------------------------


class MaxPool(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = numpy.pad(x, pad_width=[(0, 0), (1, 1), (1, 1), (0, 0)], mode="constant")
        return nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))


class SkipConnectionPool(nn.Module):
    @nn.compact
    def __call__(self, x, out_shape):
        if x.shape != out_shape:
            x = MaxPool()(x)
            x = nn.Conv(
                out_shape[-1],
                kernel_size=(1, 1),
                padding=((0, 0), (0, 0)),
                kernel_init=KERNEL_INIT,
                use_bias=False,
            )(x)

            mutable = self.is_mutable_collection("batch_stats")
            scale_init = nn.initializers.ones
            x = nn.BatchNorm(
                use_running_average=not mutable, momentum=MOMENTUM, scale_init=scale_init
            )(x)

        return x


class BasicBlockPool(nn.Module):
    features: int
    kernel: Tuple[int, int]
    pad_width: Union[Tuple[int, int]]
    downscale: bool = False
    is_act: bool = True

    @nn.compact
    def __call__(self, x):
        if self.downscale:
            x = MaxPool()(x)

        x = numpy.pad(x, pad_width=self.pad_width)
        x = nn.Conv(
            self.features, self.kernel, padding="VALID", kernel_init=KERNEL_INIT, use_bias=False
        )(x)

        mutable = self.is_mutable_collection("batch_stats")
        scale_init = nn.initializers.zeros if not self.is_act else nn.initializers.ones
        x = nn.BatchNorm(use_running_average=not mutable, momentum=MOMENTUM, scale_init=scale_init)(
            x
        )

        if self.is_act:
            x = ACTIVATION(x)

        return x


class ResNetBlockPool(nn.Module):
    features: int
    downscale: bool
    kernel: Tuple[int, int]
    pad_width: Union[Tuple[int, int]]

    @nn.compact
    def __call__(self, x):
        y = BasicBlockPool(
            features=self.features,
            downscale=self.downscale,
            kernel=self.kernel,
            pad_width=self.pad_width,
        )(x)
        y = BasicBlockPool(
            features=self.features, is_act=False, kernel=self.kernel, pad_width=self.pad_width
        )(y)
        residual = SkipConnectionPool()(x, y.shape)
        x = y + residual
        return ACTIVATION(x)


class EncoderPool(nn.Module):
    pre_layer: Sequence[int]
    layer: Sequence[int]
    kernel_size: Tuple[int, int]
    pad_width: Union[Tuple[int, int]]

    @nn.compact
    def __call__(self, x):
        # Traditional blocks
        for layer_idx, features in enumerate(self.pre_layer):
            x = BasicBlockPool(
                features=features, kernel=self.kernel_size, pad_width=self.pad_width
            )(x)
            x = MaxPool()(x)
            self.sow("intermediates", f"PreBlock_{layer_idx + 1}", x)

        # Resnet blocks
        features = x.shape[-1]  # retrieve number of features from the last conv layer
        for layer_idx, num_blocks in enumerate(self.layer):
            # double block layer
            for block_idx in range(num_blocks):
                downscale = block_idx == 0 and layer_idx != 0
                x = ResNetBlockPool(
                    features=features,
                    downscale=downscale,
                    kernel=self.kernel_size,
                    pad_width=self.pad_width,
                )(x)
                self.sow("intermediates", f"Block_{layer_idx + 1}", x)
            features *= 2  # double the number of features per new layer

        return x


# --------------------------------------------------------------------------------------------------
# OUTPUT DISTRIBUTIONS
# --------------------------------------------------------------------------------------------------


class UnivariateNormal(nn.Module):
    nb_cpt: int

    @nn.compact
    def __call__(self, x):
        logits_1 = nn.Dense(features=self.nb_cpt, name="logits_g1")(x)  # weights of gaussians dist
        mean_g1 = nn.Dense(features=self.nb_cpt, name="mean_g1")(x)
        std_g1 = 1e-5 + nn.softplus(nn.Dense(self.nb_cpt, name="std_g1")(x))
        dist_g1 = distrax.Independent(
            distrax.MixtureSameFamily(
                mixture_distribution=distrax.Categorical(logits=logits_1),
                components_distribution=distrax.Normal(loc=mean_g1, scale=std_g1),
            )
        )

        logits_2 = nn.Dense(features=self.nb_cpt, name="logits_g2")(x)
        mean_g2 = nn.Dense(features=self.nb_cpt, name="mean_g2")(x)
        std_g2 = 1e-5 + nn.softplus(nn.Dense(features=self.nb_cpt, name="std_g2")(x))
        dist_g2 = distrax.Independent(
            distrax.MixtureSameFamily(
                mixture_distribution=distrax.Categorical(logits=logits_2),
                components_distribution=distrax.Normal(loc=mean_g2, scale=std_g2),
            )
        )

        return dist_g1, dist_g2


class UnivariateNormalD(nn.Module):
    nb_cpt: int
    nb_feat: int

    @nn.compact
    def __call__(self, x):
        hidden_g1 = nn.Dense(self.nb_feat, name="dense_g1")(x)
        hidden_g1 = ACTIVATION(hidden_g1)

        logits_1 = nn.Dense(features=self.nb_cpt, name="logits_g1")(hidden_g1)
        mean_g1 = nn.Dense(features=self.nb_cpt, name="mean_g1")(hidden_g1)
        std_g1 = 1e-5 + nn.softplus(nn.Dense(self.nb_cpt, name="std_g1")(hidden_g1))
        dist_g1 = distrax.Independent(
            distrax.MixtureSameFamily(
                mixture_distribution=distrax.Categorical(logits=logits_1),
                components_distribution=distrax.Normal(loc=mean_g1, scale=std_g1),
            )
        )

        hidden_g2 = nn.Dense(self.nb_feat, name="dense_g2")(x)
        hidden_g2 = ACTIVATION(hidden_g2)

        logits_2 = nn.Dense(features=self.nb_cpt, name="logits_g2")(hidden_g2)
        mean_g2 = nn.Dense(features=self.nb_cpt, name="mean_g2")(hidden_g2)
        std_g2 = 1e-5 + nn.softplus(nn.Dense(features=self.nb_cpt, name="std_g2")(hidden_g2))
        dist_g2 = distrax.Independent(
            distrax.MixtureSameFamily(
                mixture_distribution=distrax.Categorical(logits=logits_2),
                components_distribution=distrax.Normal(loc=mean_g2, scale=std_g2),
            )
        )

        return dist_g1, dist_g2


class BivariateNormal(nn.Module):
    nb_cpt: int

    @nn.compact
    def __call__(self, x):
        logits = nn.Dense(features=self.nb_cpt, name="logits_g")(x)
        locs = nn.Dense(features=self.nb_cpt * 2, name="mean_g")(x)
        locs = locs.reshape((-1, self.nb_cpt, 2))

        tril_elements = nn.Dense(features=self.nb_cpt * 3, name="tril_g")(x)
        tril_elements = tril_elements.reshape((-1, self.nb_cpt, 3))

        scale_xx = 1e-5 + nn.softplus(tril_elements[..., 0])
        scale_yy = 1e-5 + nn.softplus(tril_elements[..., 1])
        correlation = numpy.tanh(tril_elements[..., 2])

        cholesky_factors = numpy.stack(
            [
                numpy.stack([scale_xx, numpy.zeros_like(correlation)], axis=-1),
                numpy.stack([correlation * scale_xx, scale_yy], axis=-1),
            ],
            axis=-2,
        )  # Shape: (batch_size, NB_OF_COMPONENTS, 2, 2)
        dist_g = distrax.MixtureSameFamily(
            mixture_distribution=distrax.Categorical(logits=logits),
            components_distribution=distrax.MultivariateNormalTri(
                loc=locs, scale_tri=cholesky_factors
            ),
        )

        return dist_g


class BivariateNormalD(nn.Module):
    nb_cpt: int
    nb_feat: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.nb_feat, name="dense_g")(x)
        x = ACTIVATION(x)

        logits = nn.Dense(features=self.nb_cpt, name="logits_g")(x)
        locs = nn.Dense(features=self.nb_cpt * 2, name="mean_g")(x)
        locs = locs.reshape((-1, self.nb_cpt, 2))

        tril_elements = nn.Dense(features=self.nb_cpt * 3, name="tril_g")(x)
        tril_elements = tril_elements.reshape((-1, self.nb_cpt, 3))

        scale_xx = 1e-5 + nn.softplus(tril_elements[..., 0])
        scale_yy = 1e-5 + nn.softplus(tril_elements[..., 1])
        correlation = numpy.tanh(tril_elements[..., 2])

        cholesky_factors = numpy.stack(
            [
                numpy.stack([scale_xx, numpy.zeros_like(correlation)], axis=-1),
                numpy.stack([correlation * scale_xx, scale_yy], axis=-1),
            ],
            axis=-2,
        )  # Shape: (batch_size, NB_OF_COMPONENTS, 2, 2)
        dist_g = distrax.MixtureSameFamily(
            mixture_distribution=distrax.Categorical(logits=logits),
            components_distribution=distrax.MultivariateNormalTri(
                loc=locs, scale_tri=cholesky_factors
            ),
        )

        return dist_g


# --------------------------------------------------------------------------------------------------
# BLOCKS JAX_RESNET GITHUB
# --------------------------------------------------------------------------------------------------


STAGE_SIZES = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3]}


class ConvBlock(nn.Module):
    n_filters: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    activation: Callable = ACTIVATION
    padding: Union[str, Iterable[Tuple[int, int]]] = ((0, 0), (0, 0))
    is_last: bool = False
    groups: int = 1
    kernel_init: InitFn = KERNEL_INIT
    bias_init: InitFn = nn.initializers.zeros

    conv_cls: ModuleDef = nn.Conv
    norm_cls: Optional[ModuleDef] = partial(nn.BatchNorm, momentum=MOMENTUM)

    force_conv_bias: bool = False

    @nn.compact
    def __call__(self, x):
        x = self.conv_cls(
            self.n_filters,
            self.kernel_size,
            self.strides,
            use_bias=(not self.norm_cls or self.force_conv_bias),
            padding=self.padding,
            feature_group_count=self.groups,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        if self.norm_cls:
            scale_init = nn.initializers.zeros if self.is_last else nn.initializers.ones
            mutable = self.is_mutable_collection("batch_stats")
            x = self.norm_cls(use_running_average=not mutable, scale_init=scale_init)(x)

        if not self.is_last:
            x = self.activation(x)
        return x


class ResNetStem(nn.Module):
    conv_block_cls: ModuleDef = ConvBlock

    @nn.compact
    def __call__(self, x):
        x = self.conv_block_cls(
            64, kernel_size=(KERNEL_STEM, KERNEL_STEM), strides=(2, 2), padding=[(3, 3), (3, 3)]
        )(x)
        return x


class ResNetSkipConnection(nn.Module):
    strides: Tuple[int, int]
    conv_block_cls: ModuleDef = ConvBlock

    @nn.compact
    def __call__(self, x, out_shape):
        if x.shape != out_shape:
            x = self.conv_block_cls(
                out_shape[-1], kernel_size=(1, 1), strides=self.strides, activation=lambda y: y
            )(x)
        return x


class ResNetBlock(nn.Module):
    n_hidden: int
    strides: Tuple[int, int] = (1, 1)

    activation: Callable = ACTIVATION
    conv_block_cls: ModuleDef = ConvBlock
    skip_cls: ModuleDef = ResNetSkipConnection

    @nn.compact
    def __call__(self, x):
        skip_cls = partial(self.skip_cls, conv_block_cls=self.conv_block_cls)
        y = self.conv_block_cls(self.n_hidden, padding=[(1, 1), (1, 1)], strides=self.strides)(x)
        y = self.conv_block_cls(self.n_hidden, padding=[(1, 1), (1, 1)], is_last=True)(y)
        z = self.activation(y + skip_cls(self.strides)(x, y.shape))
        self.sow("intermediates", f"Block_{self.n_hidden}", z)
        return z


def ResNet(
    block_cls: ModuleDef,
    *,
    stage_sizes: Sequence[int],
    hidden_sizes: Sequence[int] = (64, 128, 256, 512),
    conv_cls: ModuleDef = nn.Conv,
    norm_cls: Optional[ModuleDef] = partial(nn.BatchNorm, momentum=MOMENTUM),
    conv_block_cls: ModuleDef = ConvBlock,
    stem_cls: ModuleDef = ResNetStem,
    pool_fn: Callable = partial(
        nn.max_pool, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1))
    ),
) -> nn.Sequential:
    conv_block_cls = partial(conv_block_cls, conv_cls=conv_cls, norm_cls=norm_cls)
    stem_cls = partial(stem_cls, conv_block_cls=conv_block_cls)
    block_cls = partial(block_cls, conv_block_cls=conv_block_cls)

    layers = [stem_cls(), pool_fn]

    for i, (hsize, n_blocks) in enumerate(zip(hidden_sizes, stage_sizes)):
        for b in range(n_blocks):
            strides = (1, 1) if i == 0 or b != 0 else (2, 2)
            layers.append(block_cls(n_hidden=hsize, strides=strides))
    return nn.Sequential(layers)


ResNet18 = partial(ResNet, stage_sizes=STAGE_SIZES[18], stem_cls=ResNetStem, block_cls=ResNetBlock)
