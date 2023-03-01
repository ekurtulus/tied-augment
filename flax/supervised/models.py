# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flax implementation of ResNet V1."""

# See issue #620.
# pytype: disable=wrong-arg-count

from functools import partial
from typing import Any, Callable, Sequence, Tuple, Union, Optional

import jax
from flax import linen as nn
import jax.numpy as jnp
from jax.nn import initializers
import numpy as np

ModuleDef = Any

class SplitBN(nn.Module):
    first_bn: ModuleDef
    second_bn: ModuleDef
    train: bool
    scale_init: Any = initializers.ones
    
    @nn.compact
    def __call__(self, x):
        if not self.train:
            return self.first_bn(scale_init=self.scale_init)(x)
        else:
            first, second = jnp.split(x, 2, axis=0)
            first = self.first_bn(name=None, scale_init=self.scale_init)(first)
            second = self.second_bn(name=None, scale_init=self.scale_init)(second)
            return jnp.concatenate((first, second), axis=0)

class ResNetBlock(nn.Module):
  """ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x,):
    residual = x
    y = self.conv(self.filters, (3, 3), self.strides)(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3))(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters, (1, 1),
                           self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
  """Bottleneck ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    residual = x
    y = self.conv(self.filters, (1, 1))(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3), self.strides)(y)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters * 4, (1, 1))(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters * 4, (1, 1),
                           self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class ResNet(nn.Module):
  """ResNetV1."""
  stage_sizes: Sequence[int]
  block_cls: ModuleDef
  split_batchnorm : bool
  num_classes: int
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable = nn.relu
  conv: ModuleDef = nn.Conv
  
  @nn.compact
  def __call__(self, x, train: bool = True):
    conv = partial(self.conv, use_bias=False, dtype=self.dtype)
    norm = partial(nn.BatchNorm,
                   use_running_average=not train,
                   momentum=0.9,
                   epsilon=1e-5,
                   dtype=self.dtype)
    
    if self.split_batchnorm:
        norm = partial(SplitBN, first_bn=norm, second_bn=norm, train=train)
        print("using split BN ! ")
        
    x = conv(self.num_filters, (7, 7), (2, 2),
             padding=[(3, 3), (3, 3)],
             name='conv_init')(x)
    x = norm(name='bn_init')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(self.num_filters * 2 ** i,
                           strides=strides,
                           conv=conv,
                           norm=norm,
                           act=self.act)(x)
    out = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, dtype=self.dtype)(out)
    
    out = jnp.asarray(out, self.dtype)
    x = jnp.asarray(x, self.dtype)
    
    return out, x


def conv_args(kernel_size: int, nout: int):
    """Returns list of arguments which are common to all convolutions.
    Args:
        kernel_size: size of convolution kernel (single number).
        nout: number of output filters.
    Returns:
        Dictionary with common convoltion arguments.
    """
    stddev = (0.5 * kernel_size * kernel_size * nout) ** -0.5
    return dict(kernel_init=nn.initializers.normal(stddev),
                use_bias=False,
                padding='SAME')

class WRNBlock(nn.Module):
    """WideResNet block."""
    nin: int
    nout: int
    norm: ModuleDef
    stride: int = 1

    @nn.compact
    def __call__(self, x):
        x = self.norm()(x)
        o1 = nn.relu(x)
        y = nn.Conv(self.nout, (3, 3), strides=self.stride, **conv_args(3, self.nout))(o1)
        y = self.norm()(y)
        o2 = nn.relu(y)
        z = nn.Conv(self.nout, (3, 3), strides=1, **conv_args(3, self.nout))(o2)

        return z + nn.Conv(self.nout, (1, 1), strides=self.stride, **conv_args(1, self.nout))(o1) if self.nin != self.nout or self.stride > 1 else z + x


class WideResNet(nn.Module):
    """Base WideResNet implementation."""
    num_classes: int
    depth: Tuple[int]
    width: int
    split_batchnorm : bool
    norm: ModuleDef = nn.BatchNorm
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train : bool=True):
        n = (self.depth - 4) // 6
        blocks_per_group = (n,) * 3
        norm = partial(self.norm,
                    use_running_average=not train,
                    momentum=0.9,
                    epsilon=1e-5,
                    dtype=self.dtype)
        
        if self.split_batchnorm:
            norm = partial(SplitBN, first_bn=norm, second_bn=norm, train=train)
            print("using split BN ! ")

        widths = [int(v * self.width) for v in [16 * (2 ** i) for i in range(len(blocks_per_group))]]
        n = 16
        x = nn.Conv(n, (3, 3), **conv_args(3, n))(x)
        for i, (block, width) in enumerate(zip(blocks_per_group, widths)):
            stride = 2 if i > 0 else 1
            x = WRNBlock(n, width, norm, stride)(x)
            for b in range(1, block):
                x = WRNBlock(width, width, norm, 1)(x)
            n = width
        x = norm()(x)
        x = nn.relu(x)
        features = jnp.mean(x, axis=(-3, -2))
        logits = nn.Dense(self.num_classes, name="fc", kernel_init=nn.initializers.glorot_normal())(features)
        return features, logits





ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2],
                   block_cls=ResNetBlock, split_batchnorm=False)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=ResNetBlock, split_batchnorm=False)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=BottleneckResNetBlock, split_batchnorm=False)
ResNet101 = partial(ResNet, stage_sizes=[3, 4, 23, 3],
                    block_cls=BottleneckResNetBlock, split_batchnorm=False)
ResNet152 = partial(ResNet, stage_sizes=[3, 8, 36, 3],
                    block_cls=BottleneckResNetBlock, split_batchnorm=False)
ResNet200 = partial(ResNet, stage_sizes=[3, 24, 36, 3],
                    block_cls=BottleneckResNetBlock, split_batchnorm=False)

ResNet18_SBN = partial(ResNet, stage_sizes=[2, 2, 2, 2],
                   block_cls=ResNetBlock, split_batchnorm=True)
ResNet34_SBN = partial(ResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=ResNetBlock, split_batchnorm=True)
ResNet50_SBN = partial(ResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=BottleneckResNetBlock, split_batchnorm=True)
ResNet101_SBN = partial(ResNet, stage_sizes=[3, 4, 23, 3],
                    block_cls=BottleneckResNetBlock, split_batchnorm=True)
ResNet152_SBN = partial(ResNet, stage_sizes=[3, 8, 36, 3],
                    block_cls=BottleneckResNetBlock, split_batchnorm=True)
ResNet200_SBN = partial(ResNet, stage_sizes=[3, 24, 36, 3],
                    block_cls=BottleneckResNetBlock, split_batchnorm=True)

ResNet18Local = partial(ResNet, stage_sizes=[2, 2, 2, 2],
                        block_cls=ResNetBlock, conv=nn.ConvLocal)


WRN28_2 = partial(WideResNet, depth=28, width=2, split_batchnorm=False)
WRN28_8 = partial(WideResNet, depth=28, width=8, split_batchnorm=False)
WRN28_10 = partial(WideResNet, depth=28, width=10, split_batchnorm=False)
WRN37_2 = partial(WideResNet, depth=37, width=2, split_batchnorm=False)
WRN37_10 = partial(WideResNet, depth=37, width=10, split_batchnorm=False)
WRN40_2 = partial(WideResNet, depth=40, width=2, split_batchnorm=False)
WRN40_10 = partial(WideResNet, depth=40, width=10, split_batchnorm=False)

WRN28_2_SBN = partial(WideResNet, depth=28, width=2, split_batchnorm=True)
WRN28_8_SBN = partial(WideResNet, depth=28, width=8, split_batchnorm=True)
WRN28_10_SBN = partial(WideResNet, depth=28, width=10, split_batchnorm=True)
WRN37_2_SBN = partial(WideResNet, depth=37, width=2, split_batchnorm=True)
WRN37_10_SBN = partial(WideResNet, depth=37, width=10, split_batchnorm=True)
WRN40_2_SBN = partial(WideResNet, depth=40, width=2, split_batchnorm=True)
WRN40_10_SBN = partial(WideResNet, depth=40, width=10, split_batchnorm=True)





"""
ResNetRS implementations
"""

# Note: The batch norms require specialized training functions, checkout the resnetrs branch

BLOCK_ARGS = {
    50: [
        {
            "input_filters": 64,
            "num_repeats": 3
        },
        {
            "input_filters": 128,
            "num_repeats": 4
        },
        {
            "input_filters": 256,
            "num_repeats": 6
        },
        {
            "input_filters": 512,
            "num_repeats": 3
        },
    ],
    101: [
        {
            "input_filters": 64,
            "num_repeats": 3
        },
        {
            "input_filters": 128,
            "num_repeats": 4
        },
        {
            "input_filters": 256,
            "num_repeats": 23
        },
        {
            "input_filters": 512,
            "num_repeats": 3
        },
    ],
    152: [
        {
            "input_filters": 64,
            "num_repeats": 3
        },
        {
            "input_filters": 128,
            "num_repeats": 8
        },
        {
            "input_filters": 256,
            "num_repeats": 36
        },
        {
            "input_filters": 512,
            "num_repeats": 3
        },
    ],
    200: [
        {
            "input_filters": 64,
            "num_repeats": 3
        },
        {
            "input_filters": 128,
            "num_repeats": 24
        },
        {
            "input_filters": 256,
            "num_repeats": 36
        },
        {
            "input_filters": 512,
            "num_repeats": 3
        },
    ],
    270: [
        {
            "input_filters": 64,
            "num_repeats": 4
        },
        {
            "input_filters": 128,
            "num_repeats": 29
        },
        {
            "input_filters": 256,
            "num_repeats": 53
        },
        {
            "input_filters": 512,
            "num_repeats": 4
        },
    ],
    350: [
        {
            "input_filters": 64,
            "num_repeats": 4
        },
        {
            "input_filters": 128,
            "num_repeats": 36
        },
        {
            "input_filters": 256,
            "num_repeats": 72
        },
        {
            "input_filters": 512,
            "num_repeats": 4
        },
    ],
    420: [
        {
            "input_filters": 64,
            "num_repeats": 4
        },
        {
            "input_filters": 128,
            "num_repeats": 44
        },
        {
            "input_filters": 256,
            "num_repeats": 87
        },
        {
            "input_filters": 512,
            "num_repeats": 4
        },
    ],
}


def fixed_padding(inputs, kernel_size):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return jnp.pad(inputs, ((0, 0), (pad_beg, pad_end), (pad_beg, pad_end), (0, 0)))


class Conv2DFixedPadding(nn.Module):
    filters: int
    kernel_size: int
    strides: int
    name: str = None
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        if self.strides > 1:
            x = fixed_padding(x, self.kernel_size)
        return nn.Conv(
            self.filters,
            (self.kernel_size, self.kernel_size),
            self.strides,
            padding="SAME" if self.strides == 1 else "VALID",
            use_bias=False,
            kernel_init=jax.nn.initializers.variance_scaling(2.0, "fan_out", "truncated_normal"),
            name=self.name, 
            dtype=self.dtype
        )(x)


class STEM(nn.Module):
    bn_momentum: float = 0.99
    bn_epsilon: float = 1e-5
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train=True):
        x = Conv2DFixedPadding(32, kernel_size=3, strides=2, name="stem_conv_1", dtype=self.dtype,)(x)
        x = nn.BatchNorm(
            name="stem_batch_norm_1",
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon,
            use_running_average=not train
        )(x)
        x = nn.relu(x)
        x = Conv2DFixedPadding(32, kernel_size=3, strides=1, name="stem_conv_2", dtype=self.dtype,)(x)
        x = nn.BatchNorm(
            name="stem_batch_norm_2",
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon,
            use_running_average=not train
        )(x)
        x = nn.relu(x)
        x = Conv2DFixedPadding(64, kernel_size=3, strides=1, name="stem_conv_3", dtype=self.dtype,)(x)
        x = nn.BatchNorm(
            name="stem_batch_norm_3",
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon,
            use_running_average=not train
        )(x)
        x = nn.relu(x)
        x = Conv2DFixedPadding(64, kernel_size=3, strides=1, name="stem_conv_4", dtype=self.dtype,)(x)
        x = nn.BatchNorm(
            name="stem_batch_norm_4",
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon,
            use_running_average=not train
        )(x)
        x = nn.relu(x)
        return x


class ResNetRSBlockGroup(nn.Module):
    filters: int
    strides: int
    num_repeats: int
    counter: int
    se_ratio: float = 0.25
    bn_epsilon: float = 1e-5
    bn_momentum: float = 0.99
    survival_probability: float = 0.8
    name: str = None
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train=True):
        if self.name is None:
            self.name = f"block_group_{self.counter}"
        x = ResNetRSBottleneckBlock(
            self.filters,
            strides=self.strides,
            use_projection=True,
            se_ratio=self.se_ratio,
            bn_epsilon=self.bn_epsilon,
            bn_momentum=self.bn_momentum,
            survival_probability=self.survival_probability,
            name=self.name + "_block_0",
            dtype=self.dtype,
        )(x, train)
        for i in range(1, self.num_repeats):
            x = ResNetRSBottleneckBlock(
                self.filters,
                strides=1,
                use_projection=False,
                se_ratio=self.se_ratio,
                bn_epsilon=self.bn_epsilon,
                bn_momentum=self.bn_momentum,
                survival_probability=self.survival_probability,
                name=self.name + f"_block_{i}_",
                dtype=self.dtype,
            )(x, train)
        return x


class ResNetRSBottleneckBlock(nn.Module):
    filters: int
    strides: int
    use_projection: bool
    bn_momentum: float = 0.99
    bn_epsilon: float = 1e-5
    survival_probability: float = 0.8
    se_ratio: float = 0.25
    name: str
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train=True):
        shortcut = x
        if self.use_projection:
            filters_out = self.filters * 4
            if self.strides == 2:
                shortcut = nn.avg_pool(x, (2, 2), (2, 2), padding="SAME")
                shortcut = Conv2DFixedPadding(
                    filters_out,
                    kernel_size=1,
                    strides=1,
                    name=f"{self.name}_projection_conv",
                    dtype=self.dtype
                )(shortcut)
            else:
                shortcut = Conv2DFixedPadding(
                    filters_out,
                    kernel_size=1,
                    strides=self.strides,
                    name=f"{self.name}_projection_conv",
                    dtype=self.dtype
                )(shortcut)
            shortcut = nn.BatchNorm(
                axis=3, momentum=self.bn_momentum, epsilon=self.bn_epsilon, use_running_average=not train,
                name=f"{self.name}_projection_batch_norm"
            )(shortcut)
        x = Conv2DFixedPadding(self.filters, kernel_size=1, strides=1, name=self.name + "_conv_1", dtype=self.dtype)(x)
        x = nn.BatchNorm(
            axis=3,
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon, 
            use_running_average=not train,
            name=f"{self.name}_batch_norm_1"
        )(x)
        x = nn.relu(x)
        x = Conv2DFixedPadding(
            self.filters, kernel_size=3, strides=self.strides, name=self.name + "_conv_2", dtype=self.dtype
        )(x)
        x = nn.BatchNorm(
            axis=3,
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon,
            use_running_average=not train,
            name=f"{self.name}_batch_norm_2"
        )(x)
        x = nn.relu(x)
        x = Conv2DFixedPadding(
            self.filters * 4, kernel_size=1, strides=1, name=self.name + "_conv_3", dtype=self.dtype
        )(x)
        x = nn.BatchNorm(
            axis=3,
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon,
            use_running_average=not train,
            name=f"{self.name}_batch_norm_3"
        )(x)
        if 0 < self.se_ratio < 1:
            x = SE(self.filters, se_ratio=self.se_ratio, name=f"{self.name}_se", dtype=self.dtype)(x)
        if self.survival_probability:
            x = nn.Dropout(self.survival_probability, deterministic=not train, name=f"{self.name}_drop")(x)
        x = x + shortcut
        x = nn.relu(x)
        return x


class SE(nn.Module):
    in_filters: int
    se_ratio: float = 0.25
    expand_ratio: int = 1
    name: str = "se"
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        inputs = x
        x = jnp.mean(x, axis=(1,2), keepdims=True)  # global average pooling
        num_reduced_filters = max(1, int(self.in_filters * 4 * self.se_ratio))
        x = nn.Conv(
            num_reduced_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            use_bias=True,
            kernel_init=jax.nn.initializers.variance_scaling(2.0, "fan_out", "truncated_normal"),
            name=f"{self.name}_se_reduce",
            dtype=self.dtype
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            4 * self.in_filters * self.expand_ratio,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            use_bias=True,
            kernel_init=jax.nn.initializers.variance_scaling(2.0, "fan_out", "truncated_normal"),
            name=f"{self.name}_se_expand",
            dtype=self.dtype
        )(x)
        x = nn.sigmoid(x)
        return inputs * x


class ResNetRS(nn.Module):
    block_args: dict
    num_classes: int = 1000
    drop_connect_rate: float = 0.2
    dropout_rate: float = 0.25
    bn_momentum: float = 0.99
    bn_epsilon: float = 1e-5
    se_ratio: float = 0.25
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train=True):
        x = STEM(
            bn_momentum=self.bn_momentum, bn_epsilon=self.bn_epsilon, name="STEM_1", dtype=self.dtype
        )(x, train)
        for i, block_arg in enumerate(self.block_args):
            survival_probability = self.drop_connect_rate * float(i + 2) / (len(self.block_args) + 1)
            x = ResNetRSBlockGroup(
                block_arg["input_filters"],
                strides=(1 if i == 0 else 2),
                num_repeats=block_arg["num_repeats"],
                counter=i,
                se_ratio=self.se_ratio,
                bn_momentum=self.bn_momentum,
                bn_epsilon=self.bn_epsilon,
                survival_probability=survival_probability,
                name=f"ResNetRSBlockGroup{i + 2}",
                dtype=self.dtype
            )(x, train) 
        features = jnp.mean(x, axis=(1,2))  # global average pooling
        x = nn.Dropout(self.dropout_rate, deterministic=not train, name="top_dropout")(features)
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        x = nn.softmax(x)
        return features, x

ResNetRS50 = partial(ResNetRS, block_args=BLOCK_ARGS[50], drop_connect_rate=0.0, dropout_rate=0.25)
ResNetRS101 = partial(ResNetRS, block_args=BLOCK_ARGS[101], drop_connect_rate=0.0, dropout_rate=0.25)
ResNetRS152 = partial(ResNetRS, block_args=BLOCK_ARGS[152], drop_connect_rate=0.0, dropout_rate=0.25)
ResNetRS200 = partial(ResNetRS, block_args=BLOCK_ARGS[200], drop_connect_rate=0.1, dropout_rate=0.25)
ResNetRS270 = partial(ResNetRS, block_args=BLOCK_ARGS[270], drop_connect_rate=0.1, dropout_rate=0.25)
ResNetRS350 = partial(ResNetRS, block_args=BLOCK_ARGS[350], drop_connect_rate=0.1, dropout_rate=0.4)
ResNetRS420 = partial(ResNetRS, block_args=BLOCK_ARGS[420], drop_connect_rate=0.1, dropout_rate=0.4)

def posemb_sincos_2d(h, w, width, temperature=10_000., dtype=jnp.float32):
  """Follows the MoCo v3 logic."""
  y, x = jnp.mgrid[:h, :w]

  assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
  omega = jnp.arange(width // 4) / (width // 4 - 1)
  omega = 1. / (temperature**omega)
  y = jnp.einsum("m,d->md", y.flatten(), omega)
  x = jnp.einsum("m,d->md", x.flatten(), omega)
  pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
  return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
  if typ == "learn":
    return self.param(name, nn.initializers.normal(stddev=1/np.sqrt(width)),
                      (1, np.prod(seqshape), width), dtype)
  elif typ == "sincos2d":
    return posemb_sincos_2d(*seqshape, width, dtype=dtype)
  else:
    raise ValueError(f"Unknown posemb type: {typ}")


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  dropout: float = 0.0

  @nn.compact
  def __call__(self, x, deterministic=True):
    """Applies Transformer MlpBlock module."""
    inits = dict(
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
    )

    n, l, d = x.shape  # pylint: disable=unused-variable
    x = nn.Dense(self.mlp_dim or 4 * d, **inits)(x)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout)(x, deterministic)
    x = nn.Dense(d, **inits)(x)
    return x


class Encoder1DBlock(nn.Module):
  """Single transformer encoder block (MHSA + MLP)."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0

  @nn.compact
  def __call__(self, x, deterministic=True):
    out = {}
    y = nn.LayerNorm()(x)
    y = out["sa"] = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform(),
        deterministic=deterministic,
    )(y, y)
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = out["+sa"] = x + y

    y = nn.LayerNorm()(x)
    y = out["mlp"] = MlpBlock(
        mlp_dim=self.mlp_dim, dropout=self.dropout,
    )(y, deterministic)
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = out["+mlp"] = x + y
    return x, out


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""
  depth: int
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0

  @nn.compact
  def __call__(self, x, deterministic=True):
    out = {}

    # Input Encoder
    for lyr in range(self.depth):
      block = Encoder1DBlock(
          name=f"encoderblock_{lyr}",
          mlp_dim=self.mlp_dim, num_heads=self.num_heads, dropout=self.dropout)
      x, out[f"block{lyr:02d}"] = block(x, deterministic)
    out["pre_ln"] = x  # Alias for last block, but without the number in it.

    return nn.LayerNorm(name="encoder_norm")(x), out


class MAPHead(nn.Module):
  """Multihead Attention Pooling."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12

  @nn.compact
  def __call__(self, x):
    # TODO
    n, l, d = x.shape  # pylint: disable=unused-variable
    probe = self.param("probe", nn.initializers.xavier_uniform(),
                       (1, 1, d), x.dtype)
    probe = jnp.tile(probe, [n, 1, 1])

    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform())(probe, x)

    # TODO: dropout on head?
    y = nn.LayerNorm()(x)
    x = x + MlpBlock(mlp_dim=self.mlp_dim)(y)
    return x[:, 0]


class VITModel(nn.Module):
  """ViT model."""

  num_classes: Optional[int] = None
  patch_size: Sequence[int] = (16, 16)
  width: int = 768
  depth: int = 12
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  posemb: str = "learn"  # Can also be "sincos2d"
  rep_size: Union[int, bool] = False
  dropout: float = 0.0
  pool_type: str = "gap"  # Can also be "map" or "tok"
  head_zeroinit: bool = True
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, image, *, train=False):
    out = {}

    # Patch extraction
    x = out["stem"] = nn.Conv(
        self.width, self.patch_size, strides=self.patch_size,
        padding="VALID", name="embedding")(image)

    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # Add posemb before adding extra token.
    x = out["with_posemb"] = x + get_posemb(
        self, self.posemb, (h, w), c, "pos_embedding", x.dtype)

    if self.pool_type == "tok":
      cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
      x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

    n, l, c = x.shape  # pylint: disable=unused-variable
    x = nn.Dropout(rate=self.dropout)(x, not train)

    x, out["encoder"] = Encoder(
        depth=self.depth,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout=self.dropout,
        name="Transformer")(
            x, deterministic=not train)
    encoded = out["encoded"] = x

    if self.pool_type == "map":
      x = out["head_input"] = MAPHead(
          num_heads=self.num_heads, mlp_dim=self.mlp_dim)(x)
    elif self.pool_type == "gap":
      x = out["head_input"] = jnp.mean(x, axis=1)
    elif self.pool_type == "0":
      x = out["head_input"] = x[:, 0]
    elif self.pool_type == "tok":
      x = out["head_input"] = x[:, 0]
      encoded = encoded[:, 1:]
    else:
      raise ValueError(f"Unknown pool type: '{self.pool_type}'")

    x_2d = jnp.reshape(encoded, [n, h, w, -1])

    if self.rep_size:
      rep_size = self.width if self.rep_size is True else self.rep_size
      hid = nn.Dense(rep_size, name="pre_logits")
      # NOTE: In the past we did not include tanh in pre_logits.
      # For few-shot, it should not matter much, as it whitens anyways.
      x_2d = nn.tanh(hid(x_2d))
      x = nn.tanh(hid(x))

    out["pre_logits_2d"] = x_2d
    out["pre_logits"] = x

    if self.num_classes:
      kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
      head = nn.Dense(self.num_classes, name="fc", **kw)
      x_2d = out["logits_2d"] = head(x_2d)
      x = out["logits"] = head(x)

    return out["pre_logits"], x

VIT_T_16 = partial(VITModel, width=192, depth=12, mlp_dim=768, num_heads=3, patch_size=(16, 16))
VIT_S_16 = partial(VITModel, width=384, depth=12, mlp_dim=1536, num_heads=6, patch_size=(16, 16))
VIT_M_16 = partial(VITModel, width=512, depth=12, mlp_dim=2048, num_heads=8, patch_size=(16, 16))
VIT_B_16 = partial(VITModel, width=768, depth=12, mlp_dim=3072, num_heads=12, patch_size=(16, 16))
VIT_L_16 = partial(VITModel, width=1024, depth=24, mlp_dim=4096, num_heads=16, patch_size=(16, 16))
VIT_H_16 = partial(VITModel, width=1280, depth=32, mlp_dim=5120, num_heads=16, patch_size=(16, 16))
VIT_g_16 = partial(VITModel, width=1408, depth=40, mlp_dim=6144, num_heads=16, patch_size=(16, 16))
VIT_G_16 = partial(VITModel, width=1664, depth=48, mlp_dim=8192, num_heads=16, patch_size=(16, 16))
VIT_e_16 = partial(VITModel, width=1792, depth=56, mlp_dim=15360, num_heads=16, patch_size=(16, 16))

VIT_T_32 = partial(VITModel, width=192, depth=12, mlp_dim=768, num_heads=3, patch_size=(32,32))
VIT_S_32 = partial(VITModel, width=384, depth=12, mlp_dim=1536, num_heads=6, patch_size=(32,32))
VIT_M_32 = partial(VITModel, width=512, depth=12, mlp_dim=2048, num_heads=8, patch_size=(32,32))
VIT_B_32 = partial(VITModel, width=768, depth=12, mlp_dim=3072, num_heads=12, patch_size=(32,32))
VIT_L_32 = partial(VITModel, width=1024, depth=24, mlp_dim=4096, num_heads=16, patch_size=(32,32))
VIT_H_32 = partial(VITModel, width=1280, depth=32, mlp_dim=5120, num_heads=16, patch_size=(32,32))
VIT_g_32 = partial(VITModel, width=1408, depth=40, mlp_dim=6144, num_heads=16, patch_size=(32,32))
VIT_G_32 = partial(VITModel, width=1664, depth=48, mlp_dim=8192, num_heads=16, patch_size=(32,32))
VIT_e_32 = partial(VITModel, width=1792, depth=56, mlp_dim=15360, num_heads=16, patch_size=(32,32))
