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

from flax import linen as nn
import jax.numpy as jnp
from jax.nn import initializers

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
        logits = nn.Dense(self.num_classes, kernel_init=nn.initializers.glorot_normal())(features)
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

WRN28_2 = partial(WideResNet, depth=28, width=2)
WRN28_8 = partial(WideResNet, depth=28, width=8)
WRN28_10 = partial(WideResNet, depth=28, width=10)
WRN37_2 = partial(WideResNet, depth=37, width=2)
WRN37_10 = partial(WideResNet, depth=37, width=10)
WRN40_2 = partial(WideResNet, depth=40, width=2)
WRN40_10 = partial(WideResNet, depth=40, width=10)

"""
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
"""

# Used for testing only.
_ResNet1 = partial(ResNet, stage_sizes=[1], block_cls=ResNetBlock)
_ResNet1Local = partial(ResNet, stage_sizes=[1], block_cls=ResNetBlock,
                        conv=nn.ConvLocal)