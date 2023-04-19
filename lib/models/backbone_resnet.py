# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
from typing import List, Optional, Tuple

import torch.nn as nn


class BasicBlock(nn.Module):
    """
    Modified version of the original BasicBlock implementation of pytorch.
    Original version is at: torchvision/models/resnet.py
    """

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        *,
        groups: int = 1,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.bn1 = nn.BatchNorm2d(num_features=planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.bn2 = nn.BatchNorm2d(num_features=planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBase(nn.Module):
    def __init__(
        self,
        start_layers: nn.Module,
        input_size: Tuple[int, int],
        block: BasicBlock,
        layers: List[int],
        layers_in_planes: List[int],
        layers_out_planes: List[int],
        strides: Optional[List[int]] = None,
        groups: int = 1,
    ):
        super(ResNetBase, self).__init__()
        if strides is None:
            strides = [1, 2, 2, 2]
        assert (
            len(layers) == len(layers_in_planes)
            and len(layers_in_planes) == len(layers_out_planes)
            and len(layers_out_planes) == len(strides)
        )

        self._layers = nn.ModuleList()
        if start_layers:
            self._layers.append(start_layers)

        for il in range(len(layers)):
            self._layers.append(
                self._make_layer(
                    block,
                    layers_in_planes[il],
                    layers_out_planes[il],
                    layers[il],
                    stride=strides[il],
                    groups=groups,
                )
            )

        self.outshape = [
            layers_out_planes[-1] * block.expansion,
            int((input_size[0]) / 8),
            int((input_size[1]) / 8),
        ]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(
        self,
        block: BasicBlock,
        in_planes: int,
        planes: int,
        blocks: int,
        stride: int = 1,
        groups: int = 1,
    ):
        downsample = None
        if stride != 1 or in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    groups=groups,
                ),
                nn.BatchNorm2d(num_features=planes * block.expansion),
            )

        layers = []
        layers.append(block(in_planes, planes, stride, downsample, groups=groups))
        for _i in range(1, blocks):
            layers.append(block(planes * block.expansion, planes, groups=groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        for l in self._layers:
            x = l(x)
        return x


def compute_layers_in_planes(start_planes: int):
    return [start_planes, start_planes, start_planes * 2, start_planes * 4]


def compute_layers_out_planes(start_planes: int):
    return [start_planes, start_planes * 2, start_planes * 4, start_planes * 8]


def create_model(
    arch: str,
    start_layers: nn.Module,
    input_size: Tuple[int, int],
    start_planes: int,
    groups: int = 1,
    **kwargs,
):
    assert arch.startswith("resnet_layers_")
    _, start_planes_str = re.split("resnet_layers_", arch)
    assert len(start_planes_str) == 4
    """Constructs a ResNetBase model with specified layers by parsing the string."""
    layers = [int(i) for i in start_planes_str]
    model = ResNetBase(
        start_layers,
        input_size,
        BasicBlock,
        layers,
        layers_in_planes=compute_layers_in_planes(start_planes),
        layers_out_planes=compute_layers_out_planes(start_planes),
        groups=groups,
        **kwargs,
    )
    return model
