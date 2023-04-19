import math
from typing import List, Tuple

import torch
import torch.nn as nn

from . import model_utils
from .model_opts import ModelOpts


class View(nn.Module):
    def __init__(self, dims: Tuple[int, ...]) -> None:
        super(View, self).__init__()
        self.dims = dims

    def forward(self, input) -> torch.Tensor:
        x: torch.Tensor = input
        y = x.view(self.dims)
        return y


class SkeletonEncoder(nn.Module):
    def __init__(
        self,
        output_feature_map_shape: List[int],
    ) -> None:
        super(SkeletonEncoder, self).__init__()
        # We have 22 joints. For each joint, we use joint positions and joint axes as input features.
        n_skel_features = 22 * 6
        self._layers = nn.Sequential(
            nn.Linear(n_skel_features, int(math.prod(output_feature_map_shape))),
            View((-1, *output_feature_map_shape)),
            nn.BatchNorm2d(output_feature_map_shape[0]),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, joint_rotation_axes: torch.Tensor, joint_rest_positions: torch.Tensor
    ) -> torch.Tensor:
        numel = int(torch.prod(torch.tensor(joint_rotation_axes.shape[:-2])))
        skeleton_features = torch.cat(
            (joint_rotation_axes, joint_rest_positions), dim=-1
        ).reshape(numel, -1)

        skel_maps = self._layers(skeleton_features)

        return skel_maps
