# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import backbone_resnet as resnet


def procrustes_align(
    from_points: torch.Tensor,
    to_points: torch.Tensor,
) -> torch.Tensor:
    """
    Inputs have same shape `(batch_size, n_points, 3)`.
    Within each sample of the batch, `from_points` and `to_points`
    implicitly correspond to each other along dim=1.

    Returns:
    - `rot`, `translation` with shape `(batch_size, 3, 3)`
    representing transformations for each example in batch
    """
    device = from_points.device

    batch_size = from_points.shape[0]
    from_mean = from_points.mean(dim=1)
    to_mean = to_points.mean(dim=1)

    from_centered = from_points - from_mean.reshape(-1, 1, 3)
    to_centered = to_points - to_mean.reshape(-1, 1, 3)

    outer_prod = torch.matmul(torch.transpose(from_centered, 1, 2), to_centered)

    u, _, v = outer_prod.svd()
    v_m_ut = torch.matmul(v, torch.transpose(u, 1, 2))
    w = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    det = torch.det(v_m_ut)
    w[:, 2, 2] = det

    xfs = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

    xfs[:, 0:3, 0:3] = torch.matmul(torch.matmul(v, w), torch.transpose(u, 1, 2))
    xfs[:, 0:3, 3] = (
        to_mean - torch.matmul(xfs[:, 0:3, 0:3], from_mean.unsqueeze(-1)).squeeze()
    )

    return xfs


def apply_ftl_to_feature_maps(
    xfs: torch.Tensor,
    feature_maps: torch.Tensor,
    ftl_ratio: float,
) -> torch.Tensor:
    """
    This function transforms features to 3D points by reshaping along channel size.
    It treats the first/middle/last third of feature channels as X, Y, Z coordinates,
    by reshaping the features with
    `point_features_xfed = ftl_feature_maps.reshape(n_images, 3, -1)`

    This reshape operator makes the FTL operator NOT interchangeable with Feature
    split operator.

    Arg:
    `xfs`: n_images X 4 X 4, the affine transfomration matrix to transform the feature
    `feature_maps`: n_images X n_channels X H X W, the input feature maps
    `ftl_ratio`: ratio of the n_channel for FTL

    Return: n_images X n_channels X H X W, which is the transformed feature
    """
    assert ftl_ratio >= 0 and ftl_ratio <= 1

    if ftl_ratio == 0:
        return feature_maps

    n_images = feature_maps.shape[0]
    n_channels = int(feature_maps.shape[1])

    # number of ftl channels
    nc_ftl = int(round(n_channels * ftl_ratio))
    assert nc_ftl % 3 == 0
    ftl_feature_maps = feature_maps[:, 0:nc_ftl]

    # Apply feature transformations to the point features
    point_features_xfed = ftl_feature_maps.reshape(n_images, 3, -1)

    r_in = xfs[:, 0:3, 0:3]
    t_in = xfs[:, 0:3, 3]
    point_features_xfed = torch.matmul(r_in, point_features_xfed) + t_in.unsqueeze(-1)

    # Reshape back to feature maps
    ftl_feature_maps_xfed = point_features_xfed.reshape(ftl_feature_maps.shape)
    if nc_ftl != n_channels:
        cat_maps = torch.cat((ftl_feature_maps_xfed, feature_maps[:, nc_ftl:]), dim=1)
        return cat_maps
    else:
        return ftl_feature_maps_xfed


def create_backbone(
    arch_name: str,
    input_size: Tuple[int, int],
    n_out_channels: int,
) -> Tuple[nn.Module, List[int]]:
    assert arch_name.startswith("resnet")
    # The "arch" string will be directly mapped to the function name
    # in resnet.create_model so that we don't have to hard code every
    # supported architecture
    arch, start_planes_str = re.split("-f", arch_name)
    start_planes = int(start_planes_str)

    start_layers = nn.Sequential(
        nn.Conv2d(1, start_planes, kernel_size=3, padding=1),
        nn.BatchNorm2d(start_planes),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )
    resnet_input_size = (int(input_size[0] / 2), int(input_size[1] / 2))

    resnet_model = resnet.create_model(
        arch, start_layers, resnet_input_size, start_planes
    )
    n_img_maps = resnet_model.outshape[0]
    # Add an extra conv to the image feature layers since
    # the output of the base network usually has a ReLU on top
    # which doesn't have any negative number
    proj_layer = nn.Conv2d(n_img_maps, n_out_channels, kernel_size=1, padding=0)
    backbone = nn.Sequential(resnet_model, proj_layer)
    output_shape = [n_out_channels, *resnet_model.outshape[1:]]

    return backbone, output_shape


def create_multi_view_fusion_layers(
    nc_in: int,
    nc_out: int,
    n_blocks: int,
) -> nn.Module:
    """
    Linearly increase/reduce channels per block
    """
    n_channels_list = np.linspace(nc_in, nc_out, n_blocks + 1)
    fusion_layers = nn.ModuleList()

    for i in range(n_blocks):
        nc_in_cur = int(n_channels_list[i])
        nc_out_cur = int(n_channels_list[i + 1])

        fusion_layers.append(nn.Conv2d(nc_in_cur, nc_out_cur, kernel_size=1, padding=0))
        fusion_layers.append(nn.BatchNorm2d(nc_out_cur))
        fusion_layers.append(nn.ReLU(inplace=True))

    # Add an extra convolution so that features are not all positives due to ReLU
    fusion_layers.append(nn.Conv2d(nc_out, nc_out, kernel_size=1, padding=0))

    return nn.Sequential(*fusion_layers)


def compute_singlev_xfs(
    intrinsics: torch.Tensor, canonical_focal_length: float = 200
) -> torch.Tensor:
    """
    The image backbone is expected to generate 3D point features. Since predicting 3D
    points without intrinsics is infeasible, the code here factorizes the
    intrinsics into 2 parts: K = K_canonical * S. The 3D point features from the
    image itself should correspond to K_canonical so that we need to apply S to the
    points to go back to the original intrinsics.

    There are 2 ways to construct this S matrix, one is scaling the x, y components
    and the other is scaling the z components. The code here does it for the z components
    as I thought it puts ensures variations of the 3D point features.
    """
    assert len(intrinsics.shape) == 3
    n_elem = intrinsics.shape[0]
    singlev_scaled_to_orig_xf = (
        torch.eye(4, dtype=intrinsics.dtype, device=intrinsics.device)
        .unsqueeze(0)
        .repeat(n_elem, 1, 1)
    )
    singlev_scaled_to_orig_xf = singlev_scaled_to_orig_xf.reshape(n_elem, 4, 4)
    focal = intrinsics[..., 0, 0]

    singlev_scaled_to_orig_xf[..., 2, 2] = focal / canonical_focal_length

    return singlev_scaled_to_orig_xf


def create_pose_regression_layers(
    n_in_channels: int,
    n_blocks: int,
    n_out_channels: int,
) -> nn.Module:
    pose_regression_layers = []
    for _ in range(n_blocks):
        pose_regression_layers.append(resnet.BasicBlock(n_in_channels, n_in_channels))
    pose_regression_layers.append(
        nn.Conv2d(n_in_channels, n_out_channels, kernel_size=1)
    )
    pose_regression_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    pose_regression_layers = nn.Sequential(*pose_regression_layers)
    return pose_regression_layers
