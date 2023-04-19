# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Tuple

import torch
from lib.data_utils import fs

from . import (
    feature_extractor as fe,
    regressor as reg,
    skeleton_encoder as se,
    temporal as tem,
)
from .model_opts import ModelOpts
from .umetrack_model import UmeTrackModel


def _get_n_input_channels(model_opts: ModelOpts, use_skel: bool) -> int:
    n = model_opts.nImageFeatureChannels
    if use_skel:
        n = n + model_opts.nSkeletonFeatureChannels

    return n


def _create_regressor(
    model_opts: ModelOpts,
    feature_sizes: Tuple[int, int],
    use_skel: bool,
    predict_skel_scale: bool,
):
    if use_skel:
        assert model_opts.nSkeletonFeatureChannels != 0

    n_in = _get_n_input_channels(model_opts, use_skel=use_skel)
    reg_out_indices, n_out = reg.get_output_index_ranges(
        model_opts, predict_skel_scale=predict_skel_scale
    )
    return reg.PoseRegressor(
        n_channels_in=n_in,
        n_output_dims=n_out,
        output_index_ranges=reg_out_indices,
        n_blocks=model_opts.nPoseRegressionBlocks,
        n_wrist_rigid_pts=model_opts.nWristRigidPts,
        feature_map_sizes=feature_sizes,
    )


def load_pretrained_model(model_path: str):
    model_opts = ModelOpts()

    feature_extractor = fe.FeatureExtractor((96, 96), ModelOpts())
    temporal = tem.create_temporal_model(
        model_opts,
        feature_extractor.output_feature_sizes,
    )
    skeleton_encoder = se.SkeletonEncoder(
        [model_opts.nSkeletonFeatureChannels, *feature_extractor.output_feature_sizes],
    )
    regressor_k = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_u = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=False,
        predict_skel_scale=True,
    )

    umetrack_model = UmeTrackModel(
        feature_extractor=feature_extractor,
        temporal=temporal,
        skeleton_encoder=skeleton_encoder,
        regressor_k=regressor_k,
        regressor_u=regressor_u,
    )
    with fs.open(model_path, "rb") as fp:
        model_state_dict = torch.load(fp)

    umetrack_model.load_state_dict(model_state_dict)
    return umetrack_model
