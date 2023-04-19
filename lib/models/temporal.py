# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn as nn

from . import model_utils
from .model_opts import ModelOpts


class SimpleConvRNN(nn.Module):
    def __init__(
        self,
        nTemporalBlocks: int,
        nTemporalMemoryChannels: int,
        nImageFeatureChannels: int,
        temporalFTLRatio: float,
        featureMapShape: Tuple[int, int],
    ) -> None:
        super(SimpleConvRNN, self).__init__()
        self._nc_memory = nTemporalMemoryChannels
        n_temporal_channels = nImageFeatureChannels + self._nc_memory

        temporal_module = nn.ModuleList()

        for i in range(nTemporalBlocks):
            nc = n_temporal_channels
            temporal_module.append(nn.Conv2d(nc, nc, kernel_size=1, padding=0))
            # Don't add ReLU in the last block since it makes all features positives
            if i != nTemporalBlocks - 1:
                temporal_module.append(nn.ReLU(inplace=True))

        self._temporal_module = nn.Sequential(*temporal_module)
        self._temporal_ftl_ratio = float(temporalFTLRatio)
        self._input_shape = (n_temporal_channels, *featureMapShape)
        self._mem_features = torch.empty(0)
        self._prev_extrinsics = torch.empty(0)

    def input_shape(self) -> Tuple[int, int, int]:
        """
        Return: input shape to self._temporal_module
            [channels, feature_size[0], feature_size[1]]
        """
        return self._input_shape

    def transform_memory_features(
        self,
        prev_extrinsics: torch.Tensor,
        prev_mem_features: torch.Tensor,
        cur_extrinsics: torch.Tensor,
        memory_idx: torch.Tensor,
        use_memory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        use_mem_idx = memory_idx[use_memory]
        if len(use_mem_idx) != len(use_memory):
            zero_mem_idx = memory_idx[torch.logical_not(use_memory)]
            prev_mem_features[zero_mem_idx] = 0
            prev_extrinsics[zero_mem_idx] = 0

        if len(use_mem_idx) != 0:
            prev_cam0_to_world_xf = torch.inverse(prev_extrinsics[use_mem_idx])
            prev_cam0_to_cur_cam0_xf = cur_extrinsics[use_memory].bmm(
                prev_cam0_to_world_xf
            )
            prev_mem_features[use_mem_idx] = model_utils.apply_ftl_to_feature_maps(
                prev_cam0_to_cur_cam0_xf,
                prev_mem_features[use_mem_idx],
                self._temporal_ftl_ratio,
            )

        # Update prev_extrinsics with cur_extrinsics
        prev_extrinsics[memory_idx] = cur_extrinsics
        return prev_extrinsics, prev_mem_features

    def forward_one_step(
        self, prev_mem_features_xfed: torch.Tensor, cur_img_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        temporal_input = [prev_mem_features_xfed, cur_img_features]

        temporal_out = torch.cat(temporal_input, dim=1)
        temporal_out = self._temporal_module(temporal_out)

        mem_features_out = temporal_out[:, 0 : self._nc_memory]
        fused_features = temporal_out[:, self._nc_memory :]

        return mem_features_out, fused_features

    def forward_temporal_features(
        self,
        img_features: torch.Tensor,
        cur_extrinsics: torch.Tensor,
        memory_idx: torch.Tensor,
        use_memory: torch.Tensor,
        update_memory: bool = True,
    ) -> torch.Tensor:
        feat_shape = (img_features.shape[-2], img_features.shape[-1])
        required_memory_len = int(torch.max(memory_idx)) + 1
        mem_features = self._mem_features
        prev_extrinsics = self._prev_extrinsics
        if len(self._mem_features) < required_memory_len:
            mem_features = torch.zeros(
                required_memory_len,
                self._nc_memory,
                feat_shape[0],
                feat_shape[1],
                dtype=img_features.dtype,
                device=img_features.device,
            )
            prev_extrinsics = torch.zeros(
                required_memory_len,
                4,
                4,
                dtype=img_features.dtype,
                device=img_features.device,
            )
            if len(self._mem_features) != 0:
                mem_features[0 : self._mem_features.shape[0]] = self._mem_features
                prev_extrinsics[
                    0 : self._prev_extrinsics.shape[0]
                ] = self._prev_extrinsics

        prev_extrinsics, mem_features = self.transform_memory_features(
            prev_extrinsics, mem_features, cur_extrinsics, memory_idx, use_memory
        )
        mem_features_out, fused_features = self.forward_one_step(
            mem_features[memory_idx], img_features
        )
        # Update memory features
        mem_features[memory_idx] = mem_features_out

        self._prev_extrinsics = prev_extrinsics
        self._mem_features = mem_features

        return fused_features


def create_temporal_model(
    model_opts: ModelOpts, feature_map_shape: Tuple[int, int]
) -> SimpleConvRNN:
    return SimpleConvRNN(
        nTemporalBlocks=model_opts.nTemporalBlocks,
        nTemporalMemoryChannels=model_opts.nTemporalMemoryChannels,
        nImageFeatureChannels=model_opts.nImageFeatureChannels,
        temporalFTLRatio=model_opts.temporalFTLRatio,
        featureMapShape=feature_map_shape,
    )
