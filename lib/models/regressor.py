from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn

from . import model_utils
from .model_opts import ModelOpts


def _gen_rigid_features():
    rigid_samples = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            # xy plane
            [-1, -1, 0],
            # xz plane
            [-1, 0, -1],
            # yz plane
            [0, -1, -1],
        ]
    )

    rigid_samples_rescaled = np.empty(rigid_samples.shape)
    expected_norm = 0.1

    for i in range(len(rigid_samples)):
        norm = np.linalg.norm(rigid_samples[i])
        if norm == 0:
            rigid_samples_rescaled[i] = rigid_samples[i]
        else:
            rigid_samples_rescaled[i] = rigid_samples[i] / norm * expected_norm

    rigid_samples_rescaled = torch.from_numpy(rigid_samples_rescaled).float()

    return rigid_samples_rescaled


def get_output_index_ranges(
    mo: ModelOpts,
    predict_skel_scale: bool,
) -> Tuple[Dict[str, Tuple[int, int]], int]:
    rigid_samples = _gen_rigid_features()

    assert mo.nWristRigidPts <= len(rigid_samples), (
        "Max supported n_wrist_rigid_pts is "
        f"{len(rigid_samples)}, got {mo.nWristRigidPts}"
    )

    output_dims = {
        "joint_angles": 20,
        "wrist_xfs": mo.nWristRigidPts * 3,
        "skel_scales": 1 if predict_skel_scale else 0,
        "landmark_uncertainty_sigmas": 21,
    }
    n_output_dims = 0
    output_index_range = {}
    for k, v in output_dims.items():
        if v != 0:
            output_index_range[k] = (n_output_dims, n_output_dims + v)
            n_output_dims = n_output_dims + v
    return output_index_range, n_output_dims


def decode_joint_angles(finger_angles: torch.Tensor):
    wrist_angles = torch.zeros(
        finger_angles.shape[0],
        2,
        device=finger_angles.device,
        dtype=finger_angles.dtype,
    )
    joint_angles = torch.cat([finger_angles, wrist_angles], dim=1)

    return joint_angles


def decode_wrist_xfs_svd(
    pred_pts_features: torch.Tensor,
    rigid_pts_src: torch.Tensor,
) -> torch.Tensor:
    batch_size = pred_pts_features.shape[0]
    rigid_points = pred_pts_features.reshape(pred_pts_features.shape[0], -1, 3)

    from_points = rigid_pts_src.to(rigid_points.device)
    from_points = (
        from_points.unsqueeze(0)
        .expand(batch_size, from_points.shape[0], from_points.shape[1])
        .clone()
    )

    wrist_xfs = model_utils.procrustes_align(from_points, rigid_points)

    return wrist_xfs


def decode_skel_scales(
    raw_features: torch.Tensor,
) -> torch.Tensor:
    log_scales = raw_features.reshape(-1)
    # In general, The calibrated skeleton scale values are 0.8~1.2.
    # The log scale values predicted by the network will be -0.22~0.18
    skel_scales = torch.exp(log_scales)
    return skel_scales


def decode_landmark_unc_sigmas(
    raw_features: torch.Tensor,
) -> torch.Tensor:
    unc_sigmas = torch.clamp(nn.functional.softplus(raw_features), min=1e-5)
    return unc_sigmas


@dataclass
class RegressorOutput:
    joint_angles: torch.Tensor
    wrist_xfs: torch.Tensor
    skel_scales: Optional[torch.Tensor] = None
    landmark_uncertainty_sigmas: Optional[torch.Tensor] = None


class PoseRegressor(nn.Module):
    def __init__(
        self,
        n_channels_in: int,
        n_output_dims: int,
        output_index_ranges: Dict[str, Tuple[int, int]],
        n_blocks: int,
        n_wrist_rigid_pts: int,
        feature_map_sizes: Tuple[int, int],
    ):
        super().__init__()

        rigid_samples = _gen_rigid_features()
        assert n_wrist_rigid_pts <= len(rigid_samples)
        self._left_wrist_sample_points = rigid_samples[:n_wrist_rigid_pts]

        self._pose_regression_layers = model_utils.create_pose_regression_layers(
            n_in_channels=n_channels_in,
            n_blocks=n_blocks,
            n_out_channels=n_output_dims,
        )
        self._output_index_ranges = output_index_ranges
        self._input_shape = (n_channels_in, *feature_map_sizes)

    def input_shape(self) -> Tuple[int, int, int]:
        """
        Return: input shape to self._pose_regression_layers
            [channels, feature_size[0], feature_size[1]]
        """
        return self._input_shape

    def regress_poses(
        self, img_skel_features: torch.Tensor, left_hand: bool = True
    ) -> Dict[str, torch.Tensor]:
        pose_features = self._pose_regression_layers(img_skel_features)
        pose_features = torch.flatten(pose_features, 1)

        output_dict = {}
        for key, f_range in self._output_index_ranges.items():
            raw_features = pose_features[:, f_range[0] : f_range[1]]
            if key == "joint_angles":
                output_dict[key] = decode_joint_angles(raw_features)
            elif key == "wrist_xfs":
                output_dict[key] = decode_wrist_xfs_svd(
                    raw_features,
                    self._left_wrist_sample_points,
                )
            elif key == "skel_scales":
                output_dict[key] = decode_skel_scales(raw_features)
            elif key == "landmark_uncertainty_sigmas":
                output_dict[key] = decode_landmark_unc_sigmas(raw_features)
            else:
                raise ValueError(f"Unknown output key: {key}")

        return RegressorOutput(**output_dict)
