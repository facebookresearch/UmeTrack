from dataclasses import dataclass

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from . import model_utils
from .feature_extractor import FeatureExtractor
from .regressor import PoseRegressor
from .skeleton_encoder import SkeletonEncoder
from .temporal import SimpleConvRNN


@dataclass
class InputFrameData:
    """
    Each entry corresponds to data from a camera. The data here doesn't contain
    which images are observing the same hand. InputFrameDesc is used to assemble
    features observing the same hands.

    * left_images (shape [n_images, h, w])
    * intrinsics (shape [n_images, 3, 3])
    * extrinsics_xf (shape [n_images, 4, 4])
    """

    left_images: torch.Tensor
    intrinsics: torch.Tensor
    extrinsics_xf: torch.Tensor


# per-frame data descriptions, could potentially
# create another struct InputFrameDescription
@dataclass
class InputFrameDesc:
    """
    Descriptions for InputFrameData. Each tensor should
    bs: batch_size

    * sample_range (shape [bs, 2]): the 2 columns are the starting and
        ending indices. Example: a tensor [[0, 2], [2, 3]] means first sample
        corresponds to left_images[0:2] which is a multi-view sample and second
        sample corresponds to lefts_images[2:3] which is a single-view sample
    * memory_idx (shape [bs]): only applicable with a valid _temporal field. In
        rum-time if we have tracking for 2 hands, this tensor could be [0, 1].
        If the next frame left hand loses track, this memory_idx could become [1] tensor
    * use_memory (shape [bs]): a boolean tensor indicating whether to use the memory
        features for this sample
    * hand_idx (shape [bs]): hand index for each sample. There is a chance to factor this out.
    """

    sample_range: torch.Tensor
    memory_idx: torch.Tensor
    use_memory: torch.Tensor
    hand_idx: torch.Tensor


@dataclass
class InputSkeletonData:
    """
    Descriptions for InputFrameData

    * joint_rotation_axes (shape [bs, 22, 3]): 22 joint axes
    * joint_rest_positions (shape [bs, 22, 3]): 22 joint positions in rest pose
    """

    joint_rotation_axes: torch.Tensor
    joint_rest_positions: torch.Tensor


def _recover_wrist_xfs_in_world(
    hand_idx: torch.Tensor,
    cam0_extrinsics: torch.Tensor,
    left_wrist_xfs_in_cam0: torch.Tensor,
) -> torch.Tensor:
    left_wrist_xf_world = torch.inverse(cam0_extrinsics) @ left_wrist_xfs_in_cam0

    # The model only makes predictions for the left hands. In order to recover
    # the right hand transform, the x component needs to be mirrored in the final
    # transformation matrix.
    right_hand_masks = hand_idx == 1
    wrist_xf_world = left_wrist_xf_world.clone()
    wrist_xf_world[right_hand_masks, :, 0] *= -1
    return wrist_xf_world


def _get_cam0_extrinsics(
    frame_data: InputFrameData, frame_desc: InputFrameDesc
) -> torch.Tensor:
    # Extracting reference cam extrinsics
    return frame_data.extrinsics_xf[frame_desc.sample_range[:, 0]]


class UmeTrackModel(nn.Module):
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        temporal: SimpleConvRNN,
        skeleton_encoder: SkeletonEncoder,
        regressor_k: PoseRegressor,
        regressor_u: PoseRegressor,
    ):
        super().__init__()

        self._feature_extractor: FeatureExtractor = feature_extractor
        self._temporal = temporal
        self._skeleton_enc = skeleton_encoder
        self._regressor_k = regressor_k
        self._regressor_u = regressor_u

        self.eval()

    @torch.jit.export
    def getInputImageSizes(self) -> Tuple[int, int]:
        return self._feature_extractor.input_image_sizes

    def _forward_feature_extractor(
        self, frame_data: InputFrameData, sample_range: torch.Tensor
    ) -> torch.Tensor:
        # Per-view img features
        per_view_img_features = self._feature_extractor._image_backbone(
            frame_data.left_images.unsqueeze(1)
        )
        singlev_scaled_to_orig_xf = model_utils.compute_singlev_xfs(
            frame_data.intrinsics
        )

        # The following assumes that the max # views per batch is 2
        num_views = 2
        all_multiv = sample_range.shape[0] * num_views == per_view_img_features.shape[0]
        extrinsics_xf = frame_data.extrinsics_xf
        if all_multiv:
            img_features = self._feature_extractor.compute_multiv_features(
                per_view_img_features.reshape(
                    (-1, num_views) + per_view_img_features.shape[1:]
                ),
                singlev_scaled_to_orig_xf.reshape(
                    (-1, num_views) + singlev_scaled_to_orig_xf.shape[1:]
                ),
                extrinsics_xf.reshape((-1, num_views) + extrinsics_xf.shape[1:]),
            )
        else:
            img_features_list: List[torch.Tensor] = []
            for r01 in sample_range:
                r0 = int(r01[0])
                r1 = int(r01[1])
                if r1 - r0 == 1:
                    singlev_features = self._feature_extractor.compute_singlev_features(
                        per_view_img_features[r0:r1], singlev_scaled_to_orig_xf[r0:r1]
                    )
                    img_features_list.append(singlev_features)
                else:
                    multiv_features = self._feature_extractor.compute_multiv_features(
                        per_view_img_features[r0:r1].unsqueeze(0),
                        singlev_scaled_to_orig_xf[r0:r1].unsqueeze(0),
                        extrinsics_xf[r0:r1].unsqueeze(0),
                    )
                    img_features_list.append(multiv_features)

            img_features = torch.cat(img_features_list, dim=0)

        return img_features

    def _forward_feature_extractor_temporal(
        self, frame_data: InputFrameData, frame_desc: InputFrameDesc
    ) -> torch.Tensor:
        # Fused img features
        img_features = self._forward_feature_extractor(
            frame_data, frame_desc.sample_range
        )
        extrinsics = _get_cam0_extrinsics(frame_data, frame_desc)

        # Temporal features
        temporal_features = self._temporal.forward_temporal_features(
            img_features,
            extrinsics,
            frame_desc.memory_idx,
            frame_desc.use_memory,
        )
        return temporal_features

    def regress_pose_use_skeleton(
        self,
        frame_data: InputFrameData,
        frame_desc: InputFrameDesc,
        skel_data: InputSkeletonData,
    ) -> Dict[str, torch.Tensor]:
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc
        )

        skel_features = self._skeleton_enc.forward(
            joint_rotation_axes=skel_data.joint_rotation_axes,
            joint_rest_positions=skel_data.joint_rest_positions,
        )
        if skel_features.shape[0] == 1 and temporal_features.shape[0] > 1:
            # The caller only passed in a single profile and it should be used
            # for all the samples.
            skel_features = skel_features.expand(
                temporal_features.shape[0], *skel_features.shape[1:]
            )

        # Concatenate along the channel dimension (1)
        img_skel_features = torch.cat([temporal_features, skel_features], dim=1)

        regression_output = self._regressor_k.regress_poses(img_skel_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )
        return regression_output

    def regress_pose_pred_skel_scale(
        self, frame_data: InputFrameData, frame_desc: InputFrameDesc
    ) -> Dict[str, torch.Tensor]:
        singlev_masks = (
            frame_desc.sample_range[:, 1] - frame_desc.sample_range[:, 0]
        ) != 1
        assert (
            singlev_masks.all()
        ), "Unsupported: found single-view samples when calibration scale"
        temporal_features = self._forward_feature_extractor_temporal(
            frame_data, frame_desc
        )

        regression_output = self._regressor_u.regress_poses(temporal_features)

        regression_output.wrist_xfs = _recover_wrist_xfs_in_world(
            frame_desc.hand_idx,
            _get_cam0_extrinsics(frame_data, frame_desc),
            regression_output.wrist_xfs,
        )

        return regression_output
