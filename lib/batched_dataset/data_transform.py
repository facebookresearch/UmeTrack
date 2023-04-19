#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import lib.common.camera as camera

import numpy as np
import torch
from lib.common import crop
from lib.common.hand import HandModel
from lib.common.hand import mirrored_hand_model
from lib.data_utils import bundles

from .sample import parse_raw_buffers, RawSample

scalar_type = np.float32


@dataclass
class PoseData:
    joint_angles: torch.Tensor
    wrist_xfs: torch.Tensor
    left_hand_model: HandModel


@dataclass
class ModelInput:
    orig_pose_data: PoseData
    s_solved_pose_data: PoseData
    left_images: torch.Tensor
    intrinsics: torch.Tensor
    extrinsics_xf: torch.Tensor
    hand_idx: torch.Tensor


@dataclass
class PerBranchOutput:
    joint_angles: torch.Tensor
    wrist_xfs: torch.Tensor
    skel_scales: Optional[torch.Tensor] = None
    pinch_prediction: Optional[torch.Tensor] = None


@dataclass
class ModelTarget:
    gt_skel_targets: PerBranchOutput
    preds_targets: PerBranchOutput
    intrinsics: Optional[torch.Tensor] = None
    extrinsics_xf: Optional[torch.Tensor] = None


def _compute_resample_matrix(
    camera_orig: camera.PinholePlaneCameraModel,
    camera_new: camera.PinholePlaneCameraModel,
):
    """
    pylib version.
    Return `resample_xf`: taking a point in `camera_new` to a point in `camera_orig`
    """
    K_inv_new44 = np.eye(4, 4)
    K_inv_new44[0:3, 0:3] = np.linalg.inv(camera_new.uv_to_window_matrix())
    K_orig44 = np.eye(4, 4)
    K_orig44[0:3, 0:3] = camera_orig.uv_to_window_matrix()

    eye_to_world_new = camera_new.camera_to_world_xf
    world_to_eye_orig = np.linalg.inv(camera_orig.camera_to_world_xf)
    resample_xf = K_orig44 @ world_to_eye_orig @ eye_to_world_new @ K_inv_new44
    resample_xf = resample_xf.astype(np.float32)

    return resample_xf


def _resample_images_batched(images_orig, images_new, resample_xfs):
    """
    `resample_r`, `resample_t`: from pixel `(u,v,1)` in img new to pixel in img orig
    """
    resample_r = resample_xfs[:, 0:3, 0:3]
    resample_t = resample_xfs[:, 0:3, 3]

    n_images = images_orig.shape[0]
    h_orig = images_orig.shape[1]
    w_orig = images_orig.shape[2]
    h_new = images_new.shape[1]
    w_new = images_new.shape[2]

    # construct img new grid coords in format `(u,v,1)`
    grid = np.ones((h_new, w_new, 3), dtype=np.int32)
    grid[:, :, 0:2] = np.mgrid[0:w_new:1, 0:h_new:1].transpose(2, 1, 0)

    # dot grid points coords by resample_r matrices
    # get shape (n_img, h_new, w_new, #resample_r_row)
    mul_res = np.tensordot(resample_r, grid, axes=([2], [2])).transpose(0, 2, 3, 1)
    # add resample_t
    grid_xfed = mul_res.reshape(n_images, -1, 3) + np.expand_dims(resample_t, axis=1)
    grid_xfed = grid_xfed.reshape(mul_res.shape)
    # compute homogenous coords by dividing `x, y` by `z`
    homo_coords = grid_xfed[:, :, :, 0:2] / grid_xfed[:, :, :, 2:]
    # boolean flag indicating whether transformed coords are in orig image
    coord_mask = (
        (homo_coords[:, :, :, 0] >= 0)
        & (homo_coords[:, :, :, 0] < (w_orig - 1))
        & (homo_coords[:, :, :, 1] >= 0)
        & (homo_coords[:, :, :, 1] < (h_orig - 1))
    )

    # coords in new img corresponding to coords that are inside orig image
    grid_int = np.repeat(grid.reshape(-1, *grid.shape)[:, :, :, 0:2], n_images, axis=0)
    grid_int = grid_int[coord_mask]

    # keep only in-image (transformed) coords, of shape (# kept pts, 2)
    coord_masked = homo_coords[coord_mask]

    # generate the host image index for each coord
    idx_grid = np.expand_dims(np.arange(n_images, dtype=np.int32), axis=1)
    idx_grid = np.repeat(idx_grid, h_new * w_new, axis=1)
    idx_grid = idx_grid.reshape(n_images, h_new, w_new)
    img_idx = idx_grid[coord_mask]

    x = coord_masked[:, 0]
    y = coord_masked[:, 1]

    x0 = x.astype(np.int32)
    x1 = x0 + 1
    y0 = y.astype(np.int32)
    y1 = y0 + 1

    # pixel values at four neighbor pixels
    fx_00 = images_orig[(img_idx, y0, x0)]
    fx_01 = images_orig[(img_idx, y1, x0)]
    fx_10 = images_orig[(img_idx, y0, x1)]
    fx_11 = images_orig[(img_idx, y1, x1)]

    # bi-linear interpolation
    images_new[(img_idx, grid_int[:, 1], grid_int[:, 0])] = (
        fx_00 * (x1 - x) * (y1 - y)
        + fx_10 * (x - x0) * (y1 - y)
        + fx_01 * (x1 - x) * (y - y0)
        + fx_11 * (x - x0) * (y - y0)
    ) / ((x1 - x0) * (y1 - y0))


def _gen_crop_matrices(
    orig_extrinsics: np.ndarray,
    orig_intrinsics: np.ndarray,
    crop_points: np.ndarray,
    mirror_image: bool,
    crop_size: Tuple[int, int],
    focal_multiplier: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a frame data containing some (usually two) views, generate the
    corresponding crop matrices extrinsics_xf, new_intrinsics, and
    resample_xf.

    Args:
    - `orig_extrinsics`: of shape `(#views, 4, 4)` (world_to_eye transformation)
    - `orig_intrinsics`: of shape `(#views, 3, 3)`
    - `crop_points`: landmarks' 3d positions, of shape `(#pts, 3)`
    - `mirror_image`: whether to flip the image horizontally
    - `crop_size`: width/height of the cropped image
    - `focal_multiplier`: multiplier of the focal length to make hands fit inside a crop

    Return (tuple of following)
    - `extrinsics_xf`: world_to_eye transformation for new camera used to generate the
    resampled images 2d pts in new image wrt their center to take up all space of new
    image. See `crop.gen_crop_parameters_from_points()`.
    - `new_intrinsics`: `(#frames, #views, 3, 3)`, intrinsics of new camera.
    - `resample_xf`: resample_matrix used for resampling pixeles from the
    view from original camera to the view from new camera
    """

    n_views = orig_extrinsics.shape[0]
    extrinsics_xf: np.ndarray = np.empty([n_views, 4, 4], dtype=scalar_type)
    new_intrinsics: np.ndarray = np.empty([n_views, 3, 3], dtype=scalar_type)
    resample_xf: np.ndarray = np.empty([n_views, 4, 4], dtype=scalar_type)

    for view_idx in range(n_views):
        fx, fy, cx, cy = (
            orig_intrinsics[view_idx][0, 0],
            orig_intrinsics[view_idx][1, 1],
            orig_intrinsics[view_idx][0, 2],
            orig_intrinsics[view_idx][1, 2],
        )
        world_to_eye_xf = orig_extrinsics[view_idx]
        camera_orig = camera.PinholePlaneCameraModel(
            width=crop_size[0],
            height=crop_size[1],
            f=(fx, fy),
            c=(cx, cy),
            distort_coeffs=[],
            camera_to_world_xf=np.linalg.inv(world_to_eye_xf),
        )
        camera_new: camera.PinholePlaneCameraModel = (
            crop.gen_crop_parameters_from_points(
                camera_orig,
                crop_points,
                crop_size,
                mirror_image,
                camera_angle=0,
                focal_multiplier=focal_multiplier,
            )
        )
        new_world_to_eye_xf = np.linalg.inv(camera_new.camera_to_world_xf)
        extrinsics_xf[view_idx] = new_world_to_eye_xf
        new_intrinsics[view_idx] = camera_new.uv_to_window_matrix()
        resample_xf[view_idx] = _compute_resample_matrix(camera_orig, camera_new)

    return (extrinsics_xf, new_intrinsics, resample_xf)


def _perspective_crop_images(
    orig_images: np.ndarray,
    orig_extrinsics: np.ndarray,
    orig_intrinsics: np.ndarray,
    crop_points: np.ndarray,
    hand_idx: int,
    crop_size: Tuple[int, int],
) -> List[np.ndarray]:
    """
    Given a sequence data, for each frame, generate a new camera
    looking through the hand crop center, return a new image
    that encloses all given points `crop_points`. Returned images
    have intensities normalized to [0, 1].

    Args:
    - `orig_images`: of shape `(#frames, #views, height, width)`
    - `orig_extrinsics`: of shape `(#frames, #views, 4, 4)`
    - `orig_intrinsics`: of shape `(#frames, #views, 3, 3)`
    - `crop_points`: landmarks' 3d positions, of shape `(#frames, #pts, 3)`
    - `hand_idx`: 0 for left and 1 for right

    Return (tuple of following)
    - `output_images`: `(#frames, #views, crop_size, crop_size)`, pixel values are normalized
    to 0~1.
    - `extrinsics_r`, `extrinsics_t`: T_camera_world for new camera used to generate the
    resampled images 2d pts in new image wrt their center to take up all space of new
    image. See `crop.gen_crop_parameters_from_points()`.
    - `new_intrinsics`: `(#frames, #views, 3, 3)`, intrinsics of new camera.
    """

    n_frames = orig_images.shape[0]
    n_views = orig_images.shape[1]

    # for each frame, generate a hand crop region and its corresponding new camera
    # use the new camera to resample a new image
    #
    # extrinsics: T_newCam_origCam (NOTE verify with @shchhan)
    extrinsics_xf = np.empty([n_frames, n_views, 4, 4], dtype=scalar_type)
    new_intrinsics = np.empty([n_frames, n_views, 3, 3], dtype=scalar_type)
    resample_matrices = np.empty([n_frames, n_views, 4, 4], dtype=scalar_type)

    for frame_idx in range(n_frames):
        (
            extrinsics_xf[frame_idx],
            new_intrinsics[frame_idx],
            resample_matrices[frame_idx],
        ) = _gen_crop_matrices(
            orig_extrinsics[frame_idx],
            orig_intrinsics[frame_idx],
            crop_points[frame_idx],
            hand_idx == 1,
            crop_size,
        )

    cropped_left_images = np.zeros([n_frames, n_views, *crop_size], dtype=scalar_type)
    _resample_images_batched(
        *bundles.map_fields(
            lambda t: t.reshape(-1, *t.shape[2:]),
            (orig_images, cropped_left_images, resample_matrices)
        )
    )

    # Normalize the input mono image to 0~1
    cropped_left_images = cropped_left_images / 255

    return [cropped_left_images, extrinsics_xf, new_intrinsics]


def prepare_inputs_targets(
    sample: RawSample, crop_size: Tuple[int, int]
) -> Tuple[ModelInput, ModelTarget]:
    """
    Perform reshaping to input data, generate new images for hand crops,
    compute transformation matrics from each view to canonical space, finally
    return them as input sample data.

    Args:
    - `orig_images`: images per frame within a sequence, of shape
    `(seq-len, 2(left/right view), H, W)`
    - `data_dict`: data other than images for each frame, indexed by data name

    Return:
    - `ModelInput`: model input data
    - `ModelTarget`: expected output GT
    """

    def to_th(t_in: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(t_in).float()

    # Change the units from mm to meters
    sample.scaled(0.001)

    orig_images = sample.images.astype(scalar_type)
    seq_length = orig_images.shape[0]
    num_views = orig_images.shape[1]

    def repeat_seq_length(t_in: torch.Tensor) -> torch.Tensor:
        t_out = t_in.unsqueeze(0).expand(seq_length, *t_in.shape)
        return t_out

    generic_hand_model = bundles.map_fields(
        repeat_seq_length,
        sample.generic_hand_model,
        only_type=torch.Tensor,
    )
    left_generic_hand_model = mirrored_hand_model(
        generic_hand_model, to_th(sample.hand) == 1
    )
    hand_model = bundles.map_fields(
        repeat_seq_length,
        sample.hand_model,
        only_type=torch.Tensor,
    )
    left_hand_model = mirrored_hand_model(hand_model, to_th(sample.hand) == 1)
    s_solved_pose_data: PoseData = PoseData(
        wrist_xfs=to_th(sample.solved_wrist_xfs),
        joint_angles=to_th(sample.solved_joint_angles),
        left_hand_model=left_generic_hand_model,
    )
    orig_pose_data: PoseData = PoseData(
        wrist_xfs=to_th(sample.wrist),
        joint_angles=to_th(sample.joint_angles),
        left_hand_model=left_hand_model,
    )

    # Only select a subset of enclosing points for cropping
    # sample.enclosing_points.shape = (seq_len, #pts, 3)
    crop_pts = sample.enclosing_points

    # obtain resampled images into `left_images` using a new
    # cam looking at those hand crops
    (left_images, extrinsics_xf, intrinsics) = _perspective_crop_images(
        orig_images,
        sample.extrinsics,
        sample.intrinsics,
        crop_pts,
        int(sample.hand[0]),
        crop_size,
    )

    multiv_input = ModelInput(
        orig_pose_data=orig_pose_data,
        s_solved_pose_data=s_solved_pose_data,
        left_images=to_th(left_images),
        intrinsics=to_th(intrinsics),
        extrinsics_xf=to_th(extrinsics_xf),
        hand_idx=to_th(sample.hand),
    )

    multiv_gtskel_targets: PerBranchOutput = PerBranchOutput(
        joint_angles=orig_pose_data.joint_angles,
        wrist_xfs=orig_pose_data.wrist_xfs,
        skel_scales=orig_pose_data.left_hand_model.hand_scale,
        pinch_prediction=to_th(sample.pinch),
    )
    multiv_preds_targets: PerBranchOutput = PerBranchOutput(
        joint_angles=s_solved_pose_data.joint_angles,
        wrist_xfs=s_solved_pose_data.wrist_xfs,
        skel_scales=s_solved_pose_data.left_hand_model.hand_scale,
        pinch_prediction=to_th(sample.pinch),
    )

    model_target = ModelTarget(
        gt_skel_targets=multiv_gtskel_targets,
        preds_targets=multiv_preds_targets,
        intrinsics=to_th(intrinsics),
        extrinsics_xf=to_th(extrinsics_xf),
    )

    return multiv_input, model_target


def preprocess(data: Dict[str, Any], crop_size: Tuple[int, int]) -> Tuple[ModelInput, ModelTarget]:
    """
    Transform each sequence into model input & groundtruth

    Args:
    - `data`: Dict with keys "mono", "msgpack_pose_data", "msgpack_s_solved_data"
    """
    sample = parse_raw_buffers(**data)
    model_input, model_target = prepare_inputs_targets(sample, crop_size)
    return model_input, model_target
