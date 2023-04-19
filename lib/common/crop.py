#!/usr/bin/env python3
from typing import Tuple

import lib.common.camera as camera
import numpy as np

from . import affine


def gen_intrinsics_from_bounding_pts(
    pts_eye: np.ndarray, image_w: int, image_h: int, min_focal: float = 5
) -> Tuple[np.ndarray, np.ndarray]:
    pts_ndc = pts_eye[..., 0:2] / pts_eye[..., 2:]
    img_size = np.array([image_w, image_h], dtype=pts_eye.dtype)
    # Given our convention, we need to shift one pixel before dividing by 2.
    cx_cy = (img_size - 1) / 2
    fx_fy = cx_cy / np.absolute(pts_ndc).max()

    # Some sanity checks
    if np.any(pts_eye[..., 2:] < 0.0001) or np.any(fx_fy < min_focal):
        raise ValueError("Unable to create crop camera", fx_fy)

    return fx_fy, cx_cy


def gen_crop_parameters_from_points(
    camera_orig: camera.CameraModel,
    pts_world,
    new_image_size: Tuple[int, int],
    mirror_img_x: bool,
    camera_angle: float = 0,
    focal_multiplier: float = 0.95,
) -> camera.PinholePlaneCameraModel:
    """
    Given the original camera transform and a list of 3D points in the world space,
    compute the new perspective camera that makes sure after projection all the points
    can be projected inside the image.

    Auguments:
    * camera_orig: the original camera used for generating an image. The returned camera
        will have the same position but different rotation and intrinsics parameters.
    * pts_world: points in the world space that must be projected inside the image by
        the generated world to eye transform and intrinsics.
    * new_image_size: target image size
    * mirror_img_x: whether to flip the image. A typical use case is we usually mirror the
        right hand images so that a model need to handle left hand data only
    * camera_angle: how the camera is oriented physically so that we can rotate the object of
        interest to the 'upright' direction
    * focal_multiplier: when less than 1, we are zooming out a little. The effect on the image
        is some margin will be left at the boundary.
    """
    orig_world_to_eye_xf = np.linalg.inv(camera_orig.camera_to_world_xf)

    crop_center = (pts_world.min(axis=0) + pts_world.max(axis=0)) / 2.0
    new_world_to_eye = affine.make_look_at_matrix(
        orig_world_to_eye_xf, crop_center, camera_angle
    )
    if mirror_img_x:
        mirrorx = np.eye(4, dtype=np.float32)
        mirrorx[0, 0] = -1
        new_world_to_eye = mirrorx @ new_world_to_eye

    fx_fy, cx_cy = gen_intrinsics_from_bounding_pts(
        affine.transform3(new_world_to_eye, pts_world),
        new_image_size[0],
        new_image_size[1],
    )
    fx_fy = focal_multiplier * fx_fy

    return camera.PinholePlaneCameraModel(
        width=new_image_size[0],
        height=new_image_size[1],
        f=fx_fy,
        c=cx_cy,
        distort_coeffs=[],
        camera_to_world_xf=np.linalg.inv(new_world_to_eye),
    )
