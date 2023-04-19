import json
from dataclasses import dataclass

from typing import Iterator, List

import av
import lib.data_utils.fs as fs
import numpy as np
import torch
from lib.common.camera import CameraModel, read_camera_from_json
from lib.common.hand import HandModel
from lib.tracker.tracker import InputFrame, ViewData

from .tracking_result import SingleHandPose


@dataclass
class HandPoseLabels:
    cameras: List[CameraModel]
    camera_angles: List[float]
    camera_to_world_transforms: np.ndarray
    hand_model: HandModel
    joint_angles: np.ndarray
    wrist_transforms: np.ndarray
    hand_confidences: np.ndarray

    def __len__(self):
        return len(self.joint_angles)


class VideoStream:
    def __init__(self, data_path: str):
        self._data_path = data_path

    def __len__(self) -> int:
        container = av.open(self._data_path)
        # take first video stream
        stream = container.streams.video[0]
        return stream.frames

    def __iter__(self) -> Iterator[np.ndarray]:
        container = av.open(self._data_path)
        # take first video stream
        stream = container.streams.video[0]
        print(f"Opened ({int(stream.average_rate)} fps) video from {self._data_path}")

        for idx, frame in enumerate(container.decode(stream)):
            raw_mono_image_np = np.array(frame.to_image())[..., 0]
            yield raw_mono_image_np


def _load_json(p: str):
    with fs.open(p, "rb") as bf:
        return json.load(bf)


def load_hand_model_from_dict(hand_model_dict) -> HandModel:
    hand_tensor_dict = {}
    for k, v in hand_model_dict.items():
        if isinstance(v, list):
            hand_tensor_dict[k] = torch.Tensor(v)
        else:
            hand_tensor_dict[k] = v

    hand_model = HandModel(**hand_tensor_dict)
    return hand_model


def _load_hand_pose_labels(p: str) -> HandPoseLabels:
    labels = _load_json(p)
    cameras = [read_camera_from_json(c) for c in labels["cameras"]]
    camera_angles = labels["camera_angles"]
    hand_model = load_hand_model_from_dict(labels["hand_model"])
    joint_angles = np.array(labels["joint_angles"])
    wrist_transforms = np.array(labels["wrist_transforms"])
    hand_confidences = np.array(labels["hand_confidences"])
    camera_to_world_transforms = np.array(labels["camera_to_world_transforms"])

    return HandPoseLabels(
        cameras=cameras,
        camera_angles=camera_angles,
        camera_to_world_transforms=camera_to_world_transforms,
        hand_model=hand_model,
        joint_angles=joint_angles,
        wrist_transforms=wrist_transforms,
        hand_confidences=hand_confidences,
    )


class SyncedImagePoseStream:
    def __init__(self, data_path: str):
        label_path = data_path[:-4] + ".json"
        self._hand_pose_labels = _load_hand_pose_labels(label_path)
        self._image_stream = VideoStream(data_path)
        assert len(self._hand_pose_labels) == len(self._image_stream)

    def __len__(self) -> int:
        return len(self._image_stream)

    def __iter__(self):
        for frame_idx, raw_mono in enumerate(self._image_stream):
            gt_tracking = {}
            for hand_idx in range(0, 2):
                if self._hand_pose_labels.hand_confidences[frame_idx, hand_idx] > 0:
                    gt_tracking[hand_idx] = SingleHandPose(
                        joint_angles=self._hand_pose_labels.joint_angles[
                            frame_idx, hand_idx
                        ],
                        wrist_xform=self._hand_pose_labels.wrist_transforms[
                            frame_idx, hand_idx
                        ],
                        hand_confidence=self._hand_pose_labels.hand_confidences[
                            frame_idx, hand_idx
                        ],
                    )

            multi_view_images = raw_mono.reshape(
                raw_mono.shape[0], len(self._hand_pose_labels.cameras), -1
            )
            invalid_camera_to_world = (
                self._hand_pose_labels.camera_to_world_transforms[frame_idx].sum() == 0
            )
            if invalid_camera_to_world:
                assert (
                    not gt_tracking
                ), f"Cameras are not tracked, expecting no ground truth tracking!"

            views = []
            for cam_idx in range(0, len(self._hand_pose_labels.cameras)):
                cur_camera = self._hand_pose_labels.cameras[cam_idx].copy(
                    camera_to_world_xf=self._hand_pose_labels.camera_to_world_transforms[
                        frame_idx, cam_idx
                    ],
                )

                views.append(
                    ViewData(
                        image=multi_view_images[:, cam_idx, :],
                        camera=cur_camera,
                        camera_angle=self._hand_pose_labels.camera_angles[cam_idx],
                    )
                )

            input_frame = InputFrame(views=views)
            yield input_frame, gt_tracking
