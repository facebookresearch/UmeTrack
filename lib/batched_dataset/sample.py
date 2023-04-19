# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch

from lib.common.hand import HandModel, scaled_hand_model


@dataclass
class RawSample:
    images: np.ndarray
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    enclosing_points: np.ndarray
    hand: np.ndarray
    # GT skeleton
    hand_model: HandModel
    wrist: np.ndarray
    joint_angles: np.ndarray
    # Generic skeleton
    solved_wrist_xfs: np.ndarray
    solved_joint_angles: np.ndarray
    generic_hand_model: HandModel
    pinch: np.ndarray

    def scaled(self, factor: float):
        self.extrinsics[..., :3, 3] *= factor
        self.enclosing_points *= factor
        self.wrist[..., :3, 3] *= factor
        self.solved_wrist_xfs[..., :3, 3] *= factor
        self.hand_model = scaled_hand_model(self.hand_model, factor)
        self.generic_hand_model = scaled_hand_model(self.generic_hand_model, factor)


def parse_raw_buffers(mono: np.ndarray, labels: Dict[str, Any]) -> RawSample:
    labels_typed = {}
    for field, field_value in labels.items():
        if "hand_model" in field:
            labels_typed[field] = HandModel(
                **{k: torch.tensor(v) for k, v in field_value.items()}
            )
        else:
            labels_typed[field] = np.array(field_value, dtype=np.float32)

    unpacked_dict = {"images": mono, **labels_typed}
    return RawSample(**unpacked_dict)
