from typing import Dict, NamedTuple

import numpy as np

from lib.common.hand import NUM_JOINTS_PER_HAND


class SingleHandPose(NamedTuple):
    """
    A hand pose is composed of two fields:
    1) joint angles where # joints == # DoFs
    2) root-to-world rigid wrist transformation
    """

    joint_angles: np.ndarray = np.zeros(NUM_JOINTS_PER_HAND, dtype=np.float32)
    wrist_xform: np.ndarray = np.eye(4, dtype=np.float32)
    hand_confidence: float = 1.0


# Tracking result maps from hand_index to hand_pose
class TrackingResult(NamedTuple):
    hand_poses: Dict[int, SingleHandPose] = {}
    num_views: Dict[int, int] = {}
    predicted_scales: Dict[int, float] = {}
