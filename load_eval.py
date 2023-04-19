# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import fnmatch
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import lib.data_utils.fs as fs
import numpy as np
from lib.common import metric_utils
from lib.data_utils import bundles


@dataclass
class Metrics:
    keypoint_errors: np.ndarray
    keypoint_accelerations: np.ndarray
    gt_keypoint_accelerations: np.ndarray


def _compute_metrics(
    gt_keypoints: np.ndarray, tracked_keypoints: np.ndarray, valid_tracking: np.ndarray
) -> Metrics:
    def _compute_accelerations(pts: np.ndarray):
        acc = pts[:, 0:-2] + pts[:, 2:] - 2 * pts[:, 1:-1]
        return np.linalg.norm(acc, axis=-1).mean(axis=-1)

    diff_keypoints = gt_keypoints - tracked_keypoints
    keypoint_errors = np.linalg.norm(diff_keypoints, axis=-1).mean(axis=-1)
    valid_accelerations = (
        valid_tracking[:, 0:-2] & valid_tracking[:, 1:-1] & valid_tracking[:, 2:]
    )
    keypoint_accelerations = _compute_accelerations(tracked_keypoints)
    gt_keypoint_accelerations = _compute_accelerations(gt_keypoints)
    return Metrics(
        keypoint_errors=keypoint_errors[valid_tracking],
        keypoint_accelerations=keypoint_accelerations[valid_accelerations],
        gt_keypoint_accelerations=gt_keypoint_accelerations[valid_accelerations],
    )


def aggregate_metrics(output_dir: str):
    valid_tracking_all = []
    metrics_all = []
    for cur_dir, _, filenames in fs.walk(output_dir):
        json_files = fnmatch.filter(filenames, "*.npy")
        for fname in json_files:
            f_full = fs.join(cur_dir, fname)
            metric_data = {}
            with fs.open(f_full, "rb") as fp:
                metric_data = pickle.load(fp)
            valid_tracking_all.append(metric_data["valid_tracking"])
            metrics_all.append(
                _compute_metrics(
                    metric_data["gt_keypoints"],
                    metric_data["tracked_keypoints"],
                    metric_data["valid_tracking"],
                )
            )
    if len(metrics_all) != 0:
        combined_metrics = bundles.group(metrics_all, np.concatenate)
        pck_curve = (
            metric_utils.PCK_curve(
                combined_metrics.keypoint_errors, metric_utils.PCK_THRESHOLDS
            )
            * 100.0
        )
        auc_score = float(
            metric_utils.normalized_AUC(metric_utils.PCK_THRESHOLDS, pck_curve)
        )
        valid_tracking_cat = np.concatenate(valid_tracking_all, axis=1)
        n_total = valid_tracking_cat.size
        n_valid = valid_tracking_cat.sum()
        print(
            f"  Tracked {n_valid} out of {n_total}, success rate: {n_valid / n_total * 100}%"
        )
        print(f"  Mean keypoint error: {combined_metrics.keypoint_errors.mean()}")
        print(f"  AUC score: {auc_score}")
        print(
            f"  Mean keypoint accelerations: {combined_metrics.keypoint_accelerations.mean()}"
        )
        print(
            f"  GT mean keypoint accelerations: {combined_metrics.gt_keypoint_accelerations.mean()}"
        )


if __name__ == "__main__":
    root = os.path.dirname(__file__)

    for eval_mode in ["known_skeleton", "unknown_skeleton"]:
        for protocol in ["separate_hand", "hand_hand"]:
            output_dir = os.path.join(
                root,
                "tmp",
                f"eval_results_{eval_mode}",
                "real",
                protocol,
            )
            print(f"Evaluation for {eval_mode} on protocol {protocol}")
            aggregate_metrics(output_dir)
