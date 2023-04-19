# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from functools import partial

from typing import List, Tuple

import lib.data_utils.fs as fs

import numpy as np
import torch
from lib.batched_dataset.data_transform import ModelInput, preprocess
from lib.common.hand import mirrored_hand_model
from lib.common.hand_skinning import skin_landmarks
from lib.data_utils import bundles
from lib.data_utils.async_dataset import (
    AsyncToIterableDataset,
    find_dataset,
    Sampler,
    subsample,
)
from lib.data_utils.dataset_util import map_dataset
from lib.data_utils.split import Split
from lib.models.model_loader import load_pretrained_model
from lib.models.regressor import RegressorOutput
from lib.models.umetrack_model import InputFrameData, InputFrameDesc, InputSkeletonData


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def _unpack_batched_data(
    training_input: ModelInput, seq_mode: str
) -> List[Tuple[InputFrameData, InputFrameDesc, InputSkeletonData]]:
    # Construct the left hand input images, skeletons and skinned landmarks
    bs = training_input.left_images.shape[0]
    seq_len = training_input.left_images.shape[1]
    left_images = training_input.left_images.clone()
    left_hand_model = training_input.orig_pose_data.left_hand_model

    inference_inputs = []
    for i_frame in range(seq_len):
        memory_idx = torch.arange(0, bs, device=left_images.device)
        use_memory = torch.ones(bs, device=left_images.device, dtype=torch.bool)
        if i_frame == 0:
            use_memory[:] = False

        if seq_mode == "multiv":
            nv = 2
        elif seq_mode == "singlev":
            nv = 1
        else:
            raise ValueError(f"Unknown sequence mode: {seq_mode}")

        sample_range = torch.tensor(
            [(i * nv, (i + 1) * nv) for i in range(bs)], device=left_images.device
        )

        frame_data = InputFrameData(
            left_images=torch.flatten(left_images[:, i_frame, 0:nv], 0, 1),
            intrinsics=torch.flatten(training_input.intrinsics[:, i_frame, 0:nv], 0, 1),
            extrinsics_xf=torch.flatten(
                training_input.extrinsics_xf[:, i_frame, 0:nv], 0, 1
            ),
        )
        frame_desc = InputFrameDesc(
            hand_idx=training_input.hand_idx[:, i_frame].long(),
            sample_range=sample_range.long(),
            memory_idx=memory_idx.long(),
            use_memory=use_memory,
        )
        skel_data = InputSkeletonData(
            joint_rotation_axes=left_hand_model.joint_rotation_axes[:, i_frame],
            joint_rest_positions=left_hand_model.joint_rest_positions[:, i_frame],
        )
        inference_inputs.append((frame_data, frame_desc, skel_data))

    return inference_inputs


def _eval_batch(
    model, model_input, model_target, cur_mode: str, use_skel: bool, device: str
):
    hand_model = mirrored_hand_model(
        model_input.orig_pose_data.left_hand_model,
        model_input.hand_idx == 1,  # right hand is index 1
    )
    inference_inputs = _unpack_batched_data(model_input, cur_mode)
    inference_outputs = []
    for i_step, step_input in enumerate(inference_inputs):
        frame_data, frame_desc, skel_input = bundles.to_device(step_input, device)
        if use_skel:
            cur_output = model.regress_pose_use_skeleton(
                frame_data,
                frame_desc,
                skel_input,
            )
        else:
            assert (
                cur_mode == "multiv"
            ), "Skeleton scale prediction requires multiv data"
            cur_output = model.regress_pose_pred_skel_scale(frame_data, frame_desc)

        cur_output = bundles.to_device(cur_output, torch.device("cpu"))
        inference_outputs.append(cur_output)

    inference_outputs_batched = bundles.collate(inference_outputs)
    # Collate puts the sequence dim as the leading dim.
    # Do a transpose here to swap the batch dim and sequence dim.
    inference_outputs_batched = bundles.map_fields(
        lambda t: t.transpose(0, 1) if t is not None else None,
        inference_outputs_batched,
    )

    regression_target = model_target.gt_skel_targets
    gt_keypoints = skin_landmarks(
        hand_model, regression_target.joint_angles, regression_target.wrist_xfs
    )
    output_keypoints = skin_landmarks(
        hand_model,
        inference_outputs_batched.joint_angles,
        inference_outputs_batched.wrist_xfs,
    )
    keypoints_diff = gt_keypoints - output_keypoints
    keypoint_errors = keypoints_diff.norm(dim=-1).mean(dim=(1, 2))
    keypoint_errors_mm = keypoint_errors * 1000

    return keypoint_errors_mm


if __name__ == "__main__":
    root = os.path.dirname(__file__)
    device: str = "cuda" if torch.cuda.device_count() else "cpu"
    dataset_names = ["real", "synthetic"]
    print(f"Using device: {device}")
    datasets_all = [os.path.join(root, "UmeTrack_data", "torch_data", s) for s in dataset_names]

    fields = ["mono", "labels"]
    datasets = find_dataset(datasets_all, fields)
    print("Dataset stats")
    for k, v in datasets.items():
        print(f"{k}: {len(v)}")
    portion = 1
    if portion != 1:
        datasets = {k: subsample(v, portion=portion) for k, v in datasets.items()}
        print("After subsample")
        for k, v in datasets.items():
            print(f"{k}: {len(v)}")

    model_name = "pretrained_weights.torch"
    model_path = os.path.join(root, "pretrained_models", model_name)
    model = load_pretrained_model(model_path)
    model.eval()
    model.to(device)

    loaders = {}
    num_workers = 6
    batch_size = 160
    world_offset = 0
    crop_image_size = (96, 96)
    for split, dataset in datasets.items():
        sampler = Sampler(dataset, shuffle=False, drop_last=True, distrib_info=(0, 1))
        iterable_dataset = AsyncToIterableDataset(
            dataset,
            sampler,
            max_prefetch=64,
        )
        iterable_dataset = map_dataset(
            partial(preprocess, crop_size=crop_image_size), iterable_dataset
        )

        loaders[split] = torch.utils.data.DataLoader(
            iterable_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=bundles.collate,
        )

    run_splits = [Split.TEST]
    with torch.inference_mode():
        keypoint_errors = {}
        for split, loader in loaders.items():
            if split not in run_splits:
                continue
            if len(loader) == 0:
                continue
            keypoint_errors[split] = []
            for minibatch_idx, (model_input, model_target) in enumerate(loader):
                batch_keypoint_errors = _eval_batch(
                    model,
                    model_input,
                    model_target,
                    "multiv",
                    use_skel=True,
                    device=device,
                )
                keypoint_errors[split].append(batch_keypoint_errors)
            mean_error = torch.cat(keypoint_errors[split]).mean()
            print(f"Keypoint errors ({split.value}): {mean_error} mm")
