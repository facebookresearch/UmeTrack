"""
Convert the nimble internal tdg format into the release reaady format
"""
import logging
import math
import os
import shutil

from dataclasses import dataclass
from enum import Enum
from itertools import chain
from multiprocessing import Pool
from typing import Any, BinaryIO, Dict, List, Optional, overload, Tuple, Union

import lib.data_utils.fs as fs

import msgpack
import numpy as np
import torch

from lib.common.hand import HandModel, NUM_JOINTS_PER_HAND, NUM_LANDMARKS_PER_HAND
from lib.data_utils import bundles, fs
from lib.data_utils.async_dataset import (
    AsyncToIterableDataset,
    SingleFolderAsyncDataset,
)
from lib.data_utils.idxbinfile import _np_to_idx_type, BinFormat, Buffer, OBJECT_DTYPE


class IdxBinWriter:
    """
    Utility class for writing files in .torch.bin format.

    It can be used in a few different ways. To write to in-memory buffers::

        with IdxBinWriter() as w:
            w.append(np.array([1,2,3]))
            w.append(np.array([10,20,30]))

        my_idx_data: memoryview = w.idx_data()
        my_bin_data: memoryview = w.bin_data()

    To write to a named file (on local fs), pass filenames to the
    constructor. If you omit the .bin file name, it will be deduced from the
    .idx file name:

        with IdxBinWriter("manfiold://my-bucket/tree/etc/blah.torch.idx") as w:
            w.append(np.array([1,2,3]))
            w.append(np.array([10,20,30]))

    .bin file name deduction: if .idx file name ends with ".torch.idx", replace it
    with ".torch.bin" to get .bin file name. Otherwise, replace the full file extension
    suffix in .idx file name with ".torch.bin":

        /local/blah.3.torch.idx -> /local.blah.3.torch.bin
        /local/blah.3.idx.txt -> /local/blah.torch.bin

    If you want to open the files yourself, you can also pass in file-like
    objects::

        with open("my.torch.idx", "wb") as idx, open("my.torch.bin", "wb") as bin:
            w = IdxBinWriter(idx, bin)
            w.append(np.array([1,2,3]))
            w.append(np.array([10,20,30]))
            w.flush()  # must call flush() or close() or use a with-block

    The with-blocks ensure that `flush()` is called, which is necessary to
    actually write out the .idx data, since it needs to know the final size
    before it can be written.
    """

    @overload
    def __init__(self, idx: BinaryIO, bin_: BinaryIO) -> None:
        """Write .torch.idx and .torch.bin files to the given streams"""
        ...

    @overload
    def __init__(self, idx: str, bin_: Optional[str] = None) -> None:
        """Write .idx and .bin files to the given files"""
        ...

    def __init__(self, idx=None, bin_=None):
        self._closed: bool = False
        self._dims: List[Tuple[int, ...]] = []
        self._dtype = None  # None until it becomes known
        self._bin_format: Optional[BinFormat] = None
        self._itemsize = None

        # If we opened the files, we also close them. If caller gave them to us,
        # let caller close them. This is necessary to let caller extract the
        # data after writing to an io.BytesIO().
        self._own_idx: bool = False
        self._own_bin: bool = False
        self._byte_sizes: List[int] = []

        try:
            idxfile: Optional[BinaryIO] = None
            binfile: Optional[BinaryIO] = None

            if isinstance(idx, str):
                if bin_ is None:
                    # Generate default bin file path:
                    bin_ = _get_bin_path_for_idx(idx)
                    logger.info(f"bin file path {bin_}")
                idxfile = fs.open(idx, "wb")
                self._own_idx = True
            else:
                idxfile = idx

            if isinstance(bin_, str):
                binfile = fs.open(bin_, "wb")
                self._own_bin = True
            else:
                binfile = bin_

            self._idxfile: BinaryIO = idxfile
            self._binfile: BinaryIO = binfile
        except BaseException:
            self.close(False)
            raise

    def append(self, frame: np.ndarray) -> None:
        """Append one frame of data"""
        self.append_raw(frame.dtype, BinFormat.TENSOR, frame.shape, frame.tobytes())

    def append_msgpack(self, obj) -> None:
        """Append `obj` as a msgpack-encoded frame."""
        self.append_raw(
            np.dtype("object"),
            BinFormat.MSGPACK,
            (),
            msgpack.packb(obj),
        )

    def append_raw(self, dtype, bin_format: BinFormat, dims, rawbytes: Buffer) -> None:
        """Append one frame of already-packed data"""
        rawbytes = memoryview(rawbytes)
        self._dims.append(dims)
        if self._dtype is None:
            self._dtype = dtype
            self._bin_format = bin_format
            self._itemsize = 1 if dtype == OBJECT_DTYPE else dtype.itemsize
        else:
            if self._dtype != dtype:
                raise TypeError(
                    f"All frames must all have same dtype; got {self._dtype} and {dtype}"
                )
            if self._bin_format != bin_format:
                raise TypeError(
                    f"All frames must all have same binformat; got {self._bin_format} and {BinFormat.TENSOR}"
                )
        if rawbytes.nbytes % self._itemsize != 0:
            raise ValueError("buffer size is not a multiple of item size")
        self._byte_sizes.append(rawbytes.nbytes)
        self._binfile.write(rawbytes)

    def close(self, flush=True) -> None:
        if not self._closed:
            self._closed = True
            if flush:
                self.flush()
            if self._own_idx:
                self._idxfile.close()
            if self._own_bin:
                self._binfile.close()

    def flush(self, flush=True) -> None:
        del flush
        idxarray = self._generate_idx()
        self._idxfile.write(idxarray.data)

    def _generate_idx(self) -> np.ndarray:
        # pylint: disable=invalid-name, standard math notation
        N = len(self._dims)
        assert N > 0, "Cannot write empty dataset"
        S = sum(len(s) for s in self._dims)

        dtype = self._dtype
        bin_format = self._bin_format
        assert dtype is not None
        assert bin_format is not None

        # format idx as array of int64s
        idx = np.empty(6 + (N + 1) * 2 + S, dtype="int64")
        idx[0] = 0x584449544E54
        idx[1] = 1  # version
        idx[2] = _np_to_idx_type[(dtype.name, bin_format)]
        idx[3] = self._itemsize
        idx[4] = N
        idx[5] = S
        ofs = 6
        idx[ofs] = 0
        idx[ofs + 1 : ofs + N + 1] = np.cumsum([len(s) for s in self._dims])
        ofs += N + 1
        idx[ofs] = 0
        idx[ofs + 1 : ofs + N + 1] = np.cumsum(self._byte_sizes) // self._itemsize
        ofs += N + 1
        idx[ofs : ofs + S] = [*chain(*self._dims)]
        return idx

    def __enter__(self) -> "IdxBinWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(flush=exc_type is None)


def parse_skeleton_data(
    *,
    np_raw_skeleton: np.ndarray,
    np_raw_skinned_points: np.ndarray,
    np_hand_scale: Optional[np.ndarray] = None,
) -> HandModel:
    """
    Transforms raw skeleton data into a HandModel
    """
    np_raw_skeleton = np_raw_skeleton.reshape((NUM_JOINTS_PER_HAND, 10))
    np_raw_skinned_points = np_raw_skinned_points.reshape((NUM_LANDMARKS_PER_HAND, 9))
    np_hand_scale = np_hand_scale or np.array(1.0, dtype=np.float32)

    return HandModel(
        joint_rotation_axes=torch.Tensor(np_raw_skeleton[..., 0:3]),
        joint_rest_positions=torch.Tensor(np_raw_skeleton[..., 3:6]),
        joint_frame_index=torch.Tensor(np_raw_skeleton[..., 6]),
        joint_parent=torch.Tensor(np_raw_skeleton[..., 7]),
        joint_first_child=torch.Tensor(np_raw_skeleton[..., 8]),
        joint_next_sibling=torch.Tensor(np_raw_skeleton[..., 9]),
        landmark_rest_positions=torch.Tensor(np_raw_skinned_points[..., 0:3]),
        landmark_rest_bone_weights=torch.Tensor(np_raw_skinned_points[..., 3:6]),
        landmark_rest_bone_indices=torch.Tensor(np_raw_skinned_points[..., 6:9]),
        hand_scale=torch.Tensor(np_hand_scale),
    )


@dataclass
class RawSample:
    images: np.ndarray
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    enclosing_points: np.ndarray
    hand: np.ndarray
    xyz: np.ndarray
    # GT skeleton
    hand_model: HandModel
    wrist: np.ndarray
    joint_angles: np.ndarray
    # Generic skeleton
    solved_wrist_xfs: np.ndarray
    solved_joint_angles: np.ndarray
    generic_hand_model: HandModel
    is_background: Optional[np.ndarray] = None
    pinch: Optional[np.ndarray] = None

    def validate(self):
        seq_length = self.images.shape[0]
        # reshaping flattened fields
        self.extrinsics = self.extrinsics.reshape(seq_length, 2, 4, 4)
        self.intrinsics = self.intrinsics.reshape(seq_length, 2, 3, 3)
        self.enclosing_points = self.enclosing_points.reshape(seq_length, -1, 3)
        self.wrist = self.wrist.reshape(seq_length, 4, 4)
        self.joint_angles = self.joint_angles.reshape(seq_length, NUM_JOINTS_PER_HAND)
        self.joint_angles = self.joint_angles[:, :NUM_JOINTS_PER_HAND]
        self.solved_wrist_xfs = self.solved_wrist_xfs.reshape(seq_length, 4, 4)
        self.solved_joint_angles = self.solved_joint_angles.reshape(
            seq_length, NUM_JOINTS_PER_HAND
        )
        self.solved_joint_angles = self.solved_joint_angles[:, :NUM_JOINTS_PER_HAND]
        self.hand = np.repeat(self.hand, seq_length)

        if self.pinch is not None:
            self.pinch = self.pinch.reshape(seq_length, 1)
        else:
            self.pinch = np.zeros((seq_length, 1))


def parse_raw_buffers(
    mono: np.ndarray,
    msgpack_pose_data: np.ndarray,
    msgpack_s_solved_data: np.ndarray,
) -> RawSample:
    seq_length = mono.shape[0]
    pose_data = {}
    pose_data.update(msgpack.loads(msgpack_pose_data, raw=False))
    pose_data.update(msgpack.loads(msgpack_s_solved_data, raw=False))

    for k, v in pose_data.items():
        if isinstance(v, (list, float)):
            pose_data[k] = np.array(v, dtype=np.float32)
    n_views = 2
    # Only square shape is supported now
    img_size = int(np.sqrt(mono.size // seq_length // n_views))
    images = mono.reshape(seq_length, n_views, img_size, img_size)

    hand_model = parse_skeleton_data(
        np_raw_skeleton=pose_data["skeleton"],
        np_raw_skinned_points=pose_data["skinned_points"],
    )
    generic_hand_model = parse_skeleton_data(
        np_raw_skeleton=pose_data["generic_skeleton"],
        np_raw_skinned_points=pose_data["generic_skinned_points"],
        np_hand_scale=np.array(float(pose_data["solved_scales"][0]), dtype=np.float32),
    )

    target_fields = set(bundles.field_names(RawSample))

    unpacked_dict = {
        "images": images,
        "hand_model": hand_model,
        "generic_hand_model": generic_hand_model,
        **{k: v for k, v in pose_data.items() if k in target_fields},
    }
    raw_sample = RawSample(**unpacked_dict)
    raw_sample.validate()

    return raw_sample


FIELDS = ["mono", "msgpack_pose_data", "msgpack_s_solved_data"]


def _find_input_output_files(input_dir: str, output_dir: str):
    res_input_paths = []
    res_output_paths = []
    for cur_dir, _, filenames in fs.walk(input_dir):
        if not "mono.torch.bin" in filenames:
            continue

        rel_dir_path = cur_dir[len(input_dir) :]
        output_dir_path = fs.join(output_dir, rel_dir_path)
        res_input_paths.append(cur_dir)
        res_output_paths.append(output_dir_path)

    assert len(res_input_paths) == len(res_output_paths)
    print(f"Found {len(res_input_paths)} files from {input_dir}")
    return res_input_paths, res_output_paths


def _reexport_data(input_output):
    input, output = input_output
    print(f"Processing {input}")
    dataset = SingleFolderAsyncDataset(
        folder=input, fields=FIELDS, cache_in_mem_fields=FIELDS
    )
    iterable_data = AsyncToIterableDataset(
        dataset,
        sampler=range(0, len(dataset)),
    )

    if fs.exists(output):
        shutil.rmtree(output)

    os.makedirs(output)

    def _create_writer(field_name):
        return IdxBinWriter(
            fs.join(output, f"{field_name}.torch.idx"),
            fs.join(output, f"{field_name}.torch.bin"),
        )

    with _create_writer("mono") as mono_writer, _create_writer(
        "labels"
    ) as label_writer:
        for sample in iterable_data:
            processed_sample = parse_raw_buffers(**sample)
            sample_dict = bundles.asdict_rec(processed_sample)

            def _to_export_data(t_in):
                if isinstance(t_in, torch.Tensor):
                    return t_in.numpy().tolist()
                elif isinstance(t_in, np.ndarray):
                    return t_in.tolist()
                else:
                    return t_in

            images = sample_dict["images"].astype(np.uint8)
            for k0 in ["hand_model", "generic_hand_model"]:
                hand_model_keys = [*sample_dict[k0].keys()]
                for key in hand_model_keys:
                    if sample_dict[k0][key] is None:
                        del sample_dict[k0][key]

            del sample_dict["images"]
            del sample_dict["xyz"]
            dict_out = bundles.map_fields(_to_export_data, sample_dict)
            mono_writer.append(images)
            label_writer.append_msgpack(dict_out)

    return


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(__file__))

    test_run = True
    if test_run:
        input_paths, output_paths = _find_input_output_files(
            os.path.join(root, "torch_data_orig", "cropped", "00000"),
            os.path.join(root, "torch_data_test", "cropped", "00000"),
        )
        n_processed = 0
        for p_in, p_out in zip(input_paths, output_paths):
            _reexport_data((p_in, p_out))
            n_processed += 1
            if n_processed % 20 == 0:
                print(f"Processed {n_processed} out of {len(input_paths)} files")
    else:
        input_paths, output_paths = _find_input_output_files(
            os.path.join(root, "torch_data_orig"),
            os.path.join(root, "torch_data"),
        )

        pool_size = 16
        with Pool(pool_size) as p:
            error_tensors = p.map_async(
                _reexport_data, zip(input_paths, output_paths)
            ).get()
