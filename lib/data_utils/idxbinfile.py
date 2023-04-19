"""
Helpers for working with idx and bin torch files.
"""
import logging
import math
from enum import Enum
from itertools import chain
from typing import Any, BinaryIO, Dict, List, Optional, overload, Tuple, Union

import msgpack
import numpy as np
from lib.data_utils import fs

from .dataset_util import ISeq, map_dataset

logger = logging.getLogger(__name__)

Buffer = Union[bytes, memoryview]


MsgpackObject = Dict[str, Any]
RawField = Union[np.ndarray, MsgpackObject]


class BinFormat(Enum):
    TENSOR = 0
    MSGPACK = 1


def _get_bin_path_for_idx(idx_path: str):
    # if idx file path ends with ".torch.idx", then replace it with ".torch.bin" to get bin file path
    # otherwise, replace the full extension in idx file path with ".torch.bin"
    idx_suffix = ".torch.idx"
    bin_suffix = ".torch.bin"
    assert idx_path.endswith(idx_suffix)
    return idx_path[: -len(idx_suffix)] + bin_suffix


#
# Support for (.torch.idx, .torch.bin) file format
#
class TorchIdx:
    """
    Parses a .torch.idx file.

    A .torch.idx file describes the layout of a single .torch.bin file.
    Together, a ".torch.bin" and ".torch.idx" file describe an array of
    np.ndarrays or torch.Tensors::

        idx = TorchIdx("/big/data/some/where/training/file.torch.idx")

        data = idx.read_bin()
        frame17 = data[17]

    `N = len(idx)` is the number of frames.
    `idx.dims[N]` gives the shape of each frame.

    Parameters
    ----------
    path
        path to the .torch.idx file. Must be provided even if `buffer` is
        given, though it is only used for the `source` and `bin_path`
        attributes in that case.

    buffer
        bytes of .idx file, if it was already read from somewhere. If buffer
        is not provided, it will be read from `path`.

    Attributes
    ----------
    source : str
        path that this .torch.idx file was loaded from

    bin_path : str
        path to corresponding .torch.bin file

    is_uniform : bool
        true if all N frames have identical dims and dtype

    shape : Optional[Tuple[int]]
        shape of whole bin file, or `None` if per-frame shapes aren't
        all identical

    dtype : np.dtype
        dtype of array elements

    dims : Union[np.ndarray, List[Tuple]]
        dims[i] gives the shape of the i'th frame.
        Stored as a single ndarray if possible, List[Tuple] if frames
        have differing ranks.

    byte_offsets : np.ndarray
        `byte_offsets[i:i+1]` gives the span containing frame `i` in the .torch.bin file
    """

    source: str
    bin_path: str
    is_uniform: bool
    shape: Optional[Tuple[int]]
    dtype: Optional[np.dtype]
    dims: Union[np.ndarray, List[Tuple]]
    byte_offsets_array: Optional[np.ndarray]
    itemsize: int

    # TorchIdx files are formatted as an array of int64:
    #
    #  [0] magic value = 0x584449544E54
    #  [1] version = 1
    #  [2] tensor data type (index into _idx_to_np_type)
    #  [3] itemsize = np.dtype(dtype).itemsize
    #  [4] number of elements N
    #  [5] total number of dimensions
    #  [...] N+1 dim offsets (indices into our 'sizes' portion)
    #  [...] N+1 data offsets (indices into .bin file, in units of itemsize)
    #  [...] S sizes (dimensions of each element)
    #
    # Note that this supports arrays of tensors of varying sizes. However, most
    # .torchdata files have uniform sizes, and some problem code expects this.
    def __init__(
        self, path: str, bin_path: Optional[str] = None, buffer: Optional[Buffer] = None
    ) -> None:
        # pylint: disable=invalid-name, standard math notation
        if bin_path is None:
            bin_path = _get_bin_path_for_idx(path)
        if buffer is None:
            buffer = fs.read_bytes(path)
        self.source = path
        self.bin_path = bin_path

        data = self._idx_buf_to_int64(buffer)

        if data[1] == 0:
            if data[0] != 0:
                raise ValueError(f"bad magic number in .torch.idx file {path}")
        elif data[1] == 1:
            if data[0] != 0x584449544E54:
                raise ValueError(f"bad magic number in .torch.idx file {path}")
        else:
            raise ValueError(f"unsupported version {data[1]} in .torch.idx file {path}")

        code = data[2]
        self.itemsize = data[3]

        np_type, self._bin_format = _idx_to_np_type.get(code, None)
        if not np_type:
            raise KeyError(f"unrecognized type {code}")
        dtype = self.dtype = np.dtype(np_type)
        if dtype != OBJECT_DTYPE and dtype.itemsize != self.itemsize:
            raise ValueError(
                f"item size {self.itemsize} not compatible with "
                f"dtype {np_type}.itemsize={dtype.itemsize}"
            )

        N = int(data[4])
        S = int(data[5])

        ofs = 6
        dim_offsets = data[ofs : ofs + N + 1].copy()
        ofs += N + 1
        data_offsets = data[ofs : ofs + N + 1]
        ofs += N + 1
        sizes = data[ofs : ofs + S]
        dims = [tuple(sizes[dim_offsets[i] : dim_offsets[i + 1]]) for i in range(N)]

        # if possible, represent dims as a single array
        # note that self.shape is only set if dims has a uniform shape
        self.is_uniform = False
        self.shape = None
        self.byte_offsets_array: Optional[np.ndarray] = data_offsets * self.itemsize
        if self.dtype != OBJECT_DTYPE and all(len(d) == len(dims[0]) for d in dims):
            dims = _pack_reps(np.array(dims))
            if (dims == dims[0]).all():
                self.is_uniform = True
                self.shape = (N, *dims[0])
                per_tensor_length = math.prod(self.shape[1:])
                assert np.all(np.diff(data_offsets) == per_tensor_length)
                if data_offsets[0] == 0:
                    # Every element has the same size, so it's inefficient to
                    # store the byte offsets; instead just recompute them on the
                    # fly using the known fixed size.
                    self.byte_offsets_array = None

        self.dims = dims

    def __len__(self):
        return len(self.dims)

    def __repr__(self):
        if self.is_uniform:
            return f"TorchIdx(shape={self.shape}, dtype={self.dtype}, source={self.source})"
        else:
            return f"TorchIdx(N={len(self)}, dtype={self.dtype}, is_uniform=False, source={self.source})"

    def byte_offset(self, i: int) -> int:
        if self.byte_offsets_array is not None:
            return self.byte_offsets_array[i]

        shape = self.shape
        assert shape is not None
        num_items = shape[0]
        if i == -1:
            i = num_items

        assert i <= num_items
        return math.prod(shape[1:]) * self.itemsize * i

    def byte_offsets(self, start: int, end: int) -> np.ndarray:
        if self.byte_offsets_array is not None:
            return self.byte_offsets_array[start:end]

        shape = self.shape
        assert shape is not None
        num_items = shape[0]

        if end == -1:
            end = num_items

        assert start <= num_items
        # end can be one past the end of the array since it's non-inclusive:
        assert end <= (num_items + 1)
        return (
            math.prod(shape[1:])
            * self.itemsize
            * np.arange(start, end, dtype=np.ulonglong)
        )

    def read_bin(self):
        """Read the corresponding .bin file and return it as an np.ndarray."""
        s0 = self.byte_offset(0)
        s1 = self.byte_offset(-1)
        buf = fs.read_bytes(self.bin_path, s0, s1)
        return self.view_buffer(buf)

    def view_buffer(self, buffer: Buffer) -> Union[np.ndarray, ISeq[RawField]]:
        """
        View raw bytes of our .torch.bin file as output type.

        Parameters
        ----------
        buffer
            Raw bytes from our .torch.bin file.

        index
            Which frame `buffer` represents. If `slice(None)` (the default), it
            represents the whole file.

        Returns a single object representing the whole file. It will be an
        ndarray if our data is uniform, a lazy sequence of ndarrays otherwise.
        """
        self._check_buf_size(0, -1, buffer)
        shape = self.shape
        if shape is None:
            return map_dataset(self.view_buffer_at, range(len(self)), buffer=buffer)
        else:
            # uniform shape: return a single big tensor. Note that this still just
            # creates a view on the given buffer; if it is a mmapped file, this will
            # not cause it to be materialized in memory until the data is actually
            # accessed.
            return np.ndarray(shape=shape, dtype=self.dtype, buffer=buffer)

    def view_buffer_at(self, index: int, buffer: Buffer) -> RawField:
        """
        Given a buffer over the whole .bin file, return view of just one frame

        Equivalent to `self.view_buffer(buffer)[index]`
        """
        # self._check_buf_size(0, -1, buffer)
        fstart, fstop = self.byte_offsets(index, index + 2)
        return self.view_frame(index, buffer[fstart:fstop])

    def view_frame(self, index: int, buffer: Buffer) -> RawField:
        """
        Given a buffer over a single frame's data, return view of that frame

        Parameters
        ----------
        index
            Which frame `buffer` represents.

        buffer
            Raw bytes from our .torch.bin file for a single frame
        """
        self._check_buf_size(index, index + 1, buffer)
        if self._bin_format == BinFormat.MSGPACK:
            return msgpack.unpackb(buffer)
        else:
            return np.ndarray(shape=self.dims[index], dtype=self.dtype, buffer=buffer)

    def _check_buf_size(self, istart, istop, buffer):
        bufsz = memoryview(buffer).nbytes
        binsz = self.byte_offset(istop) - self.byte_offset(istart)
        if binsz != bufsz:
            raise ValueError(
                f"expected {binsz} bytes but got {bufsz} for {self.bin_path}[{istart}:{istop}]"
            )

    def data_size_bytes(self):
        """Return the total size in bytes of our .torch.bin data."""
        return self.byte_offset(-1) - self.byte_offset(0)

    def frame_size_bytes(self) -> int:
        """Return the number of bytes in a single frame."""
        if not self.is_uniform:
            raise ValueError(
                "frame_size_bytes requires that all frames are the same size"
            )
        return self.byte_offset(1) - self.byte_offset(0)

    def item_shape(self, i: Optional[int] = None) -> Tuple[int, ...]:
        """
        Return the shape of the i-th item of this dataset.

        If `i` is `None` (the default), then all items must have the same shape,
        and that shape is returned.
        """
        if i is None:
            s = self.shape
            if s is None:
                raise ValueError(f"Dataset does not have uniform shape: {self.source}")
            return s[1:]
        return tuple(self.dims[i])

    def _idx_buf_to_int64(self, buffer: Buffer) -> np.ndarray:
        # convert bytes, ndarray, etc. to uniform interface
        buffer = memoryview(buffer)
        if buffer.ndim != 1:
            raise ValueError(
                f".torch.idx data has invalid shape: {buffer.shape}; require ndim=1"
            )
        if buffer.format == "B":
            # byte or uint8 data
            if len(buffer) % 8 != 0:
                raise ValueError(
                    f".torch.idx data has invalid length: {len(buffer)}%8 != 0"
                )
        elif buffer.format not in ("q", "l") or buffer.itemsize != 8:
            # argh, int64 format code is different on Windows vs. linux!
            raise ValueError(
                f".torch.idx data has invalid format {buffer.format}: expected 'B' (bytes)  or 'q' or 'l' (int64)"
            )
        return np.ndarray(shape=(-1,), dtype=np.int64, buffer=buffer)


# Return an array that uses stride tricks to save space.
#
# If every row of the input array is identical, return an array
# with just one copy of the row, using stride=0 to make it appear
# equal to the input array.
#
# If this isn't possible, the input array will be returned as-is.
#
# This saves a bit of memory for .torch.idx files, where we store a
# large array of input dims that are usually all identical, but not
# strictly required to be.
def _pack_reps(a: np.ndarray) -> np.ndarray:
    if (a == a[0]).all():
        return np.broadcast_to(a[0].copy(), a.shape)
    else:
        return a


_idx_to_np_type: Dict[int, Tuple[str, BinFormat]] = {
    1: ("uint8", BinFormat.TENSOR),
    2: ("int8", BinFormat.TENSOR),
    3: ("int16", BinFormat.TENSOR),
    4: ("int32", BinFormat.TENSOR),
    5: ("int64", BinFormat.TENSOR),
    6: ("float32", BinFormat.TENSOR),
    7: ("float64", BinFormat.TENSOR),
    8: ("object", BinFormat.MSGPACK),  # msgpacked object
}


_np_to_idx_type: Dict[Tuple[str, BinFormat], int] = {
    s: n for n, s in _idx_to_np_type.items()
}

OBJECT_DTYPE = np.dtype("object")
