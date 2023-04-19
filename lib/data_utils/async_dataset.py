# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import hashlib
import itertools
import logging
import math
import tempfile
import time
from contextlib import closing
from datetime import timedelta
from functools import partial

from multiprocessing import pool, shared_memory, sharedctypes
from multiprocessing.dummy import Pool
from typing import (
    Any,
    AsyncIterable,
    Awaitable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import msgpack
import numpy as np
import torch
from lib.data_utils import fs, nested_async
from .idxbinfile import OBJECT_DTYPE, RawField, TorchIdx
from .split import Split

from torch.utils.data import Dataset, IterableDataset

logger = logging.getLogger(__name__)

S = TypeVar("S")
T = TypeVar("T")


# Alias for dataset of awaitables
AsyncDataset = Dataset[Awaitable[T]]

def find_torchdata_folders(root: str) -> Dict[str, List[str]]:
    """Find all torchdata folders under the given root

    Return all torchdata folders under the given root, as a dict mapping folder
    paths to lists of contained filenames.

    Args:
        root (str): root of the dataset (does not have to be the leaf)

    Returns:
        Dict[str, List[str]]: leaf folder -> files in folder mapping
    """
    logger.info(f"Looking for torchdata folders under {root}")

    original_root = root
    # Return a Dict of "leaf folder path" -> list of filename in the folder.
    out: Dict[str, List[str]] = {}
    split_names = {s.value for s in Split}
    for path, _, files in fs.walk(root, topdown=False):
        if fs.basename(path) in split_names:
            for f in files:
                if f.endswith(".torch.bin"):
                    out[path] = files
                    break

    return out


def filter_torchdata_folders(
    folders: Mapping[str, List[str]],
    fields: Iterable[str],
) -> Dict[str, List[str]]:
    """
    Filter a collection of folders for a particular dataset.

    Folders are filtered to those matching some desired set of fields.
    Each field corresponds to certain filenames, e.g. if `spam` is one
    of the input fields, then files `spam.torch.bin` and
    `spam.torch.idx` are required.

    Parameters
    ----------
    folders
        mapping from folder path to list of possible dataset files in
        that folder

    fields
        list of field names

    Returns
    -------
    A dict of the same shape as the input, but

    - Folders missing any required fields are removed.
    """
    required_fields = set(fields)
    extensions = ".torch.idx", ".torch.bin"
    all_filenames = {f + ext for f in required_fields for ext in extensions}
    required_filenames = {f + ext for f in required_fields for ext in extensions}

    out = {}
    for (folder, found) in folders.items():
        found = set(found)
        missing = required_filenames - found
        if missing:
            logger.debug(
                f"Skipping torchdata folder {folder} because some dataset files were missing: {missing}"
            )
        else:
            logger.debug(f"Adding torchdata folder {folder}")
            out[folder] = found & all_filenames
    return out


class InMemoryTorchBin(AsyncDataset[np.ndarray]):
    """
    Dataset that pre-loads a single .torch.bin file into memory.
    """

    def __init__(self, idx_file: TorchIdx) -> None:
        self.idx_file = idx_file
        bin_buffer = fs.read_bytes(str(self.idx_file.bin_path))

        self._bin_data = shared_memory.SharedMemory(create=True, size=len(bin_buffer))
        self._bin_data.buf[:] = bin_buffer[:]

    def __len__(self):
        return len(self.idx_file)

    async def __getitem__(self, i: int):
        out = self.idx_file.view_buffer_at(i, self._bin_data.buf)
        return out

    def data_size_bytes(self) -> int:
        return self.idx_file.data_size_bytes()


class AsyncTorchBin(AsyncDataset[np.ndarray], AsyncIterable[np.ndarray]):
    """
    Dataset with async access to a single .bin file.

    Parameters
    ----------

    idx_file
        A `TorchIdx` whose bin file we are loading.
    """

    def __init__(self, idx_file: TorchIdx) -> None:
        self.idx_file = idx_file

    def __repr__(self):
        return f"SingleIdxBinDataset(len={len(self)}, path={[self.idx_file.source]})"

    def __len__(self) -> int:
        return len(self.idx_file)

    async def __getitem__(self, i: int):
        """
        Implement x[i]

        Returns an awaitable to the actual element.
        """
        idx_file = self.idx_file
        start, stop = idx_file.byte_offsets(i, i + 2)
        buf = await fs.aread_bytes(
            idx_file.bin_path,
            start,
            stop,
        )
        return idx_file.view_frame(i, buf)

    def __aiter__(self):
        return self

    async def __anext__(self):
        """
        We define __aiter__ and __anext__ to support asynchronous iteration,
        though normally you would use random access instead.

            async for x in dataset:
                sum += np.sum(x)
        """
        for i in range(len(self)):
            yield await self[i]

    def data_size_bytes(self) -> int:
        """Return the total size in bytes of our .bin data."""
        return self.idx_file.data_size_bytes()


def get_field_dataset(
    folder: str,
    cache_in_mem_fields: Optional[Iterable[str]],
    field: str,
) -> Tuple[str, Optional[AsyncDataset], int]:
    """
    Load a single field dataset for a specific idx/bin pair.

    Parameters
    ----------
    folder
        see `SingleFolderAsyncDataset()` documentation.

    cache_in_mem_fields
        see `SingleFolderAsyncDataset()` documentation.

    field
        The specific field name to load (i.e. for `spam.torch.idx`/`spam.torch.bin`,
        the field name is `spam`).

    Returns
    -------
    A tuple of
    (field name, optional dataset, number of loaded samples)
    """

    if fs.exists(fs.join(folder, (field + ".torch.idx"))):
        idx_file = TorchIdx(fs.join(folder, (field + ".torch.idx")))
        n = len(idx_file)
        avg_frame_size = idx_file.data_size_bytes() / n
        if cache_in_mem_fields and field in cache_in_mem_fields:
            fieldtype = InMemoryTorchBin
        else:
            fieldtype = AsyncTorchBin

        return field, fieldtype(idx_file), n
    return field, None, 0


class SingleFolderAsyncDataset(AsyncDataset[Dict[str, np.ndarray]]):
    """
    Async dataset that encompasses a single folder path.

    Parameters
    ----------
    folder
        Which folder path this dataset covers.

    fields
        see `find_dataset()` documentation.

    cache_in_mem_fields
        see `find_dataset()` documentation.

    Attributes
    ----------
    n
        How many frames this dataset contains.
    """

    def __init__(
        self,
        folder: str,
        fields: Iterable[str],
        *,
        cache_in_mem_fields: Optional[Iterable[str]] = None,
    ) -> None:
        self._fields = {}
        self.n = None
        field_dataset_iterator = (
            get_field_dataset(
                folder,
                cache_in_mem_fields,
                field,
            )
            for field in fields
        )

        for field, field_dataset, field_n in field_dataset_iterator:
            if field_dataset is not None:
                if self.n is not None and self.n != field_n:
                    raise ValueError(f"folder has mismatched field sizes: {field}")
                self.n = field_n
                if field_dataset:
                    self._fields[field] = field_dataset
            else:
                raise ValueError(
                    f"required field {field} not found in folder {folder}, which only has {fs.listdir(folder)}"
                )

    def __repr__(self):
        return f"SingleFolderDataset(N={len(self)}, fields={self._fields})"

    def __len__(self) -> int:
        return self.n

    async def __getitem__(self, idx: int):
        return {
            name: await data[idx]
            for (name, data) in self._fields.items()
        }

    def data_size_bytes(self) -> int:
        """Return the total size in bytes of our .bin data."""
        n = 0
        for f in self._fields.values():
            n += f.data_size_bytes()
        return n

    def cacheable_size_bytes(self) -> int:
        """Return the total size in bytes this dataset might occupy in memory."""
        n = 0
        for f in self._fields.values():
            if isinstance(f, InMemoryTorchBin):
                n += f.data_size_bytes()
        return n


def find_dataset(
    roots: Union[str, List[str]],
    fields: Iterable[str],
    splits: Iterable[Split] = Split,
    *,
    cache_in_mem_fields: Optional[Iterable[str]] = None,
    num_workers: int = 5,
) -> Dict[Split, Dataset[Awaitable[Dict[str, np.ndarray]]]]:
    """
    Return dataset of all .torch.bin files under some root dir.

    Data layout is assumed to look like::

        {root}/.../{split}/{field_name}.torch.bin
        {root}/.../{split}/{field_name}.torch.idx

    Where {split} is one of "training", "testing", "heldOut".

    Arguments
    =========
    roots
        list of top level root folders. Returned data set will contain all
        the data from this list.

    fields
        list of fields (corresponding to filenames) to include in result. If a
        name "f" is in fields, then "f.torch.idx" and "f.torch.bin" must be
        found in in a folder for that folder to be included.

    splits
        list of splits to include, from {Split.TRAIN, Split.TEST,
        Split.HELDOUT}. Default is all three.

    cache_in_mem_fields
        Fields that will be pre-loaded and cached in memory. These data sets
        will be created using InMemoryTorchBin.

    num_workers
        number of workers to initialize data sets
    """
    # scan directories for valid folders
    if isinstance(roots, str):
        roots = [roots]

    folders = {}
    for root in roots:
        data_folders = find_torchdata_folders(root)
        print(f"Found {len(data_folders)} data folders in {root}")
        folders.update(data_folders)

    folders = filter_torchdata_folders(folders, fields)

    # create a single-folder dataset from each .torch.idx file
    per_folder = {s.value: [] for s in splits}
    cacheable_size_bytes = {s.value: 0 for s in splits}
    # Only keep folders that are requested according to 'splits'
    filtered_folders = [fd for fd in folders if fs.basename(fd) in per_folder]

    datasets = []
    for f in filtered_folders:
        dataset = SingleFolderAsyncDataset(
            folder=f,
            fields=fields,
            cache_in_mem_fields=cache_in_mem_fields,
        )
        datasets.append((f, dataset))

    for folder, dset in datasets:
        splitname = fs.basename(folder)
        per_folder[splitname].append(dset)
        cacheable_size_bytes[splitname] += dset.cacheable_size_bytes()

    # concatenate them into one big dataset per split
    catted = {}
    if len(roots) == 1:
        logger.info(f"Under {roots[0]}:")
    for s in splits:
        ds = per_folder[s.value]
        if len(ds) == 0:
            continue

        sz = cacheable_size_bytes[s.value]
        logger.info(f"{s}: found {len(ds)} folders")
        catted[s] = torch.utils.data.ConcatDataset(ds)
        logger.info(
            f"{s}: found total of {len(catted[s])} frames, resident size up to {sz/1e6:.3}MB"
        )

    return catted


def subsample(
    dataset: Dataset[T], *, portion: float = 1.0, max_len: Optional[int] = None
) -> Dataset[T]:
    """
    View a dataset using a randomly sampled subset of frames.

    Parameters
    ---------
    portion
        a fraction from 0 to 1 of data to include,
    max_len
        a count of frames to sample

    If both `portion` and `max_len` are given, the smaller limit will be
    used.

    >>> len(my_dataset)
    100000
    >>> p = partial_dataset(my_dataset, portion=0.05)
    >>> len(p)
    5000
    >>> p = partial_dataset(my_dataset, max_len=1000)
    >>> len(p)
    1000
    """
    n = round(len(dataset) * portion)
    if max_len is not None:
        n = min(n, max_len)
    if n >= len(dataset):
        return dataset

    indices = np.arange(len(dataset))

    # Use a constant random seed to make sure all workers see the same data so
    # to avoid data duplication when sampling.
    rng = np.random.default_rng(0)
    rng.shuffle(indices)
    indices = indices[:n].copy()
    indices.sort()
    return torch.utils.data.Subset(dataset, indices)


class Sampler(torch.utils.data.Sampler[int]):
    """
    Replacement for torch.utils.data.DistributedSampler.

    This class behaves like torch's `DistributedSampler`, except it will work
    whether created before or after starting up per-gpu processes, and whether
    iterated from a gpu process or an io worker. It will automatically assign
    each io_worker a different slice of samples.

    Parameters
    ==========
    dataset
        Dataset to sample. This is only needed for calling `len()`, so any sized
        object can be passed here. In particular, `range(N)` is fine.

    shuffle
        If true, each iteration through the sampler will yield indices in a
        different random order.

    seed
        Integer to use (together with epoch) as random number seed for
        deterministic shuffling.

    drop_last
        If the dataset size is not a multiple of the distributed world_size,
        extra samples will be repeated (if drop_last=False, the default) or
        dropped if drop_last=True.

        Note that this is separate from the loader dropping samples to make
        the last batch have an even size.

    distrib_info
        Note that you should only specify this if you want to bypass distrib.
        Otherwise you may run into weired behaviors during training.
    """

    dataset: Sized
    shuffle: bool
    seed: int
    epoch: int
    drop_last: bool
    _distrib_info: Optional[Tuple[int, int]]  # rank, world size

    def __init__(
        self,
        dataset: Union[Sized, "torch.utils.data.Dataset"],
        indices: Optional[Sequence[int]] = None,
        shuffle=True,
        seed=1,
        drop_last=False,
        distrib_info: Optional[Tuple[int, int]] = None,
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.indices = indices
        self._distrib_info = None
        if distrib_info:
            assert (
                self._distrib_info is None
            ), "Expect unspecified distrib_info: it has been obtained from env"
            self._distrib_info = distrib_info
        else:
            self._get_distrib_info(False)


    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        if self.indices is not None:
            indices = self.indices
        elif self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)  # type: ignore
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = range(len(self.dataset))

        # bump epoch for next iteration. This sampler may be copied to
        # persistent loader workers, so we can't rely on the problem to
        # increment it each epoch.
        self.epoch += 1

        # pad or drop samples so every node gets the same count
        rank, world_size = self._get_distrib_info(True)
        while True:
            r = len(indices) % world_size
            if r == 0:
                break
            if self.drop_last:
                indices = indices[:-r]
            else:
                indices = torch.as_tensor(indices)
                indices = torch.cat((indices, indices[: world_size - r]))

        # subsample for node
        indices = indices[rank::world_size]

        # subsample for io_worker
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            indices = indices[worker_info.id :: worker_info.num_workers]

        return iter(indices)

    def __len__(self):
        n = len(self.dataset)
        inf = self._get_distrib_info(False)
        if inf is None:
            return n
        _, world_size = inf

        # round to world size, rounding up or down
        if self.drop_last:
            n = n // world_size
        else:
            n = -(-n // world_size)

        # if we're on a worker node, just count samples for this worker
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return n
        return len(range(worker_info.id, n, worker_info.num_workers))

    def __getstate__(self):
        # Iteration our proper subset requires knowing both the distributed
        # worker (rank, world_size) and the io worker (id, num_workers).
        #
        # However it's possible to create the sampler before initializing
        # distributed info (e.g. in main, before creating distributed worker
        # processes), copy it out to the workers, and then have the loader copy
        # it again out to the loader workers, so when we iterate we no longer
        # have access to the info we need.
        #
        # So we grab the state we need when the sampler is pickled.
        self._get_distrib_info(False)
        return self.__dict__

    def _get_distrib_info(self, required: bool):
        if self._distrib_info is None:
            env = get_env() if required else _env
            if env:
                self._distrib_info = (env.rank, env.world_size)
        return self._distrib_info


class AsyncToIterableDataset(IterableDataset[T]):
    """
    Convert `Dataset[Awaitable[T]]` to `IterableDataset[T]`

    This conversion allows the dataset to prefetch items on iteration, since
    (unlike with a random-access dataset) it can know the iteration order.

    This in turn requires that the `Sampler` be provided here, rather than when
    the TorchDataLoader is constructed.

    Each iteration through the dataset re-shuffles the samples.

    Parameters
    ----------
    dataset
        indexable dataset

    sampler
        Iterable over sample indices. If provided, it must be specifically a
        :class:`distrib.Sampler`, not an arbitrary sampler. An arbitrary sampler
        would not work correctly if the loader has `num_workers > 1`, because
        torch assumes there is only one sampler per gpu process, not one per io
        worker.

    shuffle
        Whether to use a shuffling or sequential sampler. Only needed if
        `sampler=None`.

    max_prefetch
        The maximum number of items to prefetch per io_worker. This prefetching
        is the main purpose of the whole `Dataset[Awaitable]` system: for
        storage with high bandwidth but high latency, you may need a large
        number of reads in flight to cover latency.

        Setting a large number here should be harmless, as the prefetcher will
        ramp up to avoid sending out more reads than necessary, except that if
        your items are very large it may consume a lot of memory.

    Attributes
    ----------
    dataset
        The underlying Dataset[Awaitable[T]]

    sampler
        The associated sampler object. Exposed here so you can call
        `set_epoch()` on it to set the random number seed. (Note that the epoch
        is automatically incremented on each iteration though the sampler, so
        `set_epoch` is only needed at initialization.)
    """

    def __init__(
        self,
        dataset: Dataset[Awaitable[T]],
        sampler: Optional[Sampler] = None,
        shuffle: bool = False,
        max_prefetch=100,
    ):
        if sampler is None:
            n = len(dataset)
            sampler = Sampler(range(n), shuffle=shuffle)
        elif shuffle:
            raise ValueError("either sampler or shuffle should be specified, not both")
        self.dataset = dataset
        self.sampler = sampler
        self.max_prefetch = max_prefetch

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        samples = (self.dataset[i] for i in self.sampler)
        with closing(nested_async.prefetch_sequence(samples, self.max_prefetch)) as seq:
            yield from seq
