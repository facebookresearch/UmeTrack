"""
Helpers for working with torch datasets.
"""
from __future__ import annotations

from typing import Callable, Iterable, overload, TypeVar

import torch
from torch.utils.data import Dataset, IterableDataset
from typing_extensions import Protocol


T = TypeVar("T")
U = TypeVar("U")


class Indexable(Protocol[T]):
    def __getitem__(self, i: int) -> T:
        ...

    def __len__(self) -> int:
        ...


# ISeq is like typing.Sequence, except it's a protocol rather than
# a base class, and only requires indexing, iteration, and len().
class ISeq(Indexable[T], Iterable[T], Protocol[T]):
    ...


class IndexedMap(Dataset[T], ISeq[T]):
    """
    Return type of `map_dataset` on indexed dataset.

    Implements lazy map of an arbitrary function over a dataset.

    Unlike the built-in `map` function, MapSeq defines `[]` and `len` if the
    underlying sequence does. It also forwards `*args` and `**kwargs` to the
    function, to avoid the need for `functools.partial` or (non-picklable)
    lambdas.

    Parameters
    ----------
    func
        A function that will be called for tuple of items
        `func` will be called as `func(item0, item1, ..., **kwargs)`.

    *datasets
        One or more indexable sequences

    **kwargs
        Extra keyword args to pass to func.
    """

    def __init__(self, func, *datasets, **kwargs) -> None:
        self.datasets = datasets
        self.func = func
        self.kwargs = kwargs

    def __iter__(self):
        for items in zip(*self.datasets):
            yield self.func(*items, **self.kwargs)

    @overload
    def __getitem__(self, __i: int) -> T:
        ...

    @overload
    def __getitem__(self: IndexedMap, __s: slice) -> IndexedMap:
        ...

    def __getitem__(self, i: slice) -> IndexedMap[T]:
        if isinstance(i, slice):
            # slice access, so mapped_view[10:20] returns another
            # mapped view.
            return type(self)(self.func, *(d[i] for d in self.datasets), **self.kwargs)
        else:
            # simple indexing
            return self.func(*(d[i] for d in self.datasets), **self.kwargs)

    def __repr__(self) -> str:
        # define repr for more useful logging/debug output
        fname = getattr(self.func, "__name__", str(self.func))
        return f"{self.__class__.__name__}(func={fname})"

    def __len__(self) -> int:
        # Define len() iff the first underlying dataset does. We do this as a
        # property so `hasattr()` works correctly.
        return self.datasets[0].__len__()

    @property
    def sampler(self):
        # Define .sampler iff the underlying dataset does. We do this as a
        # property so `hasattr()` works correctly.
        return self.datasets[0].sampler


class IterableMap(IndexedMap[T], IterableDataset[T]):
    """
    Return type of `map_dataset` on IterableDataset.
    """


def map_dataset(func, *datasets, **kwargs):
    """
    Return a dataset with `func(item0, item1, ..., **kwargs)` applied itemwise.

    Keywords arguments, if any, will be stored and passed along to func
    on each call.

    This is like the builtin map(), except for random access sequences (or
    datasets) instead of just iterables, and you can pass multiple datasets
    to get the effect of zip().

    If `dataset` is an IterableDataset, the result of `map_dataset` will
    be as well.

    Parameters
    ----------
    func
        A function to apply to each element of the input sequence

    dataset
        One or more indexed or iterable input datasets.

        The first input dataset will determine the output length, and whether
        the output is an IterableDataset or not.

    kwargs
        Extra keyword arguments for func.

    Example
    -------

    >>> def f(x, y, z):
    ...     print(f"called f({x}, {y}, z={z})")
    ...     return x + y + z
    >>> s = map_dataset(f, [1, 2, 3], [10, 20, 30], z=100)
    >>> assert(s[1]==122)
    called f(2, 20, z=100)
    >>> assert(list(s)==[111, 122, 133])
    called f(1, 10, z=100)
    called f(2, 20, z=100)
    called f(3, 30, z=100)
    """
    # pytorch uses `isinstance(x, IterableDataset)` to distinguish iterable vs.
    # indexed datasets, so we need to do the same here to make sure our return
    # type matches.
    assert isinstance(datasets[0], IterableDataset)
    return IterableMap(func, *datasets, **kwargs)
