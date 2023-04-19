"""
Utilities for working generically with dicts, dataclasses, and
NamedTuples.
"""
import dataclasses
from enum import Enum
from typing import Any, Mapping

import numpy as np
import torch


def is_dictlike(obj: Any) -> bool:
    """
    Returns true if the object is a dataclass, NamedTuple, or Mapping.
    """
    return (
        dataclasses.is_dataclass(obj)
        or hasattr(obj, "_asdict")
        or isinstance(obj, Mapping)
    )


def field_names(ty):
    """
    Return list of field names of the given type.

    `ty` is a NamedTuple or dataclass type (not value).
    """
    if dataclasses.is_dataclass(ty):
        return [f.name for f in dataclasses.fields(ty)]
    else:
        return ty._fields


def asdict(obj):
    """
    Return a view or copy of "obj" as a dict.

    `obj` is any object for which is_dictlike() is True, i.e. a Mapping,
    NamedTuple, or dataclass. If it's an actual Mapping it will be
    returned as-is.
    """
    if dataclasses.is_dataclass(obj):
        return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    try:
        # namedtuple
        return obj._asdict()
    except AttributeError:
        pass

    if isinstance(obj, Mapping):
        return obj

    raise TypeError("asdict() requires a Mapping, dataclass, or NamedTuple")


def asdict_rec(obj):
    """
    Recursively convert dict-like types to actual dicts.

    This is mostly useful for data that is going to be pickled or
    otherwise saved out, where you want it to be loadable without
    dependencies on locally defined types.

    dataclasses, NamedTuples, or other Mapping types in obj will be
    recursively converted to plain `dict` types. Tuples and lists will
    be recursed into. Anything else will be left as-is.

    Example
    -------

    >>> class N(NamedTuple):
    ...   x: Any
    ...   y: Any
    >>> asdict_rec( N(x=(1,2), y={'a': 4, 'b': N(5,6)}) )
    {
        'x': (1, 2),
        'y': {
            'a': 4,
            'b': {'x': 5, 'y': 6}
        }
    }
    """
    if dataclasses.is_dataclass(obj):
        return {
            f.name: asdict_rec(getattr(obj, f.name)) for f in dataclasses.fields(obj)
        }
    try:
        # namedtuple
        obj = obj._asdict()
    except AttributeError:
        pass

    if isinstance(obj, Mapping):
        return {k: asdict_rec(v) for (k, v) in obj.items()}

    if isinstance(obj, tuple):
        return tuple(asdict_rec(v) for v in obj)

    if isinstance(obj, list):
        return [asdict_rec(v) for v in obj]

    return obj


def map_fields(func, obj, only_type=object):
    """
    map 'func' recursively over nested collection types.

    >>> map_fields(lambda x: x * 2,
    ...            {'a': 1, 'b': {'x': 2, 'y': 3}})
    {'a': 2, 'b': {'x': 4, 'y': 6}}

    E.g. to detach all tensors in a network output frame:

        frame = map_fields(torch.detach, frame, torch.Tensor)

    The optional 'only_type' parameter only calls `func` for values where
    isinstance(value, only_type) returns True. Other values are returned
    as-is.
    """
    if is_dictlike(obj):
        ty = type(obj)
        if isinstance(obj, Mapping):
            return ty((k, map_fields(func, v, only_type)) for (k, v) in obj.items())
        else:
            # NamedTuple or dataclass
            return ty(
                **{k: map_fields(func, v, only_type) for (k, v) in asdict(obj).items()}
            )
    elif isinstance(obj, tuple):
        return tuple(map_fields(func, v, only_type) for v in obj)
    elif isinstance(obj, list):
        return [map_fields(func, v, only_type) for v in obj]
    elif isinstance(obj, only_type):
        return func(obj)
    else:
        return obj


def to_device(obj, device):
    """
    Call `t.to(device)` for all tensors in obj
    """
    return map_fields(lambda t: t.to(device), obj, only_type=torch.Tensor)


def collate(batch, device=None):
    """
    Convert a list of N items to a single item containing size-N arrays.

    This is a replacement for Torch's default collator. It's a little
    simpler, and it handles dataclasses as well as NamedTuples.

    Parameters
    ----------
    batch
        List of items to combine. All items in the list should be of uniform
        type and shape.

        dicts, tuples, namedtuples, and dataclasses will be recursively collated
        per-field. Numpy arrays and torch tensors will be stacked along a
        new outermost dimension. Any other types are simply collected into
        a list.

        `None` is treated specially: if all items in the batch are `None`, the
        batched value is also just `None` instead of a list. It is an error if
        some items are `None` and others aren't.

    device
        If given, any torch tensors in the result will be moved to this
        device. Note that ndarrays are not converted to torch tensors;
        do that in your transform function if desired.

    Examples
    --------
    >>> collate([1, 2, 3, 4])
    [1, 2, 3, 4]

    >>> collate([
    ...     {'x': 1, 'y': array([0,1,2])},
    ...     {'x': 3, 'y': array([10,11,12])}
    ... ])
    {'x': [1, 3], 'y': array([[0,1,2],[10,11,12]])}
    """

    def stack_fn(t_list):
        value0 = t_list[0]
        if isinstance(value0, (np.ndarray, np.generic)):
            return np.stack(t_list)
        if isinstance(value0, torch.Tensor):
            t = torch.stack(t_list)
            if device is not None:
                t = t.to(device)
            return t

        raise TypeError(f"Can't stack tensors: unknown type {type(value0)} found")

    return group(batch, stack_fn)


def group(batch, group_fn):
    """
    Group a list of N items to a single item containing size-N arrays.
    This function is more generic than collate: collate will stack
    the tensor fields whereas this function also enables arbitrary
    grouping functions (i.e. torch.cat).
    """
    ty = type(batch[0])
    value0 = batch[0]
    if isinstance(value0, (np.ndarray, np.generic)):
        return group_fn(batch)

    if isinstance(value0, torch.Tensor):
        return group_fn(batch)

    if is_dictlike(value0):
        # Mapping, dataclass, or NamedTuple
        batch = [asdict(x) for x in batch]
        d = {k: group([v[k] for v in batch], group_fn) for k in batch[0]}
        return ty(**d)

    if isinstance(value0, tuple):
        return tuple(group([v[i] for v in batch], group_fn) for i in range(len(value0)))

    if isinstance(value0, list):
        return [group([v[i] for v in batch], group_fn) for i in range(len(value0))]

    if value0 is None:
        if any(v is not None for v in batch):
            raise TypeError("Some items are None and others are not")
        return None

    # other types are just returned as a list of items
    if any(v is None for v in batch):
        raise TypeError("Some items are None and others are not")
    return batch
