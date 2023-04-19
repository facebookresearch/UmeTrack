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
from lib.data_utils import bundles


MAX_LANDMARK_ERROR_MM = 50
PCK_THRESHOLDS = np.linspace(0, MAX_LANDMARK_ERROR_MM, 101)


def _safe_div(x, y, eps: float = 1e-6, default_val: int = 0):
    assert x.shape == y.shape
    if np.isscalar(x):
        if y < eps:
            return default_val
        else:
            return x / y

    z = x / y
    z[y < eps] = default_val
    return z


def _PCK_curve(
    errors: np.ndarray, mask: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    pcks = []
    for thresh in thresholds:
        le_threh = errors <= thresh
        pck = _safe_div((le_threh * mask).sum(axis=-1), mask.sum(axis=-1))
        pcks.append(pck)
    return np.stack(pcks).T


def PCK_curve(
    errors: np.ndarray,
    thresholds: np.ndarray,
    mask: Optional[np.ndarray] = None,
    axis: Optional[int] = None,
) -> np.ndarray:
    """
    Computes the total PCK curve. If the axis is given, computes one PCK curve
    for each element along the given axis.
    e.g., If `errors` is a 100 x 2 x 21 ndarray and axis = 1,
    return a 2 x len(thresholds) matrix.

    Parameters
    ----------
    errors : np.ndarray
        ndarray of errors
    thresholds : np.ndarray
        Thresholds for computing PCK
    mask : Optional[np.ndarray], optional
        Mask to filter invalid samples. Shape is the same as `errors`. If None,
        all error samples are assumed valid.
    axis : Optional[int], optional
        See summary, by default None

    Returns
    -------
    np.ndarray
        PCK curve(s)
    """
    if mask is None:
        mask = np.ones_like(errors)
    if axis is None:
        return _PCK_curve(errors.reshape(-1), mask.reshape(-1), thresholds)
    N = errors.shape[axis]
    return _PCK_curve(
        np.moveaxis(errors, axis, 0).reshape(N, -1),
        np.moveaxis(mask, axis, 0).reshape(N, -1),
        thresholds,
    )


def normalized_AUC(x: np.ndarray, y: np.ndarray, y_max: float = 1.0) -> np.ndarray:
    """
    Given curves sharing the same x-axis, computes normalized AUC for each of
    the curves.

    Parameters
    ----------
    x : np.ndarray
        X-axis ticks represented as a 1-D array
    y : np.ndarray
        An ndarray with the last dimension being the y-axis ticks
    y_max : float, optional
        Maximum of y value, by default 1.0

    Returns
    -------
    np.ndarray
        Normalized AUCs with shape = y.shape[:-1]
    """
    out_shape = y.shape[:-1]
    y = y.reshape(-1, y.shape[-1])
    auc = ((x[1:] - x[:-1]).reshape(1, -1) * ((y[..., 1:] + y[..., :-1]) * 0.5)).sum(
        axis=-1
    )
    max_area = (x[-1] - x[0]) * y_max
    return (auc / max_area).reshape(out_shape)
