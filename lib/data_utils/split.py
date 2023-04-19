# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Helpers for handling data splits (training / testing).
"""
import enum


class Split(enum.Enum):
    """
    Simple helper for working with training / testing splits
    of a dataset.

    Various utilities and dataset-loading functions take or return
    the splits of a dataset as a `Dict[Split, Dataset]`.
    """

    TRAIN = "training"
    TEST = "testing"
