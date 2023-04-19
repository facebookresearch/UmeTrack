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

    The enum values ("training", "testing") correspond to
    directory names of exported TDG files.
    """

    TRAIN = "training"
    TEST = "testing"
