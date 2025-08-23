import numpy as np
from typing import Callable


def rms(data: np.ndarray) -> float:
    """Root mean squared (RMS).

    Args:
        data (np.ndarray): Input data.

    Returns:
        float: Root mean squared.
    """
    return np.sqrt(np.mean(np.square(data)))


def roughness_avg(data: np.ndarray, ignore_nan: bool = True) -> float:
    """Average roughness.
    Sum of the absolute difference from the mean.

    Args:
        data (np.ndarray): DatInput data.
        ignore_nan (bool): Ignore nan values. Defaults to True.

    Returns:
        float: Average roughness.
    """
    if ignore_nan:
        data = data[~np.isnan(data)]

    return np.sum(np.abs(data - data.mean())) / data.size


def roughness_rms(data: np.ndarray, ignore_nan: bool = True) -> float:
    """RMS roughness.
    Root mean square of difference from the mean.

    Args:
        data (np.ndarray): Input data.
        ignore_nan (bool): Ignore nan values. Defaults to True.

    Returns:
        float: RMS roughness.
    """
    if ignore_nan:
        data = data[~np.isnan(data)]

    return np.sqrt(np.mean(np.square(data - data.mean()))) / data.size
