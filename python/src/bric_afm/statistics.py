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


def roughness_avg(data: np.ndarray) -> float:
    """Average roughness.
    Sum of the absolute difference from the mean.

    Args:
        data (np.ndarray): DatInput data.

    Returns:
        float: Average roughness.
    """
    return np.sum(np.abs(data - data.mean())) / data.size


def roughness_rms(data: np.ndarray) -> float:
    """RMS roughness.
    Root mean square of difference from the mean.

    Args:
        data (np.ndarray): Input data.

    Returns:
        float: RMS roughness.
    """
    return np.sqrt(np.mean(np.square(data - data.mean()))) / data.size
