from __future__ import annotations
from . import operations
import numpy as np
import scipy as sp
from typing import TYPE_CHECKING, Protocol, Any

if TYPE_CHECKING:
    from . import image


class Calculation(Protocol):
    def __call__(self, channel: image.Channel, *args, **kwargs) -> Any: ...


def rms(data: np.ndarray) -> float:
    """Root mean squared (RMS).

    Args:
        data (np.ndarray): Input data.

    Returns:
        float: Root mean squared.
    """
    return np.sqrt(np.mean(np.square(data)))


def roughness_avg(channel: image.Channel, ignore_nan: bool = True) -> float:
    """Average roughness.
    Sum of the absolute difference from the mean plane.

    Args:
        channel (image.Channel): Channel.
        ignore_nan (bool): Ignore nan values. Defaults to True.

    Returns:
        float: Average roughness.
    """
    data = channel.data
    if ignore_nan:
        raise NotImplementedError("todo")
        data = data[~np.isnan(data)]

    mean = channel.copy()
    mean.apply(operations.plane_level)
    return np.sum(np.abs(data - mean.data)) / data.size


def roughness_rms(channel: image.Channel, ignore_nan: bool = True) -> float:
    """RMS roughness.
    Root mean square of difference from the mean plane.

    Args:
        channel (image.Channel): Channel.
        ignore_nan (bool): Ignore nan values. Defaults to True.

    Returns:
        float: RMS roughness.
    """
    data = channel.data
    if ignore_nan:
        raise NotImplementedError("todo")
        data = data[~np.isnan(data)]

    mean = channel.copy()
    mean.apply(operations.plane_level)
    return rms(data - mean.data) / data.size


def histogram(channel: image.Channel) -> tuple[np.ndarray, np.ndarray]:
    """Histogram data.
    Bin edges are computed with the Freedman Diaconis Estimator.

    Args:
        channel (image.Channel): Channel.

    Returns:
        tuple[np.ndarray, np.ndarray]: (counts, bin edges)
    """
    return np.histogram(channel.data.flatten(), bins="fd")


def multi_gaussian(x: np.ndarray, params: tuple[float, ...]) -> np.ndarray:
    """Sum of multiple Gaussians.
    Number of Gaussians is inferred from the number of passed parameters.
    Each parameter triple is considered an additional Gaussian.
    e.g. If three parameters (i.e. one parameter triple) are passed, a single Gaussian will be calculated,
    if six patameters (i.e. two parameter triples) are passed, a double Gaussian will be calculated, etc.

    Args:
        x (np.ndarray): Input values.
        params (tuple[float, ...]): Parameters of (c, mu, sigma) triples.

    Raises:
        ValueError: Parameters is an invalid length.

    Returns:
        np.ndarray: Output values.
    """
    if len(params) % 3 != 0:
        raise ValueError("params must be a multiple of 3")
    n_triples = int(len(params) / 3)

    res = np.zeros_like(x)
    for idx in range(n_triples):
        start = idx * 3
        end = (idx + 1) * 3
        (c, mu, sigma) = params[start:end]
        res += c * sp.stats.norm.pdf(x, loc=mu, scale=sigma)

    return res


def multi_gaussian_residual(
    params: tuple[float, ...],
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Residuals of a multi gaussian with the given params and data values.

    Args:
        params (tuple[float, ...]): (c, mu, sigma, ...)
        x (np.ndarray): Input values for the double gaussian.
        y (np.ndarray): Data values.

    Returns:
        np.ndarray: Residuals between the double gaussian fit and y.
    """
    if len(params) % 3 != 0:
        raise ValueError("params must be a multiple of 3")

    fit = multi_gaussian(x, params)
    return fit - y
