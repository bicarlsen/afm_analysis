import numpy as np
from sklearn import linear_model
from typing import Callable

type Operation = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def min_to_zero(x: np.ndarray, y: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Zero data such that the minimum is 0.

    Args:
        data (np.ndarray): Data.

    Returns:
        np.ndarray: Modified data.
    """
    return data - data.min()


def plane_level(x: np.ndarray, y: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Level data over the average plane determined by linear regression.

    Args:
        x (np.ndarray): x index. Should be a 1D array.
        y (np.ndarray): y index. Should be a 1D array.
        data (np.ndarray): Data values. Should be a 2D array.

    Returns:
        np.ndarray: Leveled data values.
    """
    coords = np.stack(np.meshgrid(x, y), axis=-1)
    fit = linear_model.LinearRegression().fit(coords.reshape(-1, 2), data.flatten())
    plane = np.dot(coords, fit.coef_) + fit.intercept_
    return data - plane
