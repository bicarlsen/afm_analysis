import numpy as np


def xy_to_coords(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Convert list of x and y values to grid of coordinate points.

    Args:
        x (np.ndarray): x values.
        y (np.ndarray): y values.

    Returns:
        np.ndarray: Coordinate grid indexed by (x, y).
    """
    return np.stack(np.meshgrid(y, x, indexing="ij"), axis=-1)
