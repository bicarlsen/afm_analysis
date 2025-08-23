from . import operations, utils
import numpy as np
import trimesh
from typing import Optional, Any


def create_mesh(
    x: np.ndarray,
    y: np.ndarray,
    data: np.ndarray,
    colors: Optional[np.ndarray] = None,
    colormap: str = "viridis",
    **kwargs
) -> trimesh.Trimesh:
    """Creates a mesh.

    Args:
        x (np.ndarray): x index. Should be 1D.
        y (np.ndarray): y index. Should be 1D.
        data (np.ndarray): Height values. Should be 2D.
        colors (Optional[np.ndarray]): Color values. Should be 2D.
        color_map (str): Name of a `matplotlib` colormap.
        Defaults to `viridis`.
        **kwargs: Passed to `trimesh.Trimesh`.

    Raises:
        ValueError: Dimension of `x`, `y`, or `data` is incorrect.

    Returns:
        trimesh.Trimesh: Mesh.

    Notes:
        + The underlying meshing library does best with values
        values that are ~1 (i.e. not 1e-9).
    """
    if x.ndim != 1:
        raise ValueError("invalid dimension for x")
    if y.ndim != 1:
        raise ValueError("invalid dimension for y")
    if data.ndim != 2:
        raise ValueError("invalid dimension for data")
    if colors is not None and colors.ndim != 2:
        raise ValueError("invalid dimension for colors")

    # vertex norms
    dx, dy = np.gradient(data, x, y)
    dz = np.full_like(data, -1)
    norms = np.stack([dx, dy, dz], -1)
    magnitudes = np.dstack([np.linalg.norm(norms, axis=-1)] * 3)
    norms = -norms / magnitudes

    coords = utils.xy_to_coords(x, y)
    vertices = np.insert(coords, 2, data - data.min(), axis=-1).reshape(-1, 3)

    ncols = y.shape[0]
    faces = np.array(
        [
            [
                [j * ncols + i, (j + 1) * ncols + i, j * ncols + i + 1],
                [(j + 1) * ncols + i, (j + 1) * ncols + i + 1, j * ncols + i + 1],
            ]
            for i in range(ncols - 1)
            for j in range(ncols - 1)
        ]
    ).reshape(-1, 3)

    if colors is not None:
        colors = trimesh.visual.color.interpolate(colors.flat, color_map=colormap)

    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=norms,
        vertex_colors=colors,
        **kwargs
    )
