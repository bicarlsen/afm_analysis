from __future__ import annotations
from . import mesh, utils
import numpy as np
from sklearn import linear_model
import trimesh
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from . import image


class Operation(Protocol):
    def __call__(self, channel: image.Channel, *args, **kwargs) -> np.ndarray: ...


def crop(channel: image.Channel) -> np.ndarray:
    """Crop an image as large as possible removing all `np.nan` values.

    Args:
        channel (image.Channel): Channel.

    Returns:
        np.ndarray: Cropped cahnnel data.
    """
    raise NotImplementedError("todo")


def min_to_zero(channel: image.Channel) -> np.ndarray:
    """Zero data such that the minimum is 0.

    Args:
        channel (image.Channel): Channel.

    Returns:
        np.ndarray: Data translated such that the minimum value is 0.
    """
    data = channel.data
    return data - data.min()


def plane_level(channel: image.Channel) -> np.ndarray:
    """Level data over the average plane determined by linear regression.

    Args:
        channel (image.Channel): Channel.

    Returns:
        np.ndarray: Leveled data values.
    """
    coords = utils.xy_to_coords(channel.x, channel.y)
    fit = linear_model.LinearRegression().fit(
        coords.reshape(-1, 2), channel.data.flatten()
    )
    plane = np.dot(coords, fit.coef_) + fit.intercept_
    return channel.data - plane


def surface_fit(channel: image.Channel, xdeg: int = 1, ydeg: int = 1) -> np.ndarray:
    """Level data over a 2D polynomial surface.

    Args:
        channel (image.Channel): Channel.
        xdeg (int): Polynomial degree along x axis.
        ydeg (int): Polynomial degree along y axis.

    Returns:
        np.ndarray: Leveled data.
    """
    raise NotImplementedError("todo")


def add_conformal_layer(
    channel: image.Channel,
    thickness: float,
    scale: float = 1,
) -> np.ndarray:
    """Add a conformal layer to a surface.
    This is only an approximate surface constructed by
    offsetting vertices by `thickness` in the direction of their normal.

    Args:
        channel (image.Channel): Channel.
        thickness (float): Thickness of the conformal layer.
        scale (float): How to scale values.
        The underlying library used to compute the mesh does better when values are ~1.
        Defaults to 1.

    Raises:
        ValueError: Invalid `thickness`.
        ValueError: Invalid `scale`.

    Returns:
        np.ndarray: Vertices of the conformal surface indexed according to `x` and `y`.
        i.e. The returned data is sampled at the same positions as `data`.
        Values are `numpy.nan` if a sample could not be retrieved at that location.

    Notes:
        + This function can be memory intensive. If you run into a `MemoryError`
        be sure that all values are ~1 (i.e. not 1e-9).
    """
    if thickness < 0:
        raise ValueError("thickness can not be negative")
    if scale <= 0:
        raise ValueError("invalid scale, must be greater than 0")

    if thickness == 0:
        return channel.data

    x = channel.x * scale
    y = channel.y * scale
    m = mesh.create_mesh(x, y, channel.data * scale)

    offset_vertices = m.vertices + thickness * m.vertex_normals
    offset_mesh = trimesh.Trimesh(
        vertices=offset_vertices, faces=m.faces, process=False
    )

    ray_z = offset_mesh.vertices[:, -1].max() + 1
    ray_origins = m.vertices.copy()
    ray_origins[:, 2] = ray_z
    ray_directions = np.array([[0, 0, -1]] * ray_origins.shape[0])
    intersections, _, _ = offset_mesh.ray.intersects_location(
        ray_origins, ray_directions, multiple_hits=False
    )

    intersections_x = np.searchsorted(x, intersections[:, 0], side="left")
    intersections_y = np.searchsorted(y, intersections[:, 1], side="left")
    intersections_idx = np.ravel_multi_index(
        [intersections_x, intersections_y], channel.data.shape
    )
    intersections_z = intersections[:, 2]
    vertices = np.full((x.size, y.size), np.nan)
    vertices.put(intersections_idx, intersections_z)

    return vertices / scale
