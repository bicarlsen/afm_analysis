import numpy as np
from bric_afm import operations as ops, Image


def test_min_to_zero():
    x_dim = np.random.randint(1, 1000)
    y_dim = np.random.randint(1, 1000)
    x = np.random.rand(x_dim)
    y = np.random.rand(y_dim)
    data = np.random.randn(y_dim, x_dim, 1)
    d_min = data.min()
    img = Image(x, y, data, ["data"])
    res = ops.min_to_zero(img["data"])
    assert res.shape == (y_dim, x_dim)
    assert res.min() == 0
    assert (data[:, :, 0] - d_min == res).all()


def test_plane_level():
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 2])
    data = np.array([[[1]] * 3, [[2]] * 3, [[3]] * 3])
    img = Image(x, y, data, ["data"])
    res = ops.plane_level(img["data"])
    assert res.shape == (len(x), len(y))
    assert np.allclose(res, np.zeros_like(res))


def test_crop_boundary_mask():
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 2, 3, 4])
    data = np.array(
        [
            [[0], [0], [0], [0], [0]],
            [[0], [np.nan], [0], [0], [0]],
            [[0], [0], [0], [0], [0]],
            [[0], [0], [0], [0], [0]],
            [[0], [0], [0], [0], [np.nan]],
        ]
    )

    img = Image(x, y, data, ["data"])
    mask = ops.crop_boundary_mask(img["data"])

    expected = np.array(
        [
            [False, False, False, False, False],
            [False, False, False, False, False],
            [False, False, True, True, False],
            [False, False, True, True, False],
            [False, False, False, False, False],
        ]
    )
    assert (expected == mask).all()
