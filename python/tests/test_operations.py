import numpy as np
import bric_afm.operations as ops


def test_min_to_zero():
    x = np.array([])
    y = np.array([])
    n = np.random.randint(1, 1000)
    data = np.random.randn(n, n)
    res = ops.min_to_zero(x, y, data)
    assert res.shape == (n, n)
    assert res.min() == 0


def test_plane_level():
    x = np.array([0, 1, 2])
    y = np.array([3, 4, 5])
    data = np.array([[1] * 3, [2] * 3, [3] * 3])
    res = ops.plane_level(x, y, data)
    assert res.shape == (len(x), len(y))
    assert np.allclose(res, np.zeros_like(res))
