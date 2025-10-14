import pytest
from bric_afm import Image
import numpy as np


def test_image_label_map():
    y_dim = 10
    x_dim = 10
    channels_dim = 3
    labels = ["one", "two", "three"]
    img = Image(
        np.arange(y_dim),
        np.arange(x_dim),
        np.random.randn(y_dim, x_dim, channels_dim),
        labels,
    )

    img_labels = img.labels
    assert len(img_labels) == channels_dim
    assert img_labels[0] == "one"
    assert img_labels[1] == "two"
    assert img_labels[2] == "three"

    ch_one = img["one"]
    assert ch_one.label == "one"

    img.map_labels({"one": "first"})
    assert len(img_labels) == channels_dim
    assert img_labels[0] == "first"
    assert ch_one.label == "first"

    with pytest.raises(KeyError):
        img.map_labels({"not_there": "nope"})
