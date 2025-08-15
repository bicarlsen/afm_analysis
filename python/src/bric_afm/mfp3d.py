from .image import Image
import numpy as np
import igor2 as igor

def load_ibw(path: str) -> Image:
    """Load an `.ibw` image from the MFP-3D.

    Args:
        path (str): Path to the file.

    Returns:
        Image: Image.

    Raises:
        RuntimeError: If values can not be extracted.
    """
    ibw = igor.binarywave.load(path)
    try:
        wave = ibw["wave"]
    except KeyError:
        raise RuntimeError("wave could not be extracted")

    try:
        data = wave["wData"]
    except KeyError:
        raise RuntimeError("data could not be extracted")

    try:
        header = wave["wave_header"]
    except KeyError:
        raise RuntimeError("header could not be extracted")

    try:
        labels = wave["labels"][2][1:]
    except KeyError:
        raise RuntimeError("labels could not be extracted")
    labels = [label.decode() for label in labels]

    try:
        x_dim, y_dim, channels_dim, _ = header["nDim"]
    except ValueError:
        raise RuntimeError("dimensions could not be extracted")

    try:
        x_start, y_start, _, _ = header["sfB"]
        x_step, y_step, _, _ = header["sfA"]
    except ValueError:
        raise RuntimeError("indices could not be extracted")

    x_index = np.linspace(
        x_start, x_start + x_step * x_dim, num=x_dim, endpoint=True
    )
    y_index = np.linspace(
        y_start, y_start + y_step * y_dim, num=y_dim, endpoint=True
    )
    imgs = data.transpose(2, 0, 1)
    return Image(x_index, y_index, imgs, labels)
