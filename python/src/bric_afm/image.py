from .operations import Operation
import numpy as np
from typing import Optional, Callable


class ChannelHistory:
    """Operation history of an image channel."""

    def __init__(self) -> None:
        self._operations = []

    def push(self, f: Operation):
        self._operations.append(f)


class Channel:
    """An image channel.
    Tracks operations that have been applied.
    """

    def __init__(
        self, idx: int, x: np.ndarray, y: np.ndarray, data: np.ndarray
    ) -> None:
        self._idx = idx
        self._history = ChannelHistory()
        self._x = x
        self._y = y
        self._data = data.copy()

    @property
    def x(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Copy of the x index.
        """
        return self._x.copy()

    @property
    def y(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Copy of the y index.
        """
        return self._y.copy()

    @property
    def data(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: copy of the channel's data.
        """
        return self._data.copy()

    @property
    def history(self) -> list[Operation]:
        """
        Returns:
            list[Operation]: Copy of the channel's history.
        """
        return self._history._operations.copy()

    def apply(self, f: Operation):
        """Apply an operation to the channel data.

        Args:
            f (Operation): Operation to perform.
        """
        self._data = f(self._x, self._y, self._data)
        self._history.push(f)


class Image:
    """An image."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        data: np.ndarray,
        labels: list[str],
    ) -> None:
        """Create a new Image.

        Args:
            x (np.ndarray): x index points.
            y (np.ndarray): y index points.
            data (np.ndarray): Image data. Should be in (channel, x, y).
            labels (list[str]): Channel labels.

        Raises:
            ValueError: Data shape is invalid.
        """
        channels_dim, x_dim, y_dim = data.shape
        if (
            (channels_dim != len(labels))
            or (x_dim != x.shape[0])
            or (y_dim != y.shape[0])
        ):
            raise ValueError("invalid data shape")

        self._x = x
        self._y = y
        self._data = data
        self._labels = labels
        self._channels = [Channel(idx, self._x, self._y, self._data[idx]) for idx in range(channels_dim)]

    def __getitem__(self, key: str) -> Channel:
        """Get an image channel by label.

        Args:
            key (str): Channel label.

        Raises:
            KeyError: Label was not found.

        Returns:
            Channel: Image channel.
        """
        try:
            idx = self._labels.index(key)
        except ValueError:
            raise KeyError(key)

        return self._channels[idx]

    @property
    def x(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Copy of the x index.
        """
        return self._x.copy()

    @property
    def y(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Copy of the y index.
        """
        return self._y.copy()

    @property
    def shape(self) -> tuple[int, int]:
        """
        Returns:
            tuple[int, int]: (width, height).
        """
        return self._data.shape[1:]
    
    @property
    def labels(self) -> list[str]:
        return self._labels

    def label_index(self, label: str) -> Optional[int]:
        """Get the index of a label.

        Args:
            label (str): Label.

        Returns:
            Optional[int]: Index of the label, if found.
        """
        try:
            return self._labels.index(label)
        except ValueError:
            return None

    def copy_all_channels(self) -> np.ndarray:
        """Copy all channel data.

        Returns:
            np.ndarray: Image data.
        """
        return self._data.copy()

    def copy_channel(self, label: str) -> Optional[np.ndarray]:
        """Copy the data from an individual channel.

        Args:
            label (str): Channel label.

        Returns:
            Optional[np.ndarray]: Channel data, if found.
        """
        try:
            idx = self._labels.index(label)
        except ValueError:
            return None

        return self._data[idx].copy()

    def set_channel_data(self, channel: int, data: np.ndarray):
        """Sets a channel's data.

        Args:
            channel (int): Channel index.
            data (np.ndarray): Data.

        Raises:
            ValueError: Channel index is invalid.
            ValueError: Data has an invalid shape.
        """
        if channel > self._data.shape[0]:
            raise ValueError("invalid channel index")

        if data.shape != self._data.shape[1:]:
            raise ValueError("invalid data shape")

        self._data[channel] = data
