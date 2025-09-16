from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Optional, Iterable, Any

if TYPE_CHECKING:
    from .operations import Operation
    from .statistics import Calculation


class OperationInstantiation:
    def __init__(
        self, operation: Operation, args: tuple[Any], kwargs: dict[str, Any]
    ) -> None:
        self._operation = operation
        self._args = args
        self._kwargs = kwargs


class ChannelHistory:
    """Operation history of an image channel."""

    def __init__(self) -> None:
        self._operations = []

    def push(self, f: Operation, args: tuple[Any], kwargs: dict[str, Any]):
        self._operations.append(OperationInstantiation(f, args, kwargs))


class Channel:
    """An image channel.
    Tracks operations that have been applied.
    """

    def __init__(self, idx: int, image: "Image") -> None:
        self._idx = idx
        self._history = ChannelHistory()
        self._x = image._x
        self._y = image._y
        self._data = image._data[idx].copy()
        self._image_labels = image._labels

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
    def label(self) -> str:
        """
        Returns:
            str: Channel label.
        """
        return self._image_labels[self._idx]

    @property
    def history(self) -> list[Operation]:
        """
        Returns:
            list[Operation]: Copy of the channel's history.
        """
        return self._history._operations.copy()

    def copy(self) -> Channel:
        """Copy the channel.

        Returns:
            Channel: Copy with all data copied.
        """
        clone = Channel.__new__(Channel)
        clone._idx = self._idx
        clone._history = ChannelHistory()
        clone._history._operations = self._history._operations.copy()
        clone._x = self._x
        clone._y = self._y
        clone._data = self._data.copy()
        clone._image_labels = self._image_labels
        return clone

    def apply(self, f: Operation, *args, **kwargs):
        """Apply an operation to the channel data.

        Args:
            f (Operation): Operation to perform.
            args: Positional arguments to pass to the operation.
            kwargs: Keyword arguments to pass to the operation.
        """
        self._data = f(self, *args, **kwargs)
        self._history.push(f, args, kwargs)


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
        self._channels = [Channel(idx, self) for idx in range(channels_dim)]

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

    def map_labels(self, label_map: dict[str, str]):
        """Change channel labels.

        Args:
            label_map (dict[str, str]): Dictionary of {<current>: <new>} labels.

        Raises:
            KeyError: Label does not exist.
        """
        for src, dst in label_map.items():
            idx = None
            try:
                idx = self._labels.index(src)
            except ValueError:
                raise KeyError(f"label `{src}` does not exist")

            self._labels[idx] = dst

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


class ChannelGroup:
    @classmethod
    def from_image(cls, image: Image) -> ChannelGroup:
        return cls(image._channels)

    def __init__(
        self, channels: list[Channel], image_labels: Optional[list[str]] = None
    ):
        if image_labels is not None:
            if len(image_labels) != len(channels):
                raise ValueError("invalid image labels, incompatible length")

            if len(set(image_labels)) != len(image_labels):
                raise ValueError("invalid image labels, duplicate labels")

        self._channels = channels
        self._image_labels = image_labels

    def __getitem__(self, key: str) -> Channel:
        """Get a channel by its image label.

        Args:
            key (str): Image label.

        Raises:
            KeyError: Image labels are not set.
            KeyError: Label does not exist.

        Returns:
            Channel: Channel associated with the image label.
        """
        if self._image_labels is None:
            raise KeyError("image labels not set")

        try:
            idx = self._image_labels.index(key)
        except ValueError:
            raise KeyError("image label does not exist")

        return self._channels[idx]

    def __iter__(self):
        for ch in self._channels:
            yield ch

    def __len__(self) -> int:
        return len(self._channels)

    @property
    def image_labels(self) -> Optional[list[str]]:
        """
        Returns:
            Optional[list[str]]: Copy of the image labels.
        """
        if self._image_labels is None:
            return None
        else:
            return self._image_labels.copy()

    def items(self) -> Iterable[tuple[str, Channel]]:
        if self._image_labels is None:
            raise ValueError("image lables not set")

        return zip(self._image_labels, self._channels)

    def copy(self) -> ChannelGroup:
        """
        Returns:
            ChannelGroup: Copy of the channel group.
        """
        channels = [ch.copy() for ch in self._channels]
        return ChannelGroup(channels, self._image_labels)

    def apply(self, f: Operation, *args: tuple[Any], **kwargs: dict[str, Any]):
        """Apply an operation to all channels.

        Args:
            f (Operation): Operation to perform.
            args: Positional arguments to pass to the operation.
            kwargs: Keyword arguments to pass to the operation.
        """
        for ch in self._channels:
            ch.apply(f, *args, **kwargs)

    def calculate(
        self, f: Calculation, *args: tuple[Any], **kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform a calculation on all channels.

        Args:
            f (Calculation): Calculation to perform.
            args: Positional arguments to pass to the operation.
            kwargs: Keyword arguments to pass to the operation.

        Returns:
            dict[str, Any]: Calculation result for each channel.
        """
        return {lbl: f(ch, *args, **kwargs) for lbl, ch in self.items()}


class ImageGroup:
    def __init__(self, images: list[Image], labels: Optional[list[str]]) -> None:
        if labels is not None:
            if len(images) != len(labels):
                raise ValueError(f"incompatible labels, length does not match images")
            if len(set(labels)) != len(labels):
                raise ValueError("invalid labels, duplicate label found")

        self._images = images
        self._labels = labels

    def __getitem__(self, key: str) -> Image:
        """Get an image by its label.

        Args:
            key (str): Image label.

        Raises:
            KeyError: Labels are not set.
            KeyError: Label is not found.

        Returns:
            Image: Image associated with the given label.
        """
        if self._labels is None:
            raise KeyError("no labels set")

        try:
            idx = self._labels.index(key)
        except ValueError:
            raise KeyError("label not found")

        return self._images[idx]

    def __iter__(self):
        for im in self._images:
            yield im

    def __len__(self) -> int:
        return len(self._images)

    @property
    def labels(self) -> Optional[list[str]]:
        """
        Returns:
            Optional[list[str]]: Copy of labels.
        """
        if self._labels is None:
            return None

        return self._labels.copy()

    @labels.setter
    def labels(self, labels: list[str]):
        """Set image labels.

        Args:
            labels (list[str]): Labels to apply.

        Raises:
            ValueError: Labels have invalid length compared to images.
            ValueError: A label is duplicated.
        """
        if len(self._images) != len(labels):
            raise ValueError(f"incompatible labels, length does not match images")
        if len(set(labels)) != len(labels):
            raise ValueError("invalid labels, duplicate label found")

        self._labels = labels

    def map_labels(self, label_map: dict[str, str]):
        """Change labels.

        Args:
            label_map (dict[str, str]): Dictionary of {<current>: <new>}.

        Raises:
            ValueError: If labels have not been set.
            KeyError: If a label in the map does not exist.
        """
        if self._labels is None:
            raise ValueError("labels not set")

        for src, dst in label_map.items():
            try:
                idx = self._labels.index(src)
            except ValueError:
                raise KeyError(f"label {src} does not exist")

            self._labels[idx] = dst

    def channels(self, key: str) -> ChannelGroup:
        """Get specified channel from each image.

        Args:
            key (str): Channel label.

        Raises:
            ValueError: If an image does not contain a channel with the given label.

        Returns:
            ChannelGroup: Channel from each image.
        """
        channels = [image[key] for image in self._images]
        if any([ch is None for ch in channels]):
            raise ValueError("not all images contain the channel")

        return ChannelGroup(channels, image_labels=self._labels)
