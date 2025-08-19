from .image import Channel, ChannelGroup
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional


def plot(channel: Channel) -> go.Figure:
    """Plot a single channel.

    Args:
        channel (Channel): Channel to plot.

    Returns:
        go.Figure: Plot of channel.
    """
    fig = px.imshow(
        channel.data, x=channel.x, y=channel.y, labels={"color": channel.label}
    )
    return fig


def plot_group(channels: ChannelGroup, ncols: Optional[int] = None) -> go.Figure:
    """Plot all channels in a channel group.

    Args:
        channels (ChannelGroup): Channels.
        ncols (Optional[int], optional): Number of columns. If None, is set to the square root of the number of channels. Defaults to None.

    Returns:
        go.Figure: Figure.
    """
    if ncols is None:
        ncols = int(np.sqrt(len(channels)))

    fig = px.imshow(
        np.array([ch.data for ch in channels]), facet_col=0, facet_col_wrap=ncols
    )
    if channels.image_labels is not None:
        image_labels = channels.image_labels

        def channel_label(facet_label: str) -> str:
            idx = int(facet_label.split("=")[-1])
            return image_labels[idx]

        fig.for_each_annotation(lambda ann: ann.update(text=channel_label(ann.text)))

    return fig


def histogram(channel: Channel) -> go.Figure:
    fig = px.histogram(channel.data.flatten())
    return fig


def historgram_group(channels: ChannelGroup, cols: Optional[int] = None) -> go.Figure:
    """Plot histogram of all channels in a channel group.

    Args:
        channels (ChannelGroup): Channels.
        cols (Optional[int], optional): Number of columns. If None, is set to the square root of the number of channels. Defaults to None.

    Returns:
        go.Figure: Figure.
    """
    if cols is None:
        cols = int(np.sqrt(len(channels)))

    fig = px.imshow(
        np.array([ch.data for ch in channels]), facet_col=0, facet_col_wrap=cols
    )
    if channels.image_labels is not None:
        image_labels = channels.image_labels

        def channel_label(facet_label: str) -> str:
            idx = int(facet_label.split("=")[-1])
            return image_labels[idx]

        fig.for_each_annotation(lambda ann: ann.update(text=channel_label(ann.text)))

    return fig
