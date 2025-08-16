from .image import Channel
import plotly.graph_objects as go
import plotly.express as px


def plot_interactive(channel: Channel) -> go.Figure:
    fig = px.imshow(
        channel.data, x=channel.x, y=channel.y, labels={"color": channel.label}
    )
    return fig
