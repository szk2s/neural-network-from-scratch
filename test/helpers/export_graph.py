import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
from os import path
from typing import Tuple, List


def export_heatmap(z: np.ndarray, title: str, out_dir: str = './output/graphs/') -> None:
    """
    Generate heatmap graphs. Export them as html file into out_dir
    """
    data = [go.Heatmap(z=z)]

    layout = go.Layout(
        title=title,
    )

    py.plot(go.Figure(data, layout), filename=path.join(out_dir, title + '.html'))


def export_line_chart(x: np.ndarray, y: np.ndarray, title: str, out_dir: str = './output/graphs/') -> None:
    """
    Generate line chart. Export them as html file into out_dir
    """
    data = [go.Scatter(x=x, y=y, mode='markers+lines')]

    layout = go.Layout(
        title=title,
    )

    py.plot(go.Figure(data, layout), filename=path.join(out_dir, title + '.html'))
