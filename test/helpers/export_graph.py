import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
from os import path
from typing import Tuple, List


def export_graph(img: np.ndarray, title: str, out_dir: str = './output/graphs/') -> None:
    """
    Generate graphs. Export them as html file into out_dir
    """
    data = [go.Heatmap(z=img)]

    layout = go.Layout(
        title=title,
        height=800,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )

    py.plot(go.Figure(data, layout), filename=path.join(out_dir, title + '.html'))

