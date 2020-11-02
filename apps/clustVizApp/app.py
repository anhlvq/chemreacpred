import glob
import os

import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output
from sklearn.cluster import KMeans

from apps.clustVizApp.helpers import create_plot3D, create_plot2D
from dataIO.Dataset import FeatureDataset

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

DATA_PATH = './data'
all_files = glob.glob(os.path.join(DATA_PATH, '_feature*.csv'))
ds_list = list()
for fname in all_files:
    ds_list.append(FeatureDataset(fname, True))

ds0 = ds_list[0]
X0 = ds0.tsne3Comp()
FIGURE = create_plot3D(
    x=X0[:, 0],
    y=X0[:, 1],
    z=X0[:, 2],
    size=3,
    color='blue',
    name=ds0.idList,
)


def create_plot(idx, labels='blue', plot_type='scater3d'):
    ds = ds_list[idx]
    idList = ds.idList
    if plot_type == 'scatter3d':
        X = ds.tsne3Comp()
        FIG = create_plot3D(
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2],
            size=3,
            color=labels,
            name=idList,
        )
    else:
        X = ds.tsne2Comp()
        FIG = create_plot2D(
            x=X[:, 0],
            y=X[:, 1],
            size=3,
            color=labels,
            name=idList,
        )
    return FIG


# Methods for creating components in the layout code
def Card(children, **kwargs):
    return html.Section(children, className="card-style")


def NamedSlider(name, short, min, max, val):
    marks = {i: i for i in range(min, max + 1, 1)}

    return html.Div(
        style={"margin": "25px 5px 30px 0px"},
        children=[
            f"{name}:",
            html.Div(
                style={"margin-left": "5px"},
                children=[
                    dcc.Slider(
                        id=f"slider-{short}",
                        min=min,
                        max=max,
                        marks=marks,
                        # step=1,
                        value=val,
                    )
                ],
            ),
        ],
    )


def NamedInlineRadioItems(name, short, options, val, **kwargs):
    return html.Div(
        id=f"div-{short}",
        style={"display": "inline-block"},
        children=[
            f"{name}:",
            dcc.RadioItems(
                id=f"radio-{short}",
                options=options,
                value=val,
                labelStyle={"display": "inline-block", "margin-right": "7px"},
                style={"display": "inline-block", "margin-left": "7px"},
            ),
        ],
    )


header = html.Div(
    [
        html.Div(
            [
                html.H3(
                    "Molecular Descriptor Clustering",
                    className="uppercase title",
                ),
            ]
        )
    ],
    className="app__header",
)

sidebar = html.Div(
    [Card(
        [dcc.Dropdown(
            id="dropdown-dataset",
            searchable=False,
            clearable=False,
            options=[{'label': ds_list[i].dataSetName,
                      'value': i}
                     for i in range(0, len(ds_list))
                     ],
            placeholder="Select a dataset",
            value=0,
        ),
            dcc.Slider(
                id="slider-k",
                min=2,
                max=100,
                step=1,
                value=3,
            ),
            dbc.Button("Cluster", id='btn-cluster', color="primary", className="mr-1"),
        ]
    )
    ],
    className='three columns',
)

content = html.Div(
    [
        dcc.RadioItems(
            id="charts_radio",
            options=[
                {"label": "3D Scatter", "value": "scatter3d"},
                {"label": "2D Scatter", "value": "scatter"},
            ],
            labelClassName="radio__labels",
            inputClassName="radio__input",
            value="scatter3d",
            className="radio__group",
        ),
        dcc.Graph(
            id="clickable-graph",
            hoverData={"points": [{"pointNumber": 0}]},
            figure=FIGURE,
        ),
    ],
    className="nine columns",
)

body = html.Div(
    [
        sidebar, content
    ],
)
app.layout = html.Div(
    [
        header,
        body
    ],
    className="app__container",
)

_idx = 0
_plot_type = 'scatter3d'
_k = 5


@app.callback(
    Output("clickable-graph", "figure"),
    [Input("charts_radio", "value"), Input("dropdown-dataset", "value"), Input("btn-cluster", "n_clicks")],
)
def change_plot_type(plot_type, idx, n):
    global _idx
    global _plot_type
    global _k, ds_list
    _idx = idx
    _plot_type = plot_type
    if n is not None:
        X = ds_list[_idx].features
        km = KMeans(_k)
        labels = km.fit_predict(X)
    else:
        labels='blue'
    return create_plot(idx=idx, plot_type=plot_type, labels=labels)


if __name__ == "__main__":
    app.run_server(debug=True)
