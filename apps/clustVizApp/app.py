import glob
import os
import time
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_table
import plotly.express as px
from dash.dependencies import Input, Output
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

from apps.clustVizApp.dataIO.Dataset import FeatureDataset, checkExists
from apps.clustVizApp.dataIO.dbloader import listAllDataSets, loadDataSet
from apps.clustVizApp.dataIO.loader import readNumpyArrayFile, writeNumpyArrayFile

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

DATA_PATH = '../../data/3_processed'
dbfile = os.path.join(DATA_PATH, 'data.sqlite')
# all_files = glob.glob(os.path.join(DATA_PATH, '_feature*.csv'))
# ds_list = list()
# for fname in all_files:
#    ds_list.append(FeatureDataset(fname, True))
print('Load datasets...')
dsName_list, pattern_list = listAllDataSets(dbfile)

fname, df = loadDataSet(dbfile, pattern_list[0])
fname = os.path.join(DATA_PATH, fname)
currentds = FeatureDataset(fname, df=df, isNormalized=True)
print('Done.')


def create_plot(ds, labels='blue', plot_type='scater3d'):
    idList = ds.idList
    if plot_type == 'scatter3d':
        X = ds.tsne3Comp()
        FIG = px.scatter_3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], color=labels, hover_name=idList)
    else:
        X = ds.tsne2Comp()
        FIG = px.scatter(
            x=X[:, 0],
            y=X[:, 1],
            color=labels,
            hover_name=idList,
            title=ds.dataSetName + " | Number of Clusters = " + str(max(labels) + 1),
        )
    return FIG


def update_table_data(ds, labels):
    idList = ds.idList
    return [{"ID": idList[i], "Cluster": labels[i]} for i in range(0, len(idList))]


def create_table(ds, labels):
    return dash_table.DataTable(
        id='table',
        columns=[{"id": "ID", "name": "ID"}, {"id": "Cluster", "name": "Cluster"}],
        data=update_table_data(ds, labels),
        sort_action="native",
        sort_mode="multi",
        filter_action="native",
        row_selectable="single",
    )


clustering_methods = ['KMeans', 'KMedoids-Euclidean', 'KMedoids-Cosine', 'KMedoids-Manhattan']


def doCluster(ds, k, method='KMeans'):
    print(ds)
    file = ds.filePath + "." + str(k) + "." + method
    if checkExists(file):
        labels = readNumpyArrayFile(file)
        labels = labels.reshape(1, -1)[0, :]
    else:
        X = ds.features
        if method == 'KMeans':
            km = KMeans(k)
            labels = km.fit_predict(X)
        elif method == 'KMedoids-Euclidean':
            kmedoids = KMedoids(n_clusters=k, metric='euclidean').fit(X)
            labels = kmedoids.labels_
        elif method == 'KMedoids-Cosine':
            kmedoids = KMedoids(n_clusters=k, metric='cosine').fit(X)
            labels = kmedoids.labels_
        elif method == 'KMedoids-Manhattan':
            kmedoids = KMedoids(n_clusters=k, metric='manhattan').fit(X)
            labels = kmedoids.labels_
        else:
            labels = {i for i in range(0, X.shape[1])}
        writeNumpyArrayFile(file, labels)
    return labels


# Methods for creating components in the layout code
def Card(children, **kwargs):
    return html.Section(children, className="card-style")


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
        [
            html.Label("Select a Dataset"),
            dcc.Dropdown(
                id="dropdown-dataset",
                searchable=False,
                clearable=False,
                options=[{'label': dsName_list[i],
                          'value': i}
                         for i in range(0, len(dsName_list))
                         ],
                placeholder="Select a dataset",
                value=0,
            ),
            html.Div(
                [
                    html.Label("Set the number of clusters"),
                    dcc.Input(
                        id="n-clusters",
                        type="number",
                        value=2,
                        name="number of clusters",
                        min=2,
                        step=1,
                    ),
                ]
            ),
            html.Label("Select a clustering method"),
            dcc.Dropdown(
                id="dropdown-clustering-method",
                searchable=False,
                clearable=False,
                options=[{'label': clustering_methods[i],
                          'value': clustering_methods[i]}
                         for i in range(0, len(clustering_methods))
                         ],
                placeholder="Select a clustering method",
                value=clustering_methods[0],
            ),
        ]
    )
    ],
    className='three columns',
)
labels = doCluster(currentds, k=2)
FIGURE = create_plot(currentds, labels=labels, plot_type='scatter')

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
            value="scatter",
            className="radio__group",
        ),
        dcc.Graph(
            id="clickable-graph",
            hoverData={"points": [{"pointNumber": 0}]},
            figure=FIGURE,
        ),
        create_table(currentds, doCluster(currentds, 2))

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


@app.callback(
    [Output("clickable-graph", "figure"), Output("table", "data")],
    [Input("charts_radio", "value"),
     Input("dropdown-dataset", "value"),
     Input("n-clusters", "value"),
     Input("dropdown-clustering-method", "value")],
)
def change_plot_type(plot_type, idx, k, clusteringmethod):
    print('Clustering...')
    st = time.time()
    global currentds
    fname, df = loadDataSet(db_file=dbfile, patterns=pattern_list[idx])
    fname = os.path.join(DATA_PATH, fname)
    currentds = FeatureDataset(fname, df=df, isNormalized=True)
    labels = doCluster(currentds, k=k, method=clusteringmethod)
    print("--- %s seconds ---" % (time.time() - st))
    print('Ploting...')
    FIG = create_plot(currentds, plot_type=plot_type, labels=labels)
    print("--- %s seconds ---" % (time.time() - st))
    print('Update table...')
    TAB = update_table_data(currentds, labels=labels)
    print("--- %s seconds ---" % (time.time() - st))
    return [FIG, TAB]


if __name__ == "__main__":
    app.run_server(debug=True)
