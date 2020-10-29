import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from plotly.io._orca import shutdown_server

app = dash.Dash('ChemReacPred')
server = app.server
app.layout = html.Div([
    html.H2('Hello World 3 '),
    dcc.Dropdown(
        id='dropdown',
        options=[{'label': i, 'value': i} for i in ['LA', 'NYC', 'MTL']],
        value='LA'
    ),
    html.Div(id='display-value')
])


@app.callback(dash.dependencies.Output('display-value', 'children'),
              [dash.dependencies.Input('dropdown', 'value')])
def display_value(value):
    return 'You have selected "{}"'.format(value)


@app.server.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


if __name__ == '__main__':
    app.run_server(debug=True)
