import dash_html_components as html


def make_dash_table(selection, df):
    """ Return a dash defintion of an HTML table from a Pandas dataframe. """

    df_subset = df.loc[df["NAME"].isin(selection)]
    table = []

    for index, row in df_subset.iterrows():
        rows = [html.Td([row["NAME"]]), html.Td([html.Img(src=row["IMG_URL"])]), html.Td([row["FORM"]]),
                html.Td([html.A(href=row["PAGE"], children="Datasheet", target="_blank")])]
        table.append(html.Tr(rows))

    return table


def _create_axis(axis_type, variation="Linear", title=None):
    """
    Creates a 2d or 3d axis.

    :params axis_type: 2d or 3d axis
    :params variation: axis type (log, line, linear, etc)
    :parmas title: axis title
    :returns: plotly axis dictionnary
    """

    if axis_type not in ["3d", "2d"]:
        return None

    default_style = {
        "background": "rgb(230, 230, 230)",
        "gridcolor": "rgb(255, 255, 255)",
        "zerolinecolor": "rgb(255, 255, 255)",
    }

    if axis_type == "3d":
        return {
            "showbackground": True,
            "backgroundcolor": default_style["background"],
            "gridcolor": default_style["gridcolor"],
            "title": title,
            "type": variation,
            "zerolinecolor": default_style["zerolinecolor"],
        }

    if axis_type == "2d":
        return {
            "xgap": 10,
            "ygap": 10,
            "backgroundcolor": default_style["background"],
            "gridcolor": default_style["gridcolor"],
            "title": title,
            "zerolinecolor": default_style["zerolinecolor"],
            "color": "#444",
        }


def _black_out_axis(axis):
    axis["showgrid"] = False
    axis["zeroline"] = False
    axis["color"] = "white"
    return axis


def _create_layout(layout_type, xlabel, ylabel, zlabel):
    """ Return dash plot layout. """

    base_layout = {
        "font": {"family": "Raleway"},
        "hovermode": "closest",
        "margin": {"r": 20, "t": 0, "l": 0, "b": 0},
        "showlegend": False,
    }

    if layout_type == "scatter3d":
        base_layout["scene"] = {
            "xaxis": _create_axis(axis_type="3d", title=xlabel),
            "yaxis": _create_axis(axis_type="3d", title=ylabel),
            "zaxis": _create_axis(axis_type="3d", title=zlabel),
            "camera": {
                "up": {"x": 0, "y": 0, "z": 1},
                "center": {"x": 0, "y": 0, "z": 0},
                "eye": {"x": 0.08, "y": 2.2, "z": 0.08},
            },
        }
    elif layout_type == "scatter":
        base_layout["xaxis"] = _create_axis(axis_type="2d", title=xlabel)
        base_layout["yaxis"] = _create_axis(axis_type="2d", title=ylabel)
        base_layout["plot_bgcolor"] = "rgb(230, 230, 230)"
        base_layout["paper_bgcolor"] = "rgb(230, 230, 230)"

    return base_layout


def create_plot3D(x,
                  y,
                  z,
                  size,
                  color,
                  name,
                  xlabel="X",
                  ylabel="Y",
                  zlabel="Z",
                  ):
    plot_type = 'scatter3d'

    data = [
        {
            "x": x,
            "y": y,
            "z": z,
            "mode": "markers",
            "marker": {
                "line": {"color": "#444"},
                "reversescale": True,
                "sizeref": 45,
                "sizemode": "diameter",
                "opacity": 0.5,
                "size": size,
                "color": color,
            },
            "text": name,
            "type": plot_type,
        }
    ]

    layout = _create_layout(plot_type, xlabel, ylabel, zlabel)

    return {"data": data, "layout": layout}


def create_plot2D(x,
                  y,
                  size,
                  color,
                  name,
                  xlabel="X",
                  ylabel="Y",
                  ):
    plot_type = 'scatter'
    data = [
        {
            "x": x,
            "y": y,
            "mode": "markers",
            "marker": {
                "line": {"color": "#444"},
                "reversescale": True,
                "sizeref": 45,
                "sizemode": "diameter",
                "opacity": 0.7,
                "size": size,
                "color": color,
            },
            "text": name,
            "type": plot_type,
        }
    ]

    layout = _create_layout(plot_type, xlabel, ylabel, '')

    return {"data": data, "layout": layout}

