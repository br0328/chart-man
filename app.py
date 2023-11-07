
import dash_bootstrap_components as dbc
import pandas as pd
import dash

app = dash.Dash(__name__, use_pages = True, external_stylesheets = [dbc.themes.SANDSTONE])

sidebar = dbc.Nav([
        dbc.NavLink([
                dash.html.Div(page["name"], className = "ms-2"),
            ],
            href = page["path"],
            active = "exact",
        )
        for page in dash.page_registry.values()
    ],
    vertical = True,
    pills = True,
    style = {'width': '240px'}
)

app.title = 'Chartman'

app.layout = dash.html.Div([
    dash.html.Div(
        children = [
            dash.html.Img(src = 'assets/logo.png'),
            sidebar
        ],
        className = 'left_pane'
    ),
    dash.html.Div(
        children = [
            dash.page_container
        ],
        className = 'work_pane'
    ),
])

if __name__ == "__main__":
    app.run(debug = True, port = 1000)
