
from data import *
from ui import *
import dash_bootstrap_components as dbc
import pandas as pd
import dash

initialize_data()

app = dash.Dash(__name__, use_pages = True, external_stylesheets = [dbc.themes.SANDSTONE])

app.title = 'Chartman'
app.layout = get_app_layout()

app.run(debug = True, port = 1000)
