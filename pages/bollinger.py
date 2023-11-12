
from dash import dcc, html, callback, Output, Input
from ui import *
import dash_bootstrap_components as dbc
import plotly.express as px
import dash

dash.register_page(__name__, path = '/bolinger', name = 'Bollinger Bands', order = '10')

scenario_div = get_scenario_div([
])
parameter_div = get_parameter_div([
])
out_tab = get_out_tab({
})
layout = get_page_layout('Bollinger Bands', scenario_div, parameter_div, out_tab)
