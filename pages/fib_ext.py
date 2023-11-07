
from dash import dcc, html, callback, Output, Input
from ui import *
import dash_bootstrap_components as dbc
import plotly.express as px
import dash

dash.register_page(__name__, path = '/fibext', name = 'Fibonacci Extension')

title_div = get_page_title('Fibonacci|Extension')

scenario_div = get_scenario_div([
    get_symbol_input(),
    get_date_range(),
    get_interval_input(),
    get_load_button()
])

parameter_div = get_parameter_div([
    get_pivot_number_input(),
    get_merge_thres_input(),
    get_analyze_button(),
    get_backtest_button()
])

layout = html.Div(
    children = [
        title_div,
        scenario_div,
        parameter_div
    ]
)
