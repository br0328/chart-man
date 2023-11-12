
from dash import dcc, html, callback, Output, Input
from ui import *
import dash_bootstrap_components as dbc
import plotly.express as px
import dash

dash.register_page(__name__, path = '/newhigh', name = 'New Highs', order = 1)

scenario_div = get_scenario_div([
    get_symbol_input(),
    get_date_range(),
    get_interval_input()
])
parameter_div = get_parameter_div([
    get_cur_date_picker(),
    get_pivot_number_input(),
    get_merge_thres_input(),
    get_analyze_button(),
    get_backtest_button()
])
out_tab = get_out_tab(
    {
        'Plot': get_plot_div(),
        'Report': get_report_div()
    }
)
layout = get_page_layout('New Highs', scenario_div, parameter_div, out_tab)
