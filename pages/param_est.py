
from dash import dcc, callback, Output, Input, State
from datetime import date, datetime, timedelta
from plotly.subplots import make_subplots
from collections import defaultdict
from constant import *
from compute import *
from config import *
from finta import TA
from yahoo import *
from plot import *
from util import *
from data import *
from ui import *
import plotly.graph_objects as go
import numpy as np
import dash
import csv

dash.register_page(__name__, path = '/paramest', name = 'Parameter Estimation', order = '02')

# Page Layout
scenario_div = get_scenario_div([
	get_symbol_input(),
	get_date_range(),
    get_analyze_button('param-est')
])
out_tab = get_out_tab({
	'Plot': get_plot_div()
})
layout = get_page_layout('Parameter|Estimation', scenario_div, None, out_tab)

# Triggered when Analyze button clicked
@callback(
	[
		Output('alert-dlg', 'is_open', allow_duplicate = True),
		Output('alert-msg', 'children', allow_duplicate = True),
		Output('alert-dlg', 'style', allow_duplicate = True),
		Output('out-plot', 'children', allow_duplicate = True)
	],
	Input('param-est-analyze-button', 'n_clicks'),
	[
		State('symbol-input', 'value'),
		State('from-date-input', 'date'),
		State('to-date-input', 'date')
	],
	prevent_initial_call = True
)
def on_analyze_clicked(n_clicks, symbol, from_date, to_date):
    none_ret = [None] # Padding return values

    if n_clicks == 0: return alert_hide(none_ret)

    if symbol is None: return alert_error('Invalid symbol. Please select one and retry.', none_ret)
    if from_date is None: return alert_error('Invalid starting date. Please select one and retry.', none_ret)
    if to_date is None: return alert_error('Invalid ending date. Please select one and retry.', none_ret)
    if from_date > to_date: return alert_error('Invalid duration. Please check and retry.', none_ret)

    fig = runStochDivergance(symbol, from_date, to_date)
    return alert_success('Analysis Completed') + [dcc.Graph(figure = fig, className = 'param_est_graph')]