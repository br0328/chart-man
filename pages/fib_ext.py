
from dash import dcc, html, callback, Output, Input, State
from constant import *
from yahoo import *
from util import *
from ui import *
import dash_bootstrap_components as dbc
import plotly.express as px
import dash

dash.register_page(__name__, path = '/fibext', name = 'Fibonacci Extension')

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

layout = get_page_layout('Fibonacci|Extension', scenario_div, parameter_div)

@callback(
    [Output('alert-dlg', 'is_open'), Output('alert-msg', 'children'), Output('alert-dlg', 'style')],
    Input('load-button', 'n_clicks'),
    [State('symbol-input', 'value'), State('from-date-input', 'date'), State('to-date-input', 'date'), State('interval-input', 'value')]
)
def on_load_clicked(n_clicks, symbol, from_date, to_date, interval):
    if n_clicks == 0: return alert_hide()
    
    if symbol is None: return alert_error('Invalid symbol. Please select one and retry.')
    if from_date is None: return alert_error('Invalid starting date. Please select one and retry.')
    if to_date is None: return alert_error('Invalid ending date. Please select one and retry.')
    if from_date > to_date: return alert_error('Invalid duration. Please check and retry.')
    if interval is None: return alert_error('Invalid interval. Please select one and retry.')
    
    df = load_yf(symbol, from_date, to_date, interval)

    msg = 'Scenario was loaded successfully. Please analyze it.'

    return alert_warning(msg)
