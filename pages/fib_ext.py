
from dash import dcc, html, callback, Output, Input, State
from plotly.subplots import make_subplots
from constant import *
from compute import *
from config import *
from yahoo import *
from plot import *
from util import *
from data import *
from ui import *
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
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
plot_div = get_plot_div()

layout = get_page_layout('Fibonacci|Extension', scenario_div, parameter_div, plot_div)

@callback(
    [Output('alert-dlg', 'is_open'), Output('alert-msg', 'children'), Output('alert-dlg', 'style'), Output('out-plot', 'children')],
    Input('load-button', 'n_clicks'),
    [State('symbol-input', 'value'), State('from-date-input', 'date'), State('to-date-input', 'date'), State('interval-input', 'value')]
)
def on_load_clicked(n_clicks, symbol, from_date, to_date, interval):
    none_count = 1

    if n_clicks == 0: return alert_hide(none_count)
    
    if symbol is None: return alert_error('Invalid symbol. Please select one and retry.', none_count)
    if from_date is None: return alert_error('Invalid starting date. Please select one and retry.', none_count)
    if to_date is None: return alert_error('Invalid ending date. Please select one and retry.', none_count)
    if from_date > to_date: return alert_error('Invalid duration. Please check and retry.', none_count)
    if interval is None: return alert_error('Invalid interval. Please select one and retry.', none_count)
    
    if get_duration(from_date, to_date) < zigzag_window:
        return alert_error('Duration must be at least {} days for Fibonacci analysis.'.format(zigzag_window + zigzag_padding), none_count)
    
    df = load_yf(symbol, from_date, to_date, interval)
    msg = 'Scenario was loaded successfully. Please analyze it.'

    zdf = get_zigzag(df)
    downfalls = get_recent_downfalls(zdf, 4)

    print(downfalls)

    return alert_success(msg) + [update_plot(df, zdf)]

@callback(
    [Output('from-date-input', 'date')],
    Input('symbol-input', 'value'),
    [State('from-date-input', 'date')]
)
def on_symbol_changed(symbol, from_date):
    if symbol is None: return [from_date]

    ipo_date = load_stake().loc[symbol]['ipo']

    if from_date is None: return [ipo_date]
    return [ipo_date] if from_date < ipo_date else [from_date]

def update_plot(df, zdf):
    fig = make_subplots(rows = 2, cols = 1, shared_xaxes = True, vertical_spacing = 0.05, row_heights = [0.8, 0.2])

    fig.add_trace(get_candlestick(df), row = 1, col = 1)
    fig.add_trace(get_volume_bar(df), row = 2, col = 1)
    fig.add_trace(get_quick_line(zdf, 'Close'), row = 1, col = 1)

    update_shared_xaxes(fig, df, 2)

    fig.update_yaxes(type = "log", title_text = "Price", row = 1, col = 1)
    fig.update_yaxes(title_text = "Volume", row = 2, col = 1)

    fig.update_layout(
        yaxis_tickformat = "0",
        height = 1200,
        margin = dict(t = 40, b = 40),
        showlegend = False
    )
    return dcc.Graph(figure = fig)
