
from dash import dcc, callback, Output, Input, State
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from constant import *
from compute import *
from config import *
from yahoo import *
from plot import *
from util import *
from data import *
from ui import *
import plotly.graph_objects as go
import pandas_ta as ta
import numpy as np
import dash

dash.register_page(__name__, path = '/trendline', name = 'Trendline', order = '03')

scenario_div = get_scenario_div([
    get_symbol_input(),
    get_date_range(),
    get_interval_input()
])
parameter_div = get_parameter_div([
	get_level_number_input(),
	get_analyze_button('trendline')
])
out_tab = get_out_tab({
    'Plot': get_plot_div(),
    'Report': get_report_div()
})
layout = get_page_layout('Trendline', scenario_div, parameter_div, out_tab)

# Triggered when Analyze button clicked
@callback(
    [
        Output('alert-dlg', 'is_open', allow_duplicate = True),
        Output('alert-msg', 'children', allow_duplicate = True),
        Output('alert-dlg', 'style', allow_duplicate = True),
        Output('out_tab', 'value', allow_duplicate = True),
        Output('out-plot', 'children', allow_duplicate = True),
        Output('out-report', 'children', allow_duplicate = True)
    ],
    Input('trendline-analyze-button', 'n_clicks'),
    [
        State('symbol-input', 'value'),
        State('from-date-input', 'date'),
        State('to-date-input', 'date'),
        State('interval-input', 'value'),
        State('level-input', 'value')
    ],
    prevent_initial_call = True
)
def on_analyze_clicked(n_clicks, symbol, from_date, to_date, interval, level):
    none_ret = ['Plot', None, None] # Padding return values

    if n_clicks == 0: return alert_hide(none_ret)
    
    if symbol is None: return alert_error('Invalid symbol. Please select one and retry.', none_ret)
    if from_date is None: return alert_error('Invalid starting date. Please select one and retry.', none_ret)
    if to_date is None: return alert_error('Invalid ending date. Please select one and retry.', none_ret)
    if from_date > to_date: return alert_error('Invalid duration. Please check and retry.', none_ret)
    if interval is None: return alert_error('Invalid interval. Please select one and retry.', none_ret)
    if level is None: return alert_error('Invalid level. Please select one and retry.', none_ret)

    level = int(level)
    df = load_yf(symbol, from_date, to_date, interval, fit_today = True)

    return alert_success('Analysis Completed') + ['Plot', update_plot(df, level), html.Div()]

# Triggered when Symbol combo box changed
@callback(
    [
        Output('from-date-input', 'date', allow_duplicate = True)
    ],
    Input('symbol-input', 'value'),
    [
        State('from-date-input', 'date')
    ],
    prevent_initial_call = True
)
def on_symbol_changed(symbol, from_date):
    if symbol is None: return [from_date]

    # Adjust start date considering IPO date of the symbol chosen
    ipo_date = load_stake().loc[symbol]['ipo']

    if from_date is None:
        from_date = ipo_date
    elif from_date < ipo_date:
        from_date = ipo_date

    return [from_date]

# Major plotting procedure
def update_plot(df, level):
    window = 3 * level
    all_dates = list(df.index)
    peaks_x, peaks_y = [], []

    for i in range(window, len(all_dates) - window):
        h, l = df.loc[all_dates[i]]['High'], df.loc[all_dates[i]]['Low']
        is_peak_h, is_peak_l = True, True

        for j in range(i - window, i + window):
            if df.loc[all_dates[j]]['High'] > h:
                is_peak_h = False
                break

        for j in range(i - window, i + window):
            if df.loc[all_dates[j]]['Low'] < l:
                is_peak_l = False
                break

        if is_peak_h:
            peaks_x.append(all_dates[i])
            peaks_y.append(h)
        elif is_peak_l:
            peaks_x.append(all_dates[i])
            peaks_y.append(l)

    # Calculate ATR using pandas_ta
    # atr = ta.atr(high = filtered_df['High'], low=filtered_df['Low'], close=filtered_df['Close'], length=14)

    # atr_multiplier = 2
    # stop_percentage = atr.iloc[-1] * atr_multiplier / filtered_df['Close'].iloc[-1]
    # profit_percentage = (1 + (level - 1) / 4) * stop_percentage
    # print(f"stoploss: {stop_percentage:.2f}%_takeprofit: {profit_percentage:.2f}%")

	# Set two subplots: primary chart and volume chart
    fig = make_subplots(rows = 2, cols = 1, shared_xaxes = True, vertical_spacing = 0.05, row_heights = [0.8, 0.2])
    
    # Draw candlestick and volume chart
    fig.add_trace(get_candlestick(df), row = 1, col = 1)
    fig.add_trace(get_volume_bar(df), row = 2, col = 1)

    fig.add_trace(
        go.Scatter(
            x = peaks_x,
            y = peaks_y,
            mode = 'markers',
            marker = dict(size = 5, color = 'white', line_color = 'blue', line_width = 1, symbol = 'square')
        ),
        row = 1, col = 1
    )
    update_shared_xaxes(fig, df, 2)

    fig.update_yaxes(title_text = 'Price', row = 1, col = 1)
    fig.update_yaxes(title_text = 'Volume', row = 2, col = 1)

    fig.update_layout(
        yaxis_tickformat = '0',
        height = 1200,
        margin = dict(t = 40, b = 40, r = 100),
        showlegend = False
    )
    return dcc.Graph(figure = fig, className = 'trendline_graph')
