
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
import numpy as np
import dash

dash.register_page(__name__, path = '/bolinger', name = 'Bollinger Bands', order = '10')

# Page Layout
scenario_div = get_scenario_div([
	get_symbol_input(),
	get_date_range(from_date = get_jan_first(get_offset_date_str(get_today_str(), -365 * 3))),
	get_interval_input(),
	get_period_input(),
	get_analyze_button('bollinger')
])
out_tab = get_out_tab({
	'Plot': get_plot_div()
})
layout = get_page_layout('Bollinger Bands', scenario_div, None, out_tab)

# Triggered when Analyze button clicked
@callback(
	[
		Output('alert-dlg', 'is_open', allow_duplicate = True),
		Output('alert-msg', 'children', allow_duplicate = True),
		Output('alert-dlg', 'style', allow_duplicate = True),
		Output('out-plot', 'children', allow_duplicate = True)
	],
	Input('bollinger-analyze-button', 'n_clicks'),
	[
		State('symbol-input', 'value'),
		State('from-date-input', 'date'),
		State('to-date-input', 'date'),
		State('interval-input', 'value'),
		State('period-input', 'value')
	],
	prevent_initial_call = True
)
def on_analyze_clicked(n_clicks, symbol, from_date, to_date, interval, period):
	none_ret = [None] # Padding return values

	if n_clicks == 0: return alert_hide(none_ret)
	
	if symbol is None: return alert_error('Invalid symbol. Please select one and retry.', none_ret)
	if from_date is None: return alert_error('Invalid starting date. Please select one and retry.', none_ret)
	if to_date is None: return alert_error('Invalid ending date. Please select one and retry.', none_ret)
	if from_date > to_date: return alert_error('Invalid duration. Please check and retry.', none_ret)
	if interval is None: return alert_error('Invalid interval. Please select one and retry.', none_ret)
	if period is None: return alert_error('Invalid period. Please input a number and retry.', none_ret)
	
	stock_data = load_yf(symbol, from_date, to_date, interval, fit_today = True)
	stock_data.ta.bbands(length = int(period), append = True)

	fig = go.Figure(data = [
		get_candlestick(stock_data),
		go.Scatter(x = stock_data.index, y = stock_data[f'BBL_{period}_2.0'], name = 'Lower Band'),
		go.Scatter(x = stock_data.index, y = stock_data[f'BBM_{period}_2.0'], name = 'Moving Average'),
		go.Scatter(x = stock_data.index, y = stock_data[f'BBU_{period}_2.0'], name = 'Upper Band')
	])
	fig.update_layout(
		xaxis_range = [stock_data.index.min(), stock_data.index.max()],
		xaxis = dict(rangeslider = dict(visible = False)),
		yaxis = dict(rangemode = 'tozero'),
		height = 600
	)
	return alert_success('Analysis Completed') + [dcc.Graph(figure = fig, className = 'bollinger_graph')]
