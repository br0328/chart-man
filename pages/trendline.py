
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

dash.register_page(__name__, path = '/trendline', name = 'Trendline', order = '03')

scenario_div = get_scenario_div([
	get_symbol_input(),
	get_date_range(from_date = get_jan_first(get_offset_date_str(get_today_str(), -365))),
	get_interval_input()
])
parameter_div = get_parameter_div([
    get_cur_date_picker(),
	get_level_number_input('2'),
	get_analyze_button('trendline'),
	get_backtest_button('trendline')
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
		State('cur-date-input', 'date'),
		State('interval-input', 'value'),
		State('level-input', 'value')
	],
	prevent_initial_call = True
)
def on_analyze_clicked(n_clicks, symbol, from_date, to_date, cur_date, interval, level):
	none_ret = ['Plot', None, None] # Padding return values

	if n_clicks == 0: return alert_hide(none_ret)
	
	if symbol is None: return alert_error('Invalid symbol. Please select one and retry.', none_ret)
	if from_date is None: return alert_error('Invalid starting date. Please select one and retry.', none_ret)
	if to_date is None: return alert_error('Invalid ending date. Please select one and retry.', none_ret)
	if from_date > to_date: return alert_error('Invalid duration. Please check and retry.', none_ret)
	if interval is None: return alert_error('Invalid interval. Please select one and retry.', none_ret)
	if level is None: return alert_error('Invalid level. Please select one and retry.', none_ret)
	if cur_date is None: return alert_error('Invalid current date. Please select one and retry.', none_ret)
 
	if cur_date < from_date: cur_date = from_date
	if cur_date > to_date: cur_date = to_date
 
	level = int(level)
	df = load_yf(symbol, from_date, to_date, interval, fit_today = True)

	cur_date = get_nearest_backward_date(df, get_timestamp(cur_date)) # Adjust the pivot date to the nearest valid point
	if cur_date is None: return alert_warning('Nearest valid date not found. Please reselect current date.', none_ret)
 
	output = alert_trendline(df, symbol, cur_date, interval) 
	return alert_success('Analysis Completed') + ['Plot', update_plot(df, cur_date, level), html.Div(output)]

# Triggered when Symbol combo box changed
@callback(
	[
		Output('from-date-input', 'date', allow_duplicate = True),
		Output('cur-date-input', 'date', allow_duplicate = True)
	],
	Input('symbol-input', 'value'),
	[
		State('from-date-input', 'date'), State('cur-date-input', 'date')
	],
	prevent_initial_call = True
)
def on_symbol_changed(symbol, from_date, cur_date):
	if symbol is None: return [from_date, cur_date]

	# Adjust start date considering IPO date of the symbol chosen
	ipo_date = load_stake().loc[symbol]['ipo']

	if from_date is None:
		from_date = ipo_date
	elif from_date < ipo_date:
		from_date = ipo_date

	# If pivot date is not selected yet, automatically sets it as the 2/3 point of [start-date, end-date] range.
	if cur_date is None:
		from_date = get_timestamp(from_date)
		cur_date = datetime.now()
		#days = (datetime.now() - from_date).days

		#cur_date = (from_date + timedelta(days = days * 2 // 3)).strftime(YMD_FORMAT)
		from_date = from_date.strftime(YMD_FORMAT)
		cur_date = cur_date.strftime(YMD_FORMAT)

	return [from_date, cur_date]

# Triggered when Backtest button clicked
@callback(
	[
		Output('alert-dlg', 'is_open', allow_duplicate = True),
		Output('alert-msg', 'children', allow_duplicate = True),
		Output('alert-dlg', 'style', allow_duplicate = True),
		Output('out_tab', 'value', allow_duplicate = True),
		Output('out-report', 'children', allow_duplicate = True)
	],
	Input('trendline-backtest-button', 'n_clicks'),
	[
		State('symbol-input', 'value'),
		State('from-date-input', 'date'),
		State('to-date-input', 'date'),
		State('interval-input', 'value')
	],
	prevent_initial_call = True
)
def on_backtest_clicked(n_clicks, symbol, from_date, to_date, interval):
	none_ret = ['Report', None]

	if n_clicks == 0: return alert_hide(none_ret)
	
	if symbol is None: return alert_error('Invalid symbol. Please select one and retry.', none_ret)
	if from_date is None: return alert_error('Invalid starting date. Please select one and retry.', none_ret)
	if to_date is None: return alert_error('Invalid ending date. Please select one and retry.', none_ret)
	if from_date > to_date: return alert_error('Invalid duration. Please check and retry.', none_ret)
	if interval is None: return alert_error('Invalid interval. Please select one and retry.', none_ret)

	df = load_yf(symbol, from_date, to_date, interval, fit_today = True)

	# Get results of backtest for file output and visualization
	# records: table-format data
	# success_rate: accuracy of transaction positions
	# cum_profit: cumulated profit on percentage basis	
	records, success_rate, cum_profit = backtest_trendline(df, symbol, from_date, to_date, interval)
 
	csv_path = 'out/TRENDLINE-BKTEST_{}_{}_{}_{}_sr={}%_cp={}%.csv'.format(
		symbol, from_date, to_date, interval,
		'{:.1f}'.format(100 * success_rate),
		'{:.1f}'.format(100 * cum_profit)
	)
	records.to_csv(csv_path, index = False)
	report = get_report_content(records, csv_path)

	return alert_success('Backtest Complted.') + ['Report', report]

# Major plotting procedure
def update_plot(df, cur_date, level):
	df_copy = df.copy()
	df = df[:cur_date]

	df['ID'] = range(len(df))
	df['Date'] = list(df.index)	
	df.set_index('ID', inplace = True)
 
	window = 3 * level	
	df['isPivot'] = 0
	
	all_dates = list(df['Date'])
	peaks_x, peaks_y = [], []

	for i in range(window, len(df) - window):
		h, l = df.loc[i, 'High'], df.loc[i, 'Low']
		is_peak_h, is_peak_l = True, True

		for j in range(i - window, i + window):
			if df.loc[j, 'High'] > h:
				is_peak_h = False
				break

		for j in range(i - window, i + window):
			if df.loc[j, 'Low'] < l:
				is_peak_l = False
				break

		if is_peak_h:
			peaks_x.append(all_dates[i])
			peaks_y.append(h)
		elif is_peak_l:
			peaks_x.append(all_dates[i])
			peaks_y.append(l)
   
		if is_peak_h and is_peak_l:
			df.loc[i, 'isPivot'] = 3
		elif is_peak_h:
			df.loc[i, 'isPivot'] = 1
		elif is_peak_l:
			df.loc[i, 'isPivot'] = 2		

	# Calculate ATR using pandas_ta
	atr = ta.atr(high = df['High'], low = df['Low'], close = df['Close'], length = 14)
	stop_percentage = 2 * atr.iloc[-1] / df['Close'].iloc[-1]
	
	backcandles = 10 * window
	df['isBreakOut'] = 0
  
	for i in range(backcandles + window, len(df)):
		df.loc[i, 'isBreakOut'] = is_breakout(i, backcandles, window, df, stop_percentage)

	df['breakpointpos'] = df.apply(calculate_breakpoint_pos, axis = 1)	
	df_breakout = df[df['isBreakOut'] != 0]
	 
	fig = make_subplots(rows = 2, cols = 1, shared_xaxes = True, vertical_spacing = 0.05, row_heights = [0.8, 0.2])

	for candle in range(backcandles + window, len(df) - 1):
		if df.iloc[candle].isBreakOut != 0:
			best_back_l, sl_lows, interc_lows, r_sq_l, best_back_h, sl_highs, interc_highs, r_sq_h = collect_channel(candle, backcandles, window, df)
			
			extended_x = np.array(df.index[candle + 1:candle + 15])
			x1 = np.array(df.index[candle - best_back_l - window:candle + 1])
			x2 = np.array(df.index[candle - best_back_h - window:candle + 1])
   
			extended_x = np.array([extended_x[0], extended_x[-1]])
			x1 = np.array([x1[0], x1[-1]])
			x2 = np.array([x2[0], x2[-1]])

			if r_sq_l >= 0.80 and df.iloc[candle].isBreakOut == 1 and abs(sl_lows) > 0.25 :
				extended_y_lows = sl_lows * extended_x + interc_lows
				fig.add_trace(
					go.Scatter(
						x = df.loc[extended_x, 'Date'],
						y = extended_y_lows,
						mode = 'lines',
						line = dict(dash = 'dash'),
						name = 'Lower Slope',
						showlegend = False
					),
					row = 1,
					col = 1
				)
				fig.add_trace(
					go.Scatter(
						x = df.loc[x1, 'Date'],
						y = sl_lows * x1 + interc_lows,
						mode = 'lines',
						name = 'Lower Slope',
						showlegend = False
					),
					row=  1, col = 1
				)

			if r_sq_h >= 0.80 and df.iloc[candle].isBreakOut == 2:				
				extended_y_highs = sl_highs * extended_x + interc_highs
				fig.add_trace(
					go.Scatter(
						x = df.loc[extended_x, 'Date'],
						y = extended_y_highs,
						mode = 'lines',
						line = dict(dash = 'dash'),
						name = 'Max Slope',
						showlegend = False
					),
					row = 1,
					col = 1
				)
				fig.add_trace(
					go.Scatter(
						x = df.loc[x2, 'Date'],
						y = sl_highs * x2 + interc_highs,
						mode = 'lines',
						name = 'Max Slope',
						showlegend = False
					),
					row = 1, col = 1
				)

	colors = ['black', 'blue']
	df.set_index('Date', inplace = True)
	
	for color in colors:
		mask = df_breakout['isBreakOut'].map({1: 'black', 2: 'blue'}) == color
		fig.add_trace(
			go.Scatter(
				x = df_breakout['Date'][mask],
				y = df_breakout['breakpointpos'][mask],
				mode = "markers",
				marker = dict(
					size = 5,
					color = color
				),
				marker_symbol = "hexagram",
				name = "Breakout",
				legendgroup = color
			),
			row = 1,
			col = 1
		)
	
	# Draw candlestick and volume chart
	fig.add_trace(get_candlestick(df_copy), row = 1, col = 1)
	fig.add_trace(get_volume_bar(df_copy), row = 2, col = 1)

	fig.add_trace(
		go.Scatter(
			x = peaks_x,
			y = peaks_y,
			mode = 'markers',
			marker = dict(size = 7.5, color = 'MediumPurple'),
   			name = 'Pivot'
		),
		row = 1, col = 1
	)
 	
  	# Pivot date plot
	draw_vline_shape(fig, cur_date, min(df_copy['Low']), max(df_copy['High']), 'darkgreen')
	draw_annotation(fig, cur_date, min(df_copy['Low']), cur_date.strftime(' %d %b %Y') + ' →',
		xanchor = 'left', yanchor = 'bottom', color = 'darkgreen', size = 14)
 
	fig.update_traces(
		name = "Break Below",
		selector = dict(legendgroup = "black")
	)
	fig.update_traces(
		name = "Break Above",
		selector = dict(legendgroup = "blue")
	)
	update_shared_xaxes(fig, df_copy, 2)

	fig.update_yaxes(title_text = 'Price', row = 1, col = 1)
	fig.update_yaxes(title_text = 'Volume', row = 2, col = 1)

	fig.update_layout(
		yaxis_tickformat = '0',
		height = 900,
		margin = dict(t = 40, b = 40, r = 100)
	)
	return dcc.Graph(figure = fig, className = 'trendline_graph')
