
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

# Fibonacci Extension Page
dash.register_page(__name__, path = '/fibext', name = 'Fibonacci Extension', order = '07')

# Page Layout
scenario_div = get_scenario_div([
	get_symbol_input(),
	get_date_range(),
	get_interval_input()
])
parameter_div = get_parameter_div([
	#get_cur_date_picker(),
	#get_pivot_number_input(),
	get_merge_thres_input(),
	get_analyze_button('fib-ext'),
	get_backtest_button('fib-ext')
])
out_tab = get_out_tab({
	'Plot': get_plot_div(),
	'Report': get_report_div()
})
layout = get_page_layout('Fibonacci|Extension', scenario_div, parameter_div, out_tab)

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
	Input('fib-ext-analyze-button', 'n_clicks'),
	[
		State('symbol-input', 'value'),
		State('from-date-input', 'date'),
		State('to-date-input', 'date'),
		State('interval-input', 'value'),
		#State('cur-date-input', 'date'),
		#State('pivot-input', 'value'),
		State('merge-input', 'value')
	],
	prevent_initial_call = True
)
#def on_analyze_clicked(n_clicks, symbol, from_date, to_date, interval, cur_date, pivot_number, merge_thres):
def on_analyze_clicked(n_clicks, symbol, from_date, to_date, interval, merge_thres):
	none_ret = ['Plot', None, None] # Padding return values

	if n_clicks == 0: return alert_hide(none_ret)
	
	if symbol is None: return alert_error('Invalid symbol. Please select one and retry.', none_ret)
	if from_date is None: return alert_error('Invalid starting date. Please select one and retry.', none_ret)
	if to_date is None: return alert_error('Invalid ending date. Please select one and retry.', none_ret)
	if from_date > to_date: return alert_error('Invalid duration. Please check and retry.', none_ret)
	if interval is None: return alert_error('Invalid interval. Please select one and retry.', none_ret)
	
	# If duration is too short, Fibonacci analysis is not feasible.
	if get_duration(from_date, to_date) < zigzag_window + zigzag_padding:
		return alert_error('Duration must be at least {} days for Fibonacci analysis.'.format(zigzag_window + zigzag_padding), none_ret)

	# if cur_date is None: return alert_error('Invalid current date. Please select one and retry.', none_ret)
	# if cur_date < from_date or cur_date > to_date: return alert_error('Invalid current date. Please select one in scenario duration and retry.', none_ret)
	# if pivot_number is None: return alert_error('Invalid pivot number. Please select one and retry.', none_ret)
	cur_date = get_today_str()
	pivot_number = PIVOT_NUMBER_FOUR

	try:
		merge_thres = float(merge_thres) / 2 / 100
	except Exception:
		return alert_error('Invalid merge threshold. Please input correctly and retry.', none_ret)
	
	#df = load_yf(symbol, from_date, to_date, interval, fit_today = True)
	#cur_date = get_nearest_backward_date(df, get_timestamp(cur_date)) # Adjust the pivot date to the nearest valid point

	#if cur_date is None: return alert_warning('Nearest valid date not found. Please reselect current date.', none_ret)
	
	#zdf = get_zigzag(df, cur_date) # Find all possible Fibonacci pivot pairs
	pivot_number = PIVOT_NUMBER_ALL.index(pivot_number) + 1

	df, downfalls = get_recent_downfalls_old(symbol, from_date, to_date, pivot_number) # Reduce Fibonacci pivot pairs into only recent ones
	#extensions = get_fib_extensions(zdf, downfalls, get_safe_num(merge_thres), df.iloc[-1]['Close'] * 2) # Merge and sort Fibonacci extension levels
	
	extensions = get_fib_extensions(df, downfalls, get_safe_num(merge_thres), df.iloc[-1]['close'] * 0.05, df.iloc[-1]['close'] * 5) # Merge and sort Fibonacci extension levels
	behaviors = get_fib_ext_behaviors(df, extensions, cur_date, get_safe_num(merge_thres)) # Compute behaviors of each extension level

	# Generate table report
	records = analyze_fib_extension(df, extensions, behaviors, cur_date, pivot_number, merge_thres, interval, symbol)

	ddf = load_yf(symbol, from_date, to_date, interval, fit_today = True)
	cur_date = get_nearest_backward_date(ddf, get_timestamp(cur_date)) # Adjust the pivot date to the nearest valid point
 
	# CSV output and visualization
	csv_path = 'out/FIB-EXT-ANALYZE_{}_{}_{}_{}_{}_p{}_m{}%.csv'.format(
		symbol, from_date, cur_date.strftime(YMD_FORMAT), to_date, interval, pivot_number, '{:.1f}'.format(2 * 100 * merge_thres)
	)
	records.to_csv(csv_path, index = False)
	report = get_report_content(records, csv_path)

	return alert_success('Analysis Completed') + ['Plot', update_plot(df, downfalls, extensions, behaviors, cur_date), report]

# Triggered when Symbol combo box changed
@callback(
	[
		Output('from-date-input', 'date', allow_duplicate = True),
		#Output('cur-date-input', 'date', allow_duplicate = True)
	],
	Input('symbol-input', 'value'),
	[
		State('from-date-input', 'date'),
		#State('cur-date-input', 'date')
	],
	prevent_initial_call = True
)
#def on_symbol_changed(symbol, from_date, cur_date):
def on_symbol_changed(symbol, from_date):
	#if symbol is None: return [from_date, cur_date]
	if symbol is None: return [from_date]

	# Adjust start date considering IPO date of the symbol chosen
	ipo_date = load_stake().loc[symbol]['ipo']

	if from_date is None:
		from_date = ipo_date
	elif from_date < ipo_date:
		from_date = ipo_date

	# If pivot date is not selected yet, automatically sets it as the 2/3 point of [start-date, end-date] range.
	# if cur_date is None:
	# 	from_date = get_timestamp(from_date)
	# 	days = (datetime.now() - from_date).days

	# 	cur_date = (from_date + timedelta(days = days * 2 // 3)).strftime(YMD_FORMAT)
	# 	from_date = from_date.strftime(YMD_FORMAT)

	#return [from_date, cur_date]
	return [from_date]

# Triggered when Backtest button clicked
@callback(
	[
		Output('alert-dlg', 'is_open', allow_duplicate = True),
		Output('alert-msg', 'children', allow_duplicate = True),
		Output('alert-dlg', 'style', allow_duplicate = True),
		Output('out_tab', 'value', allow_duplicate = True),
		Output('out-report', 'children', allow_duplicate = True)
	],
	Input('fib-ext-backtest-button', 'n_clicks'),
	[
		State('symbol-input', 'value'),
		State('from-date-input', 'date'),
		State('to-date-input', 'date'),
		State('interval-input', 'value'),
		#State('pivot-input', 'value'),
		State('merge-input', 'value')
	],
	prevent_initial_call = True
)
#def on_backtest_clicked(n_clicks, symbol, from_date, to_date, interval, pivot_number, merge_thres):
def on_backtest_clicked(n_clicks, symbol, from_date, to_date, interval, merge_thres):
	none_ret = ['Report', None]

	if n_clicks == 0: return alert_hide(none_ret)
	
	if symbol is None: return alert_error('Invalid symbol. Please select one and retry.', none_ret)
	if from_date is None: return alert_error('Invalid starting date. Please select one and retry.', none_ret)
	if to_date is None: return alert_error('Invalid ending date. Please select one and retry.', none_ret)
	if from_date > to_date: return alert_error('Invalid duration. Please check and retry.', none_ret)
	if interval is None: return alert_error('Invalid interval. Please select one and retry.', none_ret)
	if interval == INTERVAL_QUARTERLY or interval == INTERVAL_YEARLY: return alert_error('Cannot support quarterly or monthly backtest.', none_ret)
	#if pivot_number is None: return alert_error('Invalid pivot number. Please select one and retry.', none_ret)
	
	pivot_number = PIVOT_NUMBER_FOUR
 
	# If duration is too short, Fibonacci backtest is not feasible.
	if get_duration(from_date, to_date) < zigzag_window + zigzag_padding:
		return alert_error('Duration must be at least {} days for Fibonacci analysis.'.format(zigzag_window + zigzag_padding), none_ret)

	try:
		merge_thres = float(merge_thres) / 2 / 100
	except Exception:
		return alert_error('Invalid merge threshold. Please input correctly and retry.', none_ret)
	
	df = load_yf(symbol, from_date, to_date, interval)
	pivot_number = PIVOT_NUMBER_ALL.index(pivot_number) + 1

	# Get results of backtest for file output and visualization
	# records: table-format data
	# success_rate: accuracy of transaction positions
	# cum_profit: cumulated profit on percentage basis	
	records, success_rate, cum_profit = backtest_fib_extension(
		df, interval, pivot_number, get_safe_num(merge_thres), symbol
	)
	csv_path = 'out/FIB-EXT-BKTEST_{}_{}_{}_{}_p{}_m{}%_sr={}%_cp={}%.csv'.format(
		symbol, from_date, to_date, interval, pivot_number,
		'{:.1f}'.format(2 * 100 * merge_thres),
		'{:.1f}'.format(100 * success_rate),
		'{:.1f}'.format(100 * cum_profit)
	)
	records.to_csv(csv_path, index = False)
	report = get_report_content(records, csv_path)

	return alert_success('Backtest Complted.') + ['Report', report]

# Major plotting procedure
def update_plot(df, downfalls, extensions, behaviors, cur_date):
	cur_date = df.index[-1]
	df = df.rename(columns = {"open": "Open", "high": "High", "low": "Low", "volume": "Volume", "close": "Close"})
    
	# Set two subplots: primary chart and volume chart
	fig = make_subplots(rows = 2, cols = 1, shared_xaxes = True, vertical_spacing = 0.05, row_heights = [0.8, 0.2])
	day_span = (df.index[-1] - df.index[0]).days # Total span of duration

	pivot_wid = timedelta(days = int(day_span * 0.025)) # Length of line starting from a downfall pivot point
	fold_step, dash_indent = 0.01, 0.07 # Indents and steps for clear visualization of extension level values

	cur_price = df.iloc[-1]['Close'] # The price of the pivot date
	extensions.sort(key = lambda g: abs(cur_price - (g[0][-1] + g[-1][-1]) / 2)) # Sort extension levels by cloeness to the pivot price

	upper_count, lower_count = 0, 0 # Numbers of upper and lower levels compared to the pivot price (Only 5 level pricess are annotated)

	# Loop of extension levels
	# g: each extension level group (len(g) = 1 for a single level, len(g) > 1 for a merged level)
	for g in extensions:
		b = behaviors[g[0]] # Behavior of level group g (Breakout, Support, Resistance, etc.)

		if len(g) > 1: # For a merged level
			lv = (g[0][-1] + g[-1][-1]) / 2 # Mid-price of a merged level

			# Draw tonexty filled line with merged width
			draw_tonexty_hline(fig, df.index, g[0][-1], g[-1][-1], 'darkgray', 0.5)

			for i, e in enumerate(g):
				# Plot colored markers to show which downfalls formed this merged group
				draw_marker(fig, df.index[int(len(df) * (i + 1) * fold_step)], lv, 'circle', PLOT_COLORS_DARK[e[0]])

			# bmark_x, bmark_y: The position of behavior marker
			bmark_x = df.index[int(len(df) * (len(g) + 2) * fold_step)]
			bmark_y = lv
		else:
			# For a single level g, Parse the level tuple
			# i: the index of pivot downfall
			# hd, zd: hundred date and zero date
			# hv, zv: hundred price and zero price
			# j: the index of Fibonacci level
			# lv: the price of the level
			i, hd, zd, hv, zv, j, lv = g[0]
			x = df.index[0] + timedelta(days = int(day_span * (dash_indent * (i + 2) + 0.005))) # Calculate x-indent

			draw_hline_shape(fig, x, df.index[-1], lv, PLOT_COLORS_DARK[i], dash = 'dot') # Draw dot line
			draw_text(fig, x, lv, '{:.1f}%	 '.format(FIB_EXT_LEVELS[j] * 100), 'middle left', PLOT_COLORS_DARK[i]) # Draw percent level text

			# bmark_x, bmark_y: The position of behavior marker
			bmark_x = df.index[0] + timedelta(days = int(day_span * (dash_indent * (i + 2) - 0.004)))
			bmark_y = lv

		# Mid-price of a merged level (for a single level, its price value itself)
		lv = (g[0][-1] + g[-1][-1]) / 2
		is_near = False

		if lv > cur_price:
			if upper_count < 5:
				is_near = True
				upper_count += 1
		elif lv < cur_price:
			if lower_count < 5:
				is_near = True
				lower_count += 1

		if is_near: # Only 5 upper levels and 5 lower levels are annotated
			draw_annotation(fig, 1, np.log10(lv), '- {:.1f}'.format(lv), xref = 'paper', xanchor = 'left',
				color = 'black' if len(g) > 1 else PLOT_COLORS_DARK[g[0][0]])

		# Plot behavior marks
		if b is not None:
			draw_marker(fig, bmark_x, bmark_y, FIB_EXT_MARKERS[b][0], FIB_EXT_MARKERS[b][1], 7.5, FIB_EXT_MARKERS[b][2], 'black')

	# Visualize pivot downfalls
	# Reversed loop due to overlay issues
	for j, f in enumerate(downfalls[::-1]):
		hd, zd = f # Hundred date and zero date
		i = len(downfalls) - 1 - j # Original index

		draw_hline_shape(fig, hd, zd + pivot_wid, df.loc[hd]['Close'], PLOT_COLORS_DARK[i], 1.5) # Hundred line plot
		draw_hline_shape(fig, zd, zd + pivot_wid, df.loc[zd]['Close'], PLOT_COLORS_DARK[i], 1.5) # Zero line plot

		draw_text(fig, hd, df.loc[hd]['Close'], '{:.1f}'.format(df.loc[hd]['Close']), 'top center', PLOT_COLORS_DARK[i]) # Hundred price plot
		draw_text(fig, zd, df.loc[zd]['Close'], '{:.1f}'.format(df.loc[zd]['Close']), 'bottom center', PLOT_COLORS_DARK[i]) # Zero price plot

	# Pivot price plot
	draw_hline_shape(fig, cur_date, df.index[-1], cur_price, 'red', 2, 'dash')
	draw_annotation(fig, 1, np.log10(cur_price), '------- {:.1f}'.format(cur_price), 'paper', xanchor = 'left', color = 'red', size = 14)

	# Pivot date plot
	# draw_vline_shape(fig, cur_date, min(df['Low']), max(df['High']) if len(extensions) == 0 else extensions[-1][-1][-1], 'darkgreen')
	# draw_annotation(fig, cur_date, np.log10(min(df['Low'])), cur_date.strftime(' %d %b %Y') + ' →',
	# 	xanchor = 'left', yanchor = 'bottom', color = 'darkgreen', size = 14)

	# Draw candlestick and volume chart
	fig.add_trace(get_candlestick(df), row = 1, col = 1)
	fig.add_trace(get_volume_bar(df), row = 2, col = 1)

	ma_50d = df['Close'].rolling(50).mean().round(4)
	ma_200d = df['Close'].rolling(200).mean().round(4)
    
	fig.add_trace(
		go.Scatter(
			x = df.index,
			y = ma_50d,
			name = 'MA-50D (Current: ${:.4f})'.format(ma_50d[-1]),
			line = dict(color = 'blue')
		),
		row = 1, col = 1
	)
	fig.add_trace(
		go.Scatter(
			x = df.index,
			y = ma_200d,
			name = 'MA-200D (Current: ${:.4f})'.format(ma_200d[-1]),
			line = dict(color = 'orange')
		),
		row = 1, col = 1
	)    
	update_shared_xaxes(fig, df, 2)

	# Apply logarithm to candlestick y axis
	fig.update_yaxes(type = 'log', title_text = 'Price', row = 1, col = 1)
	fig.update_yaxes(title_text = 'Volume', row = 2, col = 1)

	fig.update_layout(
		yaxis_tickformat = '0',
		height = 1200,
		margin = dict(t = 40, b = 40, r = 100),
		legend = dict(
			yanchor = "top",
			y = 0.99,
			xanchor = "left",
			x = 0.01
		)
	)
	return dcc.Graph(figure = fig, className = 'fib_ext_graph')
