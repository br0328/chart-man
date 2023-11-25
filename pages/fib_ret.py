
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
dash.register_page(__name__, path = '/fibret', name = 'Fibonacci Retracement', order = '07')

# Page Layout
scenario_div = get_scenario_div([
	get_symbol_input(),
	get_date_range(),
	get_interval_input(value = INTERVAL_MONTHLY),
    get_analyze_button('fib-ret')
])
# parameter_div = get_parameter_div([
# 	#get_cur_date_picker(),
# 	#get_pivot_number_input(),
# 	get_merge_thres_input(),
# 	get_analyze_button('fib-ext'),
# 	get_backtest_button('fib-ext')
# ])
out_tab = get_out_tab({
	'Plot': get_plot_div(),
	'Report': get_report_div()
})
layout = get_page_layout('Fibonacci|Retracement', scenario_div, None, out_tab)

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
	Input('fib-ret-analyze-button', 'n_clicks'),
	[
		State('symbol-input', 'value'),
		State('from-date-input', 'date'),
		State('to-date-input', 'date'),
		State('interval-input', 'value'),
		#State('cur-date-input', 'date'),
		#State('pivot-input', 'value'),
		#State('merge-input', 'value')
	],
	prevent_initial_call = True
)
#def on_analyze_clicked(n_clicks, symbol, from_date, to_date, interval, cur_date, pivot_number, merge_thres):
def on_analyze_clicked(n_clicks, symbol, from_date, to_date, interval):
    none_ret = ['Plot', None, None] # Padding return values

    if n_clicks == 0: return alert_hide(none_ret)

    if symbol is None: return alert_error('Invalid symbol. Please select one and retry.', none_ret)
    if from_date is None: return alert_error('Invalid starting date. Please select one and retry.', none_ret)
    if to_date is None: return alert_error('Invalid ending date. Please select one and retry.', none_ret)
    if from_date > to_date: return alert_error('Invalid duration. Please check and retry.', none_ret)
    if interval is None: return alert_error('Invalid interval. Please select one and retry.', none_ret)

    intervalSET = interval
    data, highs, lows = getPointsBest(symbol, startDate = from_date, endDate = to_date, interval = intervalSET, min_ = 0.5)
    
    y = np.log10(data["close"])
    yh = np.log10(data["high"])
    yl = np.log10(data["low"])
    x = np.linspace(0, len(data)-1, len(data))

    xScaler = Scaler(x)
    yScaler = Scaler(y)
    yy = yScaler.getScaled()
    xx = xScaler.getScaled()

    xScalerHigh = Scaler(x)
    yScalerHigh = Scaler(yh)
    yyh = yScaler.getScaled()
    xxh = xScaler.getScaled()

    xScalerLow = Scaler(x)
    yScalerLow = Scaler(yl)
    yyl = yScaler.getScaled()
    xxl = xScaler.getScaled()

    xx_array = np.array(xx)
    yy_array = np.array(yy)

    pairs0 = getPairs(xx_array, yy_array, 0.05)
    pairs1 = getPairs(xx_array, yy_array, 0.08)
    pairs2 = getPairs(xx_array, yy_array, 0.1)
    pairs11 = list(set(pairs1)|set(pairs2))

    pairs = []
    min__ = 100000000000000
    min_idx = 0
    
    for pp in pairs11:
        if yy[pp[0]] < yy[pp[1]]:
            min_chosen = yy[pp[0]]
            min_index_chosen = pp[0]
            max_chosen = yy[pp[1]]  
            max_index_chosen = pp[1]
            p = (pp[0], pp[1])  
        else:
            max_chosen = yy[pp[0]]
            max_index_chosen = pp[0]
            min_chosen = yy[pp[1]] 
            min_index_chosen = pp[1] 
            p = (pp[1], pp[0]) 
        if min__ > min_chosen:
            min__ = min_chosen 
            min_idx = min_index_chosen 

    for pp in pairs11:
        pairs.append( (min_idx, pp[1]) )

    figures = plotter1(pairs, xx, yy, xScaler, yScaler, intervalSET, data, y)
    return alert_success('Analysis Completed') + ['Plot', [dcc.Graph(figure = fig) for fig in figures], None]

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
