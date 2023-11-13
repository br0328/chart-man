
from dash import dcc, html, dash_table, callback, Output, Input, State
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
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import dash

# New High Page
dash.register_page(__name__, path = '/newhigh', name = 'New Highs', order = '01')

# Page Layout
scenario_div = get_scenario_div([
	get_symbol_input(),
	get_interval_input(),
	get_analyze_button('new-high')
])
out_tab = get_out_tab({'Plot': get_plot_div()})

layout = get_page_layout('New Highs', scenario_div, html.Div(), out_tab)

# Triggered when Analyze button clicked
@callback(
    [
        Output('alert-dlg', 'is_open', allow_duplicate = True),
        Output('alert-msg', 'children', allow_duplicate = True),
        Output('alert-dlg', 'style', allow_duplicate = True),
        Output('out-plot', 'children', allow_duplicate = True)
    ],
    Input('new-high-analyze-button', 'n_clicks'),
    [
        State('symbol-input', 'value'),
        State('interval-input', 'value')
    ],
    prevent_initial_call = True
)
def on_analyze_clicked(n_clicks, symbol, interval):
    none_ret = [None]

    if n_clicks == 0: return alert_hide(none_ret)
    
    if symbol is None: return alert_error('Invalid symbol. Please select one and retry.', none_ret)
    if interval is None: return alert_error('Invalid interval. Please select one and retry.', none_ret)
    
    df = load_yf(symbol, '1800-01-01', '2100-01-01', interval, fit_today = True)

    return alert_success('Analysis Completed') + [update_plot(df, interval)]

# If it has the new highest price today, plot both today's price and previous highest price.
# If today's price is not the new highest, plot only the highest price.
def update_plot(df, interval):
	margin = NEW_HIGH_DAY_MARGIN if interval == INTERVAL_DAILY else 5
	highs = df['High'].to_numpy()

	max_idx = highs.argmax()
	max_date = df.iloc[max_idx].name   

	if max_idx >= len(df) - margin:
		second_max_idx = highs[:-margin].argmax()
		second_max_date = df.iloc[second_max_idx].name

		df = df.iloc[second_max_idx - margin:]
	else:
		second_max_idx = None
		df = df.iloc[max_idx - margin:]

	fig = go.Figure(data = [get_candlestick(df)])

	fig.update_yaxes(type = 'log')
	fig.update_layout(xaxis_rangeslider_visible = False)
    
	fig.update_layout(
		yaxis_tickformat = "0",
		height = 600,
		margin = dict(t = 40, b = 40, r = 40),
		showlegend = False
	)
	lines = [
		get_complete_hline(highs[max_idx], 'blue', 'dot', 0.5),
		get_complete_vline(max_date, 'blue', 'dot', 0.5)
	]
	draw_annotation(fig, max_date, np.log10(min(df['Low'])), max_date.strftime(' %d %b %Y'),
		xanchor = 'left' if second_max_idx is None else 'right', yanchor = 'bottom', color = 'blue', size = 14)
	draw_annotation(fig, max_date, np.log10(highs[max_idx]), '{:.4f}'.format(highs[max_idx]),
		xanchor = 'left' if second_max_idx is None else 'right', yanchor = 'bottom', color = 'blue', size = 14)

	if second_max_idx is not None:
		lines.extend([
			get_complete_hline(highs[second_max_idx], 'brown', 'dot', 0.5),
			get_complete_vline(second_max_date, 'brown', 'dot', 0.5)
		])
		draw_annotation(fig, second_max_date, np.log10(min(df['Low'])), second_max_date.strftime(' %d %b %Y'),
			xanchor = 'left', yanchor = 'bottom', color = 'brown', size = 14)
		draw_annotation(fig, second_max_date, np.log10(highs[second_max_idx]), '{:.4f}'.format(highs[second_max_idx]),
			xanchor = 'left', yanchor = 'bottom', color = 'brown', size = 14)

	fig.update_layout(shapes = lines)

	return dcc.Graph(figure = fig, className = 'new_high_graph')
