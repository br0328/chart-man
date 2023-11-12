
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

dash.register_page(__name__, path = '/newhigh', name = 'New Highs', order = '01')

scenario_div = get_scenario_div([
	get_symbol_input(),
	get_interval_input(),
	get_analyze_button('new-high')
])
out_tab = get_out_tab({'Plot': get_plot_div()})

layout = get_page_layout('New Highs', scenario_div, html.Div(), out_tab)

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

	fig = go.Figure(
		data = [
			go.Candlestick(
				x = df.index,
				open = df['Open'],
				close = df['Close'],
				high = df['High'],
				low = df['Low']
			)
		]
	)
	fig.update_yaxes(type = 'log')
	fig.update_layout(xaxis_rangeslider_visible = False)
    
	fig.update_layout(
		yaxis_tickformat = "0",
		height = 600,
		margin = dict(t = 40, b = 40, r = 40),
		showlegend = False
	)
	lines = [
		dict(
			type = 'line',
			line = dict(
				color = 'blue',
				width = 1,
				dash = 'dot'
			),
			x0 = 0,
			x1 = 1,
			xref = 'paper',
			y0 = highs[max_idx],
			y1 = highs[max_idx]
		),
		dict(
			type = 'line',
			line = dict(
				color = 'blue',
				width = 0.5,
				dash = 'dot'
			),
			x0 = max_date,
			x1 = max_date,
			y0 = 0,
			y1 = 1,
			yref = 'paper'
		)
	]
	fig.add_annotation(
		dict(
			font = dict(color = "blue", size = 14),
			x = max_date,
			y = np.log10(min(df['Low'])),
			showarrow = False,
			text = max_date.strftime(' %d %b %Y'),
			xanchor = "left" if second_max_idx is None else "right",
			yanchor = "bottom"
		)
	)
	fig.add_annotation(
		dict(
			font = dict(color = "blue", size = 14),
			x = max_date,
			y = np.log10(highs[max_idx]),
			showarrow = False,
			text = '{:.4f}'.format(highs[max_idx]),
			xanchor = "left" if second_max_idx is None else "right",
			yanchor = "bottom"
		)
	)
	if second_max_idx is not None:
		lines.extend([
			dict(
				type = 'line',
				line = dict(
					color = 'brown',
					width = 1,
					dash = 'dot'
				),
				x0 = 0,
				x1 = 1,
				xref = 'paper',
				y0 = highs[second_max_idx],
				y1 = highs[second_max_idx]
			),
			dict(
				type = 'line',
				line = dict(
					color = 'brown',
					width = 0.5,
					dash = 'dot'
				),
				x0 = second_max_date,
				x1 = second_max_date,
				y0 = 0,
				y1 = 1,
				yref = 'paper'
			)
		])
		fig.add_annotation(
			dict(
				font = dict(color = "brown", size = 14),
				x = second_max_date,
				y = np.log10(min(df['Low'])),
				showarrow = False,
				text = second_max_date.strftime(' %d %b %Y'),
				xanchor = "left",
				yanchor = "bottom"
			)
		)
		fig.add_annotation(
			dict(
				font = dict(color = "brown", size = 14),
				x = second_max_date,
				y = np.log10(highs[second_max_idx]),
				showarrow = False,
				text = '{:.4f}'.format(highs[second_max_idx]),
				xanchor = "left",
				yanchor = "bottom"
			)
		)
	fig.update_layout(shapes = lines)

	return dcc.Graph(figure = fig, className = 'new_high_graph')
