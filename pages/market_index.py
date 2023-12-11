
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

dash.register_page(__name__, path = '/marketindex', name = 'Market Indices Divergence', order = '06')

scenario_div = get_scenario_div([
	get_date_range(from_date = get_jan_first(get_offset_date_str(get_today_str(), -365))),
    get_analyze_button('market-index'),
])
out_tab = get_out_tab({
	'Plot': html.Div([html.Div(id = 'out-plot-{}'.format(i)) for i in range(6)]),
	'Report': get_report_div()
})
layout = get_page_layout('Market Indices|Divergence', scenario_div, None, out_tab)

# Triggered when Analyze button clicked
@callback(
	[
		Output('alert-dlg', 'is_open', allow_duplicate = True),
		Output('alert-msg', 'children', allow_duplicate = True),
		Output('alert-dlg', 'style', allow_duplicate = True)
    ] +
    [
		Output('out-plot-{}'.format(i), 'children', allow_duplicate = True) for i in range(6)		
	] +
    [
        Output('out-report', 'children', allow_duplicate = True)
    ],
	Input('market-index-analyze-button', 'n_clicks'),
	[
		State('from-date-input', 'date'),
		State('to-date-input', 'date')
	],
	prevent_initial_call = True
)
def on_analyze_clicked(n_clicks, from_date, to_date):
    none_ret = [None for _ in range(7)] # Padding return values

    if n_clicks == 0: return alert_hide(none_ret)

    if from_date is None: return alert_error('Invalid starting date. Please select one and retry.', none_ret)
    if to_date is None: return alert_error('Invalid ending date. Please select one and retry.', none_ret)
    if from_date > to_date: return alert_error('Invalid duration. Please check and retry.', none_ret)

    pairs = [
        ('^NDX', '^GSPC'), ('^GSPC', '^DJI'), ('^NDX', '^DJI')
    ]    
    yf_dict, figs, report = {}, [], []

    for symbol1, symbol2 in pairs:    
        df1 = yf_dict[symbol1] if symbol1 in yf_dict.keys() else load_yf(symbol1, from_date, to_date, INTERVAL_WEEKLY, fit_today = True)
        df2 = yf_dict[symbol2] if symbol2 in yf_dict.keys() else load_yf(symbol2, from_date, to_date, INTERVAL_WEEKLY, fit_today = True)
        
        yf_dict[symbol1], yf_dict[symbol2] = df1, df2

        df1_copy, df2_copy = df1.copy(), df2.copy()
        start1, end1 = get_inter_divergence_lows(df1, df2)
        
        if start1 == -1:
            shapes, found1 = None, False
            end1 = -1
        else:
            start_date, end_date = df1.iloc[start1].name, df1.iloc[end1].name    
            df1, df2 = df1.iloc[end1 - 3:], df2.iloc[end1 - 3:]

            shapes = [dict(x0 = df1.loc[start_date].name, x1 = df1.iloc[-1].name, y0 = df1.loc[start_date].Low, y1 = df1.iloc[-1].Low, line_width = 2, type = 'line', xref = 'x', yref = 'y')]
            shapes.append(dict(x0 = df2.loc[start_date].name, x1 = df2.iloc[-1].name, y0 = df2.loc[start_date].Low, y1 = df2.iloc[-1].Low, line_width = 2, type='line', xref = 'x2', yref = 'y2'))
            
            if end_date != start_date:
                shapes.append(dict(x0 = df1.loc[end_date].name, x1 = df1.iloc[-1].name, y0 = df1.loc[end_date].Low, y1 = df1.iloc[-1].Low, line_width = 2, type = 'line', xref = 'x', yref = 'y'))
                shapes.append(dict(x0 = df2.loc[end_date].name, x1 = df2.iloc[-1].name, y0 = df2.loc[end_date].Low, y1 = df2.iloc[-1].Low, line_width = 2, type = 'line', xref = 'x2', yref = 'y2'))

            start1, end1 = start_date, df1.iloc[-1].name
            found1 = True

        fig1 = get_figure_with_candlestick_pair(df1, df2, symbol1, symbol2)

        if found1:
            fig1.update_layout(shapes = shapes)
            outputlow = f"Bullish Divergence Observed between {symbol1} and {symbol2}. Observed on weekly charts from {str(start1).split(' ')[0]} to {str(end1).split(' ')[0]}."
            append_divergence_record(symbol1, symbol2, 1, start1, end1)
        else:
            outputlow = f'No Bullish divergence found between the provided inputs.'
        
        df1, df2 = df1_copy, df2_copy
        start2, end2 = get_inter_divergence_highs(df1, df2)
        
        if start2 == -1:
            shapes, found2 = None, False
            end2 = -1
        else:
            start_date, end_date = df1.iloc[start2].name, df1.iloc[end2].name    
            df1, df2 = df1.iloc[end2 - 3:], df2.iloc[end2 - 3:]

            shapes = [dict(x0 = df1.loc[start_date].name, x1 = df1.iloc[-1].name, y0 = df1.loc[start_date].High, y1 = df1.iloc[-1].High, line_width = 2, type = 'line', xref = 'x', yref = 'y')]
            shapes.append(dict(x0 = df2.loc[start_date].name, x1 = df2.iloc[-1].name, y0 = df2.loc[start_date].High, y1 = df2.iloc[-1].High, line_width = 2, type='line', xref = 'x2', yref = 'y2'))
            
            if end_date != start_date:
                shapes.append(dict(x0 = df1.loc[end_date].name, x1 = df1.iloc[-1].name, y0 = df1.loc[end_date].High, y1 = df1.iloc[-1].High, line_width = 2, type = 'line', xref = 'x', yref = 'y'))
                shapes.append(dict(x0 = df2.loc[end_date].name, x1 = df2.iloc[-1].name, y0 = df2.loc[end_date].High, y1 = df2.iloc[-1].High, line_width = 2, type = 'line', xref = 'x2', yref = 'y2'))

            start2, end2 = start_date, df1.iloc[-1].name
            found2 = True

        fig2 = get_figure_with_candlestick_pair(df1, df2, symbol1, symbol2)
        
        if found2:
            fig2.update_layout(shapes = shapes)
            outputhigh = f"Bearish Divergence Observed between {symbol1} and {symbol2}. Observed on weekly charts from {str(start2).split(' ')[0]} to {str(end2).split(' ')[0]}."
            append_divergence_record(symbol1, symbol2, -1, start2, end2)
        else:
            outputhigh = f'No Bearish divergence found between the provided inputs.'
        
        figs.extend([
            html.Div([
                html.Div(f"{symbol1[1:]} vs {symbol2[1:]}", style = {'padding-top': '20px', 'font-weight': 'bold', 'text-align': 'center', 'font-size': '20px'}),
                html.Label(outputlow),
                dcc.Graph(figure = fig1, className = 'market_index_graph')
            ]),
            html.Div([
                html.Label(outputhigh),
                dcc.Graph(figure = fig2, className = 'market_index_graph'),
                html.Hr()
            ])
        ])
        report.append(html.Div([
                html.P(f"- {symbol1[1:]} vs {symbol2[1:]}", style = {'padding-top': '20px', 'font-weight': 'bold', 'font-size': '20px'}),
                html.P(outputlow),
                html.P(outputhigh)
            ], style = {'margin-bottom': '40px'}))

    return alert_success('Analysis Completed') + figs + [report]
