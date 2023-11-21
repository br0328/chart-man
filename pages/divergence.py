
from dash import dcc, callback, Output, Input, State
from datetime import date, datetime, timedelta
from plotly.subplots import make_subplots
from collections import defaultdict
from constant import *
from tqdm import tqdm
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
import csv

dash.register_page(__name__, path = '/divergence', name = 'Stochastic Divergence', order = '04')

# Page Layout
scenario_div = get_scenario_div([
	get_symbol_input(),
	get_date_range(),
    get_analyze_button('diver'),
	get_backtest_button('diver')
])
out_tab = get_out_tab({
	'Plot': get_plot_div(),
	'Report': get_report_div()
})
layout = get_page_layout('Stochastic|Divergence', scenario_div, None, out_tab)

# Triggered when Analyze button clicked
@callback(
	[
		Output('alert-dlg', 'is_open', allow_duplicate = True),
		Output('alert-msg', 'children', allow_duplicate = True),
		Output('alert-dlg', 'style', allow_duplicate = True),
        Output('out_tab', 'value', allow_duplicate = True),
		Output('out-plot', 'children', allow_duplicate = True)
	],
	Input('diver-analyze-button', 'n_clicks'),
	[
		State('symbol-input', 'value'),
		State('from-date-input', 'date'),
		State('to-date-input', 'date')
	],
	prevent_initial_call = True
)
def on_analyze_clicked(n_clicks, symbol, from_date, to_date):
    none_ret = ['Plot', None] # Padding return values

    if n_clicks == 0: return alert_hide(none_ret)

    if symbol is None: return alert_error('Invalid symbol. Please select one and retry.', none_ret)
    if from_date is None: return alert_error('Invalid starting date. Please select one and retry.', none_ret)
    if to_date is None: return alert_error('Invalid ending date. Please select one and retry.', none_ret)
    if from_date > to_date: return alert_error('Invalid duration. Please check and retry.', none_ret)

    fig = runStochDivergance(symbol, from_date, to_date)
    return alert_success('Analysis Completed') + ['Plot', dcc.Graph(figure = fig, className = 'diver_graph')]

# Triggered when Backtest button clicked
@callback(
	[
		Output('alert-dlg', 'is_open', allow_duplicate = True),
		Output('alert-msg', 'children', allow_duplicate = True),
		Output('alert-dlg', 'style', allow_duplicate = True),
        Output('out_tab', 'value', allow_duplicate = True),
		Output('out-report', 'children', allow_duplicate = True)
	],
	Input('diver-backtest-button', 'n_clicks'),
	[
		State('symbol-input', 'value'),
		State('from-date-input', 'date'),
		State('to-date-input', 'date')
	],
	prevent_initial_call = True
)
def on_backtest_clicked(n_clicks, symbol, from_date, to_date):
    none_ret = ['Report', None] # Padding return values

    if n_clicks == 0: return alert_hide(none_ret)

    if symbol is None: return alert_error('Invalid symbol. Please select one and retry.', none_ret)
    if from_date is None: return alert_error('Invalid starting date. Please select one and retry.', none_ret)
    if to_date is None: return alert_error('Invalid ending date. Please select one and retry.', none_ret)
    if from_date > to_date: return alert_error('Invalid duration. Please check and retry.', none_ret)

    csv_path = 'out/DIVERGENCE-REPORT_{}_{}_{}.csv'.format(
		symbol, from_date, to_date
    )
    df1, df2 = get_divergence_data(symbol, from_date, to_date, csv_path)        
    return alert_success('Analysis Completed') + ['Report', get_multi_report_content([df1, df2], ['Type-I Divergence', 'Type-II Divergence'], csv_path)]

def get_divergence_data(stock_symbol, stdate, endate, filename):
        year, month, day = map(int, stdate.split('-'))
        sdate = date(year, month, day)
        
        year1, month1, day1 = map(int, endate.split('-'))
        edate = date(year1, month1, day1 )
    
        COMMON_START_DATE = sdate
        STOCK = stock_symbol
        file_name = filename

        #days = pd.date_range(sdate, edate - timedelta(days = 1), freq = 'd').strftime('%Y-%m-%d').tolist()
        days = [(edate - timedelta(days = 1)).strftime(YMD_FORMAT)]
        TT1s, TT2s = [], []

        for dd in days:
            type1, type2 = runStochDivergance(STOCK, COMMON_START_DATE, dd, True)
            t1s = []
            
            for t in type1:
                stockPart = t[1]
                indicatorPart = t[0]
                startDate = stockPart['x0']
                endDate = stockPart['x1']
                DvalueStart = indicatorPart['y0']
                DvalueEnd = indicatorPart['y1']
                stockValueStart = stockPart['y0']
                stockValueEnd = stockPart['y1']

                t1s.append((startDate, endDate, DvalueStart, DvalueEnd, stockValueStart, stockValueEnd, dd))
            
            t2s = []
            
            for t in type2:
                stockPart = t[1]
                indicatorPart = t[0]
                startDate = stockPart['x0']
                endDate = stockPart['x1']
                DvalueStart = indicatorPart['y0']
                DvalueEnd = indicatorPart['y1']
                stockValueStart = stockPart['y0']
                stockValueEnd = stockPart['y1']

                t2s.append((startDate, endDate, DvalueStart, DvalueEnd, stockValueStart, stockValueEnd, dd))
            
            TT1s.append(t1s)
            TT2s.append(t2s)

        def find_unique_smallest_date(arrays_of_tuples):
            unique_tuples = defaultdict(list)

            for arr in arrays_of_tuples:
                for tup in arr:
                    key = tuple(tup[:-1])
                    date_str = tup[-1]

                    if key not in unique_tuples or date_str < unique_tuples[key][-1][-1]:
                        unique_tuples[key] = [(tup, date_str)]

            result = [min(tups, key = lambda x: x[-1]) for tups in unique_tuples.values()]
            return result

        out1 = find_unique_smallest_date(TT1s)
        out2 = find_unique_smallest_date(TT2s)
        
        columns = ['StartDate', 'EndDate', '%D_ValStart', '%D_ValEnd', 'Stock_ValStart', 'Stock_ValEnd', 'EndDatePut']
        rec1, rec2 = [], []
        
        with open(file_name + '.csv', "w") as csv_file:
            writer = csv.writer(csv_file, delimiter = ',')
            writer.writerow(['TYPE 1 DIVERGANCE'])
            writer.writerow(columns)
            
            for t in out1:
                tempr = []
                
                for tt in t[0]:
                    tempr.append(tt)
                
                tempr = [tempr[0].strftime(YMD_FORMAT), tempr[1].strftime(YMD_FORMAT), '{:.4f}'.format(tempr[2]), '{:.4f}'.format(tempr[3]), tempr[4], tempr[5], tempr[6]]
                writer.writerow(tempr)
                rec1.append(tempr)
                
            writer.writerow(['TYPE 2 DIVERGANCE'])
            writer.writerow(columns)
            
            for t in out2:
                tempr = []
                
                for tt in t[0]:
                    tempr.append(tt)
                    
                tempr = [tempr[0].strftime(YMD_FORMAT), tempr[1].strftime(YMD_FORMAT), '{:.4f}'.format(tempr[2]), '{:.4f}'.format(tempr[3]), tempr[4], tempr[5], tempr[6]]
                writer.writerow(tempr)
                rec2.append(tempr)
        
        df1 = pd.DataFrame(rec1, columns = columns)
        df2 = pd.DataFrame(rec2, columns = columns)

        return df1, df2
