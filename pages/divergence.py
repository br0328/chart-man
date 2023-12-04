
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
	get_date_range()
])
parameter_div = get_parameter_div([
	get_cur_date_picker(),
	get_analyze_button('diver'),
	get_backtest_button('diver')
])
out_tab = get_out_tab({
	'Plot': get_plot_div(),
	'Report': get_report_div()
})
layout = get_page_layout('Stochastic|Divergence', scenario_div, parameter_div, out_tab)

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
		State('to-date-input', 'date'),
        State('cur-date-input', 'date')
	],
	prevent_initial_call = True
)
def on_analyze_clicked(n_clicks, symbol, from_date, to_date, cur_date):
    none_ret = ['Plot', None] # Padding return values

    if n_clicks == 0: return alert_hide(none_ret)

    if symbol is None: return alert_error('Invalid symbol. Please select one and retry.', none_ret)
    if from_date is None: return alert_error('Invalid starting date. Please select one and retry.', none_ret)
    if to_date is None: return alert_error('Invalid ending date. Please select one and retry.', none_ret)
    if from_date > to_date: return alert_error('Invalid duration. Please check and retry.', none_ret)

    if cur_date is None: return alert_error('Invalid current date. Please select one and retry.', none_ret)
    
    if cur_date < from_date: cur_date = from_date
    if cur_date > to_date: cur_date = to_date
    
    cur_date = get_timestamp(cur_date)
    #fig, df = runStochDivergance(symbol, from_date, to_date, cur_date = cur_date)
    
    out1, out2 = get_divergence_data(symbol, from_date, to_date)
    df, _, _ = getPointsGivenR(symbol, 1.02, startDate = from_date, endDate = to_date)
    D = TA.STOCHD(df)
    
    fig = make_subplots(rows = 2, cols = 1, shared_xaxes = True, vertical_spacing = 0.01, subplot_titles = ('Stock prices', 'Stochastic Indicator'), row_width = [0.29,0.7])
    fig.update_yaxes(type = 'log', row = 1, col = 1)
    fig.add_trace(go.Candlestick(x = df.index, open = df['open'], high = df['high'], low = df['low'], close = df['close'], showlegend = False), row = 1, col = 1)
    fig.update_layout(xaxis_rangeslider_visible = False, yaxis_tickformat = '0')
    
    fig.add_trace(go.Scatter(x = D.index, y = D, showlegend = False), row = 2, col = 1)
    fig.add_trace(go.Scatter(x = df.index, y = df['close'].rolling(10).mean(), name = 'MA-10W'))
    fig.add_trace(go.Scatter(x = df.index, y = df['close'].rolling(40).mean(), name = 'MA-40W'))
    
    lines_to_draw = []
    
    for dStart, dEnd, _, _, _, _, _ in out1:
        lines_to_draw.extend(get_lines(dStart, dEnd, df, D, True))
    
    for dStart, dEnd, _, _, _, _, _ in out2:
        lines_to_draw.extend(get_lines(dStart, dEnd, df, D, False))
    
    lines_to_draw = [d for d in lines_to_draw if d['x1'] < cur_date]
    fig.update_layout(shapes = lines_to_draw)

    fig.update_xaxes(
        rangeslider_visible = False,
        range = [df.index[0], df.index[-1]],
        row = 1, col = 1
    )
    fig.update_xaxes(
        rangeslider_visible = False,
        range = [df.index[0], df.index[-1]],
        row = 2, col = 1
    )

    # Pivot date plot
    draw_vline_shape(fig, cur_date, min(df['low']), max(df['high']), 'darkgreen')
    draw_annotation(fig, cur_date, np.log10(min(df['low'])), cur_date.strftime(' %d %b %Y') + ' →',
        xanchor = 'left', yanchor = 'bottom', color = 'darkgreen', size = 14)

    return alert_success('Analysis Completed') + ['Plot', dcc.Graph(figure = fig, className = 'diver_graph')]

def get_lines(dStart, dEnd, df, D, is_bullish):
    return [dict(
            x0 = df.loc[dStart].name,
            y0 = D.loc[dStart],
            x1 = df.loc[dEnd].name,
            y1 = D.loc[dEnd],
            type = 'line',
            xref = 'x2',
            yref = 'y2',
            line_width = 4,
            line_color = 'blue' if is_bullish else 'black'
        ),
        dict(
            x0 = df.loc[dStart].name,
            y0 = df.loc[dStart].low if is_bullish else df.loc[dStart].high,
            x1 = df.loc[dEnd].name,
            y1 = df.loc[dEnd].low if is_bullish else df.loc[dEnd].high,
            type = 'line',
            xref = 'x',
            yref = 'y',
            line_width = 4,
            line_color = 'blue' if is_bullish else 'black'
        )
    ]

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
		days = (datetime.now() - from_date).days

		cur_date = (from_date + timedelta(days = days * 2 // 3)).strftime(YMD_FORMAT)
		from_date = from_date.strftime(YMD_FORMAT)

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

    df, acc, cum = backtest_stock_divergence(symbol, from_date, to_date)
    
    csv_path = 'out/DIVERGENCE-REPORT_{}_{}_{}_sr={:.1f}%_cp={:.1f}%.csv'.format(
		symbol, from_date, to_date, acc, cum
    )
    df.to_csv(csv_path, index = False)
    
    #return alert_success('Analysis Completed') + ['Report', get_multi_report_content([df1, df2], ['Type-I Divergence', 'Type-II Divergence'], csv_path)]
    return alert_success('Analysis Completed') + ['Report', get_report_content(df, csv_path)]

def backtest_stock_divergence(symbol, from_date, to_date):
    out1, out2 = get_divergence_data(symbol, from_date, to_date)
    df, _, _ = getPointsGivenR(symbol, 1.02, startDate = from_date, endDate = to_date)
    D = TA.STOCHD(df)
    
    out = sorted(out1 + out2, key = lambda x : (x[0], x[1]))
        
    columns = ['Position', 'Diver-Dur', 'EntryDate', 'EntryPrice', 'ExitDate', 'ExitPrice', 'Return', 'Cum-Profit']
    records, cumprof, matches = [], 0, 0
    
    for dStart, dEnd, dVStart, dVEnd, _, _, dd in out:
        r, cumprof, succ = get_trans_record(dStart, dEnd, dd, df, D, cumprof, dVEnd > dVStart)
        if r is None: continue
        records.append(r)
        if succ: matches += 1
    
    acc = matches / len(records) if len(records) > 0 else 0
    
    records.append(('', '', '', '', '', '', '', ''))
    records.append((
        '',
        f"From {change_date_format(from_date, YMD_FORMAT, DBY_FORMAT)} To {change_date_format(to_date, YMD_FORMAT, DBY_FORMAT)}",
        f"Symbol: {symbol}",
        '',
        'Success Rate:',
        '{:.1f}%'.format(100 * acc),
        '', ''
    ))
    records.append((
        '', '', '', '',
        'Cumulative Profit:',
        '{:.1f}%'.format(100 * cumprof),
        '', ''
    ))    
    records.append(('', '', '', '', '', '', '', ''))
    
    res_df = pd.DataFrame(records, columns = columns)
    return res_df, 100 * acc, 100 * cumprof

def get_trans_record(dStart, dEnd, dd, df, D, cumprof, is_bullish):
    dd = datetime.strptime(dd, YMD_FORMAT)
    di = list(D.index)
    idx = 0
    
    while idx < len(di):
        if di[idx] > dd: break
        idx += 1
    
    pv = D.iloc[idx]
    edd = None
    sg = 1 if is_bullish else -1
    
    for i in range(idx + 1, len(D)):
        if np.sign(D.iloc[i] - pv) != sg:
            edd = di[i - 1]
            if i == idx + 1: edd = None
            break
        else:
            pv = D.iloc[i]

    if edd is None: return None, cumprof, None

    ret = sg * (df.loc[edd].close - df.iloc[idx].close) / df.iloc[idx].close
    cumprof += ret
    
    return (
        'Long' if is_bullish else 'Short',
        '{} - {}'.format(dStart.strftime(DBY_FORMAT), dEnd.strftime(DBY_FORMAT)),
        di[idx].strftime(DBY_FORMAT),
        df.iloc[idx].close,
        edd.strftime(DBY_FORMAT),
        df.loc[edd].close,
        '{:.1f}%'.format(ret * 100),
        '{:.1f}%'.format(cumprof * 100)
    ), cumprof, ret > 0

# Backtest
# def get_divergence_data(stock_symbol, stdate, endate, filename = None):
#         year, month, day = map(int, stdate.split('-'))
#         sdate = date(year, month, day)
        
#         year1, month1, day1 = map(int, endate.split('-'))
#         edate = date(year1, month1, day1 )
    
#         COMMON_START_DATE = sdate
#         STOCK = stock_symbol

#         days = pd.date_range(sdate, edate - timedelta(days = 1), freq = 'd').strftime('%Y-%m-%d').tolist()
#         TT1s, TT2s = [], []

#         for dd in tqdm(days):
#             # Calculate divergence and get transactions
#             type1, type2, df = runStochDivergance(STOCK, COMMON_START_DATE, dd, True)
#             t1s = []
            
#             for t in type1:
#                 stockPart = t[1]
#                 indicatorPart = t[0]
#                 startDate = stockPart['x0']
#                 endDate = stockPart['x1']
#                 DvalueStart = indicatorPart['y0']
#                 DvalueEnd = indicatorPart['y1']
#                 stockValueStart = stockPart['y0']
#                 stockValueEnd = stockPart['y1']
#                 t1s.append((startDate, endDate, DvalueStart, DvalueEnd, stockValueStart, stockValueEnd, dd))
            
#             t2s = []
            
#             for t in type2:
#                 stockPart = t[1]
#                 indicatorPart = t[0]
#                 startDate = stockPart['x0']
#                 endDate = stockPart['x1']
#                 DvalueStart = indicatorPart['y0']
#                 DvalueEnd = indicatorPart['y1']
#                 stockValueStart = stockPart['y0']
#                 stockValueEnd = stockPart['y1']
#                 t2s.append((startDate, endDate, DvalueStart, DvalueEnd, stockValueStart, stockValueEnd, dd))
            
#             TT1s.append(t1s)
#             TT2s.append(t2s)

#         def find_unique_smallest_date(arrays_of_tuples):
#             unique_tuples = defaultdict(list)

#             for arr in arrays_of_tuples:
#                 for tup in arr:
#                     key = tuple(tup[:-1])
#                     date_str = tup[-1]

#                     if key not in unique_tuples or date_str < unique_tuples[key][-1][-1]:
#                         unique_tuples[key] = [(tup, date_str)]

#             result = [min(tups, key = lambda x: x[-1]) for tups in unique_tuples.values()]
#             return result

#         def rearrange(od):
#             od = [list(t[0]) for t in od]
#             od.sort(key = lambda x : x[0])
            
#             recs = []

#             for i in range(len(od)):
#                 tr = od[i]
#                 found_in = False
                
#                 for j in range(len(od)):
#                     if i == j: continue
#                     otr = od[j]
                    
#                     if otr[0] <= tr[0] and tr[1] <= otr[1]:
#                         found_in = True
#                         break
                
#                 if found_in: continue
#                 recs.append(tr)
            
#             while True:
#                 found_adj = False
                
#                 for i in range(len(recs) - 1):
#                     tr, otr = recs[i], recs[i + 1]
                    
#                     if tr[1] == otr[0]:
#                         tr[1] = otr[1]
#                         tr[3] = otr[3]
#                         tr[5] = otr[5]
#                         tr[6] = otr[6]
                        
#                         found_adj = True
#                         recs.pop(i + 1)
#                         break
                
#                 if not found_adj: break
            
#             return recs                

#         out1 = find_unique_smallest_date(TT1s)
#         out2 = find_unique_smallest_date(TT2s)
        
#         out1 = rearrange(out1)
#         out2 = rearrange(out2)
        
#         columns = ['Type', 'SDate', 'EDate', '%D_Start', '%D_End', 'SPrice', 'EPrice', 'EPutDate']
#         records = []

#         if True:
#             for tempr in out1:
#                 tempr = [
#                     'I',
#                     tempr[0].strftime(DBY_FORMAT),
#                     tempr[1].strftime(DBY_FORMAT),
#                     '{:.4f}'.format(tempr[2]),
#                     '{:.4f}'.format(tempr[3]),
#                     tempr[4],
#                     tempr[5],
#                     change_date_format(tempr[6], YMD_FORMAT, DBY_FORMAT)
#                 ]
#                 records.append(tempr)
                
#             for tempr in out2:
#                 tempr = [
#                     'II',
#                     tempr[0].strftime(DBY_FORMAT),
#                     tempr[1].strftime(DBY_FORMAT),
#                     '{:.4f}'.format(tempr[2]),
#                     '{:.4f}'.format(tempr[3]),
#                     tempr[4],
#                     tempr[5],
#                     change_date_format(tempr[6], YMD_FORMAT, DBY_FORMAT)
#                 ]
#                 records.append(tempr)
        
#         df = pd.DataFrame(records, columns = columns)
        
#         last_records = [
#             {},
#             {
#                 'SDate': f"Ticker: {stock_symbol}",
#                 'EDate': f"From: {sdate.strftime(DBY_FORMAT)}",
#                 '%D_Start': f"To: {edate.strftime(DBY_FORMAT)}"
#             },
#             {}
#         ]
#         for r in last_records:
#             df = pd.concat([df, pd.Series(r).to_frame().T], ignore_index = True)

#         return df
