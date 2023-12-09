
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
	get_date_range(from_date = get_offset_date_str(get_today_str(), -365)),
    get_run_button('diver'),
])
# parameter_div = get_parameter_div([
#     get_run_button('diver'),
# 	#get_cur_date_picker(),
# 	# get_analyze_button('diver'),
# 	# get_backtest_button('diver')
# ])
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
		Output('out-plot', 'children', allow_duplicate = True),
        Output('out-report', 'children', allow_duplicate = True)
	],
	Input('diver-run-button', 'n_clicks'),
	[
		State('symbol-input', 'value'),
		State('from-date-input', 'date'),
		State('to-date-input', 'date'),
        #State('cur-date-input', 'date')
	],
	prevent_initial_call = True
)
def on_run_clicked(n_clicks, symbol, from_date, to_date):
    none_ret = ['Plot', None, None] # Padding return values

    if n_clicks == 0: return alert_hide(none_ret)

    if symbol is None: return alert_error('Invalid symbol. Please select one and retry.', none_ret)
    if from_date is None: return alert_error('Invalid starting date. Please select one and retry.', none_ret)
    if to_date is None: return alert_error('Invalid ending date. Please select one and retry.', none_ret)
    if from_date > to_date: return alert_error('Invalid duration. Please check and retry.', none_ret)

    # if cur_date is None: return alert_error('Invalid current date. Please select one and retry.', none_ret)
    
    # if cur_date < from_date: cur_date = from_date
    # if cur_date > to_date: cur_date = to_date
    
    # cur_date = get_timestamp(cur_date)
    #fig, df = runStochDivergance(symbol, from_date, to_date, cur_date = cur_date)
    
    df, _, _ = getPointsGivenR(symbol, 1.02, startDate = from_date, endDate = to_date)
    D = TA.STOCHD(df)

    from_date = get_nearest_forward_date(df, get_timestamp(from_date)).strftime(YMD_FORMAT)
    out1, out2 = get_divergence_data(symbol, from_date, to_date, oldData = df)
    
    fig = make_subplots(rows = 2, cols = 1, shared_xaxes = True, vertical_spacing = 0.01, subplot_titles = ('Stock prices', 'Stochastic Indicator'), row_width = [0.29,0.7])
    fig.update_yaxes(type = 'log', row = 1, col = 1)
    fig.add_trace(go.Candlestick(x = df.index, open = df['open'], high = df['high'], low = df['low'], close = df['close'], showlegend = False), row = 1, col = 1)
    fig.update_layout(xaxis_rangeslider_visible = False, yaxis_tickformat = '0')
    
    fig.add_trace(go.Scatter(x = D.index, y = D, showlegend = False), row = 2, col = 1)
    fig.add_trace(go.Scatter(x = df.index, y = df['close'].rolling(10).mean(), name = 'MA-10W'))
    fig.add_trace(go.Scatter(x = df.index, y = df['close'].rolling(40).mean(), name = 'MA-40W'))
    
    res_df, acc, cum, full_list = backtest_stock_divergence(symbol, from_date, to_date, out1, out2)
    
    lines_to_draw = []
    smarker_x, smarker_y = [], []
    emarker_x, emarker_y = [], []
    Dsmarker_x, Dsmarker_y = [], []
    Demarker_x, Demarker_y = [], []
    
    for dStart, dEnd, dVStart, dVEnd, sv, se, dd, edd in full_list:
        lines_to_draw.extend(get_lines(dStart, dEnd, df, D, dd, edd, dVEnd > dVStart))
        smarker_x.append(dd)
        smarker_y.append(df.loc[dd].low if dVEnd > dVStart else df.loc[dd].high)
        Dsmarker_x.append(dd)
        Dsmarker_y.append(D.loc[dd])
        
        if edd is not None:
            emarker_x.append(edd)
            emarker_y.append(df.loc[edd].low if dVEnd > dVStart else df.loc[edd].high)
            Demarker_x.append(edd)
            Demarker_y.append(D.loc[edd])
    
    fig.update_layout(shapes = lines_to_draw)

    fig.add_trace(
        go.Scatter(
            x = smarker_x,
            y = smarker_y,
            mode = "markers",
            marker = dict(
                size = 7,
                color = 'purple'
            ),
            marker_symbol = "circle",
            name = 'EntryDate'
        ),
        row = 1,
        col = 1
    )
    fig.add_trace(
        go.Scatter(
            x = emarker_x,
            y = emarker_y,
            mode = "markers",
            marker = dict(
                size = 7,
                color = 'orange'
            ),
            marker_symbol = "circle",
            name = 'ExitDate'
        ),
        row = 1,
        col = 1
    )
    fig.add_trace(
        go.Scatter(
            x = Dsmarker_x,
            y = Dsmarker_y,
            mode = "markers",
            marker = dict(
                size = 7,
                color = 'purple'
            ),
            marker_symbol = "circle",
            showlegend= False
        ),
        row = 2,
        col = 1
    )
    fig.add_trace(
        go.Scatter(
            x = Demarker_x,
            y = Demarker_y,
            mode = "markers",
            marker = dict(
                size = 7,
                color = 'orange'
            ),
            marker_symbol = "circle",
            showlegend=False
        ),
        row = 2,
        col = 1
    )
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
    
    csv_path = 'out/DIVERGENCE-REPORT_{}_{}_{}_sr={:.1f}%_cp={:.1f}%.csv'.format(
		symbol, from_date, to_date, acc, cum
    )
    res_df.to_csv(csv_path, index = False)    
    return alert_success('Analysis Completed') + ['Plot', dcc.Graph(figure = fig, className = 'diver_graph'), get_report_content(res_df, csv_path)]

def get_lines(dStart, dEnd, df, D, dd, edd, is_bullish):
    lines = [dict(
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
            x0 = df.loc[dEnd].name,
            y0 = D.loc[dEnd],
            x1 = df.loc[dd].name,
            y1 = D.loc[dd],
            type = 'line',
            xref = 'x2',
            yref = 'y2',
            line_dash = 'dot',
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
        ),
        dict(
            x0 = df.loc[dEnd].name,
            y0 = df.loc[dEnd].low if is_bullish else df.loc[dEnd].high,
            x1 = df.loc[dd].name,
            y1 = df.loc[dd].low if is_bullish else df.loc[dd].high,
            type = 'line',
            xref = 'x',
            yref = 'y',
            line_dash = 'dot',
            line_color = 'blue' if is_bullish else 'black'
        )
    ]
    if edd is not None:
        lines.extend([
            dict(
                x0 = df.loc[dd].name,
                y0 = D.loc[dd],
                x1 = df.loc[edd].name,
                y1 = D.loc[edd],
                type = 'line',
                xref = 'x2',
                yref = 'y2',
                line_dash = 'dot',
                line_color = 'purple'
            ),
            dict(
                x0 = df.loc[dd].name,
                y0 = df.loc[dd].low if is_bullish else df.loc[dd].high,
                x1 = df.loc[edd].name,
                y1 = df.loc[edd].low if is_bullish else df.loc[edd].high,
                type = 'line',
                xref = 'x',
                yref = 'y',
                line_dash = 'dot',
                line_color = 'purple'
            )
        ])
    return lines

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

def backtest_stock_divergence(symbol, from_date, to_date, out1, out2):
    df, _, _ = getPointsGivenR(symbol, 1.02, startDate = from_date, endDate = to_date)
    D = TA.STOCHD(df)
    
    out = sorted(out1 + out2, key = lambda x : (x[0], x[1]))
        
    columns = ['Position', 'Diver-Dur', 'EntryDate', 'EntryPrice', 'ExitDate', 'ExitPrice', 'Return', 'Cum-Profit']
    records, full_list, cumprof, matches = [], [], 0, 0
    
    for dStart, dEnd, dVStart, dVEnd, sv, se, dd in out:
        r, cumprof, succ, edd = get_trans_record(dStart, dEnd, dd, df, D, cumprof, dVEnd > dVStart)
        
        records.append(r)
        full_list.append((dStart, dEnd, dVStart, dVEnd, sv, se, dd, edd))
        
        if succ or (edd is None): matches += 1
    
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
    return res_df, 100 * acc, 100 * cumprof, full_list

def get_trans_record(dStart, dEnd, dd, df, D, cumprof, is_bullish):
    dd = datetime.strptime(dd, YMD_FORMAT)
    di = list(D.index)
    idx = 0
    
    while idx < len(di):
        if di[idx] >= dd: break
        idx += 1
    
    idx = min(idx, len(di) - 1)
    
    pv = D.iloc[idx]
    edd = None
    sg = 1 if is_bullish else -1
    
    for i in range(max(1, idx + 1), len(D)):
        if np.sign(D.iloc[i] - pv) != sg:
            edd = di[i]
            #if i == idx + 1: edd = None
            break
        else:
            pv = D.iloc[i]

    #if edd is None: return None, cumprof, None, None
    ret = (sg * (df.loc[edd].close - df.iloc[idx].close) / df.iloc[idx].close) if edd is not None else 0
    cumprof += ret
    
    return (
        'Long' if is_bullish else 'Short',
        '{} - {}'.format(dStart.strftime(DBY_FORMAT), dEnd.strftime(DBY_FORMAT)),
        di[idx].strftime(DBY_FORMAT),
        df.iloc[idx].close,
        edd.strftime(DBY_FORMAT) if edd is not None else 'Stay Still',
        df.loc[edd].close if edd is not None else '',
        ('{:.1f}%'.format(ret * 100)) if edd is not None else '',
        '{:.1f}%'.format(cumprof * 100)
    ), cumprof, ret > 0, edd
