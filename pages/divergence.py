
from dash import dcc, callback, Output, Input, State
from datetime import date, datetime, timedelta
from plotly.subplots import make_subplots
from collections import defaultdict
from constant import *
from compute import *
from config import *
from finta import TA
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

    csv_path = 'out/DIVERGENCE-BACKTEST_{}_{}_{}.csv'.format(
		symbol, from_date, to_date
    )    
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
    records = get_divergence_data(symbol, from_date, to_date, csv_path)        
    return alert_success('Analysis Completed') + ['Report', get_report_content(records, csv_path)]

def runStochDivergance(symbol, from_date = '2000-01-01', to_date = '2022-08-07', return_csv = False):
    R = 1.02
    data, _, _ = getPointsGivenR(symbol, R, startDate = from_date, endDate = to_date)
    _, lows = getPointsGivenR(symbol, R, startDate = from_date, endDate = to_date, type_='lows')
    _, highs = getPointsGivenR(symbol, R, startDate = from_date, endDate = to_date, type_='highs')

    # highs.append(highs[-1]); highs.append(highs[-2]); highs.append(highs[-3])
    # lows.append(lows[-1]); lows.append(lows[-2]); lows.append(lows[-3])

    lows = np.asarray(lows)
    lows -= 15
    lows = lows[lows >= 0]
    lows = lows.tolist()

    highs = np.asarray(highs)
    highs -= 15
    highs = highs[highs >= 0]
    highs = highs.tolist()

    K = TA.STOCH(data, 14)
    D = TA.STOCHD(data)
    
    data = data[15:]
    D = D[15:]
    x = D.to_numpy()

    highsStoch, lowsStoch = getPointsforArray(x, 1.05)
    highsStoch.append(len(D)-1)

    rr = getReigons(lows, data['low'])
    fr = getFinalReigons(rr)
    rr1 = getReigons(highs, data['high'])
    fr1 = getFinalReigons(rr1)
    rrS1 = getReigons(highsStoch, D)
    frS1 = getFinalReigons(rrS1)
    rrS1 = getReigons(lowsStoch, D)
    frS2 = getFinalReigons(rrS1)

    type1 = getDivergance_LL_HL(fr, frS2)
    type2 = getDivergance_HH_LH(fr1, frS1)

    df = data

    if not return_csv:
        fig = make_subplots(rows = 2, cols = 1, shared_xaxes = True, vertical_spacing = 0.01, subplot_titles = ('Stock prices', 'Stochastic Indicator'), row_width = [0.29,0.7])
        fig.update_yaxes(type='log', row = 1, col = 1)
        fig.add_trace(go.Candlestick(x = df.index, open = df['open'], high = df['high'], low = df['low'], close = df['close']), row = 1, col = 1)
        fig.update_layout(xaxis_rangeslider_visible = False)
        
        fig.add_trace(go.Scatter(x = D.index, y = D), row = 2, col = 1)
        fig.add_trace(go.Scatter(x = df.index, y = df['close'].rolling(10).mean(),name = 'ma-10W'))
        fig.add_trace(go.Scatter(x = df.index, y = df['close'].rolling(40).mean(),name = 'ma-40W'))
    
    lines_to_draw, typeONEs = [], []
    
    for t in type1:
        sS, eS = t[0][0], t[1][0]
        sD, eD = t[0][1], t[1][1]
        stockS = data.iloc[t[0][0]].high
        stockE = data.iloc[t[1][0]].high

        if not eS == sS and not sD == eD:
            StockM = (stockE - stockS)/(eS-sS)
            Dm = (eD - sS)/(eD-sD)

            if StockM > 0.2 and Dm > 0.2:
                pass
            elif StockM < -0.2 and Dm < -0.2:
                pass
            else:
                start = max(t[0][1], t[0][0])
                ending = min(t[1])
                stockStart = start
                stockEnd = ending

                dStart = start
                dEnd = ending
                
                a1 = dict(
                    x0 = data.iloc[dStart].name,
                    y0 = D.iloc[dStart],
                    x1 = data.iloc[dEnd].name,
                    y1 = D.iloc[dEnd],
                    type = 'line',
                    xref = 'x2',
                    yref = 'y2',
                    line_width = 7
                )
                b1 = dict(
                    x0 = data.iloc[stockStart].name,
                    y0 = data.iloc[stockStart].low,
                    x1 = data.iloc[stockEnd].name,
                    y1 = data.iloc[stockEnd].low,
                    type = 'line',
                    xref = 'x',
                    yref = 'y',
                    line_width = 7
                )
                typeONEs.append((a1, b1))
                
                if not return_csv:
                    lines_to_draw.append(a1)
                    lines_to_draw.append(b1)                

    typeTWOs = []
    
    for t in type2:
        sS, eS = t[0][0], t[1][0]
        sD, eD = t[0][1], t[1][1]
        ss = max(sS, sD)
        ee = min(eS, eD)
        stockS = data.iloc[ss].high
        stockE = data.iloc[ee].high
        dds = D.iloc[ss]
        dde = D.iloc[ee]

        if not eS == sS and not sD == eD:
            StockM = (stockE - stockS)/(eS-sS)
            Dm = (dde - dds)/(eS-sS)

            if StockM > 0.2 and Dm > 0.2:
                pass
            elif StockM < -0.2 and Dm < -0.2:
                pass
            else:
                start = max(t[0][1], t[0][0])
                ending = min(t[1])
                stockStart = start
                stockEnd = ending

                dStart = start
                dEnd = ending
            
                a1 = dict(
                    x0 = data.iloc[dStart].name,
                    y0 = D.iloc[dStart],
                    x1 = data.iloc[dEnd].name,
                    y1 = D.iloc[dEnd],
                    type = 'line',
                    xref = 'x2',
                    yref = 'y2',
                    line_width = 7
                )
                a2 = dict(
                    x0 = data.iloc[stockStart].name,
                    y0 = data.iloc[stockStart].high,
                    x1 = data.iloc[stockEnd].name,
                    y1 = data.iloc[stockEnd].high,
                    type = 'line',
                    xref = 'x',
                    yref = 'y',
                    line_width=  7)
                typeTWOs.append((a1, a2))
                
                if not return_csv:
                    lines_to_draw.append(a1)
                    lines_to_draw.append(a2)

    if not return_csv: fig.update_layout(shapes = lines_to_draw)    
    if return_csv: return typeONEs, typeTWOs
    
    return fig

def get_divergence_data(stock_symbol, stdate, endate, filename):
        ct = datetime.now()
        year, month, day = map(int, stdate.split('-'))
        sdate = date(year, month, day)
        
        year1, month1, day1 = map(int, endate.split('-'))
        edate = date(year1, month1, day1 )
    
        COMMON_START_DATE = '2020-01-01'
        STOCK = stock_symbol
        file_name = filename

        days = pd.date_range(sdate, edate - timedelta(days = 1), freq = 'd').strftime('%Y-%m-%d').tolist()
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

                t1s.append( (startDate, endDate, DvalueStart, DvalueEnd, stockValueStart, stockValueEnd, dd) )
            
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

                t2s.append( (startDate, endDate, DvalueStart, DvalueEnd, stockValueStart, stockValueEnd, dd) )
            
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

            result = [min(tups, key=lambda x: x[-1]) for tups in unique_tuples.values()]
            return result

        out1 = find_unique_smallest_date(TT1s)
        out2 = find_unique_smallest_date(TT2s)

        with open(file_name + '.csv', "w") as csv_file:
                writer = csv.writer(csv_file, delimiter = ',')
                writer.writerow(['TYPE 1 DIVERGANCE'])
                writer.writerow(['startDate', 'endDate', '%D valueStart', '%D valueEnd', 'stock Value Start', 'stock Value End', 'END DATE PUT'])
                
                for t in out1:
                    tempr = []
                    
                    for tt in t[0]:
                        tempr.append(tt)
                        
                    writer.writerow(tempr)
                    
                writer.writerow(['TYPE 2 DIVERGANCE'])
                writer.writerow(['startDate', 'endDate', '%D valueStart', '%D valueEnd', 'stock Value Start', 'stock Value End', 'END DATE PUT'])
                
                for t in out2:
                    tempr = []
                    
                    for tt in t[0]:
                        tempr.append(tt)
                        
                    writer.writerow(tempr)

        return pd.DataFrame()
