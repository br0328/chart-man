
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

dash.register_page(__name__, path = '/fibext', name = 'Fibonacci Extension', order = '07')

scenario_div = get_scenario_div([
    get_symbol_input(),
    get_date_range(),
    get_interval_input()
])
parameter_div = get_parameter_div([
    get_cur_date_picker(),
    get_pivot_number_input(),
    get_merge_thres_input(),
    get_analyze_button('fib-ext'),
    get_backtest_button('fib-ext')
])
out_tab = get_out_tab({
    'Plot': get_plot_div(),
    'Report': get_report_div()
})
layout = get_page_layout('Fibonacci|Extension', scenario_div, parameter_div, out_tab)

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
        State('cur-date-input', 'date'),
        State('pivot-input', 'value'),
        State('merge-input', 'value')
    ],
    prevent_initial_call = True
)
def on_analyze_clicked(n_clicks, symbol, from_date, to_date, interval, cur_date, pivot_number, merge_thres):
    none_ret = ['Plot', None, None]

    if n_clicks == 0: return alert_hide(none_ret)
    
    if symbol is None: return alert_error('Invalid symbol. Please select one and retry.', none_ret)
    if from_date is None: return alert_error('Invalid starting date. Please select one and retry.', none_ret)
    if to_date is None: return alert_error('Invalid ending date. Please select one and retry.', none_ret)
    if from_date > to_date: return alert_error('Invalid duration. Please check and retry.', none_ret)
    if interval is None: return alert_error('Invalid interval. Please select one and retry.', none_ret)
    
    if get_duration(from_date, to_date) < zigzag_window + zigzag_padding:
        return alert_error('Duration must be at least {} days for Fibonacci analysis.'.format(zigzag_window + zigzag_padding), none_ret)

    if cur_date is None: return alert_error('Invalid current date. Please select one and retry.', none_ret)
    if cur_date < from_date or cur_date > to_date: return alert_error('Invalid current date. Please select one in scenario duration and retry.', none_ret)
    if pivot_number is None: return alert_error('Invalid pivot number. Please select one and retry.', none_ret)

    try:
        merge_thres = float(merge_thres) / 100
    except Exception:
        return alert_error('Invalid merge threshold. Please input correctly and retry.', none_ret)
    
    df = load_yf(symbol, from_date, to_date, interval, fit_today = True)
    cur_date = get_nearest_backward_date(df, get_timestamp(cur_date))

    if cur_date is None: return alert_warning('Nearest valid date not found. Please reselect current date.', none_ret)
    
    zdf = get_zigzag(df, cur_date)
    pivot_number = PIVOT_NUMBER_ALL.index(pivot_number) + 1

    downfalls = get_recent_downfalls(zdf, pivot_number)
    extensions = get_fib_extensions(zdf, downfalls, get_safe_num(merge_thres), df.iloc[-1]['Close'] * 2)
    behaviors = get_fib_ext_behaviors(df, extensions, cur_date, get_safe_num(merge_thres))

    records = analyze_fib_extension(df, extensions, behaviors, cur_date, pivot_number, merge_thres, interval, symbol)
    csv_path = 'out/FIB-EXT-ANALYZE_{}_{}_{}_{}_{}_p{}_m{}%.csv'.format(
        symbol, from_date, cur_date.strftime(YMD_FORMAT), to_date, interval, pivot_number, '{:.1f}'.format(100 * merge_thres)
    )
    records.to_csv(csv_path, index = False)
    report = get_report_content(records, csv_path)

    return alert_success('Analysis Completed') + ['Plot', update_plot(df, downfalls, extensions, behaviors, cur_date), report]

@callback(
    [
        Output('from-date-input', 'date'), Output('cur-date-input', 'date')
    ],
    Input('symbol-input', 'value'),
    [
        State('from-date-input', 'date'), State('cur-date-input', 'date')
    ]
)
def on_symbol_changed(symbol, from_date, cur_date):
    if symbol is None: return [from_date, cur_date]

    ipo_date = load_stake().loc[symbol]['ipo']

    if from_date is None:
        from_date = ipo_date
    elif from_date < ipo_date:
        from_date = ipo_date

    if cur_date is None:
        from_date = get_timestamp(from_date)
        days = (datetime.now() - from_date).days

        cur_date = (from_date + timedelta(days = days * 2 // 3)).strftime(YMD_FORMAT)
        from_date = from_date.strftime(YMD_FORMAT)

    return [from_date, cur_date]

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
        State('pivot-input', 'value'),
        State('merge-input', 'value')
    ],
    prevent_initial_call = True
)
def on_backtest_clicked(n_clicks, symbol, from_date, to_date, interval, pivot_number, merge_thres):
    none_ret = ['Report', None]

    if n_clicks == 0: return alert_hide(none_ret)
    
    if symbol is None: return alert_error('Invalid symbol. Please select one and retry.', none_ret)
    if from_date is None: return alert_error('Invalid starting date. Please select one and retry.', none_ret)
    if to_date is None: return alert_error('Invalid ending date. Please select one and retry.', none_ret)
    if from_date > to_date: return alert_error('Invalid duration. Please check and retry.', none_ret)
    if interval is None: return alert_error('Invalid interval. Please select one and retry.', none_ret)
    if interval == INTERVAL_QUARTERLY or interval == INTERVAL_YEARLY: return alert_error('Cannot support quarterly or monthly backtest.', none_ret)
    if pivot_number is None: return alert_error('Invalid pivot number. Please select one and retry.', none_ret)
    
    if get_duration(from_date, to_date) < zigzag_window + zigzag_padding:
        return alert_error('Duration must be at least {} days for Fibonacci analysis.'.format(zigzag_window + zigzag_padding), none_ret)

    try:
        merge_thres = float(merge_thres) / 100
    except Exception:
        return alert_error('Invalid merge threshold. Please input correctly and retry.', none_ret)
    
    df = load_yf(symbol, from_date, to_date, interval)
    pivot_number = PIVOT_NUMBER_ALL.index(pivot_number) + 1

    records, success_rate, cum_profit = backtest_fib_extension(
        df, interval, pivot_number, get_safe_num(merge_thres), symbol
    )
    csv_path = 'out/FIB-EXT-BKTEST_{}_{}_{}_{}_p{}_m{}%_sr={}%_cp={}%.csv'.format(
        symbol, from_date, to_date, interval, pivot_number,
        '{:.1f}'.format(100 * merge_thres),
        '{:.1f}'.format(100 * success_rate),
        '{:.1f}'.format(100 * cum_profit)
    )
    records.to_csv(csv_path, index = False)
    report = get_report_content(records, csv_path)

    return alert_success('Backtest Complted.') + ['Report', report]

def update_plot(df, downfalls, extensions, behaviors, cur_date):
    fig = make_subplots(rows = 2, cols = 1, shared_xaxes = True, vertical_spacing = 0.05, row_heights = [0.8, 0.2])
    day_span = (df.index[-1] - df.index[0]).days

    pivot_wid = timedelta(days = int(day_span * 0.025))
    fold_step, dash_indent = 0.01, 0.07

    cur_price = df.loc[cur_date]['Close']
    extensions.sort(key = lambda g: abs(cur_price - (g[0][-1] + g[-1][-1]) / 2))

    upper_count, lower_count = 0, 0

    for g in extensions:
        b = behaviors[g[0]]

        if len(g) > 1:
            fig.add_trace(
                go.Scatter(
                    x = df.index,
                    y = np.repeat(g[0][-1], len(df)),
                    fill = None,
                    line = dict(color = 'darkgray', width = 0.5)
                ),
                row = 1, col = 1
            )
            fig.add_trace(
                go.Scatter(
                    x = df.index,
                    y = np.repeat(g[-1][-1], len(df)),
                    fill = 'tonexty',
                    line = dict(color = 'darkgray', width = 0.5)
                ),
                row = 1, col = 1
            )
            lv = (g[0][-1] + g[-1][-1]) / 2

            for i, e in enumerate(g):
                fig.add_trace(
                    go.Scatter(
                        mode = 'markers',
                        x = [df.index[int(len(df) * (i + 1) * fold_step)]],
                        y = [lv],
                        marker = dict(color = PLOT_COLORS_DARK[e[0]], symbol = 'circle')
                    ),
                    row = 1, col = 1
                )

            bmark_x = df.index[int(len(df) * (len(g) + 2) * fold_step)]
            bmark_y = lv
        else:
            i, hd, zd, hv, zv, j, lv = g[0]
            x = df.index[0] + timedelta(days = int(day_span * (dash_indent * (i + 2) + 0.005)))

            fig.add_shape(
                type = "line",
                x0 = x,
                y0 = lv,
                x1 = df.index[-1],
                y1 = lv,
                line = dict(color = PLOT_COLORS_DARK[i], width = 1, dash = 'dot'),
                row = 1, col = 1
            )
            fig.add_trace(
                go.Scatter(
                    x = [x],
                    y = [lv],
                    mode = "text",
                    text = ['{:.1f}%     '.format(FIB_EXT_LEVELS[j] * 100)],
                    textposition = "middle left",
                    textfont = dict(color = PLOT_COLORS_DARK[i])
                ),
                row = 1, col = 1
            )
            bmark_x = df.index[0] + timedelta(days = int(day_span * (dash_indent * (i + 2) - 0.004)))
            bmark_y = lv

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

        if is_near:
            fig.add_annotation(
                dict(
                    font = dict(color = "black" if len(g) > 1 else PLOT_COLORS_DARK[g[0][0]]),
                    x = 1,
                    y = np.log10(lv),
                    showarrow = False,
                    text='- {:.1f}'.format(lv),
                    xref = "paper",
                    xanchor = 'left',
                    yref = "y"
                )
            )

        if b is not None:
            fig.add_trace(
                go.Scatter(
                    mode = "markers",
                    x = [bmark_x],
                    y = [bmark_y],
                    marker = dict(
                        symbol = FIB_EXT_MARKERS[b][0],                        
                        color = FIB_EXT_MARKERS[b][1],
                        size = 7.5,
                        angle = FIB_EXT_MARKERS[b][2],
                        line = dict(
                            color = "black",
                            width = 1
                        )
                    )
                )
            )

    for j, f in enumerate(downfalls[::-1]):
        hd, zd = f
        i = len(downfalls) - 1 - j

        fig.add_shape(
            type = "line",
            x0 = hd,
            y0 = df.loc[hd]['Close'],
            x1 = zd + pivot_wid,
            y1 = df.loc[hd]['Close'],
            line = dict(color = PLOT_COLORS_DARK[i], width = 1.5),
            row = 1, col = 1
        )
        fig.add_shape(
            type = "line",
            x0 = zd,
            y0 = df.loc[zd]['Close'],
            x1 = zd + pivot_wid,
            y1 = df.loc[zd]['Close'],
            line = dict(color = PLOT_COLORS_DARK[i], width = 1.5),
            row = 1, col = 1
        )
        fig.add_trace(
            go.Scatter(
                x = [hd],
                y = [df.loc[hd]['Close']],
                mode = "text",
                text = ['{:.1f}'.format(df.loc[hd]['Close'])],
                textposition = "top center",
                textfont = dict(color = PLOT_COLORS_DARK[i])
            ),
            row = 1, col = 1
        )
        fig.add_trace(
            go.Scatter(
                x = [zd],
                y = [df.loc[zd]['Close']],
                mode = "text",
                text = ['{:.1f}'.format(df.loc[zd]['Close'])],
                textposition = "bottom center",
                textfont = dict(color = PLOT_COLORS_DARK[i])
            ),
            row = 1, col = 1
        )

    fig.add_shape(
        type = "line",
        x0 = cur_date,
        y0 = cur_price,
        x1 = df.index[-1],
        y1 = cur_price,
        line = dict(color = 'red', width = 2, dash = 'dash'),
        row = 1, col = 1
    )
    fig.add_annotation(
        dict(
            font = dict(color = "red", size = 14),
            x = 1,
            y = np.log10(cur_price),
            showarrow = False,
            text = '------- {:.1f}'.format(cur_price),
            xref = "paper",
            xanchor = 'left'
        )
    )
    fig.add_shape(
        type = "line",
        x0 = cur_date,
        y0 = min(df['Low']),
        x1 = cur_date,
        y1 = max(df['High']) if len(extensions) == 0 else extensions[-1][-1][-1],
        line = dict(color = 'darkgreen'),
        row = 1, col = 1
    )
    fig.add_annotation(
        dict(
            font = dict(color = "darkgreen", size = 14),
            x = cur_date,
            y = np.log10(min(df['Low'])),
            showarrow = False,
            text = cur_date.strftime(' %d %b %Y') + ' →',
            xanchor = "left",
            yanchor = "bottom",
            yref = "y"
        )
    )
    fig.add_trace(get_candlestick(df), row = 1, col = 1)
    fig.add_trace(get_volume_bar(df), row = 2, col = 1)

    update_shared_xaxes(fig, df, 2)

    fig.update_yaxes(type = "log", title_text = "Price", row = 1, col = 1)
    fig.update_yaxes(title_text = "Volume", row = 2, col = 1)

    fig.update_layout(
        yaxis_tickformat = "0",
        height = 1200,
        margin = dict(t = 40, b = 40, r = 100),
        showlegend = False
    )
    return dcc.Graph(figure = fig, className = 'fib_ext_graph')
