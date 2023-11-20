
from scipy import stats
import plotly.graph_objects as go
import numpy as np

def get_candlestick(df):
	return go.Candlestick(
		x = df.index,
		open = df['Open'],
		high = df['High'],
		low = df['Low'],
		close = df['Close'],
		increasing_line_color = 'green',
		decreasing_line_color = 'red'
		)

def get_volume_bar(df):
	return go.Bar(
		x = df.index,
		y = df['Volume'],
		marker_color = df['Close'].diff().apply(lambda x: 'green' if x >= 0 else 'red')
	)

def get_quick_line(df, column, color = 'blue'):
	return go.Scatter(
		x = df.index,
		y = df[column],
		line = dict(color = color)
	)

def update_shared_xaxes(fig, df, count):
	for i in range(count):
		fig.update_xaxes(
			rangeslider_visible = False,
			range = [df.index[0], df.index[-1]],
			row = i + 1, col = 1
		)

def draw_tonexty_hline(fig, xran, y0, y1, color = 'black', width = 1, row = 1, col = 1):
	for i in range(2):
		fig.add_trace(
			go.Scatter(
				x = xran,
				y = np.repeat(y0 if i == 0 else y1, len(xran)),
				fill = None if i == 0 else 'tonexty',
				line = dict(color = color, width = width)
			),
			row = row, col = col
		)

def draw_marker(fig, x, y, symbol = 'square', color = 'black', size = None, angle = 0, line_color = None, row = 1, col = 1):
	fig.add_trace(
		go.Scatter(
			mode = 'markers',
			x = [x],
			y = [y],
			marker = dict(
				color = color,
				symbol = symbol,
				size = size,
				angle = angle,
				line = dict(color = line_color, width = 0 if line_color is None else 1)
			)
		),
		row = row, col = col
	)

def draw_hline_shape(fig, x0, x1, y, color = 'black', width = 1, dash = 'solid', row = 1, col = 1):
	fig.add_shape(
		type = "line",
		x0 = x0,
		y0 = y,
		x1 = x1,
		y1 = y,
		line = dict(color = color, width = width, dash = dash),
		row = row,
		col = col
	)

def draw_vline_shape(fig, x, y0, y1, color = 'black', width = 1, dash = 'solid', row = 1, col = 1):
	fig.add_shape(
		type = "line",
		x0 = x,
		y0 = y0,
		x1 = x,
		y1 = y1,
		line = dict(color = color, width = width, dash = dash),
		row = row,
		col = col
	)

def draw_text(fig, x, y, text, position = 'middle center', color = 'black', size = 12, row = 1, col = 1):
	fig.add_trace(
		go.Scatter(
			x = [x],
			y = [y],
			mode = "text",
			text = [text],
			textposition = position,
			textfont = dict(color = color, size = size)
		),
		row = row,
		col = col
	)

def draw_annotation(fig, x, y, text, xref = 'x', yref = 'y', xanchor = 'center', yanchor = 'middle', color = 'black', size = 12):
	fig.add_annotation(
		dict(
			font = dict(color = color, size = size),
			x = x,
			y = y,
			showarrow = False,
			text = text,
			xref = xref,
			xanchor = xanchor,
			yref = yref,
			yanchor = yanchor
		)
	)

def get_complete_hline(y, color, dash = 'solid', width = 1):
	return dict(
		type = 'line',
		line = dict(
			color = color,
			width = width,
			dash = dash
		),
		x0 = 0, x1 = 1, xref = 'paper',
		y0 = y, y1= y		   
	)

def get_complete_vline(x, color, dash = 'solid', width = 1):
	return dict(
		type = 'line',
		line = dict(
			color = color,
			width = width,
			dash = dash
		),
		x0 = x, x1 = x,
		y0 = 0, y1 = 1, yref = 'paper'
	)

def collect_channel(candle, backcandles, window, df):
	best_r_squared_low = 0
	best_r_squared_high = 0
	best_slope_low = 0
	best_intercept_low = 0
	best_slope_high = 0
	best_intercept_high = 0
	best_backcandles_low = 0
	best_backcandles_high = 0
	
	for i in range(backcandles - backcandles // 2, backcandles + backcandles // 2, window):
		local_df = df.iloc[candle - i - window: candle - window]
		
		lows = local_df[local_df['isPivot'] == 2].Low.values[-4:]
		idx_lows = local_df[local_df['isPivot'] == 2].Low.index[-4:]
		highs = local_df[local_df['isPivot'] == 1].High.values[-4:]
		idx_highs = local_df[local_df['isPivot'] == 1].High.index[-4:]

		if len(lows) >= 2:
			slope_low, intercept_low, r_value_l, _, _ = stats.linregress(idx_lows, lows)
			
			if (r_value_l ** 2) * len(lows) > best_r_squared_low and (r_value_l ** 2) > 0.85:
				best_r_squared_low = (r_value_l ** 2)*len(lows)
				best_slope_low = slope_low
				best_intercept_low = intercept_low
				best_backcandles_low = i
		
		if len(highs) >= 2:
			slope_high, intercept_high, r_value_h, _, _ = stats.linregress(idx_highs, highs)
			
			if (r_value_h ** 2)*len(highs) > best_r_squared_high and (r_value_h ** 2)> 0.85:
				best_r_squared_high = (r_value_h ** 2)*len(highs)
				best_slope_high = slope_high
				best_intercept_high = intercept_high
				best_backcandles_high = i
	
	return best_backcandles_low, best_slope_low, best_intercept_low, best_r_squared_low, best_backcandles_high, best_slope_high, best_intercept_high, best_r_squared_high

def is_breakout(candle, backcandles, window, df, stop_percentage):
	if 'isBreakOut' not in df.columns: return 0

	for i in range(1, 2):
		if df['isBreakOut'].iloc[candle - i] != 0: return 0
  
	if candle - backcandles - window < 0: return 0
	best_back_l, sl_lows, interc_lows, r_sq_l, best_back_h, sl_highs, interc_highs, r_sq_h = collect_channel(candle, backcandles, window, df)
	
	thirdback = candle - 2
	thirdback_low = df.iloc[thirdback].Low
	thirdback_high = df.iloc[thirdback].High
	thirdback_volume = df.iloc[thirdback].Volume

	prev_idx = candle - 1
	prev_high = df.iloc[prev_idx].High
	prev_low = df.iloc[prev_idx].Low
	prev_close = df.iloc[prev_idx].Close
	prev_open = df.iloc[prev_idx].Open
	
	curr_idx = candle
	curr_high = df.iloc[curr_idx].High
	curr_low = df.iloc[curr_idx].Low
	curr_close = df.iloc[curr_idx].Close
	curr_open = df.iloc[curr_idx].Open
	curr_volume= max(df.iloc[candle].Volume, df.iloc[candle-1].Volume)
	breakpclow = (sl_lows * prev_idx + interc_lows  - curr_low) / curr_open
	breakpchigh = (curr_high - sl_highs * prev_idx - interc_highs) / curr_open

	if ( 
		thirdback_high > sl_lows * thirdback + interc_lows and
		curr_volume >thirdback_volume and
		prev_close < prev_open and
		curr_close < curr_open and
		sl_lows > 0 and
		prev_close < sl_lows * prev_idx + interc_lows and
		curr_close < sl_lows * prev_idx + interc_lows):
		return 1
	elif (
		thirdback_low < sl_highs * thirdback + interc_highs and
		curr_volume > thirdback_volume and
		prev_close > prev_open and 
		curr_close > curr_open and
		sl_highs < 0 and
		prev_close > sl_highs * prev_idx + interc_highs and
		curr_close > sl_highs * prev_idx + interc_highs):
		return 2
	else:
		return 0

def calculate_breakpoint_pos(row):
	if row['isBreakOut'] == 2:
		return row['Low'] - 3e-3
	elif row['isBreakOut'] == 1:
		return row['High'] + 3e-3
	else:
		return np.nan
	