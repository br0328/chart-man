
from plotly.subplots import make_subplots
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
		decreasing_line_color = 'red',
  		showlegend = False
	)

def get_volume_bar(df):
	return go.Bar(
		x = df.index,
		y = df['Volume'],
		marker_color = df['Close'].diff().apply(lambda x: 'green' if x >= 0 else 'red'),
		showlegend = False
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

def get_figure_with_candlestick_pair(df1, df2, title1 = '', title2 = ''):
    fig = make_subplots(rows = 2, cols = 1, shared_xaxes = True, vertical_spacing = 0.01, subplot_titles = (title1, title2))
    
    fig.add_trace(go.Candlestick(x = df1.index, open = df1['Open'], high = df1['High'], low = df1['Low'], close = df1['Close']), row = 1, col = 1)
    fig.add_trace(go.Candlestick(x = df2.index, open = df2['Open'], high = df2['High'], low = df2['Low'], close = df2['Close']), row = 2, col = 1)
        
    fig.update_yaxes(type = 'log', row = 1, col = 1)
    fig.update_layout(xaxis_rangeslider_visible = False, showlegend = False)
    fig.update_layout(xaxis2_rangeslider_visible = False)
    fig.update_yaxes(type = 'log', row = 2, col = 1)
    
    return fig
