
import plotly.graph_objects as go

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
