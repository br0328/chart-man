
from plotly.subplots import make_subplots
from datetime import timedelta
from collections import deque
from scipy import stats
from constant import *
from tqdm import tqdm
from finta import TA
from yahoo import *
from data import *
from config import *
from util import *
import plotly.graph_objects as go
import pandas_ta as ta
import pandas as pd
import numpy as np
import math

"""
Core logic modules
"""

# Class to store endpoints of a line and handle required requests
class Linear:
    def __init__(self, x1, y1, x2, y2, startIndex = None, endIndex = None):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.startIndex = startIndex
        self.endIndex = endIndex
        self.m = (y1 - y2) / (x1 - x2)
        self.c = y1 - self.m * x1

    def getY(self,x):
        return self.m * x + self.c

    def isInRange(self,x):
        return self.x1 <= x and x <= self.x2

    def getAngle(self, inDegrees=True):
        tanTheta = self.m
        theta = math.atan(tanTheta)

        if not inDegrees:
            return theta
        else:
            return theta * 180 / math.pi

    def getMangnitude(self):
        return math.sqrt((self.y2 - self.y1) * (self.y2 - self.y1) + (self.x2 - self.x1) * (self.x2 - self.x1))

class Reigon:
    def __init__(self, s, e, c):
        self.start = s
        self.end = e
        self.class_ = c

def getReigons(highs, data, stoch = False):
    reigons = []
    i = 15 if stoch else 0

    while i + 1 < len(highs):
        h1 = highs[i]
        h2 = highs[i+1]
        p1 = data[h1]
        p2 = data[h2]
        
        if p2 > p1 and (p2-p1)/p2 > 0.025:
            reigons.append(Reigon(h1, h2, 1))
        elif p2 < p1 and (p1-p2)/p1 > 0.025:
            reigons.append(Reigon(h1, h2, -1))
        else:
            reigons.append(Reigon(h1, h2, 0))
            
        i += 1
    return reigons

def getFinalReigons(reigons):
    rr = reigons.copy()
    i = 0
    
    while i + 1 < len(rr):
        r1 = rr[i]
        r2 = rr[i+1]

        if not r1.class_ == r2.class_:
            i += 1
        else:
            rr[i].end = r2.end
            rr.remove(r2)
            
    return rr

def getOverlap(r1, r2, percent = 0.3):
    s1 = r1.start
    s2 = r2.start
    e1 = r1.end
    e2 = r2.end

    if s2 <= s1 and e2 <= s1:
        return False
    elif s2 >= e1 and e2 >= e1:
        return False
    elif s2 < s1 and e2 > e1:
        return True
    elif s2 > s1 and e2 < e1:
        return True
    elif s1 < s2 and e2 > s1:
        p = (e2 - s1) / (e1 - s1)
        return p > percent
    elif s2 < e1 and e2 > e1:
        p = (e1 - s2) / (e1 - s1)
        return p > percent

def getClosestPrevIndex(start, data, type):
    min_ = 10000000000
    selected = None
    
    if type == 'start':
        for d in data:
            if start - d > 0:
                if start - d < min_:
                    min_ = start - d
                    selected = d
                    
        return selected
    else:
        for d in data:
            if start - d < 0:
                if -start + d < min_:
                    min_ = -start + d
                    selected = d
                    
        return selected

# Get local peak points using Zig-Zag algorithm
def get_zigzag(df, final_date):
	pivots = []

	series = df['Close']
	init_date = df.index[0]
	
	win_dur = timedelta(days = zigzag_window)
	pad_dur = timedelta(days = zigzag_padding)

	win_end_date = final_date - pad_dur
	win_start_date = win_end_date - win_dur

	while win_start_date >= init_date:
		if len(series[win_start_date:win_end_date]) > 1:
			max_idx = series[win_start_date:win_end_date].idxmax()
			min_idx = series[win_start_date:win_end_date].idxmin()

			if max_idx < min_idx:
				if len(pivots) > 0:
					if pivots[-1][1] > 0:
						pivots.append((min_idx, -1, series[min_idx]))
						pivots.append((max_idx, 1, series[max_idx]))
					elif pivots[-1][2] < series[min_idx]:
						pivots.append((max_idx, 1, series[max_idx]))
					else:
						pivots[-1] = (min_idx, -1, series[min_idx])
						pivots.append((max_idx, 1, series[max_idx]))
				else:
					pivots.append((min_idx, -1, series[min_idx]))
					pivots.append((max_idx, 1, series[max_idx]))
			else:
				if len(pivots) > 0:
					if pivots[-1][1] < 0:
						pivots.append((max_idx, 1, series[max_idx]))
						pivots.append((min_idx, -1, series[min_idx]))
					elif pivots[-1][2] > series[max_idx]:
						pivots.append((min_idx, -1, series[min_idx]))
					else:
						pivots[-1] = (max_idx, 1, series[max_idx])
						pivots.append((min_idx, -1, series[min_idx]))
				else:
					pivots.append((max_idx, 1, series[max_idx]))
					pivots.append((min_idx, -1, series[min_idx]))

		win_end_date -= win_dur
		win_start_date -= win_dur

	pivots = pivots[::-1]

	for _ in range(zigzag_merges):
		merged_pivots = merge_zigzag_pivots(pivots)		
		if len(merged_pivots) < 4: break

		pivots = merged_pivots

	res = pd.DataFrame(columns = ['Date', 'Sign', 'Close'])
	
	for idx, sign, v in pivots:
		r = {'Date': idx, 'Sign': sign, 'Close': v}
		res = pd.concat([res, pd.Series(r).to_frame().T], ignore_index = True)

	res.set_index('Date', inplace = True)
	return res

# Refine peak points by merging Zig-Zag peaks
def merge_zigzag_pivots(pivots):
	if len(pivots) < 3: return pivots	
	res, i = [], 0

	while i < len(pivots) - 3:
		res.append(pivots[i])

		if pivots[i + 3][0] - pivots[i][0] < timedelta(days = zigzag_merge_dur_limit):
			v = [pivots[j][2] for j in range(i, i + 4)]

			if min(v[0], v[3]) < min(v[1], v[2]) and max(v[0], v[3]) > max(v[1], v[2]):
				if zigzag_merge_val_limit * (max(v[0], v[3]) - min(v[0], v[3])) > (max(v[1], v[2]) - min(v[1], v[2])):
					i += 3
				else:
					i += 1
			else:
				i += 1
		else:
			i += 1

	for j in range(i, len(pivots)):
		res.append(pivots[j])

	return res

# Get recent downfall pivot pairs from Zig-Zag peak points
def get_recent_downfalls(zdf, count):
	res = []

	for i in range(len(zdf) - 1, 1, -1):
		row, prev = zdf.iloc[i], zdf.iloc[i - 1]		

		if row['Sign'] > 0: continue		
		hv, zv = prev['Close'], row['Close']

		if (hv - zv) < hv * fibo_pivot_diff_limit: continue
		res.append((prev.name, row.name))

		if len(res) == count: break

	return res[::-1]

# Get Fibonacci extension levels from a given set of downfall pivot pairs
def get_fib_extensions(zdf, downfalls, merge_thres, suppress_level):
	all_levels = []

	for i, f in enumerate(downfalls):
		hd, zd = f
		hv, zv = zdf.loc[hd]['Close'], zdf.loc[zd]['Close']
		dv = hv - zv

		for j, l in enumerate(FIB_EXT_LEVELS):
			lv = zv + dv * l
			if lv > suppress_level: break

			all_levels.append((i, hd, zd, hv, zv, j, round(lv, 4)))

	all_levels.sort(key = lambda x: x[-1])
	res, flags = [], []

	for i, level in enumerate(all_levels):
		if i in flags: continue

		lv = level[-1]
		th = lv * merge_thres

		flags.append(i)		
		g = [level]

		for j in range(i + 1, len(all_levels)):
			if j in flags: continue
			v = all_levels[j][-1]

			if v - lv <= th:
				flags.append(j)
				g.append(all_levels[j])

				lv = v

		res.append(g)

	return res

# Compute behaviors of Fibonacci extension levels
def get_fib_ext_behaviors(df, extensions, cur_date, merge_thres):
	res = {}
	cur_price = df.loc[cur_date]['Close']

	for g in extensions:
		lv = (g[0][-1] + g[-1][-1]) / 2
		is_resist = (lv >= cur_price)

		behavior, pv, start_date = None, None, None

		for d in df.loc[cur_date:].iloc:
			v = d.High if is_resist else d.Low

			if pv is not None:
				if (pv < lv and v >= lv) or (pv > lv and v <= lv):
					start_date = d.name
					break

			pv = d.Low if is_resist else d.High

		if start_date is not None:
			milestone_forward = FIB_BEHAVIOR_MILESTONE
			milestone_date = None

			while milestone_forward >= 5 and milestone_date is None:
				milestone_date = get_nearest_forward_date(df, start_date + timedelta(days = milestone_forward))
				milestone_forward //= 2

			if milestone_date is not None:
				mlv = df.loc[milestone_date]['Close']
				thres = lv * merge_thres

				has_mid_up, has_mid_down = False, False

				for d in df.loc[df.loc[start_date:milestone_date].index[1:-1]].iloc:
					if (d.Close - lv) >= thres:
						has_mid_up = True
					elif (lv - d.Close) >= thres:
						has_mid_down = True

				if (mlv - lv) >= thres:
					if has_mid_down:
						behavior = 'Res_Semi_Break' if is_resist else 'Sup_Semi_Sup'
					else:
						behavior = 'Res_Break' if is_resist else 'Sup_Sup'
				elif (lv - mlv) >= thres:
					if has_mid_up:
						behavior = 'Res_Semi_Res' if is_resist else 'Sup_Semi_Break'
					else:
						behavior = 'Res_Res' if is_resist else 'Sup_Break'
				elif has_mid_up == has_mid_down:
					end_date = get_nearest_forward_date(df, milestone_date + timedelta(days = milestone_forward))

					if end_date is not None:
						elv = df.loc[end_date]['Close']

						if (elv - lv) >= thres:
							behavior = 'Res_Semi_Break' if is_resist else 'Sup_Semi_Sup'
						elif (lv - elv) >= thres:
							behavior = 'Res_Semi_Res' if is_resist else 'Sup_Semi_Break'
						else:
							behavior = 'Vibration'
					else:
						behavior = 'Vibration'
				elif has_mid_up:
					behavior = 'Res_Break' if is_resist else 'Sup_Sup'
				else:
					behavior = 'Res_Res' if is_resist else 'Sup_Break'

		res[g[0]] = behavior

	return res

# Generate table-format data for Fibonacci extension analysis
def analyze_fib_extension(df, extensions, behaviors, cur_date, pivot_number, merge_thres, interval, symbol):
	cols = ['ExtID', 'Level', 'Type', 'Width', 'Behavior', 'Description', ' ']
	res = pd.DataFrame(columns = cols)
	
	cur_price = df.loc[cur_date]['Close']
	i = 0

	for g in extensions:
		lv = (g[0][-1] + g[-1][-1]) / 2
		b = behaviors[g[0]]
		i += 1

		record = [
			i,
			'{:.4f}$'.format(lv),
			'Resistance' if lv >= cur_price else 'Support',
			'{:.2f}%'.format(100 * (g[-1][-1] - g[0][-1]) / g[0][-1]) if len(g) > 1 else '',
			FIB_EXT_MARKERS[b][-1] if b is not None else '',
			' & '.join(['{:.1f}% of {:.4f}-{:.4f}'.format(FIB_EXT_LEVELS[j] * 100, zv, hv) for _, _, _, hv, zv, j, _ in g]),
			''
		]
		res = pd.concat([res, pd.Series(dict(zip(cols, record))).to_frame().T], ignore_index = True)

	res = pd.concat([res, pd.Series({}).to_frame().T], ignore_index = True)
	
	res = pd.concat([res, pd.Series({
		'ExtID': 'Ticker:',
		'Level': symbol,
		'Type': 'Current Date:',
		'Width': cur_date.strftime('%d %b %Y'),
		'Behavior': 'Current Price:',
		'Description': '{:.4f}$'.format(cur_price)
	}).to_frame().T], ignore_index = True)

	res = pd.concat([res, pd.Series({
		'Level': 'From: {}'.format(df.index[0].strftime('%d %b %Y')),
		'Type': 'To: {}'.format(df.index[-1].strftime('%d %b %Y')),
		'Width': 'By: ' + interval,
		'Behavior': 'Merge: {:.1f}%'.format(merge_thres * 100),
		'Description': 'Recent Pivots: {}'.format(pivot_number)
	}).to_frame().T], ignore_index = True)

	res = pd.concat([res, pd.Series({}).to_frame().T], ignore_index = True)
	return res

# Backtest using Fibonacci extension strategy
#
# (Logic)
# For each date point, get recent downfall pivot pairs (Hundred-Zero pairs)
# Calculate Fibonacci extension levels
# If the current date point crosses an extension level, it's time to send signal.
# If it falls to cross, a Short signal is raised.
# If it rises to cross, a Long signal is raised.
# With Short signal, either put Short position or seal ongoing Long position.
# With Long signal, either put Long position or seal ongoing Short position.
#
# (Return)
# Transaction records, position accuracy rate and cumulated profit on percentage basis
def backtest_fib_extension(df, interval, pivot_number, merge_thres, symbol):
	cols = ['TransID', 'Position', 'EnterDate', 'EnterPrice', 'ExitDate', 'ExitPrice', 'Offset', 'Profit', 'CumProfit', 'X', ' ']

	enter_date, position = None, None
	trans_count, match_count, cum_profit = 0, 0, 0

	signs = deque(maxlen = 4 if interval == INTERVAL_DAILY else 1)
	res = pd.DataFrame(columns = cols)

	for cur_date in tqdm(list(df.index), desc = 'backtesting', colour = 'red'):
		cur_candle = df.loc[cur_date]
		signs.append(np.sign(cur_candle['Close'] - cur_candle['Open']))

		if enter_date is not None and (cur_date - enter_date).days < MIN_FIB_EXT_TRANS_DUR: continue

		if signs.count(1) == len(signs):
			cur_sign = 1
		elif signs.count(-1) == len(signs):
			cur_sign = -1
		else:
			cur_sign = 0

		if cur_sign == 0: continue
		if position == cur_sign: continue

		min_cur_price = min(cur_candle['Close'], cur_candle['Open'])
		max_cur_price = max(cur_candle['Close'], cur_candle['Open'])

		zdf = get_zigzag(df, cur_date)
		downfalls = get_recent_downfalls(zdf, pivot_number)
		extensions = get_fib_extensions(zdf, downfalls, get_safe_num(merge_thres), cur_candle['Close'] * 2)

		has_signal = False

		for g in extensions:
			lv = (g[0][-1] + g[-1][-1]) / 2

			if min_cur_price <= lv and lv <= max_cur_price:
				has_signal = True
				break

		if position is None:
			position = cur_sign
			enter_date = cur_date
		else:
			price_offset = cur_candle['Close'] - df.loc[enter_date]['Close']
			true_sign = np.sign(price_offset)
			trans_count += 1

			if true_sign == position: match_count += 1

			profit = position * price_offset / df.loc[enter_date]['Close']
			cum_profit += profit			

			record = [
				trans_count,
				'Long' if position > 0 else 'Short',
				enter_date.strftime('%d %b %Y'),
				'{:.4f}$'.format(df.loc[enter_date]['Close']),
				cur_date.strftime('%d %b %Y'),
				'{:.4f}$'.format(cur_candle['Close']),
				'{:.2f}%'.format(100 * price_offset / df.loc[enter_date]['Close']),
				'{:.4f}%'.format(100 * profit),
				'{:.4f}%'.format(100 * cum_profit),
				'T' if true_sign == position else 'F',
				' '
			]
			res = pd.concat([res, pd.Series(dict(zip(cols, record))).to_frame().T], ignore_index = True)
			enter_date, position = None, None

	success_rate = (match_count / trans_count) if trans_count != 0 else 0
	res = pd.concat([res, pd.Series({}).to_frame().T], ignore_index = True)
	
	res = pd.concat([res, pd.Series({
		'TransID': 'Ticker:',
		'Position': symbol,
		'EnterDate': 'From: {}'.format(df.index[0].strftime('%d %b %Y')),
		'EnterPrice': 'To: {}'.format(df.index[-1].strftime('%d %b %Y')),
		'ExitDate': 'By: ' + interval,
		'ExitPrice': 'Recent Pivots: {}'.format(pivot_number),
		'Offset': 'Merge: {:.1f}%'.format(merge_thres * 100)
	}).to_frame().T], ignore_index = True)

	res = pd.concat([res, pd.Series({
		'EnterDate': 'Success Rate:',
		'EnterPrice': '{:.1f}%'.format(success_rate * 100),
		'ExitDate': 'Cumulative Profit:',
		'ExitPrice': '{:.1f}%'.format(cum_profit * 100)
	}).to_frame().T], ignore_index = True)

	res = pd.concat([res, pd.Series({}).to_frame().T], ignore_index = True)
	return res, success_rate, cum_profit

# Get information for dashboard
def get_dashboard_info():
	cols = ['Symbol', 'State', 'Current Price', 'New Highest']
	res = pd.DataFrame(columns = cols)

	for symbol in tqdm(load_stock_symbols(), desc = 'loading', colour = 'green'):
		df = load_yf(symbol, '1800-01-01', '2100-01-01', INTERVAL_DAILY)

		highs = df['High'].to_numpy()		
		last_date = df.index[-1]

		is_new_highest = (highs.argmax() == len(highs) - 1)
		is_bullish = df.loc[last_date]['Close'] >= df.loc[last_date]['Open']

		record = [
			symbol,
			'↑ Bullish' if is_bullish else '↓ Bearish',
			'${:.4f}'.format(df.loc[last_date]['Close']),
			'√ ${:.4f}'.format(highs[-1]) if is_new_highest else ''
		]
		res = pd.concat([res, pd.Series(dict(zip(cols, record))).to_frame().T], ignore_index = True)

	res = pd.concat([res, pd.Series({}).to_frame().T], ignore_index = True)
	return res, last_date

def is_pivot(candle, window, df):
	if candle - window < 0 or candle + window >= len(df): return 0
	
	pivot_high = 1
	pivot_low = 2
	
	for i in range(candle - window, candle + window + 1):
		if df.iloc[candle].Low > df.iloc[i].Low: pivot_low = 0
		if df.iloc[candle].High < df.iloc[i].High: pivot_high = 0
	
	if pivot_high and pivot_low:
		return 3
	elif pivot_high:
		return pivot_high
	elif pivot_low:
		return pivot_low
	else:
		return 0

def calculate_point_pos(row):
	if row['isPivot'] == 2:
		return row['Low'] - 1e-3
	elif row['isPivot'] == 1:
		return row['High'] + 1e-3
	else:
		return np.nan

def backtest_trendline(df):
	combined_trades = pd.DataFrame()
	
	df['ID'] = range(len(df))
	df['Date'] = list(df.index)
	df.set_index('ID', inplace = True)
 
	atr = ta.atr(high = df['High'], low = df['Low'], close = df['Close'], length = 14)
	atr_multiplier = 2 
	stop_percentage = atr.iloc[-1] * atr_multiplier / df['Close'].iloc[-1]
        
	for level in range(2, 11, 2):
		window = 3 * level
		backcandles = 10 * window
		
		df['isPivot'] = df.apply(lambda row: is_pivot(row.name, window, df), axis = 1)
		df['isBreakOut'] = 0
  
		for i in range(backcandles + window, len(df)):
			df.loc[i, 'isBreakOut'] = is_breakout(i, backcandles, window, df, stop_percentage)

		trades_data = unit_trendline_backtest(df, level)
		combined_trades = pd.concat([combined_trades, trades_data])

	combined_trades = combined_trades.sort_values(by = 'Enter Date')
	combined_trades = combined_trades.drop_duplicates(subset = ['Enter Date', 'Exit Date'], keep = 'first')

	total_trades = len(combined_trades)
	profitable_trades = len(combined_trades[combined_trades['Profit/Loss'] > 0])
	success_rate = profitable_trades / total_trades if total_trades != 0 else 0

	valid_trades = combined_trades.dropna(subset = ['Return']).copy()
	valid_trades['Cumulative Return'] = (1 + valid_trades['Return'] / 100).cumprod()

	overall_return = valid_trades['Cumulative Return'].iloc[-1] - 1
	
	combined_trades = combined_trades.drop('Profit/Loss', axis = 1)
	combined_trades = combined_trades.round(4)

	return combined_trades, success_rate, overall_return

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
				best_r_squared_low = (r_value_l ** 2) * len(lows)
				best_slope_low = slope_low
				best_intercept_low = intercept_low
				best_backcandles_low = i
		
		if len(highs) >= 2:
			slope_high, intercept_high, r_value_h, _, _ = stats.linregress(idx_highs, highs)
			
			if (r_value_h ** 2) * len(highs) > best_r_squared_high and (r_value_h ** 2)> 0.85:
				best_r_squared_high = (r_value_h ** 2) * len(highs)
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
	prev_close = df.iloc[prev_idx].Close
	prev_open = df.iloc[prev_idx].Open
	
	curr_idx = candle
	curr_close = df.iloc[curr_idx].Close
	curr_open = df.iloc[curr_idx].Open
	curr_volume= max(df.iloc[candle].Volume, df.iloc[candle-1].Volume)

	if ( 
		thirdback_high > sl_lows * thirdback + interc_lows and
		curr_volume > thirdback_volume and
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

def getDivergance_LL_HL(r, rS):
    divs = []
    
    for rr in r:
        for rrs in rS:
            if getOverlap(rr, rrs):
                sc = rr.class_
                dc = rrs.class_

                if sc == -1 or sc == 0:
                    if dc == 1:
                        if not rr.start == rr.end and not rrs.start == rrs.end: divs.append(( (rrs.start, rr.start), (rrs.end, rr.end)))
    return divs

def getDivergance_HH_LH(r, rS):
    divs = []
    
    for rr in r:
        for rrs in rS:
            if getOverlap(rr, rrs):
                sc = rr.class_
                dc = rrs.class_

                if sc == 1 or sc == 0:
                    if dc == -1:
                        if not rr.start == rr.end and not rrs.start == rrs.end: divs.append(( (rrs.start, rr.start), (rrs.end, rr.end)))
    return divs


def calculate_breakpoint_pos(row):
	if row['isBreakOut'] == 2:
		return row['Low'] - 3e-3
	elif row['isBreakOut'] == 1:
		return row['High'] + 3e-3
	else:
		return np.nan

def unit_trendline_backtest(df, level):
    trades = []

    for i in range(1, len(df)):
        signal_type = df['isBreakOut'].iloc[i]
        signal = ""
        
        if signal_type == 2:
            signal = "Long"
            entry_date = df['Date'].iloc[i].strftime(YMD_FORMAT)
            entry_price = df['Close'].iloc[i]
            exit_price = None

            for j in range(i + 1, len(df)):
                if df['isPivot'].iloc[j] != 0:
                    exit_date = df['Date'].iloc[j].strftime(YMD_FORMAT)
                    exit_price = df['Close'].iloc[j]
                    break

            if exit_price is None:
                exit_date = df['Date'].iloc[-1].strftime(YMD_FORMAT)
                exit_price = df['Close'].iloc[-1]

            profit_or_stopped = calculate_profit_or_stopped(entry_price, exit_price, signal_type)
            trades.append((entry_date, entry_price, exit_date, exit_price, profit_or_stopped,signal,level))
        elif signal_type == 1:
              signal = "Short"
              entry_date = df['Date'].iloc[i].strftime(YMD_FORMAT)
              entry_price = df['Close'].iloc[i]
              exit_price = None
              
              for j in range(i + 1, len(df)):
                if df['isPivot'].iloc[j] != 0:
                    exit_date = df['Date'].iloc[j].strftime(YMD_FORMAT)
                    exit_price = df['Close'].iloc[j]
                    break

              if exit_price is None:
                    exit_date = df['Date'].iloc[-1].strftime(YMD_FORMAT)
                    exit_price = df['Close'].iloc[-1]

              profit_or_stopped = calculate_profit_or_stopped(entry_price, exit_price, signal_type)
              trades.append((entry_date, entry_price, exit_date, exit_price, profit_or_stopped, signal, level))

    trade_data = pd.DataFrame(trades, columns = ['Enter Date', 'Enter Price', 'Exit Date', 'Exit Price', 'Profit/Loss', 'Signal', 'Level'])
    trade_data['Return'] = trade_data['Profit/Loss'] * abs(trade_data['Enter Price'] - trade_data['Exit Price'] ) / trade_data['Enter Price']

    return trade_data

def calculate_profit_or_stopped(entry_price, exit_price, long_or_short):
  if long_or_short == 2:
    if exit_price >= entry_price :
        return 1
    else:
        return -1
  elif long_or_short == 1:
    if exit_price <= entry_price :
        return 1
    else:
        return -1

def get_inter_divergence_lows(df1, df2):
	last_low_1 = df1.iloc[-1].Low
	last_low_2 = df2.iloc[-1].Low

	i = len(df1) - 2
	starts, ends = -1, -1
    
	while i >= 0:
		dir1 = np.sign(last_low_1 - df1.iloc[i].Low)
		dir2 = np.sign(last_low_2 - df2.iloc[i].Low)
		
		if dir1 != dir2:
			if starts != -1: starts = len(df1) - i - 1
			ends = len(df1) - i - 1

		if starts != -1: break
		i -= 1

	return starts, ends

def get_inter_divergence_highs(df1, df2):
	last_high_1 = df1.iloc[-1].High
	last_high_2 = df2.iloc[-1].High
 
	i = len(df1) - 2
	starts, ends = -1, -1

	while i >= 0:
		dir1 = np.sign(last_high_1 - df1.iloc[i].High)
		dir2 = np.sign(last_high_2 - df2.iloc[i].High)

		if dir1 != dir2:
			if starts == -1: starts = i
			ends = i

		if starts != -1: break
		i -= 1

	return starts, ends

def getPointsBest(STOCK, min_ = 0.23, max_ = 0.8, getR = False, startDate = '2000-01-01', endDate = '2121-01-01',
		increment = 0.005, limit = 100, interval = INTERVAL_DAILY, returnData = 'close'):
    
    data = load_yf(STOCK, startDate, endDate, interval)
    data = data.dropna()
    data = data.rename(columns = {"Open": "open", "High": "high", "Low": "low", "Volume": "volume", "Close": "close"})
    
    date_format = "%Y-%m-%d"
    
    end_ = datetime.strptime(endDate, date_format)
    today = datetime.today()

    d = today - end_
    
    if returnData == 'close':
        Sdata = getScaledY(data["close"])
    elif returnData == 'lows':
        Sdata = getScaledY(data["low"])
    elif returnData == 'highs':
        Sdata = getScaledY(data["high"])
    else:
        print("Wrong data for argument returnData")
        return None
    
    R = 1.1
    satisfied = False
    c = 0
    
    while not satisfied and c < limit and R > 1:
        if returnData == 'close':
            highs, lows = getPointsforArray(data["close"], R)
        elif returnData == 'lows':
            highs, lows = getPointsforArray(data["low"], R)
        else:
            highs, lows = getPointsforArray(data["high"], R)
        
        if not len(highs) <2 and not len(lows) < 2:
            linears = getLinears(Sdata, sorted(highs+lows))
            MSE = getMSE(Sdata, linears)
            c += 1
            
            if min_ < MSE and MSE < max_:
                satisfied = True
            elif MSE > min_:
                R -= increment
            else:
                R += increment                
        else:
            R -= increment
    if R > 1:
        if returnData == 'close':
            h, l = getPointsforArray(data["close"], R)
        elif returnData == 'lows':
            h, l = getPointsforArray(data["low"], R)
        else:
            h, l = getPointsforArray(data["close"], R)
    else:
        if returnData == 'close':
            h, l = getPointsforArray(data["close"], 1.001)
        elif returnData == 'lows':
            h, l = getPointsforArray(data["low"], 1.001)
        else:
            h, l = getPointsforArray(data["close"], 1.001)

    if getR:
        return data, h, l, R
    else:
        return data, h, l

def getScaledY(data):
    return (np.asarray(data) - min(data)) / (max(data) - min(data))

def getScaledX(x, data):
    return np.asarray(x) / len(data)

def getUnscaledX(x, data):
    p =  x * len(data)
    return int(p)

def getLinears(data, Tps):
    linears = []
    i = 0
    
    while i + 1 < len(Tps):
        l = Linear(getScaledX(Tps[i], data), data[Tps[i]], getScaledX(Tps[i + 1], data), data[Tps[i + 1]])
        linears.append(l)
        i += 1
    return linears

def getLinearForX(lins, x):
    for l in lins:
        if l.isInRange(x): return l

    return None

def getMSE(data, lins):
    i = lins[0].x1
    E = 0
    
    while i < lins[-1].x2:
        l = getLinearForX(lins, i)
        p = data[getUnscaledX(i, data)]
        pHat = l.getY(i)
        E += abs((p - pHat)) * 1 / len(data)
        i += 1 / len(data)

    return E * 10

def getPointsforArray(series, R = 1.1):
    highs, lows = getTurningPoints(series, R, combined = False)
    return highs, lows

def getTurningPoints(closeSmall, R, combined = True):    
    markers_on = []
    highs = []
    lows = []
    
    i, markers_on = findFirst(closeSmall, len(closeSmall), R, markers_on)
    
    if i < len(closeSmall) and closeSmall[i] > closeSmall[0]:
        i, highs = finMax(i, closeSmall, len(closeSmall) - 1, R, highs)
    while i < len(closeSmall) - 1 and not math.isnan(closeSmall[i]):
        i, lows = finMin(i, closeSmall, len(closeSmall) - 1, R, lows)
        i, highs = finMax(i, closeSmall, len(closeSmall) - 1, R, highs)
    
    if combined:
        return highs + lows
    else:
        return highs, lows

def findFirst(a, n, R, markers_on):
    iMin = 1
    iMax = 1
    i = 2
    while i<n and a[i]/a[iMin]< R and a[iMax]/a[i]< R:
        if a[i] < a[iMin]:
            iMin = i
        if a[i] > a[iMax]:
            iMax = i
        i += 1
    if iMin < iMax:
        markers_on.append(iMin)
    else:
        markers_on.append(iMax)
    return i, markers_on

def finMin(i, a, n, R, markers_on):
    iMin = i
    
    while i < n and a[i] / a[iMin] < R:
        if a[i] < a[iMin]: iMin = i
        i += 1
        
    if i < n or a[iMin] < a[i]:
        markers_on.append(iMin)
        
    return i, markers_on

def finMax(i, a, n, R, markers_on):
    iMax = i
    
    while i < n and a[iMax] / a[i] < R:
        if a[i] > a[iMax]: iMax = i
        i += 1
        
    if i < n or a[iMax] > a[i]:
        markers_on.append(iMax)
        
    return i, markers_on

def getPointsGivenR(STOCK, R, startDate = '2000-01-01', endDate = '2121-01-01', interval = INTERVAL_DAILY, type_ = None, oldData = None):
	if oldData is None:
		data = load_yf(STOCK, startDate, endDate, interval)
		data = data.dropna()
		data = data.rename(columns={"Open": "open", "High": "high", "Low": "low", "Volume": "volume", "Close": "close"})
	else:
		data = oldData

	date_format = "%Y-%m-%d"
	end_ = datetime.strptime(endDate, date_format)
	today = datetime.today()

	d = today - end_

	if type_ is None:
		highs, lows = getPointsforArray(data["close"], R)
		return data, highs, lows
	elif type_== 'lows':
		_, lows = getPointsforArray(data["low"], R)
		return data, lows
	elif type_== 'highs':
		highs, _ = getPointsforArray(data["high"], R)
		return data, highs
	else:
		return None, None

def runStochDivergance(symbol, from_date = '2000-01-01', to_date = '2022-08-07', return_csv = False):
    R = 1.02
    data, _, _ = getPointsGivenR(symbol, R, startDate = from_date, endDate = to_date)
    _, lows = getPointsGivenR(symbol, R, startDate = from_date, endDate = to_date, type_='lows', oldData = data)
    _, highs = getPointsGivenR(symbol, R, startDate = from_date, endDate = to_date, type_='highs', oldData = data)

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
            StockM = (stockE - stockS) / (eS - sS)
            Dm = (eD - sS) / (eD - sD)

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

    if not return_csv: fig.update_layout(shapes = lines_to_draw, showlegend = False)    
    if return_csv: return typeONEs, typeTWOs
    
    return fig