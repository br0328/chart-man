
from datetime import timedelta
from collections import deque
from constant import *
from tqdm import tqdm
from yahoo import *
from data import *
from config import *
from util import *
import pandas as pd
import numpy as np

"""
Core logic modules
"""

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
			'{:.4f}$'.format(df.loc[last_date]['Close']),
			'√ {:.4f}'.format(highs[-1]) if is_new_highest else ''
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
