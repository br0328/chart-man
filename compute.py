
from datetime import timedelta
from config import *
import pandas as pd

def get_zigzag(df):
	pivots = []
	series = df['Close']

	init_date = df.index[0]
	final_date = df.index[-1]
	
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
