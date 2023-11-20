
from datetime import datetime, timedelta
from constant import *
from config import *
from util import *
from data import *
import yfinance as yf
import joblib
import glob
import os

def load_yf(symbol, start, end, interval, fit_today = False, for_backup = False):
	if yf_on:
		df = yf.download(symbol, start = start, end = end, interval = '1d', progress = False)
		df = df.drop('Adj Close', axis = 1)

		if for_backup:
			joblib.dump(df, './cache/{}.che'.format(symbol))
			return
	else:
		df = joblib.load('./cache/{}.che'.format(symbol))

	df = df.dropna()
	df = df.round(4)

	agg_dict = {
		'Open': 'first',
		'Close': 'last',
		'High': 'max',
		'Low': 'min',
		'Volume': 'sum'
	}
	df = df.groupby(pd.Grouper(freq = INTERVAL_LETTER_DICT[interval])).agg(agg_dict)
	df = df.dropna()

	if not fit_today:
		today = get_today_str()
		last_day = df.iloc[-1].name.strftime(YMD_FORMAT)

		if last_day > today: df = df[:-1]

	df = df.loc[start:get_offset_date_str(end, 1)]
	return df
