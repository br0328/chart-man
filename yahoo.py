
from datetime import datetime
from collections import defaultdict
from constant import *
from config import *
from util import *
from data import *
import yfinance as yf
import joblib
import glob
import os

yf_caches = defaultdict(lambda: None)

def load_yf(symbol, start, end, interval, fit_today = True, for_backup = False):
	if start is None: start = '1900-01-01'
	if end is None: end = '2100-01-01'
	if interval is None: interval = INTERVAL_DAILY
    
	df = yf_caches[symbol]
	
	if df is None:
		if yf_on:
			df = yf.download(symbol, start = start, end = get_offset_date_str(end, 1), interval = '1d', progress = False)
			df = df.drop('Adj Close', axis = 1)
		else:
			df = joblib.load('./cache/{}.che'.format(symbol))

		if for_backup:
			joblib.dump(df, './cache/{}.che'.format(symbol))
			yf_caches[symbol] = df     

	df = df.dropna()
	df = df.round(4)
 
	end = get_offset_date_str(end, 1)
	df = df.loc[start:datetime.strptime(end, YMD_FORMAT)]

	agg_dict = {
		'Open': 'first',
		'Close': 'last',
		'High': 'max',
		'Low': 'min',
		'Volume': 'sum'
	}  
	df = df.groupby(pd.Grouper(freq = INTERVAL_LETTER_DICT[interval])).agg(agg_dict)
	df = df.dropna()

	if len(df) == 0: return df
	last_day = df.iloc[-1].name.strftime(YMD_FORMAT)

	if last_day > end:
		if fit_today:
			idx_list = df.index.to_list()
			idx_list[-1] = datetime.strptime(end, YMD_FORMAT)
			df.index = idx_list
		else:
			df = df[:-1]
	
	return df
