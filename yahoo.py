
from datetime import datetime, timedelta
from constant import *
from util import *
from data import *
import yfinance as yf
import joblib
import glob
import os

def load_series(symbol, start, end, interval):
    if interval == INTERVAL_DAILY:
        interval_key = '1d'
        safe_offset = 1
    elif interval == INTERVAL_WEEKLY:
        interval_key = '1wk'
        safe_offset = 7
    else:
        interval_key = '1mo'
        safe_offset = 30

    df = load_cache(symbol, start, end, interval_key)
    
    if df is None:
        df = yf.download(symbol, start = start, end = end, interval = interval_key)
        safe_end = get_offset_date_str(end, -safe_offset)

        if start <= safe_end: add_cache(df, symbol, start, safe_end, interval_key)

    df = df.loc[start:end]

    return df

def load_cache(symbol, start, end, interval_key):
    ipo_date = load_stake().loc[symbol]['ipo']
    
    new_start = max(start, ipo_date)
    new_end = min(end, get_today_str())

    for fn in glob.glob('./cache/{}_{}_*.che'.format(get_interval_letter(interval_key), symbol)):
        segs = fn.split('_')
        sd, ed = segs[-2], segs[-1].split('.')[0]

        if sd <= new_start and new_end <= ed: return joblib.load(fn)

    return None

def add_cache(df, symbol, start, end, interval_key):
    cache_files = glob.glob('./cache/{}_{}_*.che'.format(get_interval_letter(interval_key), symbol))    
    joblib.dump(df, './cache/{}_{}_{}_{}.che'.format(get_interval_letter(interval_key), symbol, start, end))

    for fn in cache_files:
        segs = fn.split('_')
        sd, ed = segs[2], segs[3]

        if start <= sd and ed <= end: os.remove(fn)
