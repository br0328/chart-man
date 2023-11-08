
from datetime import datetime, timedelta
from constant import *
from util import *
from data import *
import yfinance as yf
import joblib
import glob
import os

def load_yf(symbol, start, end, interval):
    if interval == INTERVAL_DAILY:
        interval_key = '1d'
    elif interval == INTERVAL_WEEKLY:
        interval_key = '1wk'
    elif interval == INTERVAL_MONTHLY:
        interval_key = '1mo'
    else:
        interval_key = '3mo'

    df = load_cache(symbol, start, end, interval_key)
    
    if df is None:
        df = yf.download(symbol, start = start, end = end, interval = interval_key)
        add_cache(df, symbol, start, end, interval_key)

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

    safe_end = min(end, get_today_str())
    joblib.dump(df, './cache/{}_{}_{}_{}.che'.format(get_interval_letter(interval_key), symbol, start, safe_end))

    for fn in cache_files:
        segs = fn.split('_')
        sd, ed = segs[2], segs[3]

        if start <= sd and ed <= safe_end: os.remove(fn)
