
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
    if interval == INTERVAL_DAILY:
        interval_key = '1d'
    elif interval == INTERVAL_WEEKLY:
        interval_key = '1wk'
    elif interval == INTERVAL_MONTHLY:
        interval_key = '1mo'
    else:
        interval_key = '3mo'

    ipo_date = load_stake().loc[symbol]['ipo']

    if yf_on:
        df = yf.download(symbol, start = start, end = end, interval = interval_key)
        
        if for_backup:
            joblib.dump(df, './cache/{}_{}.che'.format(get_interval_letter(interval_key), symbol))
            return
    else:
        df = joblib.load('./cache/{}_{}.che'.format(get_interval_letter(interval_key), symbol))

    df = df.loc[start:get_offset_date_str(end, 1)]
    return df
