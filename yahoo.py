
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
        df = yf.download(symbol, start = start, end = end, interval = '1d')
        
        if for_backup:
            joblib.dump(df, './cache/{}.che'.format(symbol))
            return
    else:
        df = joblib.load('./cache/{}.che'.format(symbol))

    df = df.loc[start:get_offset_date_str(end, 1)]
    return df
