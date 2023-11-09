
from yahoo import *
from data import *
import pandas as pd
import joblib

def test_cached_df():
	df = joblib.load('./cache/AAPL.che')
	print(df[:-1])

if __name__ == '__main__':
	test_cached_df()
