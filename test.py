
from yahoo import *
from data import *
import pandas as pd
import joblib

def test_cached_df():
	df = joblib.load('./cache/M_AAPL_1900-11-30_2023-10-31.che')
	print(df)

if __name__ == '__main__':
	initialize_data()

	test_cached_df()
