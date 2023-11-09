
from yahoo import *
from data import *
import pandas as pd
import joblib

def test_cached_df():
	df = joblib.load('./cache/D_AAPL_1900-11-30_2023-11-08.che')
	print(df)

	print('###')
	print(df.iloc[-1].name.strftime('%Y-%m-%d'))

if __name__ == '__main__':
	initialize_data()

	test_cached_df()
