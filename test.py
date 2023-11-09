
from yahoo import *
from data import *
import pandas as pd
import joblib

def test_cached_df():
	df = joblib.load('./cache/AAPL.che')
	print(df)

	df = df.drop('Adj Close', axis = 1)
	print(df)

	#df = df.groupby(pd.Grouper(freq = 'Y')).sum()

	agg_dict = {'Open': 'first',
            'Close': 'last',
            'High': 'max',
            'Low': 'min',
            'Volume': 'sum'}

	grouped_df = df.groupby(pd.Grouper(freq = 'Q')).agg(agg_dict)
	print(grouped_df)

if __name__ == '__main__':
	test_cached_df()
