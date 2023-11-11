
from yahoo import *
from data import *
from util import *
import pandas as pd
import joblib

def test_cached_df():
	df = joblib.load('./cache/AAPL.che')
	print(df[:-1])
	print('###')

	print(df.loc[get_timestamp('2008-02-11'):].index[1:3])

	for r in df.loc[df.loc[get_timestamp('2008-02-11'):].index[1:3]].iloc:
		print(r)
		break

	print(df.iloc[0:3])

def test_stake():
	initialize_data()

	df = load_stake()
	print(df.loc['AAPL']['ipo'])

if __name__ == '__main__':
	test_cached_df()
	#test_stake()
