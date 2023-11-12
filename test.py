
from yahoo import *
from data import *
from util import *
import pandas as pd
import joblib

def test_cached_df():
	df = joblib.load('./cache/AAPL.che')
	print(df[:-1])
	print('###')

	for d in df.index:
		if str(df.loc[d]['Close']).lower().startswith('na'):
			print(d)

def test_stake():
	initialize_data()

	df = load_stake()
	print(df.loc['AAPL']['ipo'])

if __name__ == '__main__':
	test_cached_df()
	#test_stake()
