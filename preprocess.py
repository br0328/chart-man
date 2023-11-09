
from constant import *
from yahoo import *
from data import *
import yfinance as yf
import pandas as pd

def fill_ipo_dates():
	df = pd.read_csv('./data/stake.csv')

	for i, row in df.iterrows():
		y = yf.download(row['symbol'], interval = '1d')
		ipo = y.iloc[0].name.strftime(YMD_FORMAT)

		df.at[i, 'ipo'] = ipo

	df.to_csv('./data/stake.csv', index = False)

def backup_cache():
	initialize_data()
	df = pd.read_csv('./data/stake.csv')

	for i, row in df.iterrows():
		load_yf(row['symbol'], None, None, None, for_backup = True)

if __name__ == '__main__':
	#fill_ipo_dates()
	backup_cache()
