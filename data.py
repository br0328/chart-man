
import pandas as pd

def initialize_data():
	global df_stake

	df_stake = None

def load_stake():
	global df_stake

	if df_stake is None: df_stake = pd.read_csv('./data/stake.csv', index_col = 'symbol')

	return df_stake

def load_symbols():
	return list(load_stake().index)

def load_stock_symbols():
	return [s for s in load_symbols() if not s.startswith('^')]
