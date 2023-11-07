
def initialize_data():
	global d_symbol

	d_symbol = None

def load_symbols():
	global d_symbol

	if d_symbol is None:
		with open('./data/symbol.csv', 'r') as fp:
			d_symbol = [l.strip('\n') for l in fp]

	return d_symbol

def load_stock_symbols():
	return [s for s in load_symbols() if not s.startswith('^')]
