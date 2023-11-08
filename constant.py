
ALERT_COLOR_SUCCESS = '#4CB22C'
ALERT_COLOR_WARNING = '#E98454'
ALERT_COLOR_ERROR = '#C01D1D'

INTERVAL_ALL = ['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']
INTERVAL_DAILY, INTERVAL_WEEKLY, INTERVAL_MONTHLY, INTERVAL_QUARTERLY, INTERVAL_YEARLY = tuple(INTERVAL_ALL)

def alert_success(msg = ''):
	return True, msg, {'backgroundColor': ALERT_COLOR_SUCCESS}

def alert_warning(msg = ''):
	return True, msg, {'backgroundColor': ALERT_COLOR_WARNING}

def alert_error(msg = ''):
	return True, msg, {'backgroundColor': ALERT_COLOR_ERROR}

def alert_hide():
	return False, '', {}
