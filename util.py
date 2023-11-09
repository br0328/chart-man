
from datetime import datetime, timedelta
from constant import *

def get_today_str(format_str = '%Y-%m-%d'):
	return datetime.now().strftime(format_str)

def get_offset_date_str(base_date_str, days, format_str = YMD_FORMAT):
	base_date = datetime.strptime(base_date_str, format_str)
	return (base_date + timedelta(days = days)).strftime(format_str)

def get_interval_letter(interval_key):
	if interval_key == '3mo':
		return 'Q'
	else:
		return interval_key[1].upper()

def alert_success(msg = ''):
	return True, msg, {'backgroundColor': ALERT_COLOR_SUCCESS}

def alert_warning(msg = ''):
	return True, msg, {'backgroundColor': ALERT_COLOR_WARNING}

def alert_error(msg = ''):
	return True, msg, {'backgroundColor': ALERT_COLOR_ERROR}

def alert_hide():
	return False, '', {}
