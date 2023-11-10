
from datetime import datetime, timedelta
from constant import *

def get_today_str(format_str = YMD_FORMAT):
	return datetime.now().strftime(format_str)

def get_offset_date_str(base_date_str, days, format_str = YMD_FORMAT):
	base_date = datetime.strptime(base_date_str, format_str)
	return (base_date + timedelta(days = days)).strftime(format_str)

def get_duration(start_date_str, end_date_str, format_str = YMD_FORMAT):
	start_date = datetime.strptime(start_date_str, format_str)
	end_date = datetime.strptime(end_date_str, format_str)

	return (end_date - start_date).days

def get_timestamp(date_str, format_str = YMD_FORMAT):
	return datetime.strptime(date_str, format_str)

def alert_success(msg = '', none_count = 0):
	return [True, msg, {'backgroundColor': ALERT_COLOR_SUCCESS}] + [None for _ in range(none_count)]

def alert_warning(msg = '', none_count = 0):
	return [True, msg, {'backgroundColor': ALERT_COLOR_WARNING}] + [None for _ in range(none_count)]

def alert_error(msg = '', none_count = 0):
	return [True, msg, {'backgroundColor': ALERT_COLOR_ERROR}] + [None for _ in range(none_count)]

def alert_hide(none_count = 0):
	return [False, '', {}] + [None for _ in range(none_count)]
