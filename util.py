
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

def alert_success(msg = '', none_ret = []):
	return [True, msg, {'backgroundColor': ALERT_COLOR_SUCCESS}] + none_ret

def alert_warning(msg = '', none_ret = 0):
	return [True, msg, {'backgroundColor': ALERT_COLOR_WARNING}] + none_ret

def alert_error(msg = '', none_ret = 0):
	return [True, msg, {'backgroundColor': ALERT_COLOR_ERROR}] + none_ret

def alert_hide(none_ret = 0):
	return [False, '', {}] + none_ret

def get_safe_num(v):
	if isinstance(v, int) or isinstance(v, float): return v

	try:
		return int(v)
	except ValueError:
		try:
			return float(v)
		except ValueError:
			return 0

def get_nearest_backward_date(df, cur_date):
	res = cur_date

	while res not in df.index:
		res = res - timedelta(days = 1)

		if res < df.index[0]: return None

	return res

def get_nearest_forward_date(df, cur_date):
	res = cur_date

	while res not in df.index:
		res = res + timedelta(days = 1)

		if res > df.index[-1]: return None

	return res
