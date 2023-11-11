
ALERT_COLOR_SUCCESS = '#4CB22C'
ALERT_COLOR_WARNING = '#E98454'
ALERT_COLOR_ERROR = '#D35151'

INTERVAL_ALL = ['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']
INTERVAL_DAILY, INTERVAL_WEEKLY, INTERVAL_MONTHLY, INTERVAL_QUARTERLY, INTERVAL_YEARLY = tuple(INTERVAL_ALL)
INTERVAL_LETTER_DICT = dict(zip(INTERVAL_ALL, ['D', 'W', 'M', 'Q', 'Y']))

PIVOT_NUMBER_ALL = [
	'Recent One Pivot', 'Recent Two Pivots', 'Recent Three Pivots', 'Recent Four Pivots'
]
PIVOT_NUMBER_ONE, PIVOT_NUMBER_TWO, PIVOT_NUMBER_THREE, PIVOT_NUMBER_FOUR = tuple(PIVOT_NUMBER_ALL)

YMD_FORMAT = '%Y-%m-%d'

FIB_EXT_LEVELS = [1.618, 2.618, 4.236, 6.854, 11.09, 17.944, 29.034, 46.978, 76.012]

PLOT_COLORS_DARK = ['blue', 'orange', 'indigo', 'magenta', 'green', 'red', 'black']
