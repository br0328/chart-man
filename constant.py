
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

FIB_BEHAVIOR_MILESTONE = 60

FIB_EXT_MARKERS = {
	'Res_Break': ('triangle-up', 'black', 0, 'middle'),
	'Res_Semi_Break': ('triangle-up', 'white', 0, 'middle'),
	'Res_Res': ('arrow-bar-left', 'black', 90, 'middle'),
	'Res_Semi_Res': ('arrow-bar-left', 'white', 90, 'middle'),
	'Sup_Break': ('triangle-down', 'black', 0, 'middle'),
	'Sup_Semi_Break': ('triangle-down', 'white', 0, 'middle'),
	'Sup_Sup': ('arrow-bar-right', 'black', 90, 'middle'),
	'Sup_Semi_Sup': ('arrow-bar-right', 'white', 90, 'middle'),
	'Vibration': ('diamond', 'white', 0, 'middle')
}

PLOT_COLORS_DARK = ['blue', 'orange', 'indigo', 'magenta', 'green', 'red', 'black']
