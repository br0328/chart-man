
from dash import dcc, html, callback, Output, Input, State
from constant import *
from compute import *
from config import *
from yahoo import *
from plot import *
from util import *
from data import *
from ui import *
import plotly.graph_objects as go
import numpy as np
import dash

dash.register_page(__name__, path = '/', name = 'Dashboard', order = '00')

info, last_date = get_dashboard_info()
report = get_dashboard_content(info, last_date)

layout = get_page_layout('Dashboard', html.Div(), html.Div(),
	html.Div(
		children = [report],
		style = {
			'paddingTop': '100px',
			'text-align': 'center'
		},
		id = 'dash-info'
	)
)
