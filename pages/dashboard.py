
from dash import dcc, html, callback, Output, Input
from ui import *
import dash_bootstrap_components as dbc
import plotly.express as px
import dash

dash.register_page(__name__, path = '/', name = 'Dashboard', order = 0)

layout = get_page_title('Dahboard')
