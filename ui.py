
from dash import dcc, html, callback, Output, Input, page_container
from data import *
import dash_bootstrap_components as dbc
import dash

def get_app_layout():
	return html.Div([
	    html.Div(
	        children = [
	            html.Img(src = 'assets/logo.png'),
	            get_side_bar()
	        ],
	        className = 'left_pane'
	    ),
	    html.Div(
	        children = [
	            page_container
	        ],
	        className = 'work_pane'
	    ),
	])

def get_side_bar():
	return dbc.Nav([
	        dbc.NavLink([
	                dash.html.Div(page["name"], className = "ms-2"),
	            ],
	            href = page["path"],
	            active = "exact",
	        )
	        for page in dash.page_registry.values()
	    ],
	    vertical = True,
	    pills = True,
	    style = {'width': '240px'}
	)

def get_page_title(title_text):
	return html.Div(
	    className = 'title_div',
	    children = [
	        html.H1(
	            children = seg
	        ) for seg in title_text.split('|')
	    ]
	)

def get_scenario_title():
	return html.H2('- SCENARIO')

def get_parameter_title():
	return html.H2('- PARAMETER')

def get_symbol_input():
	return html.Div(
		className = 'scenario_block',
		children = [
			dcc.Dropdown(id = 'symbol-input', placeholder = 'Select Symbol ...', options = load_stock_symbols(), style = {'width': '210px'})
		])

def get_date_range():
	return html.Div(
		className = 'scenario_block',
		children = [
			dcc.DatePickerSingle(id = 'from-date-input', placeholder='From ...', display_format = 'YYYY-M-D', style = {'display': 'inline-block'}),
			html.Div(
				children = [
					html.Label('→', style = {'fontWeight': 'bolder'})
				],
				style = {
					'display': 'inline-block', 'padding': '0 10px 0 10px'
				}),
			dcc.DatePickerSingle(id = 'to-date-input', placeholder='To ...', display_format = 'YYYY-M-D', style = {'display': 'inline-block'})
		])

def get_interval_input():
	return html.Div(
		className = 'scenario_block',
		children = [
			dcc.Dropdown(
				id = 'interval-input', placeholder = 'By ...', options = ['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly'], style = {'width': '120px'}
			)
		])

def get_load_button():
	return html.Div(
		className = 'scenario_button',
		children = [
			html.Button(
                'LOAD',
                id = 'load-button',
                n_clicks = 0
            ),
		])

def get_analyze_button():
	return html.Div(
		className = 'scenario_button',
		children = [
			html.Button(
                'ANALYZE',
                id = 'analyze-button',
                n_clicks = 0
            ),
		])

def get_backtest_button():
	return html.Div(
		className = 'scenario_button',
		children = [
			html.Button(
                'BACKTEST',
                id = 'backtest-button',
                n_clicks = 0
            ),
		])

def get_scenario_div(children):
	return html.Div(
		className = 'scenario_div',
		children = [
			get_scenario_title(),
			html.Div(
				children = children,
				style = {'paddingLeft': '40px'}
			)
		])

def get_parameter_div(children):
	return html.Div(
		className = 'parameter_div',
		children = [
			get_parameter_title(),
			html.Div(
				children = children,
				style = {'paddingLeft': '40px'}
			)
		])

def get_pivot_number_input():
	return html.Div(
        className = 'scenario_block',
        children = [
            dcc.Dropdown(
            	id = 'pivot-input',
            	placeholder = 'Number of Pivots ...',
            	options = [
            		'Recent One Pivot',
            		'Recent Two Pivots',
            		'Recent Three Pivots',
            		'Recent Four Pivots',
            		'Recent Five Pivots',
            	],
            	style = {'width': '210px'})
    	])

def get_merge_thres_input():
	return html.Div(
        className = 'scenario_block',
        children = [
            dcc.Input(
            	id = 'merge-input',
            	placeholder = 'Merge By ...',
            	type = 'text',
            	style = {'width': '110px'}
            ),
            html.Label('%', style = {'fontWeight': 'bolder', 'paddingLeft': '5px'})
    	])
