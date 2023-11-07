
from dash import dcc, html, callback, Output, Input

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

def get_symbol_input():
	return html.Div(
		className = 'scenario_block',
		children = [
			dcc.Dropdown(id = 'symbol-input', placeholder = 'Select Symbol ...', options = ['ABC', 'DEF'], style = {'width': '170px'})
		])

def get_date_range():
	return html.Div(
		className = 'scenario_block',
		children = [
			dcc.DatePickerSingle(id = 'from-date-input', placeholder='From ...', display_format = 'YYYY-M-D', style = {'display': 'inline-block'}),
			html.Div(
				children = [
					html.Label('→', style = {'font-weight': 'bolder'})
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
			dcc.Dropdown(id = 'interval-input', placeholder = 'By ...', options = ['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly'], style = {'width': '120px'})
		])

def get_run_button():
	return html.Div(
		className = 'scenario_button',
		children = [
			html.Button(
                'Run',
                id = 'run-button',
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
				style = {'padding-left': '40px'}
			)
		])
