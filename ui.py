
from dash import dcc, html, dash_table, callback, Output, Input, page_container
from constant import *
from config import *
from util import *
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
	        className = 'work_pane',
	        style = {'height': '100%'}
	    )
	])

def get_page_layout(title, scenario_div, parameter_div, out_tab):
	return html.Div(
	    children = [
	        get_page_title(title),
	        
	        html.Div(
	        	className = 'primary_pane',
	        	children = [
			        scenario_div,
			        parameter_div,
			        out_tab
	        	]
	        ),
		    dbc.Alert([html.H3("", id = "alert-msg")], id = "alert-dlg", is_open = False, fade = True, duration = 3000)
	    ],
	    style = {'height': '100%'}
	)

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
			dcc.Dropdown(id = 'symbol-input', placeholder = 'Select Symbol', options = load_stock_symbols(), style = {'width': '210px'})
		]
	)

def get_date_range():
	return html.Div(
		className = 'scenario_block',
		children = [
			dcc.DatePickerSingle(
				id = 'from-date-input',
				placeholder = 'From',
				display_format = 'YYYY-M-D',
				style = {'display': 'inline-block'}
				),
			html.Div(
				children = [
					html.Label('→', style = {'fontWeight': 'bolder'})
				],
				style = {
					'display': 'inline-block', 'padding': '0 10px 0 10px'
				}),
			dcc.DatePickerSingle(
				id = 'to-date-input',
				placeholder = 'To',
				display_format = 'YYYY-M-D',
				style = {'display': 'inline-block'},
				date = get_today_str()
				)
		]
	)

def get_cur_date_picker():
	return html.Div(
		className = 'scenario_block',
		children = [
			dcc.DatePickerSingle(
				id = 'cur-date-input',
				placeholder = 'When',
				display_format = 'YYYY-M-D',
				style = {'display': 'inline-block'}
				)
		]
	)

def get_interval_input():
	return html.Div(
		className = 'scenario_block',
		children = [
			dcc.Dropdown(
				id = 'interval-input', placeholder = 'By', options = INTERVAL_ALL, style = {'width': '120px'}
			)
		]
	)

def get_analyze_button(prefix):
	return html.Div(
		className = 'scenario_button',
		children = [
			html.Button(
                'ANALYZE',
                id = prefix + '-analyze-button',
                n_clicks = 0
            ),
		]
	)

def get_backtest_button(prefix):
	return html.Div(
		className = 'scenario_button',
		children = [
			html.Button(
                'BACKTEST',
                id = prefix + '-backtest-button',
                n_clicks = 0
            ),
		]
	)

def get_scenario_div(children):
	return html.Div(
		className = 'scenario_div',
		children = [
			get_scenario_title(),
			html.Div(
				children = children,
				style = {'paddingLeft': '40px'}
			)
		]
	)

def get_parameter_div(children):	
	return html.Div(
		className = 'parameter_div',
		children = [
			get_parameter_title(),
			html.Div(
				children = children,
				style = {'paddingLeft': '40px'}
			)
		]
	)

def get_out_tab(children):
	return html.Div(
		className = 'out_tab_parent',
		children = [
			dcc.Tabs(
				className = 'out_tab',
				id = 'out_tab',
				children = [dcc.Tab(label = label, value = label, children = [child]) for label, child in children.items()],
				value = list(children.keys())[0] if len(children) > 0 else ''
			)
		]
	)

def get_plot_div():
	return html.Div(id = 'out-plot')

def get_report_div():
	return html.Div(id = 'out-report')

def get_pivot_number_input():
	return html.Div(
        className = 'scenario_block',
        children = [
            dcc.Dropdown(
            	id = 'pivot-input',
            	placeholder = 'Number of Pivots',
            	options = PIVOT_NUMBER_ALL,
            	style = {'width': '210px'},
            	value = PIVOT_NUMBER_FOUR
            )
    	]
    )

def get_merge_thres_input():
	return html.Div(
        className = 'scenario_block',
        children = [
            dcc.Input(
            	id = 'merge-input',
            	placeholder = 'Merge By',
            	type = 'text',
            	style = {'width': '110px'},
            	value = str(default_fibo_ext_merge_thres)
            ),
            html.Label('%', style = {'fontWeight': 'bolder', 'paddingLeft': '5px'})
    	]
    )

def get_report_content(df, path):
	return html.Div(
		children = [
			html.Div(path, style = {'text-align': 'right', 'margin-top': '10px', 'margin-bottom': '10px', 'padding-right': '20px'}),
			dash_table.DataTable(
				df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],
				fixed_rows = dict(headers = True)
			)
		]
	)
