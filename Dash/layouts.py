"""
layouts.py
Собирает интерфейс приложения
"""

from dash import html, dcc
from config import CONTINENTS,EMPLOYMENT_TYPES, GRAM_STATUSES, ANTIBIOTICS

def create_layout(datasets):
    '''Основной layout приложения'''

    gapminder = datasets["gapminder"]
    salaries = datasets["salaries"]

    return html.Div(
        children=[
            # Header
            html.H1(children='Hello Dash'),  
            html.Div('Dash: A web application framework for your data.'), 

            # Graph selector
            dcc.Dropdown(
                id = "dataset-selector",
                options=[
                {'label': 'Gapminder Data Five Year', 'value': 'gapminder'},
                {'label': 'AI Job Salaries', 'value': 'salaries'},
                {'label': 'Antibiotics Sensitivity', 'value': 'antibiotics'}
                ],
                value="gapminder",
                clearable = False,
            ),

            # Main graph
            dcc.Graph(id='main'),  

            # === GAPMINDER CONTROLS ===
            html.Div(
                id = "gapminder-controls",
                    children = [

                        # continent choice
                        html.Label('Continent/region'),
                        dcc.Dropdown(
                            id = 'gapminder-continent',
                            options = CONTINENTS,
                            value = 'Asia',
                        ),

                        # year choice
                        html.Label('Years'),
                        dcc.Slider(
                            id = 'gapminder-year',
                            min=gapminder['year'].min(),
                            max=gapminder['year'].max(),
                            step = None,
                            value = gapminder['year'].min(),
                            marks={str(year): str(year) for year in gapminder['year'].unique()},
                        ),
                    ],
            ),


            # === SALARIES CONTROLS ===
            html.Div(
                id = 'salary-controls',
                children=[

                    # continent choice
                    html.Label('Continent/region'),
                    dcc.Dropdown(
                        id = 'salary-country',
                        options = [{'label': s, 'value': s} for s in salaries['employee_residence'].unique()],
                        value = salaries['employee_residence'].iloc[0],
                    ),

                    # employment type choice
                    html.Label('Employment type'),
                    dcc.Dropdown(
                        id = 'salary-employment',
                        options = EMPLOYMENT_TYPES,
                        value = 'FT',
                    ),

                    # year choice
                    html.Label('Years'),
                    dcc.Slider(
                        id = 'salary-year',
                        min=salaries["work_year"].min(),
                        max=salaries["work_year"].max(),
                        step = 1,
                        value = salaries["work_year"].min(),
                        marks={i: str(i) for i in range(int(salaries["work_year"].min()), int(salaries["work_year"].max())+1)},
                    ),
                ],   
        ),

        # === ANTIBIOTICS CONTROLS ===
        html.Div(
            id = 'antibiotics-controls',
            children = [

                # gram status choice
                html.Label('Gram-status'),
                dcc.Dropdown(
                    id = 'antibiotics-gram',
                    options = GRAM_STATUSES,
                    value = 'positive'
                ),

                # antibiotics choice
                html.Label('Antibiotics'),
                dcc.Dropdown(
                    id = 'antibiotics-selected',
                    options = ANTIBIOTICS, 
                    value = ['Penicillin'],
                    multi=True
                ),
            ]
        ),

    ],
    style={"width": "60%", "margin": "auto"}
    )