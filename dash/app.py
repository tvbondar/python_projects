import dash
from dash import html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

app = dash.Dash(__name__)


# ====DATA====
url1 = 'https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv'
url2= 'https://raw.githubusercontent.com/plotly/datasets/master/salaries-ai-jobs-net.csv'
url3 = 'https://raw.githubusercontent.com/plotly/datasets/master/Antibiotics.csv'

life = pd.read_csv(url1)
salaries = pd.read_csv(url2)
antibiotics = pd.read_csv(url3)
antibiotics.columns = antibiotics.columns.str.strip()

# ====LAYOUT====
app.layout = html.Div(
   children=[
        html.H1(children='Hello Dash'),  
        html.Div(
        '''Dash: A web application framework for your data.'''
        ), 

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

        dcc.Graph(id='main'),  

        # graph gapminder controls
        html.Div(
            id = "gapminder-controls",
                children = [

                    # continent choice
                    html.Label('Continent/region'),
                    dcc.Dropdown(
                        id = 'gapminder-continent',
                        options = ['Asia', 'Europe', 'Africa', 'Americas', 'Oceania'],
                        value = 'Asia',
                    ),

                    # year choice
                    html.Label('Years'),
                    dcc.Slider(
                        id = 'gapminder-year',
                        min=life['year'].min(),
                        max=life['year'].max(),
                        step = None,
                        value = life['year'].min(),
                        marks={str(year): str(year) for year in life['year'].unique()},
                    ),
                ],
        ),


        # graph salaries controls
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
                    options = ["FT", "PT", "CT", "FW"],
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

        # graph antibiotics controls
        html.Div(
            id = 'antibiotics-controls',
            children = [

                # gram status choice
                html.Label('Gram-status'),
                dcc.Dropdown(
                    id = 'antibiotics-gram',
                    options = ['positive','negative'],
                    value = 'positive'
                ),

                # antibiotics choice
                html.Label('Antibiotics'),
                dcc.Dropdown(
                    id = 'antibiotics-selected',
                    options=["Penicillin", "Streptomycin", "Neomycin"], 
                    value = ['Penicillin'],
                    multi=True
                ),
            ]
        ),

    ],
    style={"width": "60%", "margin": "auto"}
)


@app.callback(
    Output("main", "figure"),
    Input("dataset-selector", "value"),
    Input("gapminder-continent", "value"),
    Input("gapminder-year", "value"),
    Input("salary-country", "value"),
    Input("salary-employment", "value"),
    Input("salary-year", "value"),
    Input("antibiotics-gram", "value"),
    Input("antibiotics-selected", "value"),
)

def update_graph(selected, gapminder_continent, gapminder_year, salary_country, salary_employment, salary_year, antibiotics_gram, antibiotics_list):
    if selected == 'gapminder':
    # filter by year, then by continent
        df = life[life['year'] == gapminder_year]
        if gapminder_continent != 'All':
            df = df[df['continent'] == gapminder_continent]
        return px.scatter(
            df, 
            x="gdpPercap", 
            y="lifeExp",
            size="pop", 
            color="continent", 
            hover_name="country",
            log_x=True, 
            size_max=60,
            template="plotly_white")
    
    
    elif selected == 'salaries':
        #filter by country, then by employment and then by year
        df = salaries.copy()
        if salary_country:
            df = df[df['employee_residence'] == salary_country]
        if salary_employment:
            df = df[df['employment_type'] == salary_employment]
        if salary_year:
            df = df[df['work_year'] == salary_year]
        return px.box(
            df, 
            x="job_title", 
            y="salary_in_usd",
            log_y = True,
            template="plotly_white")
    
    elif selected == 'antibiotics':
        # filter by gram then by antibiotics list
        df = antibiotics.copy()
        if antibiotics_gram:
            df = df[df['Gram'] == antibiotics_gram]
        if  not antibiotics_list:
            return go.Figure()
        df_plot = df.set_index('Bacteria')[antibiotics_list]
        return px.imshow( 
            df_plot.T,
            labels=dict(x="Bacteria", y="Antibiotics", color="MIC"),aspect='auto', template="plotly_white")
    return go.Figure()

@app.callback(
    Output("gapminder-controls", "style"),
    Output("salary-controls", "style"),
    Output("antibiotics-controls", "style"),
    Input("dataset-selector", "value"),
)



def hide_controls(selected):
    hide = {'display': 'none'}
    show = {'display': 'block'}
    if selected == "gapminder":
        return show, hide, hide
    elif selected == "salaries":
        return hide, show, hide
    elif selected == "antibiotics":
        return hide, hide, show
    else:
        return hide, hide, hide






# ====RUN APP====
if __name__ == '__main__':
    app.run(debug=True)



        
    
        

        


        








