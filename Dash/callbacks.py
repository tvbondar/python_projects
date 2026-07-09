"""
callbacks.py
Основная логика работы приложения
"""

from dash import Input, Output
import plotly.express as px
import plotly.graph_objects as go

def register_callbacks(app, data):
    '''Регистрация всех callbacks '''

    # CALLBACK 1 - обновление графика   
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
        '''Обновляет график в зависимости от датасета'''
        
        # Gapminder
        if selected == 'gapminder':
            # filter by year, then by continent
            df = data['gapminder']
            df = df[df['year'] == gapminder_year]
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
    
        # Salaries 
        elif selected == 'salaries':
            #filter by country, then by employment and then by year
            df = data['salaries'].copy()
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
    
        # Antibiotics
        elif selected == 'antibiotics':
            # filter by gram then by antibiotics list
            df = data['antibiotics'].copy()
            if antibiotics_gram:
                df = df[df['Gram'] == antibiotics_gram]
            if  not antibiotics_list:
                return go.Figure()
            df_plot = df.set_index('Bacteria')[antibiotics_list]
            return px.imshow( 
                df_plot.T,
                labels=dict(x="Bacteria", y="Antibiotics", color="MIC"),aspect='auto', template="plotly_white")
        return go.Figure()


    # CALLBACK 2 - показать/скрыть управление
    @app.callback(
        Output("gapminder-controls", "style"),
        Output("salary-controls", "style"),
        Output("antibiotics-controls", "style"),
        Input("dataset-selector", "value"),
    )

    def hide_controls(selected):
        '''Показывает управление только к выбранному графику'''
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

