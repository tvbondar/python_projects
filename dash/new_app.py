"""
app.py
Основная точка входа в программу
"""

import dash
from config import APP_CONFIG
from load_data import load_data
from layouts import create_layout
from callbacks import register_callbacks

# Создание приложения Dash
app = dash.Dash(__name__)

# Загрузка данных
data = load_data()

# Создание интерфейса
app.layout = create_layout(data)

# Регистрация логики
register_callbacks(app, data)

# Запуск приложения
if __name__ == '__main__':
    app.run(debug=APP_CONFIG['debug'])