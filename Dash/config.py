"""
config.py
Настройки приложения: URL, константы, параметры
"""

import pandas as pd

# URL адреса
DATASET_URLS = {
    'gapminder': 'https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv',
    'salaries': 'https://raw.githubusercontent.com/plotly/datasets/master/salaries-ai-jobs-net.csv',
    'antibiotics': 'https://raw.githubusercontent.com/plotly/datasets/master/Antibiotics.csv'
}

# Константы для dropdown и slider
CONTINENTS =  ['Asia', 'Europe', 'Africa', 'Americas', 'Oceania']
EMPLOYMENT_TYPES = ["FT", "PT", "CT", "FW"]
GRAM_STATUSES = ['positive','negative']
ANTIBIOTICS = ["Penicillin", "Streptomycin", "Neomycin"]



# конфигурация для запуска
APP_CONFIG = {
    'debug': True
}


