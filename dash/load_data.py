"""
load_data.py
Загружает данные. Возвращает готовые к использованию DataFrame
"""

import pandas as pd
from config import DATASET_URLS


def load_data():
    '''Загружает все датасеты, возвращает словарь'''
    # Загрузка датасетов
    gapminder = pd.read_csv(DATASET_URLS['gapminder'])
    salaries = pd.read_csv(DATASET_URLS['salaries'])
    antibiotics = pd.read_csv(DATASET_URLS['antibiotics'])
    
    # Очистка данных (для antibiotics - необходимо)
    antibiotics.columns = antibiotics.columns.str.strip()







    datasets = {
        'gapminder': gapminder,
        'salaries': salaries,
        'antibiotics': antibiotics
    }
    
    return datasets


