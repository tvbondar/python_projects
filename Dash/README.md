# Описание проекта

Дашборд-визуализация данных из стандартного набора датасетов библиотеки Plotly. Использованные датасеты:
    1. gapminderDataFiveYear.csv: 
        URL: https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv
    2. salaries-ai-jobs-net.csv: 
        URL: https://raw.githubusercontent.com/plotly/datasets/master/salaries-ai-jobs-net.csv
    3. Antibiotics.csv: 
        URL: https://raw.githubusercontent.com/plotly/datasets/master/Antibiotics.csv

Построенные графики:
    - Bubble plot
    - Box plot
    - Heatmap


## Инструкции по установке

Для установки зависимостей выполни команду:

```bash
pip install -r requirements.txt
```

## Инструкции по запуску

Для запуска веб-сервиса выполни команду:

```bash
python app.py
```

После запуска приложение будет доступно по адресу: `http://127.0.0.1:8050`

# Итог

Дашборд аккуратно выводит визуализации данных из датасетов. Графики информативны и визуально понятны. Добавлена возможность интерактива с графиками через фильтрацию, которая влияет на отображаемые графики. 

## Изменения

- Проект переработан в модульную структуру: настройки, загрузка данных, интерфейс и callbacks вынесены в отдельные модули.
- Добавлены пользовательские стили CSS для улучшения внешнего вида приложения.

