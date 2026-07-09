# Описание репозитория
Репозиторий содержит ряд проектов, выполненных мной на Python.

## Fraud Detection
Нейросетевая система обнаружения мошеннических транзакций
ROC-AUC 0.9989 · Keras/TensorFlow · 6.36 млн транзакций · PaySim Dataset

Система в реальном времени определяет вероятность мошенничества в мобильных платежах.
Используется полносвязная нейронная сеть (Keras), обученная с учётом всех ограничений датасета PaySim:
Избегнута утечка данных через newbalanceOrig / newbalanceDest — вместо них созданы легитимные признаки (balance_change_*, amount_to_balance_ratio и др.).

# Dash

Дашборд-визуализация данных из стандартного набора датасетов библиотеки Plotly. 

Использованные датасеты: 
1. gapminderDataFiveYear.csv: URL: https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv 
2. salaries-ai-jobs-net.csv: URL: https://raw.githubusercontent.com/plotly/datasets/master/salaries-ai-jobs-net.csv 
3. Antibiotics.csv: URL: https://raw.githubusercontent.com/plotly/datasets/master/Antibiotics.csv

Построенные графики: - Bubble plot - Box plot - Heatmap
