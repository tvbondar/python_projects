import pandas as pd
url3 = 'https://raw.githubusercontent.com/plotly/datasets/master/Antibiotics.csv'

antibiotics = pd.read_csv(url3)
print(antibiotics.columns.tolist())
antibiotics.columns = antibiotics.columns.str.strip()
print(antibiotics.columns.tolist())

print("=" * 60)
print("КОЛОНКИ В ANTIBIOTICS:")
print(antibiotics.columns.tolist())
print("=" * 60)
print("ПРОВЕРКА НАЛИЧИЯ 'Gram':")
print(f"'Gram' in columns: {'Gram' in antibiotics.columns}")
print("=" * 60)
print("ПЕРВЫЕ 5 СТРОК:")
print(antibiotics.head())
print("=" * 60)