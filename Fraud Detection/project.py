# Импорт библиотек
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
import joblib
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fraud_detection.log'),
        logging.StreamHandler()
    ]
)

# Создание класса для системы FraudDetection
class FraudDetectionSystem:
    # Инициализация класса
    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.columns = None

    # Загрузка и валидация данных
    def load_data(self, data_path):
        try:
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Файл данных не найден: {data_path}")
            df = pd.read_csv(data_path)
            logging.info(f"Данные успешно загружены. Размер: {df.shape}")
            required_columns = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                                'oldbalanceDest', 'newbalanceDest', 'isFraud']
            if not all(col in df.columns for col in required_columns):
                raise ValueError("Отсутствуют обязательные колонки в данных")
            return df
        except Exception as e:
            logging.error(f"Ошибка загрузки данных: {str(e)}")
            raise

    # Предобработка данных и feature engineering
    def preprocess_data(self, df):
        try:
            df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, errors='ignore')
            type_map = {'CASH_OUT': 1, 'PAYMENT': 2, 'CASH_IN': 3, 'TRANSFER': 4, 'DEBIT': 5}
            df['type'] = df['type'].map(type_map).fillna(0)
            if df['type'].eq(0).any():
                logging.warning("Обнаружены неизвестные типы транзакций, присвоено значение 0.")
            df['balance_change_org'] = df['oldbalanceOrg'] - df['newbalanceOrig']
            df['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
            df['is_amount_high'] = (df['amount'] > df['amount'].quantile(0.99)).astype(int)
            df['hour'] = df['step'] % 24
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
            df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1e-6)
            self.columns = df.drop('isFraud', axis=1).columns.tolist()
            return df
        except Exception as e:
            logging.error(f"Ошибка предобработки: {str(e)}")
            raise

    # Обучение и сохранение модели
    # Supervised learning - данные размечены
    def train_model(self, df, save_path='model.joblib'):
        try:
            X = df.drop('isFraud', axis=1)
            y = df['isFraud']
            # Борьба с дисбалансом
            undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
            X_balanced, y_balanced = undersample.fit_resample(X, y)
            # Разделение данных
            X_train, X_test, y_train, y_test = train_test_split(
                X_balanced, y_balanced, test_size=0.3, random_state=42)
            # Обучение модели
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_leaf=5,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            # Оценка модели
            self._evaluate_model(X_test, y_test)
            # Важность признаков
            self._calculate_feature_importance(X_train)
            # Сохранение модели
            self._save_model(save_path)
        except Exception as e:
            logging.error(f"Ошибка обучения модели: {str(e)}")
            raise

    # Оценка качества модели
    def _evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        logging.info("\nОтчет о классификации:")
        logging.info(classification_report(y_test, y_pred))
        logging.info("\nМатрица ошибок:")
        logging.info(confusion_matrix(y_test, y_pred))
        logging.info(f"\nROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # Расчет важности признаков
    def _calculate_feature_importance(self, X_train):
        self.feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)

        # Словарь для перевода названий признаков
        feature_names_ru = {
            'type': 'Тип операции',
            'amount': 'Сумма',
            'oldbalanceOrg': 'Исходный баланс (отправитель)',
            'newbalanceOrig': 'Новый баланс (отправитель)',
            'oldbalanceDest': 'Исходный баланс (получатель)',
            'newbalanceDest': 'Новый баланс (получатель)',
            'balance_change_org': 'Изменение баланса (отправитель)',
            'balance_change_dest': 'Изменение баланса (получатель)',
            'is_amount_high': 'Высокая сумма (>99%-перцентиль)',
            'hour': 'Час дня',
            'step': 'Шаг(время)',
            'is_night': 'Ночное время (22:00–6:00)',
            'amount_to_balance_ratio': 'Отношение суммы к балансу'
        }
        self.feature_importance['Feature'] = self.feature_importance['Feature'].map(
            lambda x: feature_names_ru.get(x, x)
        )

    # Сохранение модели и метаданных
    def _save_model(self, path):
        model_meta = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'columns': self.columns,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        joblib.dump(model_meta, path)
        logging.info(f"Модель сохранена в {path}")

    #  Загрузка модели из файла
    def load_model(self, path='model.joblib'):
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Файл модели не найден: {path}")
            model_meta = joblib.load(path)
            self.model = model_meta['model']
            self.feature_importance = model_meta['feature_importance']
            self.columns = model_meta['columns']
            logging.info(f"Модель загружена из {path}. Время сохранения: {model_meta['timestamp']}")
        except Exception as e:
            logging.error(f"Ошибка загрузки модели: {str(e)}")
            raise

    # Предсказание для новой транзакции
    def predict_transaction(self, transaction_data):
        try:
            # Проверка и подготовка данных
            if not isinstance(transaction_data, pd.DataFrame):
                transaction_df = pd.DataFrame([transaction_data])
            # Проверка колонок
            missing_cols = set(self.columns) - set(transaction_df.columns)
            if missing_cols:
                raise ValueError(f"Отсутствуют колонки: {missing_cols}")
            # Предсказание
            proba = self.model.predict_proba(transaction_df[self.columns])[0, 1]
            return proba
        except Exception as e:
            logging.error(f"Ошибка предсказания: {str(e)}")
            raise

    # Визуализация данных и сохранение графиков
    def visualize_data(self, df, output_dir='visualizations'):
        try:
            os.makedirs(output_dir, exist_ok=True)

            # 1. Распределение целевой переменной (isFraud)
            plt.figure(figsize=(8, 6))
            ax = sns.countplot(x=df['isFraud'])
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Легитимные (0)', 'Мошеннические (1)'])
            plt.title('Распределение мошеннических транзакций')
            plt.xlabel('Тип транзакции')
            plt.ylabel('Количество')
            plt.savefig(os.path.join(output_dir, 'fraud_distribution.png'))
            plt.close()

            # 2. Типы транзакций
            plt.figure(figsize=(12, 8))
            ax = sns.countplot(x=df['type'])
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_xticklabels([ 'Вывод наличных', 'Платеж', 'Пополнение наличными', 'Перевод', 'Дебет' ])
            plt.title('Распределение типов транзакций')
            plt.xlabel('Тип операции')
            plt.ylabel('Количество')
            plt.savefig(os.path.join(output_dir, 'transaction_types.png'))
            plt.close()

            if self.feature_importance is not None:
            # 3. Важность признаков
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Importance', y='Feature', data=self.feature_importance.head(15))
                plt.title('Топ-15 важных признаков')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
                plt.close()

            logging.info(f"Визуализации сохранены в {output_dir}: fraud_distribution.png, transaction_types.png, feature_importance.png")
        except Exception as e:
            logging.error(f"Ошибка при визуализации данных: {str(e)}")
            raise

    # Тестовые сценарии
    def run_tests(self):
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите train_model.")
        test_cases = [
            {
                'description': "Обычная дневная транзакция",
                'step': 50, 'type': 2, 'amount': 1000,
                'oldbalanceOrg': 50000, 'newbalanceOrig': 49000,
                'oldbalanceDest': 1000, 'newbalanceDest': 2000,
                'balance_change_org': 1000, 'balance_change_dest': 1000,
                'is_amount_high': 0, 'hour': 12, 'is_night': 0,
                'amount_to_balance_ratio': 0.02
            },
            {
                'description': "Подозрительная ночная транзакция",
                'step': 120, 'type': 4, 'amount': 900000,
                'oldbalanceOrg': 1000000, 'newbalanceOrig': 100000,
                'oldbalanceDest': 0, 'newbalanceDest': 900000,
                'balance_change_org': 900000, 'balance_change_dest': 900000,
                'is_amount_high': 1, 'hour': 2, 'is_night': 1,
                'amount_to_balance_ratio': 0.9
            }
        ]

        for case in test_cases:
            try:
                proba = self.predict_transaction(case)
                risk_level = "Высокий" if proba > 0.7 else " Средний" if proba > 0.4 else " Низкий"

                logging.info(f"\n{case['description']}:")
                logging.info(f"Вероятность мошенничества: {proba:.2%}")
                logging.info(f"Уровень риска: {risk_level}")

            except Exception as e:
                logging.error(f"Ошибка теста: {str(e)}")


if __name__ == "__main__":
    try:
        # Инициализация системы
        fraud_system = FraudDetectionSystem()

        save_path = 'model.joblib'

        # Загрузка данных
        df = fraud_system.load_data("PS_20174392719_1491204439457_log.csv")

        # Предобработка
        processed_df = fraud_system.preprocess_data(df)

        # Обучение модели
        if os.path.exists(save_path):
            fraud_system.load_model(save_path)
        else:
            fraud_system.train_model(processed_df, save_path)

        # Визуализация данных
        fraud_system.visualize_data(processed_df)

        # Тестовые сценарии
        fraud_system.run_tests()

    except Exception as e:
        logging.error(f"Критическая ошибка: {str(e)}")


