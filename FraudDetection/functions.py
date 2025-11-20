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
import config


class FraudDetectionSystem:
    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.columns = None
        self.high_amount_threshold = None  # Храним порог из обучения

    def load_data(self, data_path=None):
        data_path = data_path or config.DATA_PATH
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

    def preprocess_data(self, df):
        try:
            df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, errors='ignore')
            type_map = {'CASH_OUT': 1, 'PAYMENT': 2, 'CASH_IN': 3, 'TRANSFER': 4, 'DEBIT': 5}
            df['type'] = df['type'].map(type_map).fillna(0)
            if df['type'].eq(0).any():
                logging.warning("Обнаружены неизвестные типы транзакций, присвоено значение 0.")

            # Сохраняем порог для одной транзакции
            self.high_amount_threshold = df['amount'].quantile(config.HIGH_AMOUNT_PERCENTILE)

            df['balance_change_org'] = df['oldbalanceOrg'] - df['newbalanceOrig']
            df['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
            df['is_amount_high'] = (df['amount'] > self.high_amount_threshold).astype(int)
            df['hour'] = df['step'] % 24
            df['is_night'] = ((df['hour'] >= config.NIGHT_HOURS_START) | (df['hour'] <= config.NIGHT_HOURS_END)).astype(int)
            df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1e-6)

            self.columns = df.drop('isFraud', axis=1).columns.tolist()
            return df
        except Exception as e:
            logging.error(f"Ошибка предобработки: {str(e)}")
            raise

    def preprocess_single(self, transaction_dict):
        if self.high_amount_threshold is None:
            raise ValueError("Сначала вызовите preprocess_data() для обучения порога is_amount_high")

        df = pd.DataFrame([transaction_dict])
        df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, errors='ignore')

        type_map = {'CASH_OUT': 1, 'PAYMENT': 2, 'CASH_IN': 3, 'TRANSFER': 4, 'DEBIT': 5}
        df['type'] = df['type'].map(type_map).fillna(0)

        df['balance_change_org'] = df['oldbalanceOrg'] - df['newbalanceOrig']
        df['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
        df['is_amount_high'] = (df['amount'] > self.high_amount_threshold).astype(int)
        df['hour'] = df['step'] % 24
        df['is_night'] = ((df['hour'] >= config.NIGHT_HOURS_START) | (df['hour'] <= config.NIGHT_HOURS_END)).astype(int)
        df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1e-6)

        return df[self.columns]

    def train_model(self, df, save_path=None):
        save_path = save_path or config.MODEL_PATH
        try:
            X = df.drop('isFraud', axis=1)
            y = df['isFraud']
            undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
            X_balanced, y_balanced = undersample.fit_resample(X, y)
            X_train, X_test, y_train, y_test = train_test_split(
                X_balanced, y_balanced, test_size=0.3, random_state=42)

            self.model = RandomForestClassifier(**config.MODEL_PARAMS)
            self.model.fit(X_train, y_train)

            self._evaluate_model(X_test, y_test)
            self._calculate_feature_importance(X_train)
            self._save_model(save_path)
        except Exception as e:
            logging.error(f"Ошибка обучения модели: {str(e)}")
            raise

    def _evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        logging.info("\nОтчет о классификации:")
        logging.info(classification_report(y_test, y_pred))
        logging.info("\nМатрица ошибок:")
        logging.info(confusion_matrix(y_test, y_pred))
        logging.info(f"\nROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    def _calculate_feature_importance(self, X_train):
        self.feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)

        feature_names_ru = {
            'type': 'Тип операции', 'amount': 'Сумма',
            'oldbalanceOrg': 'Исходный баланс (отправитель)',
            'newbalanceOrig': 'Новый баланс (отправитель)',
            'oldbalanceDest': 'Исходный баланс (получатель)',
            'newbalanceDest': 'Новый баланс (получатель)',
            'balance_change_org': 'Изменение баланса (отправитель)',
            'balance_change_dest': 'Изменение баланса (получатель)',
            'is_amount_high': 'Высокая сумма (>99%-перцентиль)',
            'hour': 'Час дня', 'step': 'Шаг(время)',
            'is_night': 'Ночное время (22:00–6:00)',
            'amount_to_balance_ratio': 'Отношение суммы к балансу'
        }
        self.feature_importance['Feature'] = self.feature_importance['Feature'].map(
            lambda x: feature_names_ru.get(x, x)
        )

    def _save_model(self, path):
        model_meta = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'columns': self.columns,
            'high_amount_threshold': self.high_amount_threshold,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model_meta, path)
        logging.info(f"Модель сохранена в {path}")

    def load_model(self, path=None):
        path = path or config.MODEL_PATH
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Файл модели не найден: {path}")
            model_meta = joblib.load(path)
            self.model = model_meta['model']
            self.feature_importance = model_meta.get('feature_importance')
            self.columns = model_meta['columns']
            self.high_amount_threshold = model_meta.get('high_amount_threshold')
            logging.info(f"Модель загружена из {path}. Время: {model_meta['timestamp']}")
        except Exception as e:
            logging.error(f"Ошибка загрузки модели: {str(e)}")
            raise

    def predict_transaction(self, transaction_data):
        try:
            if not isinstance(transaction_data, pd.DataFrame):
                transaction_df = pd.DataFrame([transaction_data])
            missing_cols = set(self.columns) - set(transaction_df.columns)
            if missing_cols:
                raise ValueError(f"Отсутствуют колонки: {missing_cols}")
            proba = self.model.predict_proba(transaction_df[self.columns])[0, 1]
            return proba
        except Exception as e:
            logging.error(f"Ошибка предсказания: {str(e)}")
            raise

    def visualize_data(self, df, output_dir=None):
        output_dir = output_dir or config.VISUALIZATIONS_DIR
        try:
            os.makedirs(output_dir, exist_ok=True)

            # 1. Распределение isFraud
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
            ax.set_xticklabels(['Вывод наличных', 'Платеж', 'Пополнение наличными', 'Перевод', 'Дебет'])
            plt.title('Распределение типов транзакций')
            plt.xlabel('Тип операции')
            plt.ylabel('Количество')
            plt.savefig(os.path.join(output_dir, 'transaction_types.png'))
            plt.close()

            # 3. Важность признаков
            if self.feature_importance is not None and not self.feature_importance.empty:
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Importance', y='Feature', data=self.feature_importance.head(15))
                plt.title('Топ-15 важных признаков')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
                plt.close()

            logging.info(f"Визуализации сохранены в {output_dir}")
        except Exception as e:
            logging.error(f"Ошибка при визуализации данных: {str(e)}")
            raise

    def run_tests(self):
        if self.model is None:
            raise ValueError("Модель не обучена.")

        test_cases = [
            {
                'description': "Обычная дневная транзакция",
                'step': 50, 'type': 'PAYMENT', 'amount': 1000,
                'oldbalanceOrg': 50000, 'newbalanceOrig': 49000,
                'oldbalanceDest': 1000, 'newbalanceDest': 2000,
            },
            {
                'description': "Подозрительная ночная транзакция",
                'step': 120, 'type': 'TRANSFER', 'amount': 900000,
                'oldbalanceOrg': 1000000, 'newbalanceOrig': 100000,
                'oldbalanceDest': 0, 'newbalanceDest': 900000,
            }
        ]

        low = config.RISK_THRESHOLDS["low"]
        med = config.RISK_THRESHOLDS["medium"]

        for case in test_cases:
            try:
                input_df = self.preprocess_single(case)
                proba = self.model.predict_proba(input_df)[0, 1]
                risk_level = "Высокий" if proba > med else "Средний" if proba > low else "Низкий"
                logging.info(f"\n{case['description']}:")
                logging.info(f"Вероятность мошенничества: {proba:.2%}")
                logging.info(f"Уровень риска: {risk_level}")
            except Exception as e:
                logging.error(f"Ошибка теста: {str(e)}")