import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
import joblib
import logging
from datetime import datetime
import config

# Основной класс системы обнаружения мошеннических транзакций.
class FraudDetectionSystem:
    def __init__(self):
        self.model = None # Обученная нейронная сеть
        self.feature_importance = None # Важность признаков
        self.columns = None # Список признаков после предобработки
        self.high_amount_threshold = None  # Порог для признака "большая сумма"
        self.scaler = None   # StandardScaler для нормализации данных

    # Загружает данные из CSV-файла.
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

    # Предобработка полного датасета
    def preprocess_data(self, df):
        try:
            df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, errors='ignore')
            # Кодирование типа транзакции по словарю из config
            df['type'] = df['type'].map(config.TRANSACTION_TYPE_MAP).fillna(config.UNKNOWN_TYPE_VALUE)
            if df['type'].eq(0).any():
                logging.warning("Обнаружены неизвестные типы транзакций, присвоено значение 0.")

            # Определяем порог "большой суммы"
            self.high_amount_threshold = df['amount'].quantile(config.HIGH_AMOUNT_PERCENTILE)
            # Новые признаки
            df['balance_change_org'] = df['oldbalanceOrg'] - df['newbalanceOrig']
            df['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
            df['is_amount_high'] = (df['amount'] > self.high_amount_threshold).astype(int)
            df['hour'] = df['step'] % 24
            df['is_night'] = ((df['hour'] >= config.NIGHT_HOURS_START) | (df['hour'] <= config.NIGHT_HOURS_END)).astype(int)
            df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1e-6)
            # Сохраняем порядок и названия признаков
            self.columns = df.drop('isFraud', axis=1).columns.tolist()
            return df
        except Exception as e:
            logging.error(f"Ошибка предобработки: {str(e)}")
            raise

    # Предобработка одной транзакции
    def preprocess_single(self, transaction_dict):
        if self.high_amount_threshold is None or not hasattr(self, 'scaler'):
            raise ValueError("Модель не обучена или scaler не загружен")

        df = pd.DataFrame([transaction_dict])
        df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, errors='ignore')
        df['type'] = df['type'].map(config.TRANSACTION_TYPE_MAP).fillna(config.UNKNOWN_TYPE_VALUE)

        df['balance_change_org'] = df['oldbalanceOrg'] - df['newbalanceOrig']
        df['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
        df['is_amount_high'] = (df['amount'] > self.high_amount_threshold).astype(int)
        df['hour'] = df['step'] % 24
        df['is_night'] = ((df['hour'] >= config.NIGHT_HOURS_START) | (df['hour'] <= config.NIGHT_HOURS_END)).astype(int)
        df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1e-6)

        df = df[self.columns].copy()
        df_scaled = self.scaler.transform(df.values)
        return df_scaled

    # Обучение нейронной сети
    def train_model(self, df, save_path=None):
        save_path = save_path or config.MODEL_PATH
        try:
            X = df.drop('isFraud', axis=1).values.astype('float32')
            y = df['isFraud'].values.astype('float32')

            # Балансировка: уменьшаем количество легитимных транзакций
            undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
            X_balanced, y_balanced = undersample.fit_resample(X, y)

            X_train, X_test, y_train, y_test = train_test_split(
                X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
            )

            # Масштабирование — обязательно для нейросетей!
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            self.scaler = scaler # Сохраняем для предсказаний

            input_dim = X_train.shape[1]

            # Архитектура нейросети (из config)
            model = Sequential([
                Dense(config.MODEL_PARAMS["hidden_layers"][0], activation='relu', input_shape=(input_dim,)),
                BatchNormalization(),
                Dropout(config.MODEL_PARAMS["dropout"]),

                Dense(config.MODEL_PARAMS["hidden_layers"][1], activation='relu'),
                BatchNormalization(),
                Dropout(config.MODEL_PARAMS["dropout"]),

                Dense(config.MODEL_PARAMS["hidden_layers"][2], activation='relu'),
                Dropout(config.MODEL_PARAMS["dropout"] // 2),

                Dense(1, activation='sigmoid')  # выход — вероятность мошенничества
            ])

            model.compile(
                optimizer=Adam(learning_rate=config.MODEL_PARAMS["learning_rate"]),
                loss='binary_crossentropy',
                metrics=['accuracy', 'auc']
            )

            early_stopping = EarlyStopping(
                monitor='val_auc', mode='max', patience=10, restore_best_weights=True
            )

            logging.info(f"Начинаем обучение нейросети... признаков: {input_dim}")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=config.MODEL_PARAMS["epochs"],
                batch_size=config.MODEL_PARAMS["batch_size"],
                callbacks=[early_stopping],
                verbose=1
            )

            self.model = model
            self.history = history

            # Оценка качества
            y_pred_proba = model.predict(X_test, verbose=0).ravel()
            y_pred = (y_pred_proba > 0.5).astype(int)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            logging.info(f"ROC-AUC на тесте: {roc_auc_score(y_test, y_pred_proba):.4f}")

            self._save_model(save_path)
            logging.info("Нейросеть успешно обучена и сохранена!")
            print("Обучение завершено. Результаты:")
            print(f"ROC-AUC на тестовой выборке:     {auc_score:.4f}")
            print(f"Accuracy:                        {accuracy_score(y_test, y_pred):.4f}")
            print(f"Precision (мошенничество):       {precision_score(y_test, y_pred):.4f}")
            print(f"Recall (мошенничество):          {recall_score(y_test, y_pred):.4f}")
            print(f"F1-score:                        {f1_score(y_test, y_pred):.4f}")
            print(f"Всего тестовых примеров:         {len(y_test)}")
            print(f"Мошеннических в тесте:           {int(y_test.sum())} ({y_test.mean():.2%})")
            print("=" * 60)
            print("Модель сохранена → models/model.joblib")
            print("Готово к использованию!\n")
        except Exception as e:
            logging.error(f"Ошибка обучения нейросети: {str(e)}")
            raise

    # Сохраняет модель, scaler, метаданные в один joblib-файл
    def _save_model(self, path):
        model_meta = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance if hasattr(self, 'feature_importance') else None,
            'columns': self.columns,
            'high_amount_threshold': self.high_amount_threshold,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model_meta, path)
        logging.info(f"Нейросеть сохранена в {path}")

    # Загружает сохранённую модель и все метаданные
    def load_model(self, path=None):
        path = path or config.MODEL_PATH
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Файл модели не найден: {path}")
            model_meta = joblib.load(path)
            self.model = model_meta['model']
            self.scaler = model_meta['scaler']
            self.feature_importance = model_meta.get('feature_importance')
            self.columns = model_meta['columns']
            self.high_amount_threshold = model_meta.get('high_amount_threshold')
            logging.info(f"Нейросеть загружена из {path}. Время: {model_meta['timestamp']}")
        except Exception as e:
            logging.error(f"Ошибка загрузки нейросети: {str(e)}")
            raise

    # Предсказывает вероятность мошенничества для одной транзакции.
    def predict_transaction(self, transaction_data):
        try:
            if isinstance(transaction_data, dict):
                X = self.preprocess_single(transaction_data)
            else:
                X = transaction_data
            proba = self.model.predict(X, verbose=0)[0][0]
            return float(proba)
        except Exception as e:
            logging.error(f"Ошибка предсказания: {str(e)}")
            raise

    # Создаёт и сохраняет три графика
    def visualize_data(self, df, output_dir=None):
        output_dir = output_dir or config.VISUALIZATIONS_DIR
        try:
            os.makedirs(output_dir, exist_ok=True)

            # 1. Распределение isFraud
            plt.figure(figsize=(8, 6))
            ax = sns.countplot(x=df['isFraud'])
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Легитимные (0)', 'Мошеннические (1)'])
            plt.title(config.VISUALIZATION_TITLES['fraud_distribution'])
            plt.xlabel('Тип транзакции')
            plt.ylabel('Количество')
            plt.savefig(os.path.join(output_dir, 'fraud_distribution.png'))
            plt.close()

            # 2. Типы транзакций
            plt.figure(figsize=(12, 8))
            ax = sns.countplot(data=df, x='type', order=[1, 2, 3, 4, 5])
            ax.set_xticks([0, 1, 2, 3, 4])
            ax.set_xticklabels(config.TYPE_LABELS)
            plt.title(config.VISUALIZATION_TITLES['transaction_types'])
            plt.xlabel('Тип операции')
            plt.ylabel('Количество')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'transaction_types.png'))
            plt.close()

            # 3. Важность признаков
            if self.feature_importance is not None and not self.feature_importance.empty:
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Importance', y='Feature', data=self.feature_importance.head(15))
                plt.title(config.VISUALIZATION_TITLES['feature_importance'])
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
                plt.close()

            logging.info(f"Визуализации сохранены в {output_dir}")
        except Exception as e:
            logging.error(f"Ошибка при визуализации данных: {str(e)}")
            raise

    # Запускает тестовые сценарии из config.py
    def run_tests(self):
        if self.model is None:
            raise ValueError("Модель не обучена.")

        low = config.RISK_THRESHOLDS["low"]
        med = config.RISK_THRESHOLDS["medium"]

        print("\n" + "=" * 60)
        print("ТЕСТОВЫЕ ПРЕДСКАЗАНИЯ")
        print("=" * 60)

        for case in config.TEST_CASES:
            try:
                X_scaled = self.preprocess_single(case)
                proba = float(self.model.predict(X_scaled, verbose=0)[0][0])

                risk_level = "ВЫСОКИЙ" if proba > med else "СРЕДНИЙ" if proba > low else "НИЗКИЙ"

                logging.info(f"\n{case['description']}:")
                logging.info(f"Вероятность мошенничества: {proba:.4%}")
                logging.info(f"Уровень риска: {risk_level}")

                print(f"{case['description']}:")
                print(f"Вероятность: {proba:.4%} → {risk_level}")

            except Exception as e:
                logging.error(f"Ошибка теста: {str(e)}")
                print(f"Ошибка при тесте: {e}")