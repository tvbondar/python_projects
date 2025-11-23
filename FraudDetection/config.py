import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Пути
DATA_PATH = os.path.join(BASE_DIR, "data", "PS_20174392719_1491204439457_log.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, "visualizations")
LOG_FILE = os.path.join(BASE_DIR, "fraud_detection.log")

# Модель
MODEL_PARAMS = {
    "hidden_layers": [128, 68, 32],
    "dropout": 0.3,
    "learning_rate": 0.001,
    "epochs": 50,
    "batch_size": 1024,
    "validation_split": 0.2
}

# Предобработка
HIGH_AMOUNT_PERCENTILE = 0.99
NIGHT_HOURS_START = 22
NIGHT_HOURS_END = 6

# маппинг типов транзакций
TRANSACTION_TYPE_MAP = {
    'CASH_OUT': 1,
    'PAYMENT': 2,
    'CASH_IN': 3,
    'TRANSFER': 4,
    'DEBIT': 5
}
UNKNOWN_TYPE_VALUE = 0 # что подставить, если тип неизвестен

# названия новых признаков
FEATURE_NAMES = {
    'balance_change_org': 'balance_change_org',
    'balance_change_dest': 'balance_change_dest',
    'is_amount_high': 'is_amount_high',
    'hour': 'hour',
    'is_night': 'is_night',
    'amount_to_balance_ratio': 'amount_to_balance_ratio'
}

# Риск
RISK_THRESHOLDS = {
    "low": 0.4,
    "medium": 0.7,
    "high": 1.0
}

# Логирование
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# Тестовые кейсы
TEST_CASES = [
    {
        'description': "Обычная дневная транзакция (маленький платёж)",
        'step': 50, 'type': 'PAYMENT', 'amount': 1200,
        'oldbalanceOrg': 34200, 'newbalanceOrig': 33000,
        'oldbalanceDest': 0, 'newbalanceDest': 0,
    },
    {
        'description': "Подозрительный перевод ночью всей суммы на счёте",
        'step': 5, 'type': 'TRANSFER', 'amount': 1250000,
        'oldbalanceOrg': 1250000, 'newbalanceOrig': 0,
        'oldbalanceDest': 0, 'newbalanceDest': 1250000,
    },
    {
        'description': "Обычное пополнение днём (CASH_IN)",
        'step': 180, 'type': 'CASH_IN', 'amount': 50000,
        'oldbalanceOrg': 15000, 'newbalanceOrig': 65000,
        'oldbalanceDest': 0, 'newbalanceDest': 0,
    },
    {
        'description': "Очень большая сумма ночью + обнуление счёта (классическое мошенничество)",
        'step': 23, 'type': 'TRANSFER', 'amount': 2800000,
        'oldbalanceOrg': 2800000, 'newbalanceOrig': 0,
        'oldbalanceDest': 120000, 'newbalanceDest': 2920000,
    },
    {
        'description': "Мелкий вывод наличных днём — норма",
        'step': 95, 'type': 'CASH_OUT', 'amount': 8000,
        'oldbalanceOrg': 45000, 'newbalanceOrig': 37000,
        'oldbalanceDest': 0, 'newbalanceDest': 0,
    }
]

# Визуализация
VISUALIZATION_TITLES = {
    'fraud_distribution': 'Распределение мошеннических транзакций',
    'transaction_types': 'Распределение типов транзакций',
    'feature_importance': 'Топ-15 важных признаков'
}
TYPE_LABELS = ['Вывод наличных', 'Платеж', 'Пополнение наличными', 'Перевод', 'Дебет']