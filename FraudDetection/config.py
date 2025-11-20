import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Пути
DATA_PATH = os.path.join(BASE_DIR, "data", "PS_20174392719_1491204439457_log.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, "visualizations")
LOG_FILE = os.path.join(BASE_DIR, "fraud_detection.log")

# Модель
MODEL_PARAMS = {
    "n_estimators": 150,
    "max_depth": 12,
    "min_samples_leaf": 5,
    "random_state": 42,
    "class_weight": "balanced",
    "n_jobs": -1
}

# Предобработка
HIGH_AMOUNT_PERCENTILE = 0.99
NIGHT_HOURS_START = 22
NIGHT_HOURS_END = 6

# Риск
RISK_THRESHOLDS = {
    "low": 0.4,
    "medium": 0.7,
    "high": 1.0
}

# Логирование
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'