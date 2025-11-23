import os
import logging
import config
from functions import FraudDetectionSystem

# Логирование
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)

if __name__ == "__main__":
    try:
        fraud_system = FraudDetectionSystem()

        # Используем config
        df = fraud_system.load_data()
        processed_df = fraud_system.preprocess_data(df)

        # Папки
        os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
        os.makedirs(config.VISUALIZATIONS_DIR, exist_ok=True)

        # Модель
        if os.path.exists(config.MODEL_PATH):
            fraud_system.load_model()
        else:
            fraud_system.train_model(processed_df)

        # Визуализация
        fraud_system.visualize_data(processed_df)

        # Тесты
        fraud_system.run_tests()

    except Exception as e:
        logging.error(f"Критическая ошибка: {str(e)}")


