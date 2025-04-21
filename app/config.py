import os

class Settings:
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    ZERO_SHOT_MODEL: str = "facebook/bart-large-mnli"
    SETFIT_MODEL: str = "sentence-transformers/paraphrase-mpnet-base-v2"
    LABEL_THRESHOLD: int = 8
    ST_TRAIN_TRIGGER_COUNT: int = 4

settings = Settings()
