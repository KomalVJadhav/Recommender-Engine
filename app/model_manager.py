import os
import uuid
import mlflow
from setfit import SetFitModel, SetFitTrainer
from datasets import Dataset
from app.config import settings
from app.logger import logging

class ModelManager:
    def __init__(self):
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        self.logger = logging.getLogger("recommender")

    def train_setfit(self, labeled_data: dict):
        self.logger.info("Training SetFit model...")
        texts, labels = zip(*labeled_data.items())
        dataset = Dataset.from_dict({"text": texts, "label": list(map(int, labels))})
        model = SetFitModel.from_pretrained(settings.SETFIT_MODEL)
        trainer = SetFitTrainer(model=model, train_dataset=dataset, metric="f1")
        trainer.train()

        run_id = str(uuid.uuid4())
        model_path = f"models/setfit_{run_id}"
        model.save_pretrained(model_path)

        with mlflow.start_run(run_name=f"setfit_{run_id}"):
            mlflow.set_tag("model", "SetFit")
            mlflow.log_param("num_samples", len(labeled_data))
            mlflow.log_artifacts(model_path, artifact_path="setfit_model")

        self.logger.info(f"SetFit model trained and saved at {model_path}")
        return model
