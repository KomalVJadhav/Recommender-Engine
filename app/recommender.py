from transformers import pipeline
from app.model_manager import ModelManager
from app.config import settings
import numpy as np
from app.logger import logging

class RecommenderEngine:
    def __init__(self):
        self.model_manager = ModelManager()
        self.zero_shot = pipeline("zero-shot-classification", model=settings.ZERO_SHOT_MODEL)
        self.logger = logging.getLogger("recommender")

    def recommend(self, topic: str, labeled_data: dict):
        pos_samples = [text for text, label in labeled_data.items() if label]
        neg_samples = [text for text, label in labeled_data.items() if not label]

        if len(labeled_data) < settings.LABEL_THRESHOLD:
            self.logger.info("Using zero-shot model due to insufficient labels.")
            return self._zero_shot_recommend(topic)

        self.logger.info("Using SetFit model for few-shot classification.")
        model = self.model_manager.train_setfit(labeled_data)
        all_texts = pos_samples + neg_samples
        predictions = model.predict(all_texts)
        ranked = [text for text, pred in zip(all_texts, predictions) if pred == 1]
        return {"topic": topic, "recommendations": ranked}

    def _zero_shot_recommend(self, topic):
        # Ideally replace with database or streamed data
        sample_texts = [
            "This is a sample sentence about diabetes.",
            "A new medication was discussed.",
            "The patient showed signs of asthma."
        ]
        results = self.zero_shot(sample_texts, candidate_labels=[topic], multi_label=False)
        scores = [r['scores'][0] for r in results]
        top_indices = np.argsort(scores)[::-1]
        recommendations = [sample_texts[i] for i in top_indices[:3]]
        return {"topic": topic, "recommendations": recommendations}
