#classifier wrapper

from email.mime import text
import json
from pathlib import Path
import numpy as np
import joblib
from app.perceptron import Perceptron

ARTIFACTS_dir = Path(__file__).parent / "artifacts"

class Classifier:
    def __init__(self):
        self.tfidf = joblib.load(ARTIFACTS_dir / "tfidf.joblib")
        with open(ARTIFACTS_dir / "model_metadata.json") as f:
            self.metadata = json.load(f)

        self.model = Perceptron(
            learning_rate=self.metadata["learning_rate"],
            max_iterations=self.metadata["max_iterations"],
            class_weight=self.metadata["class_weight"],)

        self.model.weights = joblib.load(ARTIFACTS_dir / "weights.joblib")

        self.model.bias = joblib.load(ARTIFACTS_dir / "bias.joblib")

        self.labels = {int(k): v for k, v in self.metadata["labels"].items()}



    def predict(self, text: str) -> dict:

        X = self.tfidf.transform([text]).toarray()

        score = self.model.decision_score(X)[0]

        label_id = 1 if score >= 0 else 0

        
        confidence = 1 / (1 + np.exp(-score))
        confidence = float(confidence if label_id == 1 else 1 - confidence)

        return {
            "label": self.labels[label_id],
            "confidence": round(confidence, 4),
            "raw_score": round(float(score), 4),
        }


# Loaded once when the API process starts, not per-request.
classifier = Classifier()