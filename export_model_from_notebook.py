import joblib
import json
import os

os.makedirs("app/artifacts", exist_ok=True)

joblib.dump(vectorizer, "app/artifacts/tfidf.joblib")
joblib.dump(perceptron.weights, "app/artifacts/weights.joblib")
joblib.dump(perceptron.bias, "app/artifacts/bias.joblib")

metadata = {
    "learning_rate": perceptron.learning_rate,
    "max_iterations": perceptron.max_iterations,
    "class_weight": perceptron.class_weight,
    "labels": {0: "ham", 1: "spam"},
}
with open("app/artifacts/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Saved directly into app/artifacts/")