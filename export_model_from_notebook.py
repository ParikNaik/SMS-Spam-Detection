# =========================================================================
# COPY THIS CODE AS A NEW CELL AT THE VERY BOTTOM OF spam_detection.ipynb
# (after the cell where perceptron.fit(X_train_dense, Y_train) has run)
# This file itself does not run on its own - vectorizer and perceptron
# only exist as trained objects inside the notebook's kernel.
# =========================================================================

import joblib
import json
import os

# Since the notebook lives at the repo root, and app/artifacts/ is a
# subfolder of the repo root, we can save directly there instead of
# saving here and moving files afterward.
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