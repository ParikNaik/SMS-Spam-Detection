# SMS Spam Detection

A spam classifier for text messages, built around a **perceptron written from scratch in NumPy** and served as a REST API with a browser interface.

**Live demo:** _add your Render URL here_ (free tier: the first load after a quiet period can take 30-60 seconds while the server wakes up)

Paste a text message, and the model returns a verdict (spam or ham) along with its raw decision score.

## Why this project

Scam and phishing texts are getting more convincing, and a spam filter is only useful if people can actually use it. This project started as a notebook experiment and was turned into a working service: a trained model behind an API, with a web page on top.

## Results

Evaluated on a held-out test set of 1,115 messages (966 ham, 149 spam) from the UCI SMS Spam Collection:

| Metric | Ham | Spam |
|---|---|---|
| Precision | 1.00 | 0.75 |
| Recall | 0.95 | 0.97 |
| F1-score | 0.97 | 0.85 |

Overall accuracy: **95.3%**.

```
Confusion matrix (rows = true, columns = predicted)

             ham   spam
true ham     917     49
true spam      4    145
```

The model catches 145 of 149 spam messages. The cost is 49 real messages flagged as spam, which is why spam precision is 0.75. See the next section for why accuracy alone is a misleading metric here.

## The class imbalance problem

The dataset is about 87% ham (4,825 ham vs 747 spam). A model that answers "ham" every time would score roughly 87% accuracy without learning anything, so accuracy is not a trustworthy metric on this data.

The perceptron's update rule made this worse. Ham mistakes vastly outnumber spam mistakes during training, so the bias term drifted toward "ham" and the model ignored the rare class.

The fix was a class-weighted update rule, where mistakes on the rare class move the weights further. Tuning the weight mattered:

| Spam class weight | Accuracy | Spam precision | Spam recall |
|---|---|---|---|
| Full inverse frequency (about 6.5) | 0.72 | 0.32 | 0.98 |
| Square root of the ratio (about 2.5) | 0.95 | 0.75 | 0.97 |

The full inverse-frequency weight overcorrected and flagged far too many real messages. The softer square-root weight is what the final model uses.

## How it works

```
text message
    -> TF-IDF vectorizer (unigrams + bigrams, 3,000 features)
    -> perceptron: score = w . x + b
    -> score >= 0 means spam, otherwise ham
```

- **Features:** `TfidfVectorizer` with English stop words removed, bigrams, `min_df=2`, `max_df=0.95`, ASCII accent stripping.
- **Model:** a perceptron implemented by hand (`app/perceptron.py`), with no ML library doing the learning.
- **Decision score:** the raw `w . x + b` value. Because TF-IDF values are small, scores are small too, so look at the sign rather than the magnitude.

## API

Run the server and open `/docs` for interactive documentation.

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Web interface |
| GET | `/health` | Liveness check, returns `{"status": "ok"}` |
| POST | `/classify` | Classify one message |
| GET | `/docs` | Auto-generated Swagger UI |

**Request**

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "URGENT! You have won a free prize, call now"}'
```

**Response**

```json
{
  "label": "spam",
  "confidence": 0.5062,
  "raw_score": 0.0246
}
```

- `label` is `"spam"` or `"ham"`.
- `raw_score` is the perceptron's decision value. Above 0 means spam.
- `confidence` is a sigmoid squash of the raw score. It is **not a calibrated probability**, and it sits near 0.5 even for correct predictions because the raw scores are small. Treat `label` and `raw_score` as the real outputs.

Input must be between 1 and 2,500 characters. Anything else returns a `422` validation error.

## Project structure

```
SMS-Spam-Detection/
├── spam_detection.ipynb          # data loading, training, evaluation, model export
├── SMSSpamCollection             # UCI dataset
├── requirements.txt
├── app/
│   ├── main.py                   # FastAPI app and routes
│   ├── model.py                  # loads saved artifacts, exposes predict()
│   ├── schemas.py                # Pydantic request/response models
│   ├── perceptron.py             # the perceptron class
│   ├── static/
│   │   └── index.html            # browser interface
│   └── artifacts/                # exported model files (created by the notebook)
│       ├── tfidf.joblib
│       ├── weights.joblib
│       ├── bias.joblib
│       └── model_metadata.json
```

Code and learned parameters are kept separate on purpose. The `Perceptron` class lives in `app/perceptron.py`, and only the trained numbers (weights and bias) are saved to disk and loaded at startup.

## Run it locally

```bash
# create and activate a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # Mac/Linux

pip install -r requirements.txt
```

1. Open `spam_detection.ipynb` and run all cells, including the final export cell. This writes the four model files into `app/artifacts/`.
2. Start the server from the repo root:

```bash
uvicorn app.main:app --reload
```

3. Open http://127.0.0.1:8000 for the web interface or http://127.0.0.1:8000/docs for the API docs.

**Note:** use the same scikit-learn version for training and serving. The saved vectorizer is a pickled scikit-learn object, and loading it under a different version can fail or behave differently.

## Deploying (Render)

| Setting | Value |
|---|---|
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |

Commit `app/artifacts/` to the repository, since the deployed server needs the trained model files to start.

## Limitations

- **Dataset age and style.** The model was trained on the UCI SMS Spam Collection, a small, older, UK-flavored dataset. It has not been trained or tested on modern or AI-generated scam messages, so performance on those is unknown.
- **Precision trade-off.** About 1 in 4 messages flagged as spam is a real message. The class weight can be tuned toward fewer false alarms at some cost to recall.
- **Uncalibrated scores.** A perceptron does not produce probabilities, so `confidence` is only an approximation.
- **Single message at a time.** There is no batch endpoint yet.

## Possible next steps

- Batch classification endpoint and a table view in the UI
- Rate limiting and request logging
- Threshold tuning exposed as a setting, so users can trade precision against recall
- Evaluation on newer, more varied message data
- Comparison against a scikit-learn baseline (Naive Bayes, logistic regression)

## Tech stack

Python, NumPy, scikit-learn (TF-IDF only), FastAPI, Uvicorn, Pydantic, vanilla HTML/CSS/JavaScript.

## Dataset

[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection): 5,572 labeled messages (4,825 ham, 747 spam).
