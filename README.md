## Car Price Predictor (Flask + scikit-learn)

An educational, end-to-end regression project showing how different ML models learn to predict car prices. It includes:

- A clean preprocessing pipeline (imputation, scaling, one-hot encoding)
- Multiple models trained and compared automatically
- An ensemble (Voting Regressor) to combine models
- A Flask app with UI to predict prices and view comparison metrics

This repo is intentionally simple and readable to help newcomers (and busy managers) understand how ML fits together.

### Demo: What happens

1. Data is loaded (your CSV at `data/cars.csv` or a synthetic dataset is generated).
2. We split into train/test and build a preprocessing pipeline.
3. We train several models and evaluate them:
    - Linear Regression
    - Ridge (L2)
    - Lasso (L1)
    - Random Forest
    - Gradient Boosting
    - Ensemble (Voting over top performers)
4. We compute metrics (R², MAE, RMSE) and pick the best model.
5. The best model is saved to `model.joblib`. A full report is saved to `metrics.json`.
6. The Flask app serves:
    - `/` — prediction form
    - `/compare` — comparison table of models
    - `/train` — retrain and refresh metrics
    - `/metrics` — raw JSON of results

### Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: put your dataset at data/cars.csv with target column price/Price/selling_price
python app.py  # this will train on first run if model is missing
# Open http://127.0.0.1:5000
```

### Using the API

- Predict via JSON:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{"brand":"Toyota","year":2018,"mileage":65000,"engine_size":1.6}'
```

- Compare models (JSON): `GET /metrics`

### Educational Notes

- Why multiple models? Each algorithm has different bias/variance trade-offs. We compare them to learn which works best
  for this data.
- Why a pipeline? Consistent preprocessing avoids data leakage and keeps code maintainable.
- What metrics mean:
    - R²: How much variance we explain (closer to 1 is better)
    - MAE: Average absolute error in the same units as the price (lower is better)
    - RMSE: Penalizes large errors more strongly (lower is better)
- Ensemble: A simple average (Voting Regressor) of strong models often performs as well as or better than the single
  best model.

### Project Layout

- `app.py` — Flask app (predict, compare, retrain, metrics)
- `train.py` — training and comparison logic; saves `model.joblib` and `metrics.json`
- `templates/` — UI pages (`index.html`, `compare.html`)
- `requirements.txt` — Python dependencies

### Bring Your Own Data

Place a CSV at `data/cars.csv` with a target column named one of:

- `price` (preferred)
- `Price`
- `selling_price`

Feature columns can include categorical (e.g., brand) and numeric (e.g., year, mileage, engine_size). The code will
infer types and handle missing values.

### Reproducibility

We fix `random_state=42` and use a consistent test split (`test_size=0.25`). You can change these in
`train_and_compare()` in `train.py`.

### License

MIT