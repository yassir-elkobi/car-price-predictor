import json
import time
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/cars.csv")
MODEL_PATH = "model.joblib"
METRICS_PATH = "metrics.json"


def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        target_candidates = ["price", "Price", "selling_price"]
        target = next((c for c in target_candidates if c in df.columns), None)
        if target is None:
            raise ValueError("No target column (price/Price/selling_price).")
        feats = [c for c in df.columns if c != target]
        return df[feats + [target]].rename(columns={target: "price"})
    # fallback synthetic dataset
    rng = np.random.default_rng(7)
    n = 800
    brands = np.array(["Toyota", "Ford", "BMW", "VW", "Renault", "Peugeot", "Tesla"])
    brand = rng.choice(brands, size=n, p=[.22, .18, .12, .16, .12, .12, .08])
    year = rng.integers(2008, 2024, size=n)
    mileage = rng.normal(60000, 25000, size=n).clip(5000, 200000)
    engine = rng.choice([1.2, 1.4, 1.6, 2.0, 2.5, 3.0], size=n)
    base = 30000 + (year - 2008) * 800 - mileage * 0.08 + (engine - 1.2) * 1500
    brand_adj = {"Toyota": 0, "Ford": -1200, "BMW": 6000, "VW": 800, "Renault": -800, "Peugeot": -600, "Tesla": 9000}
    price = base + np.vectorize(brand_adj.get)(brand) + rng.normal(0, 2500, size=n)
    return pd.DataFrame(
        {"brand": brand, "year": year, "mileage": mileage.astype(int), "engine_size": engine, "price": price})


def train_and_compare(test_size: float = 0.25, random_state: int = 42):
    df = load_data()
    y = df["price"].astype(float)
    X = df.drop(columns=["price"])

    categorical_features = [c for c in X.columns if X[c].dtype == "object"]
    numeric_features = [c for c in X.columns if c not in categorical_features]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric_features),
        ("cat",
         Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]),
         categorical_features),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    candidates = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, max_iter=10000),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    results = []
    best = {"name": None, "r2": -1e9, "pipeline": None}

    for name, estimator in candidates.items():
        pipeline = Pipeline([("pre", preprocessor), ("reg", estimator)])
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_time = time.perf_counter() - start
        y_pred = pipeline.predict(X_test)
        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(mean_squared_error(y_test, y_pred, squared=False))

        if r2 > best["r2"]:
            best = {"name": name, "r2": r2, "pipeline": pipeline}

        # Keep small sample for simple plotting in UI
        sample_n = min(150, len(y_test))
        results.append({
            "model": name,
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "fit_time_s": round(fit_time, 4),
            "y_true_sample": [float(v) for v in y_test.iloc[:sample_n].tolist()],
            "y_pred_sample": [float(v) for v in y_pred[:sample_n].tolist()],
        })

    # Simple averaging ensemble over the top 3 by R2
    top3 = sorted(results, key=lambda r: r["r2"], reverse=True)[:3]
    ensemble_estimators = []
    for r in top3:
        est = candidates[r["model"]]
        ensemble_estimators.append((r["model"], est))
    if ensemble_estimators:
        ensemble = Pipeline([
            ("pre", preprocessor),
            ("reg", VotingRegressor(ensemble_estimators))
        ])
        start = time.perf_counter()
        ensemble.fit(X_train, y_train)
        fit_time = time.perf_counter() - start
        ens_pred = ensemble.predict(X_test)
        ens_r2 = float(r2_score(y_test, ens_pred))
        ens_mae = float(mean_absolute_error(y_test, ens_pred))
        ens_rmse = float(mean_squared_error(y_test, ens_pred, squared=False))
        sample_n = min(150, len(y_test))
        results.append({
            "model": "Ensemble(Voting)",
            "r2": ens_r2,
            "mae": ens_mae,
            "rmse": ens_rmse,
            "fit_time_s": round(fit_time, 4),
            "y_true_sample": [float(v) for v in y_test.iloc[:sample_n].tolist()],
            "y_pred_sample": [float(v) for v in ens_pred[:sample_n].tolist()],
        })
        if ens_r2 > best["r2"]:
            best = {"name": "Ensemble(Voting)", "r2": ens_r2, "pipeline": ensemble}

    # Persist best model
    joblib.dump(best["pipeline"], MODEL_PATH)

    # Persist metrics
    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "n_rows": int(len(df)),
        "test_size": test_size,
        "random_state": random_state,
        "features": list(X.columns),
        "target": "price",
        "best_model": best["name"],
        "results": results,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {MODEL_PATH}. Best={best['name']}  BestR2={best['r2']:.3f}")


def main():
    train_and_compare()


if __name__ == "__main__":
    main()
