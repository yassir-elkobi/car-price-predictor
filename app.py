import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for

from train import train_and_compare

MODEL = None
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
METRICS_PATH = os.getenv("METRICS_PATH", "metrics.json")
app = Flask(__name__)


def get_model():
    global MODEL
    if MODEL is None:
        if not os.path.exists(MODEL_PATH):
            train_and_compare()
        MODEL = joblib.load(MODEL_PATH)
    return MODEL


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    data = request.get_json() if request.is_json else request.form.to_dict()
    payload = {
        "brand": data.get("brand", "Toyota"),
        "year": int(data.get("year", 2018)),
        "mileage": float(data.get("mileage", 65000)),
        "engine_size": float(data.get("engine_size", 1.6)),
    }
    X = pd.DataFrame([payload])
    model = get_model()
    price = float(model.predict(X)[0])
    if request.is_json:
        return jsonify({"predicted_price": price})
    return render_template("index.html", result={"price": f"{price:,.0f}"})


@app.post("/train")
def train():
    train_and_compare()
    # refresh in-memory model after retrain
    global MODEL
    MODEL = None
    return redirect(url_for("compare"))


@app.get("/metrics")
def metrics_json():
    if not os.path.exists(METRICS_PATH):
        train_and_compare()
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    return jsonify(metrics)


@app.get("/compare")
def compare():
    if not os.path.exists(METRICS_PATH):
        train_and_compare()
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    return render_template("compare.html", summary=metrics)


if __name__ == "__main__":
    app.run(debug=True)
