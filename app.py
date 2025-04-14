from flask import Flask, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import random

app = Flask(__name__)

# Load model and features
model = joblib.load("hybrid_model.pkl")
features = joblib.load("model_features.pkl")

# Load RF and GB for feature importance
rf = model.named_estimators_['rf']
gb = model.named_estimators_['gb']
rf_importance = rf.feature_importances_
gb_importance = gb.feature_importances_
avg_importance = (rf_importance + gb_importance) / 2
feature_importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": avg_importance
}).sort_values(by="Importance", ascending=False)

# Simulate one row of data
def simulate_data():
    return {
        "Footfall": random.randint(20, 100),
        "TempMode": random.choice([0, 1]),
        "AQ": random.uniform(50, 200),
        "USS": random.uniform(1, 10),
        "CS": random.uniform(0, 1),
        "VOC": random.uniform(100, 500),
        "RP": random.uniform(1000, 3000),
        "IP": random.uniform(2, 20),
        "Temperature": random.uniform(20, 90)
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/predict", methods=["GET"])
def predict():
    data = simulate_data()
    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    risk = "High" if prob > 0.8 else "Medium" if prob > 0.5 else "Low"

    result = {
        "data": data,
        "prediction": int(prediction),
        "probability": round(prob, 2),
        "risk": risk
    }

    if prediction == 1:
        top_features = feature_importance_df.head(3)["Feature"].tolist()
        result["fault_causes"] = top_features

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
