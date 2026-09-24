"""
Flask web app that serves the trained attrition model.
Run:  python app.py   ->  http://127.0.0.1:5000
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from train import add_features  # same feature logic as training

app = Flask(__name__)
model = joblib.load(ROOT / "models" / "attrition_model.joblib")
meta = json.loads((ROOT / "models" / "meta.json").read_text())

# Fields shown in the form; everything else is filled with dataset median/mode.
NUMERIC_FIELDS = [
    ("Age", "Age", 18, 60), ("MonthlyIncome", "Monthly Income (USD)", 1000, 20000),
    ("DistanceFromHome", "Distance From Home (km)", 1, 30), ("YearsAtCompany", "Years at Company", 0, 40),
    ("TotalWorkingYears", "Total Working Years", 0, 40), ("YearsSinceLastPromotion", "Years Since Last Promotion", 0, 15),
    ("YearsWithCurrManager", "Years With Current Manager", 0, 17), ("NumCompaniesWorked", "Companies Worked At", 0, 10),
]
RATING_FIELDS = [("JobSatisfaction", "Job Satisfaction"), ("EnvironmentSatisfaction", "Environment Satisfaction"),
                 ("WorkLifeBalance", "Work-Life Balance")]
CHOICE_FIELDS = [("OverTime", "Works Overtime"), ("JobRole", "Job Role"), ("Department", "Department"),
                 ("BusinessTravel", "Business Travel"), ("MaritalStatus", "Marital Status")]
LEVEL_FIELDS = [("JobLevel", "Job Level", [1, 2, 3, 4, 5]), ("StockOptionLevel", "Stock Option Level", [0, 1, 2, 3])]


def risk_band(p: float) -> str:
    return "High" if p >= meta["threshold"] else "Medium" if p >= meta["threshold"] * 0.6 else "Low"


@app.route("/")
def index():
    return render_template("index.html", numeric=NUMERIC_FIELDS, ratings=RATING_FIELDS,
                           choices=CHOICE_FIELDS, levels=LEVEL_FIELDS, options=meta["options"],
                           defaults=meta["defaults"], metrics=meta["metrics"],
                           top_features=meta["top_features"])


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    row = dict(meta["defaults"])
    for key, value in data.items():
        if key in row:
            row[key] = value if isinstance(row[key], str) else float(value)
    df = add_features(pd.DataFrame([row]))[meta["columns"]]
    prob = float(model.predict_proba(df)[0, 1])
    return jsonify({"probability": round(prob * 100, 1), "risk": risk_band(prob),
                    "will_leave": prob >= meta["threshold"]})


if __name__ == "__main__":
    app.run(debug=False)
