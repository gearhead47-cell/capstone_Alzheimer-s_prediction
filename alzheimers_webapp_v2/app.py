"""
Flask web app (v2): input raw patient data -> engineer features ->
scale -> predict with the tuned, pruned XGBoost model.

Setup:
    1. Copy your model.joblib, scaler.joblib, feature_list.json
       (saved from the notebook) into ./model/
    2. python app.py
    3. Open http://127.0.0.1:5000
"""

import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request

from feature_engineering import engineer_features, RAW_FEATURES

app = Flask(__name__)

# Load trained artifacts
model = joblib.load("model/model.joblib")
scaler = joblib.load("model/scaler.joblib")
with open("model/feature_list.json") as f:
    FINAL_FEATURES = json.load(f)  # exact column order the model expects

# Threshold chosen during tuning (favors recall for a screening use case)
PREDICTION_THRESHOLD = 0.40

# Form field metadata (labels, input types, ranges) for all 32 raw inputs
FEATURE_SPEC = {
    "Age":                       {"type": "number", "min": 60, "max": 90, "step": 1, "label": "Age (years)"},
    "Gender":                    {"type": "select", "options": {"0": "Male", "1": "Female"}, "label": "Gender"},
    "Ethnicity":                 {"type": "select", "options": {"0": "Caucasian", "1": "African American", "2": "Asian", "3": "Other"}, "label": "Ethnicity"},
    "EducationLevel":            {"type": "select", "options": {"0": "None", "1": "High School", "2": "Bachelor's", "3": "Higher"}, "label": "Education Level"},
    "BMI":                       {"type": "number", "min": 15, "max": 40, "step": 0.1, "label": "BMI"},
    "Smoking":                   {"type": "select", "options": {"0": "No", "1": "Yes"}, "label": "Smoking"},
    "AlcoholConsumption":        {"type": "number", "min": 0, "max": 20, "step": 0.1, "label": "Alcohol (weekly units)"},
    "PhysicalActivity":          {"type": "number", "min": 0, "max": 10, "step": 0.1, "label": "Physical Activity (hrs/week)"},
    "DietQuality":               {"type": "number", "min": 0, "max": 10, "step": 0.1, "label": "Diet Quality (0-10)"},
    "SleepQuality":              {"type": "number", "min": 4, "max": 10, "step": 0.1, "label": "Sleep Quality (4-10)"},
    "FamilyHistoryAlzheimers":   {"type": "select", "options": {"0": "No", "1": "Yes"}, "label": "Family History of Alzheimer's"},
    "CardiovascularDisease":     {"type": "select", "options": {"0": "No", "1": "Yes"}, "label": "Cardiovascular Disease"},
    "Diabetes":                  {"type": "select", "options": {"0": "No", "1": "Yes"}, "label": "Diabetes"},
    "Depression":                {"type": "select", "options": {"0": "No", "1": "Yes"}, "label": "Depression"},
    "HeadInjury":                 {"type": "select", "options": {"0": "No", "1": "Yes"}, "label": "History of Head Injury"},
    "Hypertension":              {"type": "select", "options": {"0": "No", "1": "Yes"}, "label": "Hypertension"},
    "SystolicBP":                {"type": "number", "min": 90, "max": 180, "step": 1, "label": "Systolic BP (mmHg)"},
    "DiastolicBP":               {"type": "number", "min": 60, "max": 120, "step": 1, "label": "Diastolic BP (mmHg)"},
    "CholesterolTotal":          {"type": "number", "min": 150, "max": 300, "step": 1, "label": "Total Cholesterol (mg/dL)"},
    "CholesterolLDL":            {"type": "number", "min": 50, "max": 200, "step": 1, "label": "LDL Cholesterol (mg/dL)"},
    "CholesterolHDL":            {"type": "number", "min": 20, "max": 100, "step": 1, "label": "HDL Cholesterol (mg/dL)"},
    "CholesterolTriglycerides":  {"type": "number", "min": 50, "max": 400, "step": 1, "label": "Triglycerides (mg/dL)"},
    "MMSE":                      {"type": "number", "min": 0, "max": 30, "step": 0.1, "label": "MMSE Score (0-30, lower = more impaired)"},
    "FunctionalAssessment":      {"type": "number", "min": 0, "max": 10, "step": 0.1, "label": "Functional Assessment (0-10, lower = more impaired)"},
    "MemoryComplaints":          {"type": "select", "options": {"0": "No", "1": "Yes"}, "label": "Memory Complaints"},
    "BehavioralProblems":        {"type": "select", "options": {"0": "No", "1": "Yes"}, "label": "Behavioral Problems"},
    "ADL":                       {"type": "number", "min": 0, "max": 10, "step": 0.1, "label": "Activities of Daily Living (0-10, lower = more impaired)"},
    "Confusion":                 {"type": "select", "options": {"0": "No", "1": "Yes"}, "label": "Confusion"},
    "Disorientation":            {"type": "select", "options": {"0": "No", "1": "Yes"}, "label": "Disorientation"},
    "PersonalityChanges":        {"type": "select", "options": {"0": "No", "1": "Yes"}, "label": "Personality Changes"},
    "DifficultyCompletingTasks": {"type": "select", "options": {"0": "No", "1": "Yes"}, "label": "Difficulty Completing Tasks"},
    "Forgetfulness":             {"type": "select", "options": {"0": "No", "1": "Yes"}, "label": "Forgetfulness"},
}


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", feature_order=RAW_FEATURES, feature_spec=FEATURE_SPEC,
                            result=None, values={})


@app.route("/predict", methods=["POST"])
def predict():
    values = {}
    raw_row = {}
    for feat in RAW_FEATURES:
        raw = request.form.get(feat)
        raw_row[feat] = float(raw)
        values[feat] = raw

    # Build a 1-row DataFrame, engineer features, align to training column order
    raw_df = pd.DataFrame([raw_row])
    engineered_df = engineer_features(raw_df)
    engineered_df = engineered_df.reindex(columns=FINAL_FEATURES)  # exact order + drop extras

    X_scaled = scaler.transform(engineered_df.values)

    prob = model.predict_proba(X_scaled)[0][1]  # probability of AD=1
    pred = int(prob >= PREDICTION_THRESHOLD)

    result = {
        "prediction": "Alzheimer's (AD)" if pred == 1 else "No Alzheimer's",
        "probability": round(prob * 100, 1),
        "is_positive": bool(pred == 1),
        "threshold": PREDICTION_THRESHOLD,
    }

    return render_template("index.html", feature_order=RAW_FEATURES, feature_spec=FEATURE_SPEC,
                            result=result, values=values)


if __name__ == "__main__":
    app.run(debug=True)
