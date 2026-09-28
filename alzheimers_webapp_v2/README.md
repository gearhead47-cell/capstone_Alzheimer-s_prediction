# Alzheimer's Risk Predictor — v2 (Real Trained Model)

This version uses your actual notebook-trained XGBoost model (tuned + pruned,
33 features, ~95% test accuracy, 0.40 decision threshold).

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Copy your trained model files here

From your notebook, you already ran:
```python
joblib.dump(xgb_pruned, "model.joblib")
joblib.dump(scaler_p, "scaler.joblib")
with open("feature_list.json", "w") as f:
    json.dump(final_features, f, indent=2)
```

Find those 3 files (likely in your Jupyter working directory — the same
folder as your notebook, or wherever `alzheimers_project` was created) and
copy them into this app's `model/` folder:

```
alzheimers_webapp_v2/
└── model/
    ├── model.joblib
    ├── scaler.joblib
    └── feature_list.json
```

You can copy them manually, or from Jupyter:
```python
import shutil
shutil.copy("model.joblib", "/path/to/alzheimers_webapp_v2/model/model.joblib")
shutil.copy("scaler.joblib", "/path/to/alzheimers_webapp_v2/model/scaler.joblib")
shutil.copy("feature_list.json", "/path/to/alzheimers_webapp_v2/model/feature_list.json")
```

## 3. Run the app

```bash
python app.py
```

Open **http://127.0.0.1:5000**, fill out the form (all 32 raw patient fields),
click Predict.

## How it works

- `feature_engineering.py` reproduces the *exact* same transformations from
  the notebook (PulsePressure, CholesterolRatio, SymptomCount, ComorbidityCount,
  CognitiveFunctionalScore, BPCategoryEncoded, Gender/Ethnicity one-hot) so a
  raw form submission turns into the same 33 columns the model was trained on.
- The form still asks for fields like Diabetes, HeadInjury, Disorientation
  even though those exact columns were pruned — they're still needed to
  *compute* ComorbidityCount/SymptomCount before being dropped.
- Predictions use a **0.40 probability threshold** (not the default 0.50),
  matching the recall-optimized cutoff chosen during tuning — appropriate for
  a screening tool where missing a case is costlier than a false alarm.

## Notes

- If you retrain or re-tune the model later, just re-save the three files in
  `model/` — no code changes needed, as long as the feature list stays the same.
- If you change the feature engineering itself (e.g. drop more features, add
  new ones), update `feature_engineering.py` to match your notebook exactly,
  or predictions will be wrong.
