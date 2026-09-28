"""
Feature engineering pipeline — must exactly match what was done in the
training notebook, so a raw patient record produces the same 33 columns
the model was trained on.
"""

import pandas as pd

# The 32 raw inputs the form collects from the user.
# (Some of these, like Diabetes/HeadInjury/Disorientation, are only used to
# compute ComorbidityCount/SymptomCount below and are dropped afterward —
# they still need to be collected as inputs.)
RAW_FEATURES = [
    "Age", "Gender", "Ethnicity", "EducationLevel", "BMI", "Smoking",
    "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality",
    "FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes", "Depression",
    "HeadInjury", "Hypertension", "SystolicBP", "DiastolicBP",
    "CholesterolTotal", "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides",
    "MMSE", "FunctionalAssessment", "MemoryComplaints", "BehavioralProblems", "ADL",
    "Confusion", "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks",
    "Forgetfulness",
]

SYMPTOM_COLS = [
    "MemoryComplaints", "BehavioralProblems", "Confusion",
    "Disorientation", "PersonalityChanges",
    "DifficultyCompletingTasks", "Forgetfulness",
]

COMORBIDITY_COLS = [
    "FamilyHistoryAlzheimers", "CardiovascularDisease",
    "Diabetes", "Depression", "HeadInjury", "Hypertension",
]

BP_ORDER = {
    "Normal": 0,
    "Elevated": 1,
    "Stage 1 Hypertension": 2,
    "Stage 2 Hypertension": 3,
    "Hypertensive Crisis": 4,
}


def _bp_category(sys, dia):
    if sys > 180 or dia > 120:
        return "Hypertensive Crisis"
    elif sys >= 140 or dia >= 90:
        return "Stage 2 Hypertension"
    elif sys >= 130 or dia >= 80:
        return "Stage 1 Hypertension"
    elif 120 <= sys <= 129 and dia < 80:
        return "Elevated"
    elif sys < 120 and dia < 80:
        return "Normal"
    else:
        return "Stage 1 Hypertension"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame with the 32 RAW_FEATURES columns and returns a
    DataFrame with the final 33 engineered columns the model expects
    (order doesn't matter here — the caller reindexes to feature_list.json).
    """
    df = df.copy()

    # --- Numeric engineered features ---
    df["PulsePressure"] = df["SystolicBP"] - df["DiastolicBP"]
    df["CholesterolRatio"] = df["CholesterolLDL"] / df["CholesterolHDL"]
    df["SymptomCount"] = df[SYMPTOM_COLS].sum(axis=1)
    df["ComorbidityCount"] = df[COMORBIDITY_COLS].sum(axis=1)
    df["CognitiveFunctionalScore"] = df["MMSE"] + df["FunctionalAssessment"] + df["ADL"]

    # --- BP category (ordinal) ---
    df["BPCategoryEncoded"] = df.apply(
        lambda row: BP_ORDER[_bp_category(row["SystolicBP"], row["DiastolicBP"])],
        axis=1
    )

    # --- Gender / Ethnicity one-hot (matches pd.get_dummies(..., drop_first=True)) ---
    df["Gender_Male"] = (df["Gender"] == 0).astype(int)
    df["Ethnicity_Asian"] = (df["Ethnicity"] == 2).astype(int)
    df["Ethnicity_Other"] = (df["Ethnicity"] == 3).astype(int)
    # Note: Ethnicity_Caucasian was dropped during pruning (zero importance),
    # and African American / Caucasian are implicitly represented by both
    # Ethnicity_Asian=0 and Ethnicity_Other=0.

    # --- Drop raw columns that only existed to feed engineered features,
    #     and drop Gender/Ethnicity now that we have their encoded versions ---
    drop_cols = [
        "Gender", "Ethnicity",
        "Diabetes", "Depression", "HeadInjury",
        "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks",
    ]
    df = df.drop(columns=drop_cols)

    return df
