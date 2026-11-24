"""
HealthSense Prediction Engine & What-If Risk Scenario Simulator.
Executes patient-level inferences, applies clinical risk categorization,
and computes dynamic comparative counterfactuals.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import joblib
from utils.config import (
    BEST_MODEL_PATH, PREPROCESSOR_PATH, RISK_CATEGORIES,
    CANONICAL_FEATURES, DEFAULT_DATASET_PATH
)
from utils.database import log_prediction


def get_risk_category(probability: float) -> Tuple[str, Dict[str, Any]]:
    """Determine risk category and associated styling based on predicted probability."""
    for category, meta in RISK_CATEGORIES.items():
        if meta["min"] <= probability < meta["max"] or (category == "HIGH" and probability >= 1.0):
            return category, meta
    return "HIGH", RISK_CATEGORIES["HIGH"]


def get_cohort_baseline_medians() -> Dict[str, float]:
    """Retrieve healthy cohort median values for feature comparisons."""
    try:
        df = pd.read_csv(DEFAULT_DATASET_PATH)
        # Healthy cohort medians
        df_healthy = df[df["Outcome"] == 0] if "Outcome" in df.columns else df
        medians = {}
        for feat in CANONICAL_FEATURES:
            if feat in df_healthy.columns:
                series = df_healthy[feat].replace(0, np.nan) if feat != "Pregnancies" else df_healthy[feat]
                medians[feat] = float(series.median())
        return medians
    except Exception:
        # Fallback default clinical normal values
        return {
            "Pregnancies": 1.0,
            "Glucose": 100.0,
            "BloodPressure": 72.0,
            "SkinThickness": 20.0,
            "Insulin": 80.0,
            "BMI": 24.5,
            "DiabetesPedigreeFunction": 0.35,
            "Age": 28.0,
        }


def predict_patient_risk(patient_dict: Dict[str, float],
                         model: Optional[Any] = None,
                         preprocessor: Optional[Any] = None,
                         model_name: str = "Trained Classifier",
                         save_to_db: bool = True) -> Dict[str, Any]:
    """
    Execute risk prediction for a single patient input profile.
    Returns probability, risk category, model info, and attribution dictionary.
    """
    # Load model and preprocessor if not passed
    if model is None:
        if not BEST_MODEL_PATH.exists():
            raise FileNotFoundError("Model artifact not found. Please train models in the Data Mining module first.")
        model = joblib.load(BEST_MODEL_PATH)

    if preprocessor is None:
        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError("Preprocessor artifact not found. Please train models in the Data Mining module first.")
        preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Prepare features matching expected order
    features_ordered = [feat for feat in CANONICAL_FEATURES if feat in patient_dict]
    input_df = pd.DataFrame([patient_dict])[features_ordered]

    # Preprocess (Impute & Scale using fitted training parameters)
    X_processed = preprocessor.transform(input_df)

    # Predict Probability
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X_processed)[0, 1])
    else:
        # Decision function sigmoid mapping for linear SVM
        decision = float(model.decision_function(X_processed)[0])
        prob = float(1.0 / (1.0 + np.exp(-decision)))

    prob = max(0.0, min(1.0, prob))
    category, cat_meta = get_risk_category(prob)

    # Save to SQLite
    prediction_id = None
    if save_to_db:
        prediction_id = log_prediction(
            algorithm=model_name,
            probability=round(prob, 4),
            category=category,
            patient_features=patient_dict,
            notes="Standard prediction"
        )

    return {
        "prediction_id": prediction_id,
        "probability": prob,
        "probability_percent": round(prob * 100, 1),
        "category": category,
        "category_label": cat_meta["label"],
        "category_color": cat_meta["color"],
        "badge": cat_meta["badge"],
        "model_used": model_name,
        "patient_inputs": patient_dict
    }


def simulate_what_if_scenario(baseline_inputs: Dict[str, float],
                              scenario_inputs: Dict[str, float],
                              model: Any,
                              preprocessor: Any,
                              model_name: str = "Trained Classifier") -> Dict[str, Any]:
    """
    Execute comparative counterfactual scenario simulation.
    Compares baseline patient state vs modified hypothetical lifestyle/biomarker intervention.
    """
    base_res = predict_patient_risk(baseline_inputs, model=model, preprocessor=preprocessor,
                                    model_name=model_name, save_to_db=False)
    scen_res = predict_patient_risk(scenario_inputs, model=model, preprocessor=preprocessor,
                                    model_name=model_name, save_to_db=False)

    abs_diff = scen_res["probability"] - base_res["probability"]
    abs_diff_pct = abs_diff * 100

    if base_res["probability"] > 0:
        rel_diff_pct = ((scen_res["probability"] - base_res["probability"]) / base_res["probability"]) * 100
    else:
        rel_diff_pct = 0.0

    return {
        "baseline": base_res,
        "scenario": scen_res,
        "absolute_change_percent": round(abs_diff_pct, 1),
        "relative_change_percent": round(rel_diff_pct, 1),
        "direction": "REDUCTION" if abs_diff < 0 else ("INCREASE" if abs_diff > 0 else "NO CHANGE"),
        "disclaimer": (
            "This counterfactual analysis is a statistical model-based scenario simulation. "
            "It does not constitute a guaranteed clinical treatment outcome."
        )
    }
