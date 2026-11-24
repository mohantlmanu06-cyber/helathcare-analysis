"""
HealthSense Explainability Engine (XAI).
Provides global model feature importances (Tree MDI & Permutation Importance)
and local individual patient-level risk factor breakdowns with waterfall attributions.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.inspection import permutation_importance
from utils.config import THEME


def get_global_feature_importance(model: Any, feature_names: List[str],
                                  X_test: Optional[np.ndarray] = None,
                                  y_test: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Extract global feature importance.
    Uses native tree MDI if available; falls back to Permutation Importance or coefficients.
    """
    importances = None

    # Tree-based models (Random Forest, Decision Tree)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    # Linear models (Logistic Regression)
    elif hasattr(model, "coef_"):
        coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        importances = np.abs(coef)
    # Fallback to Permutation Importance for SVM/KNN
    elif X_test is not None and y_test is not None:
        perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
        importances = np.maximum(perm.importances_mean, 0.0)
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)

    # Normalize to percentages
    sum_imp = np.sum(importances)
    norm_imp = (importances / sum_imp) if sum_imp > 0 else importances

    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": norm_imp,
        "Importance_Pct": [f"{v * 100:.1f}%" for v in norm_imp]
    }).sort_values(by="Importance", ascending=True).reset_index(drop=True)

    return df_imp


def plot_feature_importance_bar(df_imp: pd.DataFrame, model_name: str = "Model") -> go.Figure:
    """Plot horizontal bar chart of global feature importances."""
    fig = px.bar(
        df_imp,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance_Pct",
        title=f"Global Feature Importance: {model_name}",
        color="Importance",
        color_continuous_scale=[[0, THEME["primary_light"]], [1, THEME["primary"]]]
    )
    fig.update_layout(
        xaxis_title="Relative Importance Weight",
        yaxis_title="Health Feature",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family="Inter"),
        showlegend=False
    )
    return fig


def explain_single_patient_prediction(patient_dict: Dict[str, float],
                                      global_importance_df: pd.DataFrame,
                                      baseline_medians: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Compute local feature impact attributions for a single patient relative to healthy baseline.
    Estimates directional contribution (Risk Elevating vs Risk Protective).
    """
    imp_map = dict(zip(global_importance_df["Feature"], global_importance_df["Importance"]))
    attributions = []

    for feat, val in patient_dict.items():
        if feat not in baseline_medians:
            continue
        base = baseline_medians[feat]
        weight = imp_map.get(feat, 0.1)

        # Relative deviation from healthy baseline
        dev = (val - base) / (base + 1e-6)

        # For diabetes risk, higher values in glucose, BMI, age, insulin elevate risk
        impact_score = dev * weight

        direction = "ELEVATING" if impact_score > 0 else "PROTECTIVE"
        level = "High" if abs(impact_score) > 0.08 else ("Moderate" if abs(impact_score) > 0.03 else "Low")

        attributions.append({
            "Feature": feat,
            "Patient_Value": val,
            "Baseline_Median": round(base, 2),
            "Impact_Score": round(float(impact_score), 4),
            "Direction": direction,
            "Impact_Level": f"{level} contribution",
            "Absolute_Impact": abs(float(impact_score))
        })

    # Sort by absolute magnitude of contribution
    attributions.sort(key=lambda x: x["Absolute_Impact"], reverse=True)
    return attributions


def plot_local_explanation_diverging(attributions: List[Dict[str, Any]]) -> go.Figure:
    """
    Plot diverging horizontal bar chart of patient-specific factor impacts.
    Green = protective (reduces risk), Red = elevating (increases risk).
    """
    df_local = pd.DataFrame(attributions).sort_values(by="Impact_Score", ascending=True)

    colors = [THEME["danger"] if score > 0 else THEME["success"] for score in df_local["Impact_Score"]]

    fig = go.Figure(go.Bar(
        x=df_local["Impact_Score"],
        y=df_local["Feature"],
        orientation='h',
        marker_color=colors,
        text=[f"{lvl} ({sc:+.3f})" for lvl, sc in zip(df_local["Impact_Level"], df_local["Impact_Score"])],
        textposition="auto"
    ))

    fig.update_layout(
        title="Patient-Specific Factor Attributions (Local Impact vs Cohort Baseline)",
        xaxis_title="Directional Risk Contribution (← Protective | Elevating Risk →)",
        yaxis_title="Biomarker / Feature",
        height=360,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family="Inter")
    )
    return fig


def generate_natural_language_explanation(probability: float, category: str,
                                          attributions: List[Dict[str, Any]]) -> str:
    """Generate plain-English summary of patient risk factors."""
    elevating = [a for a in attributions if a["Direction"] == "ELEVATING"]
    protective = [a for a in attributions if a["Direction"] == "PROTECTIVE"]

    top_elevating_names = [f"**{a['Feature']}** ({a['Patient_Value']})" for a in elevating[:2]]
    top_protective_names = [f"**{a['Feature']}** ({a['Patient_Value']})" for a in protective[:2]]

    summary = f"Based on the trained classification model, the patient is classified as **{category} RISK** with an estimated probability of **{probability:.1%}**.\n\n"

    if top_elevating_names:
        summary += f"- **Primary Risk Drivers**: The chief factors elevating the risk probability are {', '.join(top_elevating_names)} due to significant upward divergence from cohort normal baseline.\n"
    if top_protective_names:
        summary += f"- **Protective / Favorable Factors**: Conversely, {', '.join(top_protective_names)} remain within favorable ranges, mitigating an even higher risk trajectory.\n"

    summary += "\n*Note: This attribution illustrates statistical influence within the predictive model and does not establish biological causality.*"
    return summary
