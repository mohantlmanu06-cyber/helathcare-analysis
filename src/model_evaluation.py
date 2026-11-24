"""
HealthSense Model Evaluation and Diagnostic Metrics Module.
Generates confusion matrices, ROC curves, Precision-Recall curves, and clinical safety interpretations.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from utils.config import THEME


def compute_confusion_matrix_details(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute TN, FP, FN, TP and derived clinical diagnostic rates."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0    # PPV
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0          # Negative Predictive Value
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "raw_cm": cm,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "sensitivity_recall": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "precision": round(float(precision), 4),
        "npv": round(float(npv), 4),
        "accuracy": round(float(accuracy), 4),
    }


def plot_interactive_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> go.Figure:
    """Render an annotated, interactive confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    z_text = [
        [f"TN: {tn}<br>(Healthy correctly identified)", f"FP: {fp}<br>(False Alarm)"],
        [f"FN: {fn}<br>(MISSED DIAGNOSIS)", f"TP: {tp}<br>(True Diabetic)"]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted Healthy", "Predicted Diabetic"],
        y=["Actual Healthy", "Actual Diabetic"],
        text=z_text,
        texttemplate="%{text}",
        colorscale=[[0, "#E0F2F1"], [1, THEME["primary"]]],
        showscale=False
    ))

    fig.update_layout(
        title=f"Confusion Matrix: {model_name}",
        height=340,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family="Inter")
    )
    return fig


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, model_name: str = "Model") -> go.Figure:
    """Plot Receiver Operating Characteristic (ROC) curve with AUC indicator."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {roc_auc:.3f})',
        line=dict(color=THEME["primary"], width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Chance (AUC = 0.500)',
        line=dict(color=THEME["muted"], dash='dash')
    ))

    fig.update_layout(
        title=f"ROC Curve: {model_name}",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity / Recall)",
        height=360,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family="Inter"),
        legend=dict(x=0.55, y=0.1)
    )
    return fig


def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, model_name: str = "Model") -> go.Figure:
    """Plot Precision-Recall curve with average precision."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = np.mean(y_true)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name=f'{model_name} (AP = {ap:.3f})',
        line=dict(color=THEME["danger"], width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[baseline, baseline],
        mode='lines',
        name=f'Baseline Prevalence ({baseline:.1%})',
        line=dict(color=THEME["muted"], dash='dash')
    ))

    fig.update_layout(
        title=f"Precision-Recall Curve: {model_name}",
        xaxis_title="Recall (Sensitivity)",
        yaxis_title="Precision (Positive Predictive Value)",
        height=360,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family="Inter"),
        legend=dict(x=0.55, y=0.1)
    )
    return fig


CLINICAL_RECALL_IMPORTANCE_TEXT = """
### Clinical Significance of Recall & False Negatives in Disease Screening
In medical screening systems like **HealthSense**, metric priorities differ fundamentally from standard classification problems:

1. **The Cost of False Negatives (FN)**:
   - A False Negative occurs when a patient with underlying or early-stage diabetes is erroneously classified as **LOW RISK**.
   - **Clinical Consequence**: The individual delays lifestyle changes, glycemic monitoring, and preventative care. Over months or years, unmanaged hyperglycemia leads to irreversible microvascular and macrovascular complications (e.g., neuropathy, retinopathy, nephropathy, and cardiovascular disease).
   - Therefore, **high Recall (Sensitivity)** is paramount in health screening to minimize missed cases.

2. **The Role of False Positives (FP)**:
   - A False Positive occurs when a healthy person is categorized as **HIGH RISK**.
   - **Clinical Consequence**: The patient is referred for a low-cost, definitive confirmatory laboratory test (e.g., Fasting Plasma Glucose or HbA1c test). While it causes temporary anxiety and modest administrative cost, it causes no biological harm.

3. **Conclusion for Model Selection**:
   - A model with 85% Accuracy but 55% Recall is medically dangerous.
   - We prioritize models maintaining high Recall and high ROC-AUC while preserving acceptable Precision.
"""
