"""
HealthSense Exploratory Data Analysis & Automated Health Insight Generator.
Generates 12 interactive Plotly visualizations, summary KPIs, and dynamically computes
epidemiological findings directly from actual dataset statistics.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.config import THEME


# ==============================================================================
# 1. SUMMARY STATISTICAL KPIS
# ==============================================================================

def compute_eda_kpis(df: pd.DataFrame, target_column: str = "Outcome") -> Dict[str, Any]:
    """Compute high-level statistical health indicators from the dataset."""
    total_patients = len(df)
    has_target = target_column in df.columns

    prevalence_pct = 0.0
    if has_target:
        # Check if binary (0, 1)
        positives = (df[target_column] == 1).sum()
        prevalence_pct = (positives / total_patients) * 100 if total_patients > 0 else 0.0

    avg_age = df["Age"].mean() if "Age" in df.columns else 0.0
    avg_bmi = df["BMI"].mean() if "BMI" in df.columns else 0.0
    avg_glucose = df["Glucose"].mean() if "Glucose" in df.columns else 0.0

    # High risk criteria (e.g. Glucose >= 140 or BMI >= 30)
    high_risk_count = 0
    if "Glucose" in df.columns and "BMI" in df.columns:
        high_risk_count = int(((df["Glucose"] >= 140) | (df["BMI"] >= 30.0)).sum())
    high_risk_pct = (high_risk_count / total_patients) * 100 if total_patients > 0 else 0.0

    return {
        "total_patients": total_patients,
        "disease_prevalence": round(prevalence_pct, 1),
        "avg_age": round(avg_age, 1),
        "avg_bmi": round(avg_bmi, 1),
        "avg_glucose": round(avg_glucose, 1),
        "high_risk_pct": round(high_risk_pct, 1),
    }


# ==============================================================================
# 2. DYNAMIC HEALTH INSIGHT GENERATOR
# ==============================================================================

def generate_health_insights(df: pd.DataFrame, target_column: str = "Outcome") -> List[Dict[str, str]]:
    """
    Generate automated clinical insights directly derived from actual data statistics.
    Never fabricates fake statistics.
    """
    insights = []
    n_patients = len(df)

    if n_patients == 0:
        return [{"title": "No Data", "text": "Dataset contains zero records."}]

    has_target = target_column in df.columns and df[target_column].nunique() == 2

    # Insight 1: Cohort Overview & Baseline Prevalence
    if has_target:
        pos_count = int((df[target_column] == 1).sum())
        pos_rate = (pos_count / n_patients) * 100
        insights.append({
            "title": "Cohort Composition & Baseline Prevalence",
            "text": (
                f"The analyzed dataset comprises {n_patients:,} patient records. "
                f"A total of {pos_count:,} individuals ({pos_rate:.1f}%) were diagnosed with diabetes "
                f"or flagged as high-risk, establishing the baseline epidemiological prevalence in this sample."
            )
        })

    # Insight 2: Plasma Glucose Differential
    if "Glucose" in df.columns and has_target:
        mean_g_pos = df[df[target_column] == 1]["Glucose"].mean()
        mean_g_neg = df[df[target_column] == 0]["Glucose"].mean()
        diff_g = mean_g_pos - mean_g_neg
        pct_g = (diff_g / mean_g_neg) * 100 if mean_g_neg > 0 else 0
        insights.append({
            "title": "Plasma Glucose Disparity",
            "text": (
                f"Patients with diabetes displayed a substantially elevated average oral glucose level "
                f"of {mean_g_pos:.1f} mg/dL, compared to {mean_g_neg:.1f} mg/dL in the non-diabetic cohort "
                f"(a {pct_g:.1f}% relative elevation, +{diff_g:.1f} mg/dL difference)."
            )
        })

    # Insight 3: Adiposity & BMI Impact
    if "BMI" in df.columns and has_target:
        mean_bmi_pos = df[df[target_column] == 1]["BMI"].mean()
        mean_bmi_neg = df[df[target_column] == 0]["BMI"].mean()
        diff_bmi = mean_bmi_pos - mean_bmi_neg
        obese_pos = (df[df[target_column] == 1]["BMI"] >= 30.0).mean() * 100
        insights.append({
            "title": "Body Mass Index & Adiposity Risk",
            "text": (
                f"The diabetic group exhibited an average BMI of {mean_bmi_pos:.1f} kg/m² vs "
                f"{mean_bmi_neg:.1f} kg/m² in the healthy cohort. Notably, {obese_pos:.1f}% of diabetic "
                f"patients met the clinical threshold for obesity (BMI ≥ 30 kg/m²)."
            )
        })

    # Insight 4: Age Gradient & Correlation
    if "Age" in df.columns and has_target:
        mean_age_pos = df[df[target_column] == 1]["Age"].mean()
        mean_age_neg = df[df[target_column] == 0]["Age"].mean()
        insights.append({
            "title": "Age Demographics & Vulnerability",
            "text": (
                f"Age is strongly associated with disease onset: diabetic patients averaged "
                f"{mean_age_pos:.1f} years, while non-diabetic individuals averaged {mean_age_neg:.1f} years. "
                f"This reflects an upward risk trajectory associated with age-related insulin resistance."
            )
        })

    # Insight 5: Top Statistical Predictors (Correlation)
    if has_target:
        numeric_df = df.select_dtypes(include=["number"])
        if target_column in numeric_df.columns:
            corrs = numeric_df.corr()[target_column].drop(target_column).sort_values(ascending=False)
            top_3 = corrs.head(3)
            top_str = ", ".join([f"{col} (r = {val:.2f})" for col, val in top_3.items()])
            insights.append({
                "title": "Top Linear Statistical Predictors",
                "text": (
                    f"Evaluating Pearson correlation with the disease outcome identifies the strongest "
                    f"linear associations as: {top_str}. These features represent primary targets for "
                    f"classification algorithms."
                )
            })

    return insights


# ==============================================================================
# 3. INTERACTIVE PLOTLY VISUALIZATIONS
# ==============================================================================

def plot_disease_distribution(df: pd.DataFrame, target_column: str = "Outcome") -> go.Figure:
    """1. Disease Distribution Donut Chart."""
    if target_column not in df.columns:
        return go.Figure()
    counts = df[target_column].value_counts().reset_index()
    counts.columns = [target_column, "Count"]
    counts["Label"] = counts[target_column].map({0: "Non-Diabetic (0)", 1: "Diabetic (1)"}).fillna(counts[target_column].astype(str))

    fig = px.pie(
        counts,
        names="Label",
        values="Count",
        hole=0.45,
        color="Label",
        color_discrete_map={"Non-Diabetic (0)": THEME["primary"], "Diabetic (1)": THEME["danger"]},
        title="Disease Outcome Distribution"
    )
    fig.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20), font=dict(family="Inter"))
    return fig


def plot_age_distribution(df: pd.DataFrame, target_column: str = "Outcome") -> go.Figure:
    """2. Age Distribution with Outcome Overlay."""
    if "Age" not in df.columns:
        return go.Figure()
    color_col = target_column if target_column in df.columns else None
    fig = px.histogram(
        df,
        x="Age",
        color=color_col,
        barmode="overlay",
        nbins=25,
        opacity=0.75,
        color_discrete_sequence=[THEME["primary"], THEME["danger"]],
        title="Age Distribution Stratified by Disease Status"
    )
    fig.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20), font=dict(family="Inter"))
    return fig


def plot_feature_by_outcome_box(df: pd.DataFrame, feature: str, target_column: str = "Outcome") -> go.Figure:
    """3, 4, 5. Box/Violin plot comparing physiological feature by outcome."""
    if feature not in df.columns or target_column not in df.columns:
        return go.Figure()

    df_plot = df.copy()
    df_plot["Status"] = df_plot[target_column].map({0: "Non-Diabetic", 1: "Diabetic"}).fillna(df_plot[target_column].astype(str))

    fig = px.box(
        df_plot,
        x="Status",
        y=feature,
        color="Status",
        points="outliers",
        color_discrete_map={"Non-Diabetic": THEME["primary"], "Diabetic": THEME["danger"]},
        title=f"{feature} Distribution by Outcome"
    )
    fig.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20), showlegend=False, font=dict(family="Inter"))
    return fig


def plot_bivariate_scatter(df: pd.DataFrame, x_col: str, y_col: str, target_column: str = "Outcome") -> go.Figure:
    """6, 7, 8, 9. Interactive scatter comparing two health features against risk."""
    if x_col not in df.columns or y_col not in df.columns:
        return go.Figure()

    df_plot = df.copy()
    color_col = None
    color_map = None
    if target_column in df.columns:
        df_plot["Outcome_Label"] = df_plot[target_column].map({0: "Non-Diabetic", 1: "Diabetic"}).fillna(df_plot[target_column].astype(str))
        color_col = "Outcome_Label"
        color_map = {"Non-Diabetic": THEME["primary"], "Diabetic": THEME["danger"]}

    fig = px.scatter(
        df_plot,
        x=x_col,
        y=y_col,
        color=color_col,
        color_discrete_map=color_map,
        opacity=0.7,
        title=f"{x_col} vs {y_col} Relationship",
        trendline="ols" if len(df_plot) > 5 else None
    )
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), font=dict(family="Inter"))
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """10. Interactive Pearson Correlation Heatmap."""
    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr().round(2)

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        text=corr.values,
        texttemplate="%{text}",
        hoverongaps=False
    ))
    fig.update_layout(
        title="Feature Correlation Matrix (Pearson r)",
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family="Inter")
    )
    return fig


def plot_all_feature_distributions(df: pd.DataFrame) -> go.Figure:
    """11. Normalized Boxplot Comparison across all numerical features."""
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols:
        return go.Figure()

    # Min-max scale purely for comparative visual spread
    norm_df = (df[num_cols] - df[num_cols].min()) / (df[num_cols].max() - df[num_cols].min() + 1e-6)

    fig = go.Figure()
    for col in num_cols:
        fig.add_trace(go.Box(y=norm_df[col], name=col, boxmean=True))

    fig.update_layout(
        title="Comparative Feature Spreads (Normalized [0, 1])",
        height=380,
        yaxis_title="Normalized Range",
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family="Inter")
    )
    return fig


def plot_risk_profile_radar(df: pd.DataFrame, target_column: str = "Outcome") -> go.Figure:
    """12. Radar Chart: Mean Normalized Profile (Diabetic vs Non-Diabetic)."""
    if target_column not in df.columns or df[target_column].nunique() != 2:
        return go.Figure()

    numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != target_column]
    if len(numeric_cols) < 3:
        return go.Figure()

    # Normalize columns to 0-1
    min_vals = df[numeric_cols].min()
    max_vals = df[numeric_cols].max()
    norm_df = (df[numeric_cols] - min_vals) / (max_vals - min_vals + 1e-6)
    norm_df[target_column] = df[target_column]

    mean_neg = norm_df[norm_df[target_column] == 0][numeric_cols].mean().tolist()
    mean_pos = norm_df[norm_df[target_column] == 1][numeric_cols].mean().tolist()

    # Close the radar loop
    radar_cols = numeric_cols + [numeric_cols[0]]
    mean_neg = mean_neg + [mean_neg[0]]
    mean_pos = mean_pos + [mean_pos[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=mean_neg,
        theta=radar_cols,
        fill='toself',
        name='Non-Diabetic Cohort',
        line=dict(color=THEME["primary"])
    ))
    fig.add_trace(go.Scatterpolar(
        r=mean_pos,
        theta=radar_cols,
        fill='toself',
        name='Diabetic Cohort',
        line=dict(color=THEME["danger"])
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Multivariate Risk Profile Comparison (Mean Normalized)",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family="Inter")
    )
    return fig
