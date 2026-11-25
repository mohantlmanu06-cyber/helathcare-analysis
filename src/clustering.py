"""
HealthSense Unsupervised Clustering Module.
Implements K-Means clustering, automated Elbow & Silhouette analysis,
cluster persona profiling, and 2D/3D interactive visual exploration.
"""

from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from utils.config import RANDOM_STATE, THEME


def compute_clustering_metrics(df: pd.DataFrame, feature_cols: List[str], max_k: int = 7) -> Dict[str, Any]:
    """Compute WCSS (Inertia) and Silhouette scores for k = 2 to max_k."""
    clean_df = df[feature_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean_df)

    k_values = list(range(2, max_k + 1))
    inertias = []
    silhouettes = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        sil = silhouette_score(X_scaled, kmeans.labels_)
        silhouettes.append(sil)

    # Optimal k by highest silhouette
    best_k_idx = int(np.argmax(silhouettes))
    optimal_k = k_values[best_k_idx]

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouettes": [round(s, 4) for s in silhouettes],
        "optimal_k": optimal_k,
        "X_scaled": X_scaled,
        "clean_df": clean_df
    }


def perform_kmeans_clustering(df: pd.DataFrame, feature_cols: List[str], n_clusters: int = 3) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fit K-Means with chosen k and compute cluster profiling summaries.
    Returns clustered DataFrame and summary profile dictionary.
    """
    clean_df = df.copy().dropna(subset=feature_cols).reset_index(drop=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean_df[feature_cols])

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    clean_df["Cluster"] = [f"Cluster {c+1}" for c in clusters]

    # Cluster Profiles (Means)
    profiles = clean_df.groupby("Cluster")[feature_cols].mean().round(2)
    counts = clean_df["Cluster"].value_counts().to_dict()

    # Dynamic Persona Generation
    descriptions = {}
    for c_name in sorted(clean_df["Cluster"].unique()):
        row = profiles.loc[c_name]
        traits = []
        if "Glucose" in row:
            if row["Glucose"] > 135:
                traits.append("High Plasma Glucose")
            elif row["Glucose"] < 110:
                traits.append("Low/Optimal Glucose")
            else:
                traits.append("Moderate Glucose")

        if "BMI" in row:
            if row["BMI"] >= 32:
                traits.append("Elevated Adiposity (Obese)")
            elif row["BMI"] < 25:
                traits.append("Normal BMI")
            else:
                traits.append("Overweight Profile")

        if "Age" in row:
            if row["Age"] > 45:
                traits.append("Older Demographic")
            else:
                traits.append("Younger Cohort")

        descriptions[c_name] = {
            "size": counts.get(c_name, 0),
            "percentage": round(counts.get(c_name, 0) / len(clean_df) * 100, 1),
            "summary": " • ".join(traits) if traits else "Balanced physiological measures",
            "means": row.to_dict()
        }

    clustering_summary = {
        "n_clusters": n_clusters,
        "profiles": profiles,
        "descriptions": descriptions,
        "silhouette": round(silhouette_score(X_scaled, clusters), 4),
        "cluster_centers": kmeans.cluster_centers_,
    }

    return clean_df, clustering_summary


def plot_elbow_and_silhouette(metrics: Dict[str, Any]) -> go.Figure:
    """Plot twin-axis Elbow Method (Inertia) and Silhouette Scores."""
    fig = go.Figure()

    # Inertia line
    fig.add_trace(go.Scatter(
        x=metrics["k_values"],
        y=metrics["inertias"],
        mode="lines+markers",
        name="WCSS (Inertia)",
        line=dict(color=THEME["primary"], width=2.5),
        yaxis="y1"
    ))

    # Silhouette line
    fig.add_trace(go.Scatter(
        x=metrics["k_values"],
        y=metrics["silhouettes"],
        mode="lines+markers",
        name="Silhouette Score",
        line=dict(color=THEME["danger"], width=2.5, dash="dot"),
        yaxis="y2"
    ))

    fig.update_layout(
        title="Optimal Cluster Evaluation: Elbow Curve & Silhouette Scores",
        xaxis=dict(title=dict(text="Number of Clusters (k)")),
        yaxis=dict(
            title=dict(text="Inertia (WCSS)", font=dict(color=THEME["primary"])),
            tickfont=dict(color=THEME["primary"])
        ),
        yaxis2=dict(
            title=dict(text="Silhouette Score", font=dict(color=THEME["danger"])),
            tickfont=dict(color=THEME["danger"]),
            overlaying="y",
            side="right"
        ),
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family="IBM Plex Sans"),
        legend=dict(x=0.55, y=0.95)
    )
    return fig


def plot_clusters_scatter(df_clustered: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    """Plot 2D interactive scatter of clusters."""
    fig = px.scatter(
        df_clustered,
        x=x_col,
        y=y_col,
        color="Cluster",
        color_discrete_sequence=[THEME["primary"], THEME["warning"], THEME["danger"], THEME["accent"]],
        title=f"Patient Clusters: {x_col} vs {y_col}",
        hover_data=[c for c in ["Age", "BMI", "Glucose", "BloodPressure"] if c in df_clustered.columns]
    )
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20), font=dict(family="Inter"))
    return fig
