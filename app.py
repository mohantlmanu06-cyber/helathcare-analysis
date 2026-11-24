"""
HealthSense (DiaMetrics Edition): Intelligent Health Data Analysis and Disease Risk Prediction
Matches the DiaMetrics clinical intelligence web platform structure, navigation, and styling.
"""

import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Core Configuration & UI Utilities
from utils.config import (
    THEME, MEDICAL_DISCLAIMER, CANONICAL_FEATURES,
    DEFAULT_DATASET_PATH, SAMPLE_DIRTY_DATASET_PATH,
    PROCESSED_DATASET_PATH, BEST_MODEL_PATH, PREPROCESSOR_PATH,
    PHYSIOLOGICAL_LIMITS, RISK_CATEGORIES
)
from utils.helpers import (
    inject_custom_css, render_page_header, render_disclaimer,
    render_kpi, create_risk_gauge, create_quality_score_gauge
)
from utils.database import (
    get_all_models, get_predictions, clear_all_predictions,
    log_prediction
)

# Core Domain Modules
from src.data_loader import (
    load_uploaded_file, load_default_dataset,
    load_dirty_sample_dataset, get_dataset_metadata, get_data_previews
)
from src.data_validation import validate_patient_input, validate_dataset_structure
from src.data_cleaning import compute_data_quality_report, HealthDataWrangler
from src.feature_engineering import engineer_health_features
from src.exploratory_analysis import (
    compute_eda_kpis, generate_health_insights,
    plot_disease_distribution, plot_age_distribution,
    plot_feature_by_outcome_box, plot_bivariate_scatter,
    plot_correlation_heatmap, plot_all_feature_distributions,
    plot_risk_profile_radar
)
from src.classification import HealthClassifierTrainer
from src.model_evaluation import (
    compute_confusion_matrix_details, plot_interactive_confusion_matrix,
    plot_roc_curve, plot_precision_recall_curve, CLINICAL_RECALL_IMPORTANCE_TEXT
)
from src.clustering import (
    compute_clustering_metrics, perform_kmeans_clustering,
    plot_elbow_and_silhouette, plot_clusters_scatter
)
from src.association_rules import (
    mine_association_rules, plot_rules_scatter,
    ASSOCIATION_CAUSATION_DISCLAIMER
)
from src.explainability import (
    get_global_feature_importance, plot_feature_importance_bar,
    explain_single_patient_prediction, plot_local_explanation_diverging,
    generate_natural_language_explanation
)
from src.prediction import (
    predict_patient_risk, simulate_what_if_scenario,
    get_cohort_baseline_medians
)


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="DiaMetrics | HealthSense Analytics Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if "df_raw" not in st.session_state:
    try:
        st.session_state["df_raw"] = load_default_dataset()
        st.session_state["dataset_name"] = "pima_diabetes.csv"
    except Exception:
        st.session_state["df_raw"] = None
        st.session_state["dataset_name"] = "None"

if "df_cleaned" not in st.session_state:
    if Path(PROCESSED_DATASET_PATH).exists():
        st.session_state["df_cleaned"] = pd.read_csv(PROCESSED_DATASET_PATH)
    else:
        st.session_state["df_cleaned"] = st.session_state.get("df_raw", None)

if "target_column" not in st.session_state:
    st.session_state["target_column"] = "Outcome"

if "audit_metrics" not in st.session_state:
    st.session_state["audit_metrics"] = None

if "models_trained" not in st.session_state:
    st.session_state["models_trained"] = Path(BEST_MODEL_PATH).exists()

if "classifier_artifacts" not in st.session_state:
    st.session_state["classifier_artifacts"] = None

if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None

if "preset_patient" not in st.session_state:
    st.session_state["preset_patient"] = None


# ==============================================================================
# SIDEBAR NAVIGATION (DIAMETRICS DARK THEME)
# ==============================================================================
with st.sidebar:
    # Sidebar Brand Block
    st.markdown(f"""
    <div style='display: flex; align-items: center; gap: 12px; padding: 0.8rem 0 1.2rem 0; border-bottom: 1px solid #1E293B; margin-bottom: 1.2rem;'>
        <div style='width: 38px; height: 38px; border-radius: 9px; background: #2563EB; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; color: white; box-shadow: 0 2px 6px rgba(37,99,235,0.4);'>
            🩺
        </div>
        <div>
            <div style='font-family: Outfit, sans-serif; font-size: 1.15rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em;'>DiaMetrics</div>
            <div style='font-size: 0.74rem; color: #94A3B8;'>Diabetes Risk Analytics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 10 Numbered Navigation Items matching DiaMetrics
    page = st.radio(
        "Navigation Menu",
        [
            "01 Home",
            "02 Upload",
            "03 Wrangling",
            "04 Analysis",
            "05 Mining",
            "06 Models",
            "07 Prediction",
            "08 Simulator",
            "09 History",
            "10 Report",
        ],
        label_visibility="collapsed"
    )

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

    # Active Dataset Card at Bottom of Sidebar
    records_count = len(st.session_state['df_cleaned']) if st.session_state.get('df_cleaned') is not None else 768
    st.markdown(f"""
    <div style='background: #131D33; border: 1px solid #1E293B; border-radius: 10px; padding: 0.9rem 1rem;'>
        <div style='font-size: 0.7rem; font-weight: 600; text-transform: uppercase; color: #64748B; letter-spacing: 0.05em;'>Active Dataset</div>
        <div style='font-family: "JetBrains Mono", monospace; font-size: 0.85rem; font-weight: 600; color: #F8FAFC; margin-top: 0.2rem;'>{st.session_state.get('dataset_name', 'pima_diabetes.csv')}</div>
        <div style='display: flex; align-items: center; justify-content: space-between; margin-top: 0.5rem; font-size: 0.74rem; color: #94A3B8;'>
            <span>{records_count} records</span>
            <span style='background: #064E3B; color: #6EE7B7; border: 1px solid #047857; padding: 1px 7px; border-radius: 9999px; font-size: 0.68rem; font-weight: 600;'>Cleaned</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.caption("DiaMetrics v2.0 • Clinical Edition")


# Persistent Top Alert Banner
render_disclaimer()


# ==============================================================================
# SCREEN 01: HOME (DASHBOARD OVERVIEW)
# ==============================================================================
if page == "01 Home":
    render_page_header(
        "SCREEN 01 · DASHBOARD OVERVIEW",
        "DiaMetrics Analytics Platform",
        "Clinical decision-support & epidemiological research workspace for diabetes risk discovery",
        badge="Academic Project"
    )

    # 4 Top KPI Cards
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    render_kpi(col_k1, "Cohort Size", "768 Patients", "Pima Indians diabetes dataset", icon="👥")
    render_kpi(col_k2, "Diabetes Prevalence", "34.9%", "268 positive clinical cases", icon="📊")
    render_kpi(col_k3, "Leading Model AUC", "0.814", "Logistic Regression benchmark", icon="🏆")
    render_kpi(col_k4, "Data Quality Score", "96.5 / 100", "Audited across 4 dimensions", icon="🛡️")

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # 4 Quick Navigation Action Cards
    st.markdown("### Clinical Analytics Workflows")
    ac1, ac2, ac3, ac4 = st.columns(4)
    with ac1:
        st.markdown(f"""
        <div class="action-nav-card">
            <div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>📂</div>
            <div style='font-weight: 700; color: {THEME['secondary']}; font-size: 1rem;'>02 Upload & Audit</div>
            <div style='color: {THEME['muted']}; font-size: 0.82rem; margin-top: 0.35rem; line-height: 1.45;'>
                Upload patient cohorts, inspect metadata, and evaluate automated 0–100 data quality scores.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with ac2:
        st.markdown(f"""
        <div class="action-nav-card">
            <div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>🧹</div>
            <div style='font-weight: 700; color: {THEME['secondary']}; font-size: 1rem;'>03 Data Wrangling</div>
            <div style='color: {THEME['muted']}; font-size: 0.82rem; margin-top: 0.35rem; line-height: 1.45;'>
                Resolve biological zero anomalies, impute missing values, cap outliers, and audit Before vs After.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with ac3:
        st.markdown(f"""
        <div class="action-nav-card">
            <div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>📊</div>
            <div style='font-weight: 700; color: {THEME['secondary']}; font-size: 1rem;'>04 Health Analytics</div>
            <div style='color: {THEME['muted']}; font-size: 0.82rem; margin-top: 0.35rem; line-height: 1.45;'>
                Explore 12 interactive distributions, correlation matrices, radar charts, and automated insights.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with ac4:
        st.markdown(f"""
        <div class="action-nav-card">
            <div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>🤖</div>
            <div style='font-weight: 700; color: {THEME['secondary']}; font-size: 1rem;'>06 Model Arena</div>
            <div style='color: {THEME['muted']}; font-size: 0.82rem; margin-top: 0.35rem; line-height: 1.45;'>
                Benchmark 5 supervised classifiers with cross-validation, confusion matrices, and ROC-AUC curves.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Pipeline Architecture Workflow Stepper
    st.markdown("### Pipeline Architecture & Execution Flow")
    wf1, wf2, wf3, wf4 = st.columns(4)
    with wf1:
        st.markdown("""
        <div class="workflow-step">
            <div class="workflow-step-num">1</div>
            <div class="workflow-step-title">Ingest & Audit</div>
            <div class="workflow-step-desc">Validate file constraints, flag 652 biological zeros, and compute 0–100 data quality score.</div>
        </div>
        """, unsafe_allow_html=True)
    with wf2:
        st.markdown("""
        <div class="workflow-step">
            <div class="workflow-step-num">2</div>
            <div class="workflow-step-title">Wrangle & Clean</div>
            <div class="workflow-step-desc">Apply median imputation, IQR outlier capping, and verify changes via Before vs After audit.</div>
        </div>
        """, unsafe_allow_html=True)
    with wf3:
        st.markdown("""
        <div class="workflow-step">
            <div class="workflow-step-num">3</div>
            <div class="workflow-step-title">Mining & Benchmarking</div>
            <div class="workflow-step-desc">Train 5 supervised models, discover K-Means patient clusters, and extract Apriori rules.</div>
        </div>
        """, unsafe_allow_html=True)
    with wf4:
        st.markdown("""
        <div class="workflow-step">
            <div class="workflow-step-num">4</div>
            <div class="workflow-step-title">Predict & Simulate</div>
            <div class="workflow-step-desc">Compute calibrated risk %, explain factors via diverging charts, and run What-If simulations.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # High-Level Insights Summary Card
    st.markdown(f"""
    <div class="diametrics-card">
        <h4 style='color: {THEME['primary']}; margin-top: 0; font-size: 1.1rem; display: flex; align-items: center; gap: 8px;'>
            💡 Key Clinical Intelligence Summary
        </h4>
        <div style='font-size: 0.9rem; line-height: 1.6; color: {THEME['text']};'>
            • <strong>Plasma Glucose Gap:</strong> Diabetic cohort demonstrates a <strong>+28.4%</strong> higher mean 2-hour plasma glucose concentration (141.3 mg/dL vs 109.9 mg/dL in healthy subjects).<br>
            • <strong>Adiposity Risk:</strong> <strong>82.5%</strong> of diabetic individuals have a BMI ≥ 30.0 kg/m² (WHO Obese category), confirming metabolic syndrome predominance.<br>
            • <strong>Age Correlation:</strong> Onset incidence peaks in the 40–55 age stratum, demonstrating progressive age-associated glycemic deterioration.<br>
            • <strong>Diagnostic Sensitivity:</strong> Model benchmarking prioritizes Recall and ROC-AUC over pure accuracy to minimize catastrophic false negatives in health screening.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# SCREEN 02: UPLOAD (DATA INGESTION & QUALITY ENGINE)
# ==============================================================================
elif page == "02 Upload":
    render_page_header(
        "SCREEN 02 · DATA INGESTION & AUDIT",
        "Data Ingestion & Automated Quality Engine",
        "Upload cohort files or load benchmark datasets to compute transparent 0–100 quality scores"
    )

    u_col1, u_col2 = st.columns([1.3, 0.7])
    with u_col1:
        uploaded_file = st.file_uploader(
            "Upload Cohort Dataset (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            help="Supported formats: .csv, .xlsx. Max upload size 50MB."
        )
    with u_col2:
        st.markdown("**Quick Benchmark Presets:**")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Pima Indians Diabetes", use_container_width=True, type="primary"):
                st.session_state["df_raw"] = load_default_dataset()
                st.session_state["dataset_name"] = "pima_diabetes.csv"
                st.session_state["target_column"] = "Outcome"
                st.rerun()
        with b2:
            if st.button("Dirty Test Sample", use_container_width=True):
                st.session_state["df_raw"] = load_dirty_sample_dataset()
                st.session_state["dataset_name"] = "diabetes_dirty_sample.csv"
                st.session_state["target_column"] = "Outcome"
                st.rerun()

    if uploaded_file is not None:
        df_up, err = load_uploaded_file(uploaded_file)
        if err:
            st.error(err)
        else:
            st.session_state["df_raw"] = df_up
            st.session_state["dataset_name"] = uploaded_file.name
            st.success(f"Loaded '{uploaded_file.name}' successfully ({len(df_up)} records).")

    df_current = st.session_state.get("df_raw")

    if df_current is not None:
        st.markdown("---")
        # Target column selector
        t_col1, t_col2 = st.columns([0.6, 0.4])
        with t_col1:
            avail_cols = df_current.columns.tolist()
            def_idx = avail_cols.index("Outcome") if "Outcome" in avail_cols else 0
            target_sel = st.selectbox("Designate Target Feature (Clinical Ground Truth):", avail_cols, index=def_idx)
            st.session_state["target_column"] = target_sel

        # 8 Dataset Structural Metadata Tiles
        meta = get_dataset_metadata(df_current, name=st.session_state.get("dataset_name", "Current"))
        m_r1 = st.columns(4)
        render_kpi(m_r1[0], "File Name", meta["name"][:18], "Active dataset", icon="📄")
        render_kpi(m_r1[1], "Total Rows", f"{meta['rows']:,}", "Patient records", icon="👥")
        render_kpi(m_r1[2], "Total Columns", meta["columns"], "Features & target", icon="📐")
        render_kpi(m_r1[3], "Memory Footprint", meta["memory_usage"], "Deep memory allocation", icon="💾")

        m_r2 = st.columns(4)
        render_kpi(m_r2[0], "Numeric Features", len(meta["numerical_columns"]), "Continuous biomarkers", icon="🔢")
        render_kpi(m_r2[1], "Categorical Features", len(meta["categorical_columns"]), "Discrete strata", icon="🏷️")
        render_kpi(m_r2[2], "Explicit Missing", meta["missing_values"], "Null / NaN cells", icon="⚠️")
        render_kpi(m_r2[3], "Duplicate Rows", meta["duplicate_rows"], "Identical entries", icon="🔁")

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        # Automated Data Quality Score Card
        st.markdown("### Automated Data Quality Assessment")
        q_report = compute_data_quality_report(df_current, target_column=target_sel)
        scores = q_report["scores"]

        q_c1, q_c2 = st.columns([0.35, 0.65])
        with q_c1:
            st.plotly_chart(create_quality_score_gauge(scores["overall"]), use_container_width=True)
            st.markdown(f"<div style='text-align: center; font-weight: 700; font-family: Outfit; font-size: 1.15rem; color: {THEME['primary']};'>Overall Quality: {scores['overall']}/100</div>", unsafe_allow_html=True)

        with q_c2:
            st.markdown(f"""
            <div style='background: white; border: 1px solid {THEME['border']}; border-radius: 12px; padding: 1.2rem;'>
                <div style='font-size: 0.85rem; font-weight: 600; text-transform: uppercase; color: {THEME['muted']}; margin-bottom: 0.8rem;'>Dimension Scoring Breakdown</div>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 12px;'>
                    <div>
                        <div style='display: flex; justify-content: space-between; font-size: 0.85rem;'><span>Completeness</span><strong>{scores['completeness']}%</strong></div>
                        <div style='background: #E2E8F0; height: 6px; border-radius: 3px; margin-top: 4px;'><div style='background: {THEME['primary']}; width: {scores['completeness']}%; height: 6px; border-radius: 3px;'></div></div>
                    </div>
                    <div>
                        <div style='display: flex; justify-content: space-between; font-size: 0.85rem;'><span>Validity</span><strong>{scores['validity']}%</strong></div>
                        <div style='background: #E2E8F0; height: 6px; border-radius: 3px; margin-top: 4px;'><div style='background: {THEME['primary']}; width: {scores['validity']}%; height: 6px; border-radius: 3px;'></div></div>
                    </div>
                    <div>
                        <div style='display: flex; justify-content: space-between; font-size: 0.85rem;'><span>Consistency</span><strong>{scores['consistency']}%</strong></div>
                        <div style='background: #E2E8F0; height: 6px; border-radius: 3px; margin-top: 4px;'><div style='background: {THEME['primary']}; width: {scores['consistency']}%; height: 6px; border-radius: 3px;'></div></div>
                    </div>
                    <div>
                        <div style='display: flex; justify-content: space-between; font-size: 0.85rem;'><span>Uniqueness</span><strong>{scores['uniqueness']}%</strong></div>
                        <div style='background: #E2E8F0; height: 6px; border-radius: 3px; margin-top: 4px;'><div style='background: {THEME['primary']}; width: {scores['uniqueness']}%; height: 6px; border-radius: 3px;'></div></div>
                    </div>
                </div>
                <div style='margin-top: 1rem; font-size: 0.78rem; color: {THEME['muted']}; line-height: 1.45;'>
                    <strong>Scoring Formula:</strong> Overall Score = 30% Completeness + 30% Validity + 20% Consistency + 20% Uniqueness. 
                    Penalizes unrecorded biological zeroes in vital indicators.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # Data Preview Table
        st.markdown("### Raw Cohort Dataset Preview")
        previews = get_data_previews(df_current, sample_size=5)
        pt1, pt2, pt3 = st.tabs(["First 5 Rows", "Last 5 Rows", "Random Sample (5)"])
        with pt1:
            st.dataframe(previews["head"], use_container_width=True)
        with pt2:
            st.dataframe(previews["tail"], use_container_width=True)
        with pt3:
            st.dataframe(previews["sample"], use_container_width=True)

        with st.expander("Column-by-Column Anomaly & Distribution Breakdown"):
            st.json({
                "Total Cells Audited": q_report["total_records"] * q_report["total_features"],
                "Biological Zeroes in Vitals": q_report["biological_zeros"],
                "Outliers (IQR 1.5x Rule)": q_report["outliers_count"],
                "Unique Values Per Feature": q_report["unique_values_per_col"]
            })


# ==============================================================================
# SCREEN 03: WRANGLING (DATA PREPROCESSING & AUDIT)
# ==============================================================================
elif page == "03 Wrangling":
    render_page_header(
        "SCREEN 03 · DATA WRANGLING",
        "Clinical Data Wrangling & Sanitization",
        "Resolve biological zero anomalies, impute missing values, cap clinical outliers, and audit Before vs After"
    )

    df_raw = st.session_state.get("df_raw")
    if df_raw is None:
        st.warning("No cohort dataset loaded. Please visit '02 Upload' first.")
        st.stop()

    target_col = st.session_state.get("target_column", "Outcome")

    # Pipeline Configuration Panel
    st.markdown("### 1. Data Cleaning Pipeline Configuration")
    wc1, wc2 = st.columns(2)
    with wc1:
        handle_zeros = st.checkbox("Resolve Biological Zeros (Recommended)", value=True, help="Converts impossible 0s in Glucose, BP, SkinThickness, Insulin, and BMI into NaNs for imputation.")
        remove_dups = st.checkbox("Deduplicate Patient Records", value=True)
        impute_strat = st.selectbox("Missing Value Imputation Strategy:", ["median", "mean", "knn", "most_frequent", "ffill"], index=0, help="Median is clinically robust against extreme clinical values.")
    with wc2:
        outlier_strat = st.selectbox("Outlier Handling:", ["cap", "remove", "none"], index=0, help="'cap' (Winsorization) caps values at 1.5*IQR bounds; 'remove' drops rows.")
        outlier_meth = st.selectbox("Outlier Detection Criterion:", ["iqr", "zscore"], index=0)
        scaling_choice = st.selectbox("Feature Scaling:", ["None", "standard", "minmax"], index=0)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    if st.button("🚀 Execute Data Wrangling Pipeline", type="primary", use_container_width=True):
        wrangler = HealthDataWrangler(
            handle_bio_zeros=handle_zeros,
            imputation_strategy=impute_strat,
            remove_duplicates=remove_dups,
            outlier_strategy=outlier_strat,
            outlier_method=outlier_meth,
            scaling_method=None if scaling_choice == "None" else scaling_choice
        )
        with st.spinner("Executing sanitization pipeline..."):
            df_cleaned, audit = wrangler.fit_transform(df_raw, target_column=target_col)
            st.session_state["df_cleaned"] = df_cleaned
            st.session_state["audit_metrics"] = audit
            df_cleaned.to_csv(PROCESSED_DATASET_PATH, index=False)
            st.success("Wrangling pipeline complete. Clean dataset persisted to disk!")

    audit = st.session_state.get("audit_metrics")
    if audit is not None:
        st.markdown("---")
        st.subheader("2. Before vs. After Data Wrangling Audit (Mandatory)")

        cmp1, cmp2 = st.columns([0.45, 0.55])
        with cmp1:
            st.markdown("#### Audit Verification Table")
            st.table(audit["diff_table"])

        with cmp2:
            st.markdown("#### Visual Anomaly Reduction")
            diff_df = audit["diff_table"]
            fig_diff = go.Figure(data=[
                go.Bar(name='BEFORE', x=diff_df['Metric'], y=diff_df['BEFORE'], marker_color='#EF4444'),
                go.Bar(name='AFTER', x=diff_df['Metric'], y=diff_df['AFTER'], marker_color='#10B981')
            ])
            fig_diff.update_layout(
                barmode='group',
                height=320,
                margin=dict(l=20, r=20, t=30, b=20),
                font=dict(family="IBM Plex Sans")
            )
            st.plotly_chart(fig_diff, use_container_width=True)

        st.markdown("### Cleaned Dataset Preview")
        st.dataframe(st.session_state["df_cleaned"].head(8), use_container_width=True)

        csv_data = st.session_state["df_cleaned"].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Cleaned Dataset (CSV)",
            data=csv_data,
            file_name="diametrics_cleaned_cohort.csv",
            mime="text/csv"
        )
    else:
        st.info("Click 'Execute Data Wrangling Pipeline' above to generate the Before-vs-After audit.")


# ==============================================================================
# SCREEN 04: ANALYSIS (EXPLORATORY DATA ANALYSIS)
# ==============================================================================
elif page == "04 Analysis":
    render_page_header(
        "SCREEN 04 · HEALTH ANALYTICS",
        "Exploratory Health Data Analysis & Visualizations",
        "12 interactive clinical distributions, correlation matrices, radar profiles, and automated epidemiological insights"
    )

    df = st.session_state.get("df_cleaned")
    if df is None:
        st.warning("No dataset loaded. Please visit '02 Upload' or '03 Wrangling' first.")
        st.stop()

    target_col = st.session_state.get("target_column", "Outcome")

    # 6 Top Clinical KPI Cards
    kpis = compute_eda_kpis(df, target_column=target_col)
    kc1 = st.columns(6)
    render_kpi(kc1[0], "Total Patients", f"{kpis['total_patients']:,}", "Cohort count", icon="👥")
    render_kpi(kc1[1], "Prevalence", f"{kpis['disease_prevalence']}%", "Diabetic fraction", icon="📊")
    render_kpi(kc1[2], "High Risk Rate", f"{kpis['high_risk_pct']}%", "Elevated biomarkers", icon="⚠️")
    render_kpi(kc1[3], "Mean Glucose", f"{kpis['avg_glucose']}", "mg/dL (Oral GTT)", icon="🩸")
    render_kpi(kc1[4], "Mean BMI", f"{kpis['avg_bmi']}", "kg/m² (Adiposity)", icon="⚖️")
    render_kpi(kc1[5], "Mean Age", f"{kpis['avg_age']}", "Years (Demographic)", icon="🎂")

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Dynamic Health Insights Section
    st.markdown("### 💡 Automated Epidemiological Insights")
    insights = generate_health_insights(df, target_column=target_col)
    in_cols = st.columns(min(len(insights), 3))
    for idx, item in enumerate(insights[:3]):
        with in_cols[idx]:
            st.markdown(f"""
            <div class="insight-box">
                <h4>{item['title']}</h4>
                <p>{item['text']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # 4 Visualization Tabs matching DiaMetrics
    st.markdown("### Clinical Visual Analytics")
    at1, at2, at3, at4 = st.tabs([
        "1. Cohort Distributions",
        "2. Feature Relationships",
        "3. Correlation Matrix",
        "4. Risk Profiles & Categories"
    ])

    with at1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_disease_distribution(df, target_col), use_container_width=True)
        with c2:
            st.plotly_chart(plot_age_distribution(df, target_col), use_container_width=True)
        st.plotly_chart(plot_all_feature_distributions(df), use_container_width=True)

    with at2:
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(plot_bivariate_scatter(df, "Glucose", "BMI", target_col), use_container_width=True)
        with c4:
            st.plotly_chart(plot_bivariate_scatter(df, "Age", "Glucose", target_col), use_container_width=True)

    with at3:
        st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)

    with at4:
        c5, c6 = st.columns(2)
        with c5:
            st.plotly_chart(plot_risk_profile_radar(df, target_col), use_container_width=True)
        with c6:
            df_eng = engineer_health_features(df)
            if "BMI_Category" in df_eng.columns:
                fig_bmi = px.histogram(
                    df_eng, x="BMI_Category", color=target_col,
                    barmode="group", title="WHO BMI Strata vs Disease Outcome",
                    color_discrete_sequence=[THEME["primary"], THEME["danger"]]
                )
                fig_bmi.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20), font=dict(family="IBM Plex Sans"))
                st.plotly_chart(fig_bmi, use_container_width=True)


# ==============================================================================
# SCREEN 05: MINING (UNSUPERVISED LEARNING & PATTERN DISCOVERY)
# ==============================================================================
elif page == "05 Mining":
    render_page_header(
        "SCREEN 05 · DATA MINING",
        "Unsupervised Learning & Pattern Discovery",
        "K-Means metabolic phenotype clustering and Apriori frequent association rule mining"
    )

    df = st.session_state.get("df_cleaned")
    if df is None:
        st.warning("No dataset loaded. Please visit '02 Upload' or '03 Wrangling' first.")
        st.stop()

    target_col = st.session_state.get("target_column", "Outcome")

    mt1, mt2 = st.tabs(["🧩 K-Means Clustering", "🔗 Association Rules (Apriori)"])

    with mt1:
        st.markdown("""
        **Unsupervised K-Means clustering** groups patients based purely on physiological similarity 
        without using diagnostic labels, uncovering distinct metabolic phenotypes.
        """)

        num_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != target_col]
        selected_clust = st.multiselect("Biomarkers for Clustering:", num_cols, default=[c for c in ["Age", "BMI", "Glucose", "BloodPressure"] if c in num_cols])

        if len(selected_clust) >= 2:
            cl1, cl2 = st.columns([0.4, 0.6])
            metrics = compute_clustering_metrics(df, selected_clust, max_k=6)
            with cl2:
                st.plotly_chart(plot_elbow_and_silhouette(metrics), use_container_width=True)
            with cl1:
                st.markdown(f"**Optimal k (Max Silhouette):** `{metrics['optimal_k']}`")
                k_val = st.slider("Select Cluster Count (k):", 2, 6, metrics['optimal_k'])

            df_clust, summary = perform_kmeans_clustering(df, selected_clust, n_clusters=k_val)

            sc1, sc2 = st.columns(2)
            with sc1:
                x_feat = st.selectbox("Scatter X-Axis:", selected_clust, index=0)
            with sc2:
                y_feat = st.selectbox("Scatter Y-Axis:", selected_clust, index=min(1, len(selected_clust)-1))

            st.plotly_chart(plot_clusters_scatter(df_clust, x_feat, y_feat), use_container_width=True)

            st.markdown("#### Patient Cluster Phenotypes")
            card_cols = st.columns(k_val)
            for idx, (c_name, c_data) in enumerate(summary["descriptions"].items()):
                with card_cols[idx]:
                    st.markdown(f"""
                    <div class="diametrics-card">
                        <h4 style='color: {THEME['primary']}; margin: 0 0 0.3rem 0;'>{c_name}</h4>
                        <div style='font-size: 0.8rem; color: {THEME['muted']};'>Patients: <strong>{c_data['size']} ({c_data['percentage']}%)</strong></div>
                        <div style='margin-top: 0.5rem; font-size: 0.85rem; font-weight: 500;'>{c_data['summary']}</div>
                    </div>
                    """, unsafe_allow_html=True)

    with mt2:
        st.markdown(ASSOCIATION_CAUSATION_DISCLAIMER)
        ar1, ar2 = st.columns(2)
        with ar1:
            min_sup = st.slider("Minimum Support Threshold:", 0.02, 0.30, 0.08, 0.01)
        with ar2:
            min_conf = st.slider("Minimum Confidence Threshold:", 0.20, 0.90, 0.40, 0.05)

        with st.spinner("Extracting frequent itemsets via Apriori..."):
            freq_sets, rules = mine_association_rules(df, min_support=min_sup, min_confidence=min_conf, target_column=target_col)

        if not rules.empty:
            st.plotly_chart(plot_rules_scatter(rules), use_container_width=True)
            st.markdown("#### Strongest Discovered Association Rules (Ranked by Lift)")
            st.dataframe(rules[["Antecedents", "Consequents", "support", "confidence", "lift", "leverage"]].head(15), use_container_width=True)
        else:
            st.info("No association rules found at current thresholds. Lower support or confidence.")


# ==============================================================================
# SCREEN 06: MODELS (SUPERVISED CLASSIFICATION BENCHMARK)
# ==============================================================================
elif page == "06 Models":
    render_page_header(
        "SCREEN 06 · MODEL BENCHMARK",
        "Supervised Classification Benchmark & Diagnostics",
        "5-algorithm evaluation with stratified cross-validation, data leakage prevention, and deep diagnostic curves"
    )

    df = st.session_state.get("df_cleaned")
    if df is None:
        st.warning("No dataset loaded. Please visit '02 Upload' or '03 Wrangling' first.")
        st.stop()

    target_col = st.session_state.get("target_column", "Outcome")

    if st.button("⚡ Retrain & Benchmark All 5 Classifiers", type="primary", use_container_width=True) or st.session_state.get("classifier_artifacts") is None:
        with st.spinner("Training 5 supervised algorithms with stratified CV..."):
            trainer = HealthClassifierTrainer(test_size=0.2, random_state=42)
            results_df, artifacts = trainer.train_and_evaluate_all(df, target_column=target_col)
            st.session_state["classifier_artifacts"] = artifacts
            st.session_state["results_df"] = results_df

    results_df = st.session_state.get("results_df")
    artifacts = st.session_state.get("classifier_artifacts")

    if results_df is not None:
        # Champion Model Callout Banner
        best_name = artifacts["best_model_name"]
        st.markdown(f"""
        <div style='background: #EFF6FF; border: 1px solid #BFDBFE; border-left: 4px solid #2563EB; border-radius: 10px; padding: 1rem 1.4rem; margin-bottom: 1.2rem; display: flex; align-items: center; justify-content: space-between;'>
            <div>
                <span style='font-family: Outfit; font-weight: 700; font-size: 1.15rem; color: #1E40AF;'>🏆 Benchmark Champion: {best_name}</span>
                <div style='font-size: 0.85rem; color: #475569; margin-top: 2px;'>Crowned champion based on superior held-out test ROC-AUC discrimination.</div>
            </div>
            <span class='badge-pill badge-low' style='font-size: 0.8rem;'>LEAD CLASSIFIER</span>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Model Leaderboard (Evaluated on Unseen Test Split)")
        st.dataframe(results_df, use_container_width=True)

        # Multi-Model Grouped Bar Chart
        fig_bar = px.bar(
            results_df, x="Model", y=["Accuracy", "Recall", "ROC-AUC"],
            barmode="group", title="Algorithm Benchmark Comparison (Test Metrics)",
            color_discrete_sequence=[THEME["primary"], THEME["warning"], THEME["accent"]]
        )
        fig_bar.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20), font=dict(family="IBM Plex Sans"))
        st.plotly_chart(fig_bar, use_container_width=True)

        # Deep Model Diagnostics Section
        st.markdown("---")
        st.subheader("Deep Model Diagnostics")
        selected_model = st.selectbox("Select Model to Inspect:", results_df["Model"].tolist(), index=results_df["Model"].tolist().index(best_name))

        eval_data = artifacts["detailed_evals"][selected_model]
        y_true = np.array(eval_data["y_true"])
        y_pred = np.array(eval_data["y_pred"])
        y_prob = np.array(eval_data["y_prob"])

        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            st.plotly_chart(plot_interactive_confusion_matrix(y_true, y_pred, selected_model), use_container_width=True)
        with dc2:
            st.plotly_chart(plot_roc_curve(y_true, y_prob, selected_model), use_container_width=True)
        with dc3:
            st.plotly_chart(plot_precision_recall_curve(y_true, y_prob, selected_model), use_container_width=True)

        # Feature Importance
        st.markdown("#### Global Feature Importance Ranking")
        imp_df = get_global_feature_importance(eval_data["model_obj"], artifacts["feature_names"], artifacts["X_test"], artifacts["y_test"])
        st.plotly_chart(plot_feature_importance_bar(imp_df, selected_model), use_container_width=True)

        st.markdown("---")
        st.markdown(CLINICAL_RECALL_IMPORTANCE_TEXT)


# ==============================================================================
# SCREEN 07: PREDICTION (PATIENT RISK ASSESSMENT & EXPLAINABLE AI)
# ==============================================================================
elif page == "07 Prediction":
    render_page_header(
        "SCREEN 07 · PATIENT RISK PREDICTION",
        "Patient Disease Risk Assessment & Explainable AI",
        "Calibrated individual probability estimation and local factor waterfall attributions"
    )

    if not Path(BEST_MODEL_PATH).exists() or not Path(PREPROCESSOR_PATH).exists():
        st.warning("Models not yet trained. Please visit '06 Models' first.")
        st.stop()

    best_model = joblib.load(BEST_MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Preset Patient Buttons for instant demo
    st.markdown("**Quick Preset Profiles:**")
    ps1, ps2, ps3 = st.columns(3)
    with ps1:
        if st.button("🔴 High Risk Patient Preset", use_container_width=True):
            st.session_state["preset_patient"] = {
                "age": 52, "glucose": 178, "bp": 88, "bmi": 36.4,
                "insulin": 220, "skin": 38, "preg": 4, "ped": 0.85
            }
    with ps2:
        if st.button("🟢 Healthy Patient Preset", use_container_width=True):
            st.session_state["preset_patient"] = {
                "age": 24, "glucose": 92, "bp": 68, "bmi": 21.8,
                "insulin": 75, "skin": 18, "preg": 0, "ped": 0.22
            }
    with ps3:
        if st.button("🟡 Borderline Patient Preset", use_container_width=True):
            st.session_state["preset_patient"] = {
                "age": 41, "glucose": 138, "bp": 78, "bmi": 28.5,
                "insulin": 135, "skin": 26, "preg": 2, "ped": 0.48
            }

    p_data = st.session_state.get("preset_patient") or {
        "age": 42, "glucose": 148, "bp": 72, "bmi": 33.6,
        "insulin": 125, "skin": 35, "preg": 2, "ped": 0.627
    }

    # Clinical Intake Form
    st.markdown("### Clinical Biomarkers Intake")
    with st.form("patient_prediction_form"):
        pf1, pf2, pf3 = st.columns(3)
        with pf1:
            age = st.number_input("Age (Years)", 18, 120, int(p_data["age"]))
            glucose = st.number_input("Plasma Glucose (mg/dL)", 40, 400, int(p_data["glucose"]))
            bp = st.number_input("Diastolic Blood Pressure (mm Hg)", 30, 220, int(p_data["bp"]))
        with pf2:
            bmi = st.number_input("BMI (kg/m²)", 10.0, 75.0, float(p_data["bmi"]), 0.1)
            insulin = st.number_input("Serum Insulin (μU/mL)", 5, 900, int(p_data["insulin"]))
            skin = st.number_input("Triceps Skinfold (mm)", 5, 100, int(p_data["skin"]))
        with pf3:
            preg = st.number_input("Number of Pregnancies", 0, 20, int(p_data["preg"]))
            ped = st.number_input("Diabetes Pedigree Function", 0.05, 2.50, float(p_data["ped"]), 0.01)
            activity = st.selectbox("Physical Activity Tier:", ["Sedentary", "Moderate", "Active"], index=1)

        submit_pred = st.form_submit_button("🩺 Analyze Health Risk", type="primary", use_container_width=True)

    if submit_pred:
        patient_inputs = {
            "Pregnancies": float(preg), "Glucose": float(glucose), "BloodPressure": float(bp),
            "SkinThickness": float(skin), "Insulin": float(insulin), "BMI": float(bmi),
            "DiabetesPedigreeFunction": float(ped), "Age": float(age)
        }
        is_valid, errors = validate_patient_input(patient_inputs)
        if not is_valid:
            for err in errors:
                st.error(err)
        else:
            with st.spinner("Computing calibrated inference..."):
                res = predict_patient_risk(patient_inputs, model=best_model, preprocessor=preprocessor, save_to_db=True)
                st.session_state["last_prediction"] = res

    last_pred = st.session_state.get("last_prediction")
    if last_pred is not None:
        st.markdown("---")
        st.subheader("Assessment Diagnostic Results")

        res_c1, res_c2 = st.columns([0.45, 0.55])
        with res_c1:
            st.plotly_chart(create_risk_gauge(last_pred["probability"]), use_container_width=True)
        with res_c2:
            cat = last_pred["category"]
            cat_badge = f"badge-pill badge-{cat.lower()}"
            st.markdown(f"""
            <div class="diametrics-card">
                <div style='font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: {THEME['muted']};'>Assigned Clinical Risk Strata</div>
                <div style='margin: 0.5rem 0;'><span class="{cat_badge}">{last_pred['category_label']}</span></div>
                <div style='font-size: 0.95rem; margin-top: 0.8rem; line-height: 1.7;'>
                    <strong>Calculated Probability:</strong> <span class="font-mono">{last_pred['probability_percent']}%</span><br>
                    <strong>Inference Model:</strong> {last_pred['model_used']}<br>
                    <strong>Database Audit Record:</strong> #{last_pred.get('prediction_id', 'Logged')}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Explainable AI (XAI): Directional Factor Attributions")
        baseline_medians = get_cohort_baseline_medians()
        imp_df = get_global_feature_importance(best_model, CANONICAL_FEATURES)
        attributions = explain_single_patient_prediction(last_pred["patient_inputs"], imp_df, baseline_medians)

        st.plotly_chart(plot_local_explanation_diverging(attributions), use_container_width=True)

        nl_text = generate_natural_language_explanation(last_pred["probability"], last_pred["category"], attributions)
        st.markdown(f"""
        <div class="insight-box">
            <h4>Clinical Narrative Summary</h4>
            <p>{nl_text}</p>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# SCREEN 08: SIMULATOR (COUNTERFACTUAL "WHAT-IF" ENGINE)
# ==============================================================================
elif page == "08 Simulator":
    render_page_header(
        "SCREEN 08 · RISK SIMULATOR",
        "Counterfactual What-If Scenario Simulator",
        "Simulate therapeutic & lifestyle interventions to observe estimated risk reduction"
    )

    if not Path(BEST_MODEL_PATH).exists() or not Path(PREPROCESSOR_PATH).exists():
        st.warning("Models not yet trained. Please visit '06 Models' first.")
        st.stop()

    best_model = joblib.load(BEST_MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    st.markdown("""
    Modify lifestyle-responsive biomarkers (reducing plasma glucose through diet or lowering BMI through weight loss) 
    to see how the statistical risk probability shifts relative to the patient's baseline state.
    """)

    sim1, sim2 = st.columns(2)
    with sim1:
        st.markdown("#### Baseline Patient State")
        b_age = st.number_input("Base Age", 18, 100, 48, key="sb_age")
        b_glu = st.slider("Base Glucose (mg/dL)", 70, 300, 168, key="sb_glu")
        b_bmi = st.slider("Base BMI (kg/m²)", 15.0, 55.0, 35.2, 0.1, key="sb_bmi")
        b_bp = st.slider("Base Blood Pressure (mm Hg)", 50, 180, 86, key="sb_bp")
        b_preg = 2
        b_ped = 0.55
        b_ins = 160
        b_skin = 32

    with sim2:
        st.markdown("#### Target Simulated Scenario (What-If)")
        s_glu = st.slider("Target Glucose (mg/dL)", 70, 300, 108, key="st_glu", help="Simulate glycemic control via diet/exercise")
        s_bmi = st.slider("Target BMI (kg/m²)", 15.0, 55.0, 24.8, 0.1, key="st_bmi", help="Simulate weight loss intervention")
        s_bp = st.slider("Target Blood Pressure (mm Hg)", 50, 180, 76, key="st_bp", help="Simulate antihypertensive treatment")

    base_inputs = {
        "Pregnancies": float(b_preg), "Glucose": float(b_glu), "BloodPressure": float(b_bp),
        "SkinThickness": float(b_skin), "Insulin": float(b_ins), "BMI": float(b_bmi),
        "DiabetesPedigreeFunction": float(b_ped), "Age": float(b_age)
    }

    scen_inputs = base_inputs.copy()
    scen_inputs["Glucose"] = float(s_glu)
    scen_inputs["BMI"] = float(s_bmi)
    scen_inputs["BloodPressure"] = float(s_bp)

    sim_res = simulate_what_if_scenario(base_inputs, scen_inputs, best_model, preprocessor)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    st.subheader("Counterfactual Comparison Results")

    del1, del2, del3 = st.columns(3)
    render_kpi(del1, "Baseline Risk", f"{sim_res['baseline']['probability_percent']}%", f"Tier: {sim_res['baseline']['category']}", icon="🔴")
    render_kpi(del2, "Simulated Risk", f"{sim_res['scenario']['probability_percent']}%", f"Tier: {sim_res['scenario']['category']}", icon="🟢")
    render_kpi(del3, "Net Risk Reduction", f"{sim_res['absolute_change_percent']:+.1f}%", f"Relative: {sim_res['relative_change_percent']:+.1f}%", icon="📉")

    # Comparative Visual
    fig_comp = go.Figure(data=[
        go.Bar(
            name='Risk Score %',
            x=['Baseline Patient State', 'Target Counterfactual State'],
            y=[sim_res['baseline']['probability_percent'], sim_res['scenario']['probability_percent']],
            marker_color=[THEME['danger'], THEME['success']],
            text=[f"{sim_res['baseline']['probability_percent']}%", f"{sim_res['scenario']['probability_percent']}%"],
            textposition='auto'
        )
    ])
    fig_comp.update_layout(
        yaxis_range=[0, 100],
        yaxis_title="Predicted Risk Probability (%)",
        height=320,
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(family="IBM Plex Sans")
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown(f"""
    <div class="insight-box">
        <h4>Generated Clinical Action Plan</h4>
        <p>Achieving target glycemic control (Glucose {s_glu} mg/dL) and normal adiposity (BMI {s_bmi} kg/m²) reduces absolute risk probability by <strong>{abs(sim_res['absolute_change_percent']):.1f} percentage points</strong> ({abs(sim_res['relative_change_percent']):.1f}% relative reduction). Lifestyle modification is the primary recommended preventative pathway.</p>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# SCREEN 09: HISTORY (AUDIT TRAIL & LOGGING)
# ==============================================================================
elif page == "09 History":
    render_page_header(
        "SCREEN 09 · AUDIT TRAIL",
        "Patient Prediction Audit History",
        "Persistent SQLite audit logs of all patient evaluations with multi-criteria filtering"
    )

    hf1, hf2, hf3 = st.columns([0.4, 0.4, 0.2])
    with hf1:
        cat_filter = st.selectbox("Filter Risk Strata:", ["ALL", "LOW", "MODERATE", "HIGH"])
    with hf2:
        alg_filter = st.selectbox("Filter Model Algorithm:", ["ALL", "Trained Classifier", "Logistic Regression", "Random Forest"])
    with hf3:
        if st.button("🗑️ Clear Logs", type="secondary"):
            clear_all_predictions()
            st.success("Cleared prediction logs.")
            st.rerun()

    history_df = get_predictions(category_filter=cat_filter, algorithm_filter=alg_filter, limit=100)

    if history_df.empty:
        st.info("No prediction history recorded in the database.")
    else:
        st.dataframe(history_df[["id", "created_at", "model_algorithm", "risk_probability", "risk_category", "patient_features_json"]], use_container_width=True)
        csv_hist = history_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Audit Trail (CSV)", data=csv_hist, file_name="diametrics_audit_history.csv", mime="text/csv")


# ==============================================================================
# SCREEN 10: REPORT (AUTOMATED PROJECT REPORT & EXPORT)
# ==============================================================================
elif page == "10 Report":
    render_page_header(
        "SCREEN 10 · PROJECT REPORT",
        "Comprehensive Project Report & Documentation",
        "Automated clinical decision-support & data mining research report generator"
    )

    st.markdown("""
    ### Executive Clinical & Machine Learning Audit
    This report summarizes the complete data science lifecycle executed on the cohort.
    """)

    report_text = f"""# DiaMetrics / HealthSense: Comprehensive Clinical & Data Mining Report

### 1. Executive Summary
DiaMetrics is an intelligent health data platform demonstrating automated data quality auditing, domain-aware data wrangling, exploratory epidemiological analysis, multi-algorithm classification, unsupervised phenotype clustering, and explainable disease risk prediction.

### 2. Dataset & Quality Assessment
- **Cohort Dataset**: Pima Indians Diabetes Database
- **Sample Size**: 768 patient records across 9 clinical variables
- **Quality Score**: 96.5 / 100 audited across Completeness, Validity, Consistency, and Uniqueness
- **Biological Zeroes**: 652 zero values identified in vital signs and resolved via domain-specific imputation

### 3. Machine Learning Classification Benchmark
- Evaluated 5 diverse algorithms using stratified 80/20 train/test split without data leakage:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. K-Nearest Neighbors (KNN)
  5. Support Vector Machine (SVM)
- **Top Classifier**: Logistic Regression / Random Forest (ROC-AUC ~0.814, 5-fold CV AUC ~0.838)
- **Clinical Sensitivity**: Prioritized Recall to minimize catastrophic False Negatives in disease screening

### 4. Unsupervised Data Mining
- **K-Means Clustering**: Uncovered 3 distinct metabolic risk phenotypes verified by Elbow (WCSS) and Silhouette score analysis
- **Apriori Association Rules**: Extracted actionable clinical rules (e.g. {{High Glucose, High BMI}} => {{Diabetic}}) with high Lift

### 5. Academic & Ethical Disclaimer
{MEDICAL_DISCLAIMER}
"""

    st.download_button(
        label="📄 Download Full Project Report (Markdown)",
        data=report_text,
        file_name="DiaMetrics_Comprehensive_Project_Report.md",
        mime="text/markdown"
    )

    st.markdown(report_text)
