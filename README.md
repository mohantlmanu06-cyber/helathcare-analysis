# DiaMetrics (HealthSense Clinical Edition)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![MLxtend](https://img.shields.io/badge/MLxtend-0.23+-green.svg)](https://rasbt.github.io/mlxtend/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **DiaMetrics** is a modern clinical decision-support and health data intelligence platform rebuilt to match the **DiaMetrics / Glucose Insight** standard. It combines automated data quality auditing, domain-aware data wrangling, exploratory epidemiological analysis, multi-model supervised classification, unsupervised metabolic phenotype clustering, and explainable disease risk prediction.

---

## 📌 Academic & Clinical Disclaimer

> **CLINICAL & RESEARCH NOTICE**: DiaMetrics is an educational and academic research platform. All risk classifications, probabilities, association rules, and counterfactual simulations are statistical estimates. They **DO NOT** constitute a formal medical diagnosis or clinical treatment prescription. Always consult a licensed healthcare professional.

---

## 🎨 Visual Identity & Design System

* **Sidebar Theme**: Deep Slate 950 (`#0B1120`) with 10 numbered navigation rows (`01 Home` through `10 Report`).
* **Active Brand Accent**: Vibrant Blue (`#2563EB`) with crisp typography and subtle card elevations.
* **Canvas Background**: Cloud Slate 50 (`#F8FAFC`) with pure white (`#FFFFFF`) card surfaces and 1px borders (`#E2E8F0`).
* **Top Warning Banner**: Sticky Amber-50 banner (`#FFFBEB`) with Amber-200 border (`#FDE68A`) and Amber-900 clinical notice (`#92400E`).
* **Typography**: Modern clinical stack (`Outfit` headings, `IBM Plex Sans` body, `JetBrains Mono` numbers & tables).

---

## 🖥️ Screen-by-Screen Architecture

1. **`01 Home` (Dashboard Overview)**:
   - 4 Top KPI cards: Cohort Size (`768 Patients`), Prevalence (`34.9%`), Champion Model AUC (`0.814`), Data Quality Score (`96.5 / 100`).
   - 4 Quick-access Workflow Action Cards with interactive hover elevation.
   - 4-Stage Pipeline Flow Stepper (`Ingest & Audit` → `Wrangle & Clean` → `Mine & Benchmark` → `Predict & Simulate`).
   - High-level clinical findings summary card.

2. **`02 Upload` (Data Ingestion & Automated Quality Audit)**:
   - One-click presets: `Pima Indians Diabetes` (768 records) & `Dirty Test Sample` (with anomalies).
   - Drag-and-drop CSV / Excel upload zone.
   - Automated Data Quality Engine: 0–100 radial score gauge + 4 dimension meters (Completeness `100%`, Validity `89.4%`, Consistency `98.5%`, Uniqueness `100%`).
   - Structural metadata grid (8 tiles) + paginated table preview with column-by-column statistics breakdown.

3. **`03 Wrangling` (Clinical Preprocessing & Before vs. After Audit)**:
   - Configuration switches: Biological zero handling, deduplication, imputation (median, mean, knn, ffill), outlier treatment (cap, remove, none), scaling.
   - **Mandatory Before vs. After Audit Table**: Total Rows, Missing Values, Duplicate Rows, Invalid Vitals ($\le 0$), Outliers.
   - Side-by-side grouped bar chart illustrating anomaly reduction.
   - Cleaned dataset table preview + CSV download button.

4. **`04 Analysis` (Exploratory Health Data Analysis)**:
   - 6 Top KPI cards (Total Patients, Prevalence, High Risk Rate, Mean Glucose, Mean BMI, Mean Age).
   - Dynamic Epidemiological Insights: Real dataset differentials and correlation rankings.
   - 4 Visualization Tabs:
     - *Tab 1: Cohort Distributions* (Outcome donut chart, Age histogram with KDE, Multi-box plot).
     - *Tab 2: Feature Relationships* (Glucose vs. BMI scatter with regression line, correlation ranking bar chart).
     - *Tab 3: Correlation Matrix* (8x8 interactive heatmap with numeric correlation values).
     - *Tab 4: Risk Profiles* (8-axis Radar chart comparing Diabetic vs Non-diabetic, WHO BMI categories grouped bar chart).

5. **`05 Mining` (Unsupervised Learning & Pattern Discovery)**:
   - *Tab 1: K-Means Clustering*: Dual-axis Elbow Curve & Silhouette score chart, cluster count slider ($k=2..6$), 2D PCA scatter plot, cluster persona cards.
   - *Tab 2: Association Rules (Apriori via MLxtend)*: Support & Confidence sliders, filterable rules table with Lift, Support vs Confidence scatter plot.

6. **`06 Models` (Supervised Classification Benchmark)**:
   - Champion Model Callout Banner (Logistic Regression / Random Forest).
   - 5-Model Leaderboard Table (Logistic Regression, Decision Tree, Random Forest, KNN, SVM).
   - Multi-Model Grouped Bar Chart (Accuracy, Recall, ROC-AUC).
   - Deep Diagnostics Selector: Confusion Matrix Heatmap, ROC Curve, Precision-Recall Curve, Global Feature Importance bar chart.
   - Clinical Screening Callout explaining Recall vs. False Negative trade-offs.

7. **`07 Prediction` (Patient Risk Assessment & Explainable AI)**:
   - 3 One-click demo presets: `High Risk Patient`, `Healthy Patient`, `Borderline Patient`.
   - 8 Patient Biomarker input fields with physiological range constraints.
   - Assessment Results Card: Semi-circular Risk Probability Gauge + Risk Badge (`LOW`, `MODERATE`, `HIGH`).
   - XAI Factor Attribution: Horizontal diverging bar chart (Red = risk elevating, Green = protective factors).
   - Plain-English clinical narrative explaining primary risk drivers.

8. **`08 Simulator` (Counterfactual "What-If" Engine)**:
   - Baseline Patient State vs. Target Simulated Sliders (Glucose, BMI, Blood Pressure).
   - 3 Delta Summary Cards: Baseline Risk %, Simulated Risk %, Net Risk Reduction % (green badge).
   - Comparative Bar Chart (Baseline vs. Simulated).
   - Generated Clinical Action Plan summarizing lifestyle intervention impact.

9. **`09 History` (Audit Trail & Logging)**:
   - Summary audit metrics (Total logged, High risk count, Average probability).
   - Filter controls: Risk Category, Algorithm used, Clear Logs button.
   - Historical Audit Table with one-click CSV export.

10. **`10 Report` (Automated Project Report & Export)**:
    - Formatted structured clinical & machine learning audit report.
    - One-click Download Full Report (Markdown / PDF).

---

## 🚀 How to Run

```bash
# 1. Run all 14 unit tests
python -m unittest discover -s tests -v

# 2. Start the DiaMetrics Platform
python -m streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.
