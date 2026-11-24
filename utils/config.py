"""
HealthSense Configuration Module.
Defines system-wide constants, file paths, clinical reference ranges, and display themes.
Matches DiaMetrics clinical intelligence design tokens.
"""

import os
from pathlib import Path

# Silence joblib loky cpu core warning in Windows sandboxes
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "healthsense.db"

DEFAULT_DATASET_PATH = RAW_DATA_DIR / "diabetes.csv"
SAMPLE_DIRTY_DATASET_PATH = RAW_DATA_DIR / "diabetes_dirty_sample.csv"
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "diabetes_cleaned.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, DATABASE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_STATE = 42

# Primary Disease Feature Names
CANONICAL_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET_COLUMN = "Outcome"

# Biological features where 0 is physiologically impossible in a living patient
BIOLOGICAL_ZERO_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

# Clinical Reference & Validation Limits
PHYSIOLOGICAL_LIMITS = {
    "Pregnancies": {"min": 0, "max": 20, "unit": "count", "label": "Pregnancies", "step": 1},
    "Glucose": {"min": 40, "max": 400, "unit": "mg/dL", "label": "Plasma Glucose", "step": 1},
    "BloodPressure": {"min": 30, "max": 220, "unit": "mm Hg", "label": "Diastolic Blood Pressure", "step": 1},
    "SkinThickness": {"min": 5, "max": 100, "unit": "mm", "label": "Triceps Skinfold", "step": 1},
    "Insulin": {"min": 5, "max": 900, "unit": "μU/mL", "label": "Serum Insulin", "step": 1},
    "BMI": {"min": 10.0, "max": 75.0, "unit": "kg/m²", "label": "Body Mass Index", "step": 0.1},
    "DiabetesPedigreeFunction": {"min": 0.05, "max": 2.50, "unit": "score", "label": "Pedigree Function", "step": 0.01},
    "Age": {"min": 18, "max": 120, "unit": "years", "label": "Age", "step": 1},
}

# Clinical Thresholds for Categorization
CLINICAL_THRESHOLDS = {
    "Glucose": {
        "Normal": (0, 140),
        "Impaired": (140, 200),
        "Diabetic": (200, 1000),
    },
    "BloodPressure": {
        "Normal": (0, 80),
        "Prehypertension": (80, 90),
        "Hypertension": (90, 300),
    },
    "BMI": {
        "Underweight": (0, 18.5),
        "Normal": (18.5, 25.0),
        "Overweight": (25.0, 30.0),
        "Obese": (30.0, 100.0),
    },
    "Age": {
        "Young Adult": (18, 30),
        "Middle-aged": (31, 45),
        "Older Adult": (46, 60),
        "Senior": (61, 150),
    },
}

# Risk Categories & Score Cutoffs
RISK_CATEGORIES = {
    "LOW": {"min": 0.0, "max": 0.35, "color": "#10B981", "bg": "#ECFDF5", "text": "#065F46", "border": "#A7F3D0", "label": "Low Risk", "badge": "success"},
    "MODERATE": {"min": 0.35, "max": 0.65, "color": "#F59E0B", "bg": "#FFFBEB", "text": "#92400E", "border": "#FDE68A", "label": "Moderate Risk", "badge": "warning"},
    "HIGH": {"min": 0.65, "max": 1.0, "color": "#EF4444", "bg": "#FEF2F2", "text": "#991B1B", "border": "#FECACA", "label": "High Risk", "badge": "danger"},
}

# DiaMetrics Match Design System
THEME = {
    "primary": "#2563EB",         # Tailwind Blue-600
    "primary_dark": "#1D4ED8",    # Tailwind Blue-700
    "primary_light": "#EFF6FF",   # Tailwind Blue-50
    "sidebar_bg": "#0B1120",      # Deep Slate 950
    "sidebar_border": "#1E293B",  # Slate 800
    "sidebar_text": "#94A3B8",    # Slate 400
    "sidebar_active": "#2563EB",  # Blue 600
    "secondary": "#0F172A",       # Slate 900
    "accent": "#6366F1",          # Indigo 500
    "success": "#10B981",         # Emerald 500
    "warning": "#F59E0B",         # Amber 500
    "danger": "#EF4444",          # Rose 500
    "neutral_bg": "#F8FAFC",      # Slate 50
    "card_bg": "#FFFFFF",         # Pure White
    "border": "#E2E8F0",          # Slate 200
    "text": "#0F172A",            # Slate 900
    "muted": "#64748B",           # Slate 500
}

# Academic Disclaimer
MEDICAL_DISCLAIMER = (
    "⚠️ **CLINICAL & RESEARCH NOTICE**: HealthSense / DiaMetrics is an academic clinical decision-support and epidemiological research platform. "
    "All risk classifications, probabilities, association rules, and counterfactual simulations are statistical estimates. "
    "They **DO NOT** constitute a formal medical diagnosis or clinical treatment prescription. Always consult a licensed healthcare professional."
)
