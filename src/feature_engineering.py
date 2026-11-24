"""
HealthSense Feature Engineering Module.
Derives clinically validated categorical and interaction features from raw physiological data.
"""

from typing import Tuple
import pandas as pd
import numpy as np


def engineer_health_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate derived clinical indicator features for exploratory analysis and mining.
    
    Clinical Features Derived:
    1. BMI_Category: WHO standard classifications (Underweight, Normal, Overweight, Obese)
    2. Age_Group: Epidemiological life-stage strata (18-30, 31-45, 46-60, 60+)
    3. Glucose_Risk: ADA/WHO oral glucose tolerance strata (Normal, Impaired, High)
    4. BP_Category: AHA diastolic pressure classifications (Normal, Prehypertension, Stage 1, Stage 2)
    5. Metabolic_Interaction: Product of Glucose and BMI (combined metabolic strain)
    6. HOMA_IR_Proxy: Glucose-Insulin interaction index (when insulin > 0)
    """
    df_feat = df.copy()

    # 1. BMI Category (WHO Standards)
    if "BMI" in df_feat.columns:
        df_feat["BMI_Category"] = pd.cut(
            df_feat["BMI"],
            bins=[-np.inf, 18.5, 24.99, 29.99, np.inf],
            labels=["Underweight", "Normal", "Overweight", "Obese"]
        )

    # 2. Age Group
    if "Age" in df_feat.columns:
        df_feat["Age_Group"] = pd.cut(
            df_feat["Age"],
            bins=[0, 30, 45, 60, np.inf],
            labels=["18-30", "31-45", "46-60", "60+"]
        )

    # 3. Glucose Risk (WHO/ADA 2-hr Oral Glucose Tolerance Criteria)
    if "Glucose" in df_feat.columns:
        df_feat["Glucose_Risk"] = pd.cut(
            df_feat["Glucose"],
            bins=[-np.inf, 139.99, 199.99, np.inf],
            labels=["Normal (<140)", "Impaired (140-199)", "High (>=200)"]
        )

    # 4. Blood Pressure Category (Diastolic mm Hg)
    if "BloodPressure" in df_feat.columns:
        df_feat["BP_Category"] = pd.cut(
            df_feat["BloodPressure"],
            bins=[-np.inf, 79.99, 89.99, 99.99, np.inf],
            labels=["Normal (<80)", "Prehypertension (80-89)", "Stage 1 (90-99)", "Stage 2 (>=100)"]
        )

    # 5. Composite Metabolic Risk Interaction
    if "Glucose" in df_feat.columns and "BMI" in df_feat.columns:
        df_feat["Glucose_BMI_Index"] = (df_feat["Glucose"] * df_feat["BMI"]) / 100.0

    # 6. HOMA-IR Surrogate (Homeostatic Model Assessment of Insulin Resistance)
    # Standard formula: (Fasting Glucose mg/dL * Fasting Insulin μU/mL) / 405
    if "Glucose" in df_feat.columns and "Insulin" in df_feat.columns:
        df_feat["Insulin_Glucose_Ratio"] = np.where(
            (df_feat["Insulin"] > 0) & (df_feat["Glucose"] > 0),
            (df_feat["Glucose"] * df_feat["Insulin"]) / 405.0,
            np.nan
        )

    return df_feat
