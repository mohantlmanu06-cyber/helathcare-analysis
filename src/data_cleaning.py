"""
HealthSense Data Wrangling and Automated Data Quality Engine.
Implements quality scoring, biological zero handling, missing value imputation,
outlier detection (IQR / Z-score), scaling, and Before-vs-After audit metrics.
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.config import BIOLOGICAL_ZERO_COLUMNS, PHYSIOLOGICAL_LIMITS


# ==============================================================================
# 1. AUTOMATED DATA QUALITY ENGINE
# ==============================================================================

def compute_data_quality_report(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate an automated data quality assessment with a 0-100 score.
    
    Scoring Methodology:
    - Completeness (30%): Proportion of non-null cells across the dataset.
    - Validity (30%): Proportion of cells that conform to biological physiological ranges.
    - Consistency (20%): Non-constant columns, proper data types, reasonable variance.
    - Uniqueness (20%): Proportion of non-duplicate records.
    """
    total_rows, total_cols = df.shape
    total_cells = total_rows * total_cols if total_rows > 0 and total_cols > 0 else 1

    # 1. Completeness
    explicit_missing = int(df.isnull().sum().sum())
    # Count biological zeroes as potential missing
    bio_zeros = 0
    for col in BIOLOGICAL_ZERO_COLUMNS:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            bio_zeros += int((df[col] == 0).sum())

    total_missing_concept = explicit_missing + bio_zeros
    completeness_score = max(0.0, min(100.0, 100.0 * (1.0 - (total_missing_concept / total_cells))))

    # 2. Validity
    invalid_count = 0
    for col, limits in PHYSIOLOGICAL_LIMITS.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            series = df[col].dropna()
            # Count negative or out-of-boundary values
            col_invalids = ((series < 0) | (series > limits["max"] * 1.5)).sum()
            invalid_count += int(col_invalids)

    validity_score = max(0.0, min(100.0, 100.0 * (1.0 - (invalid_count / total_cells))))

    # 3. Consistency
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    constant_penalty = (len(constant_cols) / max(total_cols, 1)) * 50.0
    consistency_score = max(0.0, min(100.0, 100.0 - constant_penalty))

    # 4. Uniqueness
    duplicate_rows = int(df.duplicated().sum())
    uniqueness_score = max(0.0, min(100.0, 100.0 * (1.0 - (duplicate_rows / max(total_rows, 1)))))

    # Overall Weighted Score
    overall_score = round(
        (0.30 * completeness_score) +
        (0.30 * validity_score) +
        (0.20 * consistency_score) +
        (0.20 * uniqueness_score),
        1
    )

    # Outlier count via IQR
    outliers_iqr = count_total_outliers(df, method="iqr")

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    return {
        "total_records": total_rows,
        "total_features": total_cols,
        "explicit_missing": explicit_missing,
        "biological_zeros": bio_zeros,
        "duplicate_records": duplicate_rows,
        "invalid_values": invalid_count,
        "outliers_count": outliers_iqr,
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "constant_columns": constant_cols,
        "unique_values_per_col": {col: int(df[col].nunique()) for col in df.columns},
        "scores": {
            "overall": overall_score,
            "completeness": round(completeness_score, 1),
            "validity": round(validity_score, 1),
            "consistency": round(consistency_score, 1),
            "uniqueness": round(uniqueness_score, 1),
        },
        "formula_explanation": (
            "Overall Quality = (30% Completeness) + (30% Validity) + "
            "(20% Consistency) + (20% Uniqueness). Completeness penalizes NaNs and "
            "physiologically impossible 0s in vital signs. Validity checks limits. "
            "Uniqueness penalizes duplicate records."
        )
    }


# ==============================================================================
# 2. OUTLIER DETECTION (IQR & Z-SCORE)
# ==============================================================================

def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """Return boolean mask of outliers using Interquartile Range (IQR)."""
    clean_series = pd.to_numeric(series, errors='coerce').dropna()
    if len(clean_series) < 4:
        return pd.Series(False, index=series.index)
    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (factor * iqr)
    upper_bound = q3 + (factor * iqr)
    return (series < lower_bound) | (series > upper_bound)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Return boolean mask of outliers using standard Z-score (|z| > threshold)."""
    clean_series = pd.to_numeric(series, errors='coerce').dropna()
    if len(clean_series) < 4 or clean_series.std() == 0:
        return pd.Series(False, index=series.index)
    mean = clean_series.mean()
    std = clean_series.std()
    z_scores = (series - mean).abs() / std
    return z_scores > threshold


def count_total_outliers(df: pd.DataFrame, method: str = "iqr") -> int:
    """Count total outlier cells across all numeric columns."""
    total_outliers = 0
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        mask = detect_outliers_iqr(df[col]) if method == "iqr" else detect_outliers_zscore(df[col])
        total_outliers += int(mask.sum())
    return total_outliers


# ==============================================================================
# 3. DATA WRANGLING PIPELINE
# ==============================================================================

class HealthDataWrangler:
    """
    Comprehensive Data Wrangling Pipeline.
    Encapsulates dirty data cleaning, biological zero resolution,
    missing value imputation, duplicate removal, outlier handling, and scaling.
    """

    def __init__(self,
                 handle_bio_zeros: bool = True,
                 imputation_strategy: str = "median",
                 remove_duplicates: bool = True,
                 outlier_strategy: str = "cap",
                 outlier_method: str = "iqr",
                 scaling_method: Optional[str] = None):
        self.handle_bio_zeros = handle_bio_zeros
        self.imputation_strategy = imputation_strategy  # mean, median, mode, knn, ffill
        self.remove_duplicates = remove_duplicates
        self.outlier_strategy = outlier_strategy        # none, cap, remove
        self.outlier_method = outlier_method            # iqr, zscore
        self.scaling_method = scaling_method            # standard, minmax, None
        self.scaler = None
        self.imputer = None
        self.numeric_cols = []
        self.categorical_cols = []

    def fit_transform(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute full data wrangling pipeline on DataFrame.
        Returns cleaned DataFrame and before/after audit metrics.
        """
        df_before = df.copy()
        df_cleaned = df.copy()

        # Audit Before metrics
        audit_before = self._calculate_audit_metrics(df_before)

        # 1. Deduplication
        if self.remove_duplicates:
            df_cleaned = df_cleaned.drop_duplicates().reset_index(drop=True)

        # 2. Biological Zero Handling
        # In health records, Glucose=0, BP=0, BMI=0, etc. are invalid/missing.
        if self.handle_bio_zeros:
            for col in BIOLOGICAL_ZERO_COLUMNS:
                if col in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    df_cleaned.loc[df_cleaned[col] <= 0, col] = np.nan

        # 3. Invalid health data filter (e.g. negative age, negative BP)
        for col in df_cleaned.select_dtypes(include=["number"]).columns:
            if col != target_column:
                # Any negative value for physiological measures is invalid
                df_cleaned.loc[df_cleaned[col] < 0, col] = np.nan

        # 4. Outlier Handling (Cap or Drop)
        feature_cols = [c for c in df_cleaned.select_dtypes(include=["number"]).columns if c != target_column]
        if self.outlier_strategy != "none":
            for col in feature_cols:
                mask = (detect_outliers_iqr(df_cleaned[col]) if self.outlier_method == "iqr"
                        else detect_outliers_zscore(df_cleaned[col]))

                if self.outlier_strategy == "cap":
                    # Winsorize / Cap at bounds
                    q1 = df_cleaned[col].quantile(0.25)
                    q3 = df_cleaned[col].quantile(0.75)
                    iqr = q3 - q1
                    low = q1 - 1.5 * iqr
                    high = q3 + 1.5 * iqr
                    df_cleaned[col] = df_cleaned[col].clip(lower=low, upper=high)
                elif self.outlier_strategy == "remove":
                    df_cleaned = df_cleaned[~mask].reset_index(drop=True)

        # 5. Missing Value Imputation
        num_features = [c for c in df_cleaned.select_dtypes(include=["number"]).columns if c != target_column]
        if len(num_features) > 0 and df_cleaned[num_features].isnull().sum().sum() > 0:
            if self.imputation_strategy == "knn":
                self.imputer = KNNImputer(n_neighbors=5)
                df_cleaned[num_features] = self.imputer.fit_transform(df_cleaned[num_features])
            elif self.imputation_strategy in ["mean", "median", "most_frequent"]:
                self.imputer = SimpleImputer(strategy=self.imputation_strategy)
                df_cleaned[num_features] = self.imputer.fit_transform(df_cleaned[num_features])
            elif self.imputation_strategy == "ffill":
                df_cleaned[num_features] = df_cleaned[num_features].ffill().bfill()
            else:
                self.imputer = SimpleImputer(strategy="median")
                df_cleaned[num_features] = self.imputer.fit_transform(df_cleaned[num_features])

        # Any remaining NaNs in target if present
        if target_column and target_column in df_cleaned.columns:
            df_cleaned = df_cleaned.dropna(subset=[target_column]).reset_index(drop=True)

        # Ensure no duplicate rows remain if deduplication was requested
        if self.remove_duplicates:
            df_cleaned = df_cleaned.drop_duplicates().reset_index(drop=True)

        # 6. Optional Feature Scaling
        if self.scaling_method:
            if self.scaling_method == "standard":
                self.scaler = StandardScaler()
            elif self.scaling_method == "minmax":
                self.scaler = MinMaxScaler()

            if self.scaler:
                df_cleaned[num_features] = self.scaler.fit_transform(df_cleaned[num_features])

        # Audit After metrics
        audit_after = self._calculate_audit_metrics(df_cleaned)

        comparison_audit = {
            "before": audit_before,
            "after": audit_after,
            "diff_table": pd.DataFrame({
                "Metric": ["Total Rows", "Missing Values", "Duplicate Rows", "Invalid Values (<=0)", "Outliers (IQR)"],
                "BEFORE": [
                    audit_before["rows"],
                    audit_before["missing"],
                    audit_before["duplicates"],
                    audit_before["invalid_zeros"],
                    audit_before["outliers"]
                ],
                "AFTER": [
                    audit_after["rows"],
                    audit_after["missing"],
                    audit_after["duplicates"],
                    audit_after["invalid_zeros"],
                    audit_after["outliers"]
                ]
            })
        }

        return df_cleaned, comparison_audit

    @staticmethod
    def _calculate_audit_metrics(df: pd.DataFrame) -> Dict[str, int]:
        """Compute key audit numbers for comparison table."""
        invalid_zeros = 0
        for col in BIOLOGICAL_ZERO_COLUMNS:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                invalid_zeros += int((df[col] <= 0).sum())

        return {
            "rows": len(df),
            "missing": int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "invalid_zeros": invalid_zeros,
            "outliers": count_total_outliers(df, method="iqr")
        }
