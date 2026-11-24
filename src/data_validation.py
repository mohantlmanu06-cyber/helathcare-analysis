"""
HealthSense Data Validation Module.
Performs pre-flight checks on dataset files, target distributions, and clinical patient inputs.
"""

from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
from utils.config import PHYSIOLOGICAL_LIMITS

MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = [".csv", ".xlsx", ".xls"]


def validate_uploaded_file_size(file_size: int) -> Tuple[bool, Optional[str]]:
    """Validate that uploaded file is within safe memory limits."""
    if file_size > MAX_UPLOAD_SIZE_BYTES:
        return False, f"File size ({file_size / (1024*1024):.1f}MB) exceeds 50MB security threshold."
    return True, None


def validate_dataset_structure(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate dataset readiness for Data Mining and Wrangling.
    Checks row count, column count, target existence, and class balance.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if df is None or df.empty:
        errors.append("Dataset is empty.")
        return {"is_valid": False, "errors": errors, "warnings": warnings}

    if len(df) < 30:
        errors.append(f"Insufficient sample size ({len(df)} rows). At least 30 records required for reliable mining.")

    if df.shape[1] < 2:
        errors.append(f"Insufficient features ({df.shape[1]} columns). Dataset requires features and a target.")

    # Target column validation
    if target_column:
        if target_column not in df.columns:
            errors.append(f"Selected target column '{target_column}' is not present in the dataset.")
        else:
            unique_targets = df[target_column].dropna().unique()
            if len(unique_targets) < 2:
                errors.append(f"Target column '{target_column}' must have at least 2 distinct classes (found {len(unique_targets)}).")
            elif len(unique_targets) > 10:
                warnings.append(f"Target column has {len(unique_targets)} distinct values. Is this a classification task?")

    # Check for completely empty columns
    all_null_cols = df.columns[df.isnull().all()].tolist()
    if all_null_cols:
        warnings.append(f"Columns with 100% missing values detected: {all_null_cols}")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def validate_patient_input(inputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate individual patient feature inputs against biological boundaries.
    Ensures non-negative, plausible clinical ranges.
    """
    errors: List[str] = []

    for feature, value in inputs.items():
        if value is None:
            errors.append(f"Missing required input for {feature}.")
            continue

        try:
            val_num = float(value)
        except (ValueError, TypeError):
            errors.append(f"Feature '{feature}' must be a valid numeric quantity.")
            continue

        # Check negative values
        if val_num < 0:
            errors.append(f"'{feature}' cannot be negative (provided: {val_num}).")

        # Check physiological plausibility
        if feature in PHYSIOLOGICAL_LIMITS:
            limits = PHYSIOLOGICAL_LIMITS[feature]
            if val_num < limits["min"] or val_num > limits["max"]:
                errors.append(
                    f"{limits['label']} ({val_num} {limits['unit']}) is outside clinical plausibility range "
                    f"[{limits['min']} - {limits['max']} {limits['unit']}]."
                )

    return len(errors) == 0, errors
