"""
HealthSense Data Loader Module.
Handles safe ingestion of CSV and Excel datasets, metadata extraction, and sample previews.
"""

import io
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
from utils.config import DEFAULT_DATASET_PATH, SAMPLE_DIRTY_DATASET_PATH


def format_memory_size(bytes_val: int) -> str:
    """Format memory bytes into human-readable string (KB, MB)."""
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.2f} KB"
    else:
        return f"{bytes_val / (1024 * 1024):.2f} MB"


def get_dataset_metadata(df: pd.DataFrame, name: str = "Uploaded Dataset") -> Dict[str, Any]:
    """Extract fundamental structural metadata from DataFrame."""
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    memory_bytes = df.memory_usage(deep=True).sum()

    return {
        "name": name,
        "rows": len(df),
        "columns": df.shape[1],
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage": format_memory_size(memory_bytes),
        "memory_bytes": int(memory_bytes),
        "column_names": df.columns.tolist(),
    }


def load_uploaded_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Safely parse an uploaded Streamlit file object (CSV or Excel).
    Returns (DataFrame, error_message).
    """
    if uploaded_file is None:
        return None, "No file provided."

    file_name = uploaded_file.name.lower()
    try:
        if file_name.endswith(".csv"):
            # Try utf-8 first, fallback to latin-1
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin-1")
        elif file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file format. Please upload a .csv, .xlsx, or .xls file."

        if df.empty:
            return None, "Uploaded dataset is empty."

        return df, None
    except Exception as e:
        return None, f"Failed to parse file: {str(e)}"


def load_default_dataset() -> pd.DataFrame:
    """Load the standard canonical Pima Indians Diabetes dataset."""
    if not DEFAULT_DATASET_PATH.exists():
        raise FileNotFoundError(f"Canonical dataset not found at {DEFAULT_DATASET_PATH}")
    return pd.read_csv(DEFAULT_DATASET_PATH)


def load_dirty_sample_dataset() -> pd.DataFrame:
    """Load the sample dirty dataset with known duplicates, NaNs, and anomalies."""
    if not SAMPLE_DIRTY_DATASET_PATH.exists():
        raise FileNotFoundError(f"Dirty sample dataset not found at {SAMPLE_DIRTY_DATASET_PATH}")
    return pd.read_csv(SAMPLE_DIRTY_DATASET_PATH)


def get_data_previews(df: pd.DataFrame, sample_size: int = 5) -> Dict[str, pd.DataFrame]:
    """Return dictionary with head, tail, and random sample previews."""
    sample_n = min(sample_size, len(df))
    return {
        "head": df.head(sample_n),
        "tail": df.tail(sample_n),
        "sample": df.sample(sample_n, random_state=42) if len(df) > 0 else df,
    }
