"""
HealthSense Database Layer.
Manages SQLite storage for datasets, model versions, prediction history, and data quality reports.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from utils.config import DATABASE_PATH


def get_db_connection():
    """Create and return a thread-safe connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_PATH, timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables if they do not already exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Datasets Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            rows_count INTEGER NOT NULL,
            cols_count INTEGER NOT NULL,
            target_col TEXT,
            file_hash TEXT,
            notes TEXT
        )
    """)

    # 2. Data Quality Reports Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS data_quality_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_records INTEGER,
            completeness_score REAL,
            validity_score REAL,
            consistency_score REAL,
            uniqueness_score REAL,
            overall_score REAL,
            report_json TEXT,
            FOREIGN KEY (dataset_id) REFERENCES datasets(id)
        )
    """)

    # 3. Models Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            dataset_id INTEGER,
            features_json TEXT,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1_score REAL,
            roc_auc REAL,
            cv_mean REAL,
            cv_std REAL,
            training_time REAL,
            is_best INTEGER DEFAULT 0,
            hyperparameters_json TEXT
        )
    """)

    # 4. Predictions Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_algorithm TEXT NOT NULL,
            risk_probability REAL NOT NULL,
            risk_category TEXT NOT NULL,
            patient_features_json TEXT NOT NULL,
            top_factors_json TEXT,
            notes TEXT
        )
    """)

    conn.commit()
    conn.close()


# Ensure tables exist at module load
init_db()


def log_dataset(filename: str, rows: int, cols: int, target_col: str, file_hash: str = "", notes: str = "") -> int:
    """Log an uploaded dataset record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO datasets (filename, rows_count, cols_count, target_col, file_hash, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (filename, rows, cols, target_col, file_hash, notes))
    dataset_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return dataset_id


def log_quality_report(dataset_id: Optional[int], total_records: int, completeness: float,
                       validity: float, consistency: float, uniqueness: float,
                       overall: float, report_dict: Dict[str, Any]) -> int:
    """Save automated data quality report."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO data_quality_reports
        (dataset_id, total_records, completeness_score, validity_score, consistency_score, uniqueness_score, overall_score, report_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (dataset_id, total_records, completeness, validity, consistency, uniqueness, overall, json.dumps(report_dict)))
    report_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return report_id


def log_model_metadata(version: str, algorithm: str, features: List[str], metrics: Dict[str, Any],
                       is_best: bool = False, dataset_id: Optional[int] = None, params: Optional[Dict] = None) -> int:
    """Save trained model metadata into SQLite."""
    conn = get_db_connection()
    cursor = conn.cursor()

    if is_best:
        cursor.execute("UPDATE models SET is_best = 0")

    cursor.execute("""
        INSERT INTO models
        (model_version, algorithm, dataset_id, features_json, accuracy, precision, recall, f1_score, roc_auc, cv_mean, cv_std, training_time, is_best, hyperparameters_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        version,
        algorithm,
        dataset_id,
        json.dumps(features),
        metrics.get("accuracy"),
        metrics.get("precision"),
        metrics.get("recall"),
        metrics.get("f1"),
        metrics.get("roc_auc"),
        metrics.get("cv_mean"),
        metrics.get("cv_std"),
        metrics.get("training_time"),
        1 if is_best else 0,
        json.dumps(params or {})
    ))
    model_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return model_id


def get_all_models() -> pd.DataFrame:
    """Retrieve all trained model records as a DataFrame."""
    conn = get_db_connection()
    query = """
        SELECT id, model_version, algorithm, training_date, accuracy, precision, recall, f1_score, roc_auc, cv_mean, training_time, is_best
        FROM models
        ORDER BY id DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def log_prediction(algorithm: str, probability: float, category: str,
                   patient_features: Dict[str, Any], top_factors: Optional[List[Dict]] = None,
                   notes: str = "") -> int:
    """Save a patient risk prediction into the SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions
        (model_algorithm, risk_probability, risk_category, patient_features_json, top_factors_json, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        algorithm,
        probability,
        category,
        json.dumps(patient_features),
        json.dumps(top_factors or []),
        notes
    ))
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return prediction_id


def get_predictions(category_filter: Optional[str] = None,
                    algorithm_filter: Optional[str] = None,
                    limit: int = 100) -> pd.DataFrame:
    """Retrieve prediction history with optional filters."""
    conn = get_db_connection()
    query = "SELECT * FROM predictions WHERE 1=1"
    params = []

    if category_filter and category_filter != "ALL":
        query += " AND risk_category = ?"
        params.append(category_filter)

    if algorithm_filter and algorithm_filter != "ALL":
        query += " AND model_algorithm = ?"
        params.append(algorithm_filter)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def delete_prediction(prediction_id: int):
    """Delete a single prediction record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
    conn.commit()
    conn.close()


def clear_all_predictions():
    """Clear all prediction history."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
