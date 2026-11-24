"""
HealthSense Classification and Machine Learning Engine.
Implements 5 supervised algorithms with rigorous data leakage prevention:
Preprocessing (Imputation & Scaling) is fit strictly on training splits.
Calculates real test evaluation metrics, performs 5-fold cross-validation, and persists models.
"""

import time
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# 5 Required Classification Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from utils.config import (
    RANDOM_STATE, BEST_MODEL_PATH, PREPROCESSOR_PATH,
    BIOLOGICAL_ZERO_COLUMNS
)
from utils.database import log_model_metadata


class HealthClassifierTrainer:
    """
    Supervised learning workbench for comparing 5 diagnostic models.
    Guarantees no data leakage by encapsulating preprocessing within training splits.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = RANDOM_STATE):
        self.test_size = test_size
        self.random_state = random_state
        self.feature_names: List[str] = []
        self.target_name: str = ""
        self.preprocessor = None
        self.trained_models: Dict[str, Any] = {}
        self.results_df: Optional[pd.DataFrame] = None
        self.best_model_name: str = ""
        self.best_model = None

    def prepare_data(self, df: pd.DataFrame, target_column: str = "Outcome") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Extract features/target and execute stratified train-test split.
        Converts biological zeros to NaNs before splitting.
        """
        self.target_name = target_column
        df_clean = df.copy()

        # Mark biological zeros as missing
        for col in BIOLOGICAL_ZERO_COLUMNS:
            if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean.loc[df_clean[col] <= 0, col] = np.nan

        # Filter out rows with missing target
        df_clean = df_clean.dropna(subset=[target_column]).reset_index(drop=True)

        # Feature selection (all numerical except target)
        self.feature_names = [c for c in df_clean.select_dtypes(include=["number"]).columns if c != target_column]

        X = df_clean[self.feature_names]
        y = df_clean[target_column].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        return X_train, X_test, y_train, y_test

    def train_and_evaluate_all(self, df: pd.DataFrame, target_column: str = "Outcome",
                               scaling: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Train all 5 classifiers using identical stratified train-test split.
        Fits preprocessing (Imputer + Scaler) ONLY on training data.
        """
        X_train_raw, X_test_raw, y_train, y_test = self.prepare_data(df, target_column)

        # Create preprocessing pipeline: Impute (median) -> Scale (StandardScaler)
        # FIT ONLY ON TRAINING SPLIT TO PREVENT DATA LEAKAGE
        steps = [('imputer', SimpleImputer(strategy='median'))]
        if scaling:
            steps.append(('scaler', StandardScaler()))

        self.preprocessor = Pipeline(steps)
        X_train = self.preprocessor.fit_transform(X_train_raw)
        X_test = self.preprocessor.transform(X_test_raw)

        # Save fitted preprocessor
        joblib.dump(self.preprocessor, PREPROCESSOR_PATH)

        # Define 5 Classifiers
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=self.random_state),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=self.random_state),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=self.random_state),
            "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
            "Support Vector Machine (SVM)": SVC(probability=True, kernel='rbf', random_state=self.random_state)
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        benchmark_records = []
        detailed_evals = {}

        for name, clf in models.items():
            start_time = time.time()
            clf.fit(X_train, y_train)
            train_duration = time.time() - start_time

            # Test predictions
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)

            # Test Metrics (real test data)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_test, y_prob)
            except Exception:
                auc = 0.5

            # 5-fold Cross-Validation on training set
            cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='roc_auc')

            self.trained_models[name] = clf

            record = {
                "Model": name,
                "Accuracy": round(float(acc), 4),
                "Precision": round(float(prec), 4),
                "Recall": round(float(rec), 4),
                "F1 Score": round(float(f1), 4),
                "ROC-AUC": round(float(auc), 4),
                "CV Score (Mean)": round(float(cv_scores.mean()), 4),
                "CV Score (Std)": round(float(cv_scores.std()), 4),
                "Training Time (s)": round(float(train_duration), 4),
            }
            benchmark_records.append(record)

            detailed_evals[name] = {
                "y_true": y_test.tolist(),
                "y_pred": y_pred.tolist(),
                "y_prob": y_prob.tolist(),
                "metrics": record,
                "model_obj": clf
            }

        results_df = pd.DataFrame(benchmark_records).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
        self.results_df = results_df

        # Identify best model by ROC-AUC
        self.best_model_name = results_df.iloc[0]["Model"]
        self.best_model = self.trained_models[self.best_model_name]

        # Persist best model to disk
        joblib.dump(self.best_model, BEST_MODEL_PATH)

        # Log metadata to SQLite database
        for record in benchmark_records:
            is_best = (record["Model"] == self.best_model_name)
            log_model_metadata(
                version="1.0.0",
                algorithm=record["Model"],
                features=self.feature_names,
                metrics={
                    "accuracy": record["Accuracy"],
                    "precision": record["Precision"],
                    "recall": record["Recall"],
                    "f1": record["F1 Score"],
                    "roc_auc": record["ROC-AUC"],
                    "cv_mean": record["CV Score (Mean)"],
                    "cv_std": record["CV Score (Std)"],
                    "training_time": record["Training Time (s)"]
                },
                is_best=is_best
            )

        artifacts = {
            "results_df": results_df,
            "best_model_name": self.best_model_name,
            "best_model": self.best_model,
            "feature_names": self.feature_names,
            "detailed_evals": detailed_evals,
            "X_test": X_test,
            "y_test": y_test,
            "X_train": X_train,
            "y_train": y_train,
            "preprocessor": self.preprocessor
        }

        return results_df, artifacts
