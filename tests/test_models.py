"""
Unit tests for HealthSense Classification, Clustering, and Association Rules Engines.
"""

import unittest
import numpy as np
import pandas as pd
from src.data_loader import load_default_dataset
from src.classification import HealthClassifierTrainer
from src.clustering import compute_clustering_metrics, perform_kmeans_clustering
from src.association_rules import mine_association_rules
from src.explainability import get_global_feature_importance


class TestModelsAndMining(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = load_default_dataset()
        cls.trainer = HealthClassifierTrainer(test_size=0.2, random_state=42)
        cls.results_df, cls.artifacts = cls.trainer.train_and_evaluate_all(cls.df, target_column="Outcome")

    def test_classifier_training_metrics(self):
        # Must train 5 models
        self.assertEqual(len(self.results_df), 5)
        expected_models = {
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "K-Nearest Neighbors (KNN)",
            "Support Vector Machine (SVM)"
        }
        self.assertEqual(set(self.results_df["Model"]), expected_models)

        # Check that metrics are reasonable real numbers
        for _, row in self.results_df.iterrows():
            self.assertGreater(row["Accuracy"], 0.50)
            self.assertGreater(row["ROC-AUC"], 0.50)
            self.assertGreater(row["Precision"], 0.0)
            self.assertGreater(row["Recall"], 0.0)

    def test_best_model_selection(self):
        self.assertIsNotNone(self.artifacts["best_model"])
        self.assertTrue(len(self.artifacts["best_model_name"]) > 0)

    def test_global_feature_importance(self):
        best_clf = self.artifacts["best_model"]
        features = self.artifacts["feature_names"]
        imp_df = get_global_feature_importance(best_clf, features, self.artifacts["X_test"], self.artifacts["y_test"])
        self.assertEqual(len(imp_df), len(features))
        self.assertAlmostEqual(imp_df["Importance"].sum(), 1.0, places=2)

    def test_kmeans_clustering(self):
        feature_cols = ["Age", "BMI", "Glucose", "BloodPressure"]
        metrics = compute_clustering_metrics(self.df, feature_cols, max_k=4)
        self.assertIn("optimal_k", metrics)
        self.assertGreater(len(metrics["silhouettes"]), 0)

        df_clustered, summary = perform_kmeans_clustering(self.df, feature_cols, n_clusters=3)
        self.assertIn("Cluster", df_clustered.columns)
        self.assertEqual(len(summary["descriptions"]), 3)

    def test_association_rules_mining(self):
        freq_sets, rules = mine_association_rules(self.df, min_support=0.05, min_confidence=0.30)
        self.assertFalse(freq_sets.empty)
        if not rules.empty:
            self.assertIn("Antecedents", rules.columns)
            self.assertIn("Consequents", rules.columns)
            self.assertIn("lift", rules.columns)
            self.assertGreater(rules["lift"].iloc[0], 0)


if __name__ == "__main__":
    unittest.main()
