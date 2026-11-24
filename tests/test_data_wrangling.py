"""
Unit tests for HealthSense Data Wrangling and Automated Data Quality Engine.
"""

import unittest
import numpy as np
import pandas as pd
from src.data_loader import load_default_dataset, load_dirty_sample_dataset
from src.data_cleaning import (
    compute_data_quality_report, HealthDataWrangler,
    detect_outliers_iqr, detect_outliers_zscore
)
from src.feature_engineering import engineer_health_features


class TestDataWrangling(unittest.TestCase):

    def setUp(self):
        self.df_clean = load_default_dataset()
        self.df_dirty = load_dirty_sample_dataset()

    def test_default_dataset_loaded(self):
        self.assertEqual(len(self.df_clean), 768)
        self.assertIn("Outcome", self.df_clean.columns)

    def test_data_quality_report_clean_vs_dirty(self):
        report_clean = compute_data_quality_report(self.df_clean)
        report_dirty = compute_data_quality_report(self.df_dirty)

        # Both scores should be in [0, 100]
        self.assertTrue(0 <= report_clean["scores"]["overall"] <= 100)
        self.assertTrue(0 <= report_dirty["scores"]["overall"] <= 100)

        # Dirty dataset has duplicates and explicit invalid values so quality should be lower
        self.assertGreater(report_dirty["duplicate_records"], 0)
        self.assertGreater(report_dirty["invalid_values"], 0)

    def test_data_wrangler_pipeline(self):
        wrangler = HealthDataWrangler(
            handle_bio_zeros=True,
            imputation_strategy="median",
            remove_duplicates=True,
            outlier_strategy="cap",
            outlier_method="iqr"
        )
        df_wrangled, audit = premature_result = wrangler.fit_transform(self.df_dirty, target_column="Outcome")

        # Duplicates should be 0 after cleaning
        self.assertEqual(audit["after"]["duplicates"], 0)
        # Missing values should be 0 after imputation
        self.assertEqual(audit["after"]["missing"], 0)
        # Invalid values (<=0 in vital columns) should be 0
        self.assertEqual(audit["after"]["invalid_zeros"], 0)

        # Diff table must contain 5 rows
        diff_table = audit["diff_table"]
        self.assertEqual(len(diff_table), 5)
        self.assertIn("BEFORE", diff_table.columns)
        self.assertIn("AFTER", diff_table.columns)

    def test_outlier_detection(self):
        s = pd.Series([10, 12, 11, 10, 12, 11, 10, 150])  # 150 is an outlier
        iqr_mask = detect_outliers_iqr(s)
        self.assertTrue(iqr_mask.iloc[-1])
        self.assertFalse(iqr_mask.iloc[0])

    def test_feature_engineering(self):
        df_feat = engineer_health_features(self.df_clean)
        self.assertIn("BMI_Category", df_feat.columns)
        self.assertIn("Age_Group", df_feat.columns)
        self.assertIn("Glucose_Risk", df_feat.columns)
        self.assertIn("BP_Category", df_feat.columns)
        self.assertIn("Glucose_BMI_Index", df_feat.columns)


if __name__ == "__main__":
    unittest.main()
