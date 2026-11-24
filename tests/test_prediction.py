"""
Unit tests for HealthSense Patient Risk Prediction and What-If Simulation.
"""

import unittest
from src.data_validation import validate_patient_input
from src.prediction import (
    predict_patient_risk, simulate_what_if_scenario,
    get_cohort_baseline_medians
)
from utils.database import get_predictions, clear_all_predictions


class TestPrediction(unittest.TestCase):

    def setUp(self):
        self.valid_patient = {
            "Pregnancies": 2,
            "Glucose": 155,
            "BloodPressure": 80,
            "SkinThickness": 25,
            "Insulin": 120,
            "BMI": 33.5,
            "DiabetesPedigreeFunction": 0.55,
            "Age": 42,
        }

    def test_patient_input_validation(self):
        # Valid input
        is_valid, errors = validate_patient_input(self.valid_patient)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

        # Invalid negative age
        invalid_patient = self.valid_patient.copy()
        invalid_patient["Age"] = -5
        is_valid, errors = validate_patient_input(invalid_patient)
        self.assertFalse(is_valid)
        self.assertTrue(any("cannot be negative" in e for e in errors))

        # Extreme impossible value
        extreme_patient = self.valid_patient.copy()
        extreme_patient["Glucose"] = 800
        is_valid, errors = validate_patient_input(extreme_patient)
        self.assertFalse(is_valid)
        self.assertTrue(any("clinical plausibility" in e for e in errors))

    def test_prediction_inference(self):
        pred = predict_patient_risk(self.valid_patient, save_to_db=True)
        self.assertIn("probability", pred)
        self.assertIn("category", pred)
        self.assertTrue(0.0 <= pred["probability"] <= 1.0)
        self.assertIn(pred["category"], ["LOW", "MODERATE", "HIGH"])

    def test_what_if_scenario_simulation(self):
        healthy_scenario = self.valid_patient.copy()
        healthy_scenario["Glucose"] = 95  # Lowered glucose
        healthy_scenario["BMI"] = 23.0    # Lowered BMI

        from utils.config import BEST_MODEL_PATH, PREPROCESSOR_PATH
        import joblib
        model = joblib.load(BEST_MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)

        sim = simulate_what_if_scenario(self.valid_patient, healthy_scenario, model, preprocessor)
        self.assertIn("baseline", sim)
        self.assertIn("scenario", sim)
        # Improving glucose and BMI should reduce risk
        self.assertLessEqual(sim["scenario"]["probability"], sim["baseline"]["probability"])
        self.assertEqual(sim["direction"], "REDUCTION")

    def test_database_logging_and_filtering(self):
        # Clear predictions table
        clear_all_predictions()

        # Run 2 predictions
        predict_patient_risk(self.valid_patient, save_to_db=True)
        predict_patient_risk(self.valid_patient, save_to_db=True)

        history = get_predictions()
        self.assertEqual(len(history), 2)


if __name__ == "__main__":
    unittest.main()
