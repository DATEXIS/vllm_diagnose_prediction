import sys
import os
import unittest

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluate import normalize_icd, calculate_metrics

class TestEvaluation(unittest.TestCase):
    
    def test_normalization(self):
        test_cases = [
            ("K86.0", "K86"),
            ("k86.0", "K86"),
            ("I10", "I10"),
            ("A12.34", "A12"),
            ("123.45", "123"),
            ("", ""),
            (None, ""),
            ("Abc.def", "ABC"),
        ]
        
        for input_code, expected in test_cases:
            with self.subTest(input_code=input_code):
                self.assertEqual(normalize_icd(input_code), expected)

    def test_metrics_calculation(self):
        # Basic case: Perfect match
        y_true = [["A10", "B20"]]
        y_pred = [["A10", "B20"]]
        metrics = calculate_metrics(y_true, y_pred)
        self.assertEqual(metrics['micro']['f1'], 1.0)
        
        # Case: Partial match
        # Patient 1: True [A, B], Pred [A] -> 1 TP, 1 FN, 0 FP
        # Patient 2: True [C], Pred [C, D] -> 1 TP, 0 FN, 1 FP
        y_true = [["A", "B"], ["C"]]
        y_pred = [["A"], ["C", "D"]]
        metrics = calculate_metrics(y_true, y_pred)
        
        # Micro F1: 
        # Total TPs = 1 (A) + 1 (C) = 2
        # Total FPs = 1 (D) = 1
        # Total FNs = 1 (B) = 1
        # Precision = 2 / (2 + 1) = 0.666...
        # Recall = 2 / (2 + 1) = 0.666...
        # F1 = 0.666...
        self.assertAlmostEqual(metrics['micro']['f1'], 0.66666666666)

if __name__ == "__main__":
    unittest.main()
