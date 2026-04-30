"""Tests for ICD normalization and metric calculation."""

import pytest

from src.data.evaluate import calculate_metrics, normalize_icd, safe_parse_true_labels


class TestNormalizeICD:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("K86.0", "K86"),
            ("k86.0", "K86"),
            ("I10", "I10"),
            ("A12.34", "A12"),
            ("123.45", "123"),
            ("", ""),
            (None, ""),
            ("Abc.def", "ABC"),
        ],
    )
    def test_normalize(self, raw, expected):
        assert normalize_icd(raw) == expected


class TestSafeParseTrueLabels:
    def test_list(self):
        assert safe_parse_true_labels(["A10", "B20"]) == ["A10", "B20"]

    def test_stringified_list(self):
        assert safe_parse_true_labels("['A10', 'B20']") == ["A10", "B20"]

    def test_comma_separated_fallback(self):
        assert safe_parse_true_labels("A10, B20") == ["A10", "B20"]

    def test_none(self):
        assert safe_parse_true_labels(None) == []


class TestCalculateMetrics:
    def test_perfect_match(self):
        m = calculate_metrics([["A10", "B20"]], [["A10", "B20"]])
        assert m["micro"]["f1"] == 1.0

    def test_partial_match(self):
        # Patient 1: True [A,B], Pred [A] -> 1 TP, 1 FN, 0 FP
        # Patient 2: True [C], Pred [C,D] -> 1 TP, 0 FN, 1 FP
        # Micro: TP=2, FP=1, FN=1; P=R=2/3 => F1=2/3
        m = calculate_metrics([["A", "B"], ["C"]], [["A"], ["C", "D"]])
        assert m["micro"]["f1"] == pytest.approx(2 / 3)
