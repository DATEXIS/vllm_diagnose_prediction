"""Tests for the minimal JSON parser.

Contract:
  * Find the LAST balanced top-level JSON object in the response.
  * Validate it against ICDsModel.
  * No repair, no regex fallbacks. Raise on invalid input.
"""

import json

import pytest

from src.utils.parsing_utils import (
    JSONExtractionError,
    extract_last_json_object,
    parse_prediction,
    parse_prediction_codes,
)


def _payload(codes):
    return json.dumps({"diagnoses": [{"icd_code": c, "reason": "r"} for c in codes]})


class TestExtractLastJsonObject:
    def test_clean_single_object(self):
        s = _payload(["I10"])
        assert extract_last_json_object(s) == s

    def test_picks_last_when_multiple_objects(self):
        first = _payload(["E11"])
        last = _payload(["I10", "K35"])
        s = f"thinking... {first}\nfinal: {last}"
        assert extract_last_json_object(s) == last

    def test_braces_inside_strings_are_ignored(self):
        target = '{"diagnoses": [{"icd_code": "I10", "reason": "BP {high}"}]}'
        assert extract_last_json_object(target) == target

    def test_no_json_raises(self):
        with pytest.raises(JSONExtractionError):
            extract_last_json_object("just thinking, no JSON here")

    def test_unbalanced_truncated_raises(self):
        # Truncated payload, no closing brace at all.
        with pytest.raises(JSONExtractionError):
            extract_last_json_object('{"diagnoses": [{"icd_code": "I10"')


class TestParsePrediction:
    def test_valid_prediction(self):
        s = _payload(["I10", "E11"])
        model = parse_prediction(s)
        assert [d.icd_code for d in model.diagnoses] == ["I10", "E11"]

    def test_takes_last_block(self):
        first = _payload(["E11"])
        last = _payload(["I10"])
        s = f"thinking... {first}\nactual: {last}"
        assert parse_prediction_codes(s) == ["I10"]

    def test_schema_mismatch_raises(self):
        # Valid JSON, wrong shape.
        with pytest.raises(JSONExtractionError):
            parse_prediction(json.dumps({"not_diagnoses": []}))

    def test_empty_response_raises(self):
        with pytest.raises(JSONExtractionError):
            parse_prediction("")
