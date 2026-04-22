import sys
import os
import logging

# Add src to path so we can import parsing_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.parsing_utils import safe_parse_json

# Configure logging to see failures
logging.basicConfig(level=logging.INFO)

def test_parsing():
    test_cases = [
        {
            "name": "Clean JSON",
            "input": '{"diagnoses": [{"icd_code": "I10", "reason": "High blood pressure"}]}',
            "expected": ["I10"]
        },
        {
            "name": "Thinking + JSON",
            "input": 'I have analyzed the note. Here are the codes: {"diagnoses": [{"icd_code": "E11.9", "reason": "Diabetes"}]}',
            "expected": ["E11.9"]
        },
        {
            "name": "Unescaped Newlines",
            "input": '{"diagnoses": [{"icd_code": "K86.0", "reason": "Chronic pancreatitis\nPatient has symptoms"}]}',
            "expected": ["K86.0"]
        },
        {
            "name": "Truncated Mid-String (User Example)",
            "input": '{"diagnoses": [{"icd_code": "K86.0", "reason": "The patient has a history of chronic pancreatitis, which is a known condition, and presents with worsening symptoms including nausea, vomiting, and ab',
            "expected": ["K86.0"]
        },
        {
            "name": "Truncated Mid-Object (Multiple Items)",
            "input": '{"diagnoses": [{"icd_code": "I10", "reason": "BP"}, {"icd_code": "E11", "reason": "Diab',
            "expected": ["I10", "E11"]
        },
        {
            "name": "Deeply Truncated (Only Header)",
            "input": '{"diagnoses": [',
            "expected": []
        }
    ]

    print("\n--- Running Parsing Tests ---")
    passed = 0
    for case in test_cases:
        result = safe_parse_json(case["input"])
        if result == case["expected"]:
            print(f"[PASS] {case['name']}")
            passed += 1
        else:
            print(f"[FAIL] {case['name']}")
            print(f"       Input: {case['input'][:100]}...")
            print(f"       Expected: {case['expected']}")
            print(f"       Got:      {result}")

    print(f"\nSummary: {passed}/{len(test_cases)} cases passed.")
    
    if passed == len(test_cases):
        print("All tests passed! 🚀")
    else:
        print("Some tests failed. ❌")
        sys.exit(1)

if __name__ == "__main__":
    test_parsing()
