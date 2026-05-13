"""Tests for the admission note section parser."""

import pytest

from src.utils.admission_note_parser import filter_sections, parse_sections

SAMPLE_NOTE = """\
CHIEF COMPLAINT: ___ pain, fever

PRESENT ILLNESS: PCP:  ___

MEDICAL HISTORY: T1DM

MEDICATION ON ADMISSION:

ALLERGIES: No Known Allergies / Adverse Drug Reactions

PHYSICAL EXAM: Gen: young woman laying bed in no acute distress, answering questions appropriately  Vitals:T98.8, BP92/56, HR 89, RR16, O2Sat 100RA     HEENT: Anicteric, eyes conjugate, MMM, no JVD Cardiovascular: RRR no MRG, nl. S1 and S2 Pulmonary: Lung fields clear to auscultation throughout Gastroinestinal: Soft, non-tender, non-distended, bowel sounds present, no HSM, mild left sided CVA tenderness  MSK: No edema Skin: No rashes or ulcerations evident Neurological: Alert, interactive, speech fluent, face symmetric, moving all extremities Psychiatric: pleasant, appropriate affect

FAMILY HISTORY: maternal aunt with thyroid issues - otherwise no autoimmune in family, no CVA/MI/cancer.

SOCIAL HISTORY: ___ FAMILY HISTORY:  maternal aunt with thyroid issues - otherwise no autoimmune in family, no CVA/MI/cancer.\
"""

SECTION_NAMES = [
    "CHIEF COMPLAINT",
    "PRESENT ILLNESS",
    "MEDICAL HISTORY",
    "MEDICATION ON ADMISSION",
    "ALLERGIES",
    "PHYSICAL EXAM",
    "FAMILY HISTORY",
    "SOCIAL HISTORY",
]

IGNORE_PHRASES = [
    "No Known Allergies / Adverse Drug Reactions",
    "None",
    "Not applicable",
    "N/A",
    "Unknown",
]


class TestParseSections:
    def test_all_known_sections_found(self):
        sections = parse_sections(SAMPLE_NOTE, SECTION_NAMES)
        # All 8 headers appear in the sample note
        assert set(SECTION_NAMES) == set(sections.keys())

    def test_chief_complaint_content(self):
        sections = parse_sections(SAMPLE_NOTE, SECTION_NAMES)
        assert "___ pain, fever" in sections["CHIEF COMPLAINT"]

    def test_physical_exam_content(self):
        sections = parse_sections(SAMPLE_NOTE, SECTION_NAMES)
        assert "Gen:" in sections["PHYSICAL EXAM"]

    def test_case_insensitive_header_matching(self):
        note = "chief complaint: chest pain\npresent illness: acute onset"
        sections = parse_sections(note, SECTION_NAMES)
        assert "CHIEF COMPLAINT" in sections
        assert "PRESENT ILLNESS" in sections

    def test_empty_note_returns_empty_dict(self):
        assert parse_sections("", SECTION_NAMES) == {}

    def test_no_section_names_returns_empty_dict(self):
        assert parse_sections(SAMPLE_NOTE, []) == {}

    def test_note_with_no_matching_headers(self):
        sections = parse_sections("just some plain text", SECTION_NAMES)
        assert sections == {}

    def test_medication_on_admission_empty(self):
        sections = parse_sections(SAMPLE_NOTE, SECTION_NAMES)
        # "MEDICATION ON ADMISSION:" has nothing after it before the next header
        assert sections["MEDICATION ON ADMISSION"].strip() == ""


class TestFilterSections:
    def test_drops_allergies_ignore_phrase(self):
        sections = parse_sections(SAMPLE_NOTE, SECTION_NAMES)
        filtered = filter_sections(sections, IGNORE_PHRASES)
        assert "ALLERGIES" not in filtered

    def test_drops_empty_medication_section(self):
        sections = parse_sections(SAMPLE_NOTE, SECTION_NAMES)
        filtered = filter_sections(sections, IGNORE_PHRASES)
        assert "MEDICATION ON ADMISSION" not in filtered

    def test_keeps_physical_exam(self):
        sections = parse_sections(SAMPLE_NOTE, SECTION_NAMES)
        filtered = filter_sections(sections, IGNORE_PHRASES)
        assert "PHYSICAL EXAM" in filtered

    def test_drops_section_below_min_chars(self):
        sections = {"CHIEF COMPLAINT": "ok", "PRESENT ILLNESS": "much longer content here"}
        filtered = filter_sections(sections, [], min_chars=10)
        assert "CHIEF COMPLAINT" not in filtered
        assert "PRESENT ILLNESS" in filtered

    def test_ignore_phrase_comparison_is_case_insensitive(self):
        sections = {"ALLERGIES": "no known allergies / adverse drug reactions"}
        filtered = filter_sections(sections, IGNORE_PHRASES)
        assert "ALLERGIES" not in filtered

    def test_empty_ignore_phrases_drops_only_short_sections(self):
        sections = {"A": "long enough text here", "B": ""}
        filtered = filter_sections(sections, [], min_chars=5)
        assert "A" in filtered
        assert "B" not in filtered

    def test_sections_with_sufficient_content_kept(self):
        # CHIEF COMPLAINT and PHYSICAL EXAM have enough content in the sample.
        # PRESENT ILLNESS ("PCP:  ___" = 9 chars) and MEDICAL HISTORY ("T1DM" = 4 chars)
        # are below min_chars=10 and are correctly filtered out.
        sections = parse_sections(SAMPLE_NOTE, SECTION_NAMES)
        filtered = filter_sections(sections, IGNORE_PHRASES)
        assert "CHIEF COMPLAINT" in filtered
        assert "PHYSICAL EXAM" in filtered
        assert "PRESENT ILLNESS" not in filtered   # too short after stripping
        assert "MEDICAL HISTORY" not in filtered   # too short after stripping
