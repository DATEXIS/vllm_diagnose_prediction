"""Per-case rareness factors for efficacy score updates.

Each case gets a scalar derived from how rare its ground-truth 3-digit
codes are in the training corpus (mean inverse document frequency).
Rare codes in a case → higher factor → larger efficacy updates when an
instruction helps fix that case.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, List, Sequence

from src.data.evaluate import normalize_icd, safe_parse_true_labels


def code_document_frequencies(
    label_lists: Sequence[Sequence[str]],
) -> Counter[str]:
    """Count how many cases contain each 3-digit code (document frequency)."""
    counts: Counter[str] = Counter()
    for labels in label_lists:
        codes = {normalize_icd(c) for c in labels if normalize_icd(c)}
        for code in codes:
            counts[code] += 1
    return counts


def idf_weight(code: str, doc_freq: Counter[str], n_docs: int) -> float:
    """Smoothed IDF: log((N + 1) / (df + 1)) + 1."""
    df = doc_freq.get(code, 0)
    return math.log((n_docs + 1) / (df + 1)) + 1.0


def case_rareness_factor(
    true_codes: Iterable[str],
    doc_freq: Counter[str],
    n_docs: int,
) -> float:
    """Mean IDF over the case's ground-truth codes; 1.0 if no codes."""
    codes = [normalize_icd(c) for c in true_codes if normalize_icd(c)]
    if not codes:
        return 1.0
    weights = [idf_weight(c, doc_freq, n_docs) for c in codes]
    return float(sum(weights) / len(weights))


def compute_rareness_factors(
    label_lists: Sequence[Sequence[str]],
) -> List[float]:
    """One rareness factor per case, aligned with label_lists order."""
    doc_freq = code_document_frequencies(label_lists)
    n_docs = len(label_lists)
    return [case_rareness_factor(labels, doc_freq, n_docs) for labels in label_lists]


def labels_from_column(series_values) -> List[List[str]]:
    """Parse a dataframe column of ICD label lists into normalized 3-digit codes."""
    out: List[List[str]] = []
    for val in series_values:
        parsed = safe_parse_true_labels(val)
        out.append([normalize_icd(c) for c in parsed if normalize_icd(c)])
    return out
