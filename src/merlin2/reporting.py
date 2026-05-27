"""MERLIN 2 reporting utilities.

Formatting helpers for per-iteration metrics and retrieval logs.
All functions are pure (no side effects) so they can be called from main.py
or from tests without touching wandb or the filesystem.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from src.data.evaluate import calculate_metrics, normalize_icd, sample_prf
from src.merlin2.pipeline import CaseState, PipelineCaseResult
from src.merlin2.retriever import THRESHOLD_FPR, is_semantic_path


# --------------------------------------------------------- per-iteration metrics

def compute_per_iteration_metrics(
    results: List[PipelineCaseResult],
    ground_truth: List[List[str]],
) -> List[dict]:
    """Compute eval metrics per iteration t under two groupings.

    Returns one dict per iteration t, each with:
      - 'all':       metrics over every sample that has a prediction at t.
      - 'last_iter': metrics over only the samples whose final iteration is t.

    At t=0 both groupings are identical. Metric keys: f1_micro, f1_macro,
    precision_micro, recall_micro, precision_macro, recall_macro.
    """
    max_iters = max(r.iterations for r in results) if results else 0
    out = []
    for t in range(max_iters):
        entry = _iteration_entry(t, results, ground_truth)
        if entry:
            out.append(entry)
    return out


def _iteration_entry(
    t: int,
    results: List[PipelineCaseResult],
    ground_truth: List[List[str]],
) -> dict:
    entry: dict = {}

    all_metrics = _metrics_at_t_all(t, results, ground_truth)
    if all_metrics:
        entry["all"] = all_metrics

    last_metrics = _metrics_at_t_last(t, results, ground_truth)
    if last_metrics:
        entry["last_iter"] = last_metrics

    return entry


def _metrics_at_t_all(
    t: int,
    results: List[PipelineCaseResult],
    ground_truth: List[List[str]],
) -> dict:
    """Metrics over all samples that have a prediction at iteration t."""
    y_pred, y_true = [], []
    parse_failures = 0
    for r, truth in zip(results, ground_truth):
        if t >= len(r.history.predictions):
            continue
        y_pred.append(_norm_codes_from_pred(r.history.predictions[t]))
        y_true.append([normalize_icd(c) for c in truth if normalize_icd(c)])
        if r.halt_reason == "parse_failure" and len(r.history.predictions) - 1 == t:
            parse_failures += 1

    if not y_pred:
        return {}
    m = _metrics_dict(calculate_metrics(y_true, y_pred))
    m["parse_failures"] = parse_failures
    m["n_samples"] = len(y_pred)
    return m


def _metrics_at_t_last(
    t: int,
    results: List[PipelineCaseResult],
    ground_truth: List[List[str]],
) -> dict:
    """Metrics over samples whose final iteration is t (i.e. they halted after this wave)."""
    y_pred, y_true = [], []
    for r, truth in zip(results, ground_truth):
        if len(r.history.predictions) - 1 == t:
            y_pred.append(_norm_codes_from_pred(r.history.predictions[t]))
            y_true.append([normalize_icd(c) for c in truth if normalize_icd(c)])

    if not y_pred:
        return {}
    return _metrics_dict(calculate_metrics(y_true, y_pred))


def _norm_codes_from_pred(pred_model) -> List[str]:
    return [normalize_icd(d.icd_code) for d in pred_model.diagnoses if normalize_icd(d.icd_code)]


def _metrics_dict(m: dict) -> dict:
    return {
        "f1_micro": m["micro"]["f1"],
        "f1_macro": m["macro"]["f1"],
        "precision_micro": m["micro"]["precision"],
        "recall_micro": m["micro"]["recall"],
        "precision_macro": m["macro"]["precision"],
        "recall_macro": m["macro"]["recall"],
    }


# ----------------------------------------------------- retrieval event table

def flatten_retrieval_events(results: List[PipelineCaseResult]) -> pd.DataFrame:
    """Flatten all retrieval events across results into a long-format DataFrame."""
    rows = []
    for r in results:
        for t, events in enumerate(r.history.retrieval_events):
            for ev in events:
                rows.append({
                    "hadm_id": r.hadm_id,
                    "iteration": t,
                    "instruction_id": ev.instruction_id,
                    "path": ev.path,
                    "trigger_value": ev.trigger_value,
                    "efficacy_score": ev.efficacy_score,
                    "target_codes": ",".join(ev.target_codes),
                })
    return pd.DataFrame(rows)


# ------------------------------------------------------- retrieval log string

def format_retrieval_log(result: PipelineCaseResult) -> str:
    """Build a human-readable retrieval log for wandb inspection.

    Layout:
        True labels: A, B, C

        T=0 (zero-shot): X1, Y1  R=0.50  P=0.67  F1=0.57
        T=1 (final): X2, Y2      R=0.67  P=0.80  F1=0.73

        --- Retrieved Instructions ---
        T=1:
          Rethink codes (FPR):
          * B18    fpr=1.00
          Semantic Similarity:
          * [E11]  [note]  sim=0.85  "If the note mentions long-standing DM2..."
    """
    true_codes = result.history.ground_truth_codes or []
    lines: List[str] = [
        f"True labels: {', '.join(sorted(true_codes)) if true_codes else '—'}",
        "",
    ]
    lines.extend(_format_prediction_lines(result, true_codes))
    lines.extend(_format_instruction_lines(result))
    return "\n".join(lines)


def _format_prediction_lines(result: PipelineCaseResult, true_codes: List[str]) -> List[str]:
    n_preds = len(result.history.predictions)
    lines = []
    for t, pred_model in enumerate(result.history.predictions):
        pred_codes = _norm_codes_from_pred(pred_model)
        pred_str = ", ".join(pred_codes) if pred_codes else "—"

        labels = []
        if t == 0:
            labels.append("zero-shot")
        if t == n_preds - 1 and n_preds > 1:
            labels.append("final")
        label_suffix = f" ({', '.join(labels)})" if labels else ""

        score_str = ""
        if true_codes:
            prec, rec, f1 = sample_prf(true_codes, pred_codes)
            score_str = f"  R={rec:.2f}  P={prec:.2f}  F1={f1:.2f}"

        lines.append(f"T={t}{label_suffix}: {pred_str}{score_str}")
    return lines


def _format_instruction_lines(result: PipelineCaseResult) -> List[str]:
    retrieval_pairs = list(zip(result.history.retrieval_events, result.history.instructions_used))
    if not any(events for events, _ in retrieval_pairs):
        return []

    lines = ["\n--- Retrieved Instructions ---"]
    for t, (events, instrs) in enumerate(retrieval_pairs):
        if not events:
            continue
        ev_by_id = {ev.instruction_id: ev for ev in events}
        fpr_lines, sem_lines = _classify_instruction_lines(instrs, ev_by_id)

        lines.append(f"\nT={t}:")
        if fpr_lines:
            lines.append("  Rethink codes (FPR):")
            lines.extend(fpr_lines)
        if sem_lines:
            lines.append("  Semantic Similarity:")
            lines.extend(sem_lines)
    return lines


def _classify_instruction_lines(instrs, ev_by_id) -> tuple:
    fpr_lines, sem_lines = [], []
    for instr in instrs:
        ev = ev_by_id.get(instr.instruction_id)
        if ev is None:
            continue
        codes_tag = ", ".join(ev.target_codes) if ev.target_codes else "?"
        snippet = (instr.instruction_text or "")[:90].replace("\n", " ")

        if ev.path == THRESHOLD_FPR:
            fpr_lines.append(f"  * {codes_tag:<6}  fpr={ev.trigger_value:.2f}")
        elif is_semantic_path(ev.path):
            section_tag = ev.path.removeprefix("sem_")
            sem_lines.append(
                f"  * [{codes_tag}]  [{section_tag}]"
                f"  sim={ev.trigger_value:.2f}  \"{snippet}\""
            )
    return fpr_lines, sem_lines
