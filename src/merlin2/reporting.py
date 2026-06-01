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
    """Compute eval metrics per iteration t over all samples that have a prediction at t.

    Returns one flat metrics dict per iteration t.
    Metric keys: f1_micro, f1_macro, precision_micro, recall_micro,
    precision_macro, recall_macro, parse_failures, n_samples.
    """
    max_iters = max(r.iterations for r in results) if results else 0
    out = []
    for t in range(max_iters):
        entry = _metrics_at_t_all(t, results, ground_truth)
        if entry:
            out.append(entry)
    return out


def _metrics_at_t_all(
    t: int,
    results: List[PipelineCaseResult],
    ground_truth: List[List[str]],
) -> dict:
    """Metrics over all samples at iteration t.

    Samples that halted before t carry forward their last prediction, so
    every iteration covers the full population and t=max equals summary charts.
    """
    y_pred, y_true = [], []
    n_samples_at_t = 0
    parse_failures_at_t = 0
    for r, truth in zip(results, ground_truth):
        effective_t = min(t, len(r.history.predictions) - 1)
        y_pred.append(_norm_codes_from_pred(r.history.predictions[effective_t]))
        y_true.append([normalize_icd(c) for c in truth if normalize_icd(c)])
        if t < len(r.history.predictions):
            n_samples_at_t += 1
            if r.halt_reason == "parse_failure" and len(r.history.predictions) - 1 == t:
                parse_failures_at_t += 1

    if not y_pred:
        return {}
    m = _metrics_dict(calculate_metrics(y_true, y_pred))
    m["n_samples"] = n_samples_at_t
    m["parse_failure_pct"] = parse_failures_at_t / n_samples_at_t * 100 if n_samples_at_t else 0.0
    return m


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

    # Pre-compute code sets for annotation; empty when ground truth is absent.
    gt_set: set = {normalize_icd(c) for c in (result.history.ground_truth_codes or [])}
    predictions = result.history.predictions

    lines = ["\n--- Retrieved Instructions ---"]
    for t, (events, instrs) in enumerate(retrieval_pairs):
        if not events:
            continue
        ev_by_id = {ev.instruction_id: ev for ev in events}

        # Instructions retrieved at wave t are shown to the model, which then
        # produces predictions[t].  predictions[t-1] is what it said before.
        pred_t_codes: set = (
            set(_norm_codes_from_pred(predictions[t])) if t < len(predictions) else set()
        )
        pred_prev_codes: set = (
            set(_norm_codes_from_pred(predictions[t - 1]))
            if t > 0 and (t - 1) < len(predictions)
            else set()
        )

        fpr_lines, sem_lines = _classify_instruction_lines(
            instrs, ev_by_id, gt_set, pred_t_codes, pred_prev_codes
        )

        lines.append(f"\nT={t}:")
        if fpr_lines:
            lines.append("  Rethink codes (FPR):")
            lines.extend(fpr_lines)
        if sem_lines:
            lines.append("  Semantic Similarity:")
            lines.extend(sem_lines)
    return lines


def _classify_instruction_lines(
    instrs,
    ev_by_id,
    gt_set: set,
    pred_t_codes: set,
    pred_prev_codes: set,
) -> tuple:
    fpr_lines, sem_lines = [], []
    for instr in instrs:
        ev = ev_by_id.get(instr.instruction_id)
        if ev is None:
            continue
        codes_tag = ", ".join(ev.target_codes) if ev.target_codes else "?"
        snippet = (instr.instruction_text or "")[:90].replace("\n", " ")
        annotation = _eval_annotation(instr, gt_set, pred_t_codes, pred_prev_codes)

        if ev.path == THRESHOLD_FPR:
            fpr_lines.append(f"  * {codes_tag:<6}  fpr={ev.trigger_value:.2f}{annotation}")
        elif is_semantic_path(ev.path):
            section_tag = ev.path.removeprefix("sem_")
            sem_lines.append(
                f"  * [{codes_tag}]  [{section_tag}]"
                f"  sim={ev.trigger_value:.2f}  \"{snippet}\"{annotation}"
            )
    return fpr_lines, sem_lines


def _eval_annotation(instr, gt_set: set, pred_t_codes: set, pred_prev_codes: set) -> str:
    """Return a short '  ✓  +added' / '  ✗  ignored' tag for one retrieved instruction.

    Empty string when gt_set is absent (test/eval run without ground truth)
    or target_codes is empty.

    Correctness (✓ / ✗):
        add    — TP if any target code is in GT
        remove — TP if any target code is *not* in GT (i.e. the model was
                 right to be warned against adding it)

    Adoption status (what the model did at this wave):
        +added      — add instruction, code was absent at t-1 and present at t
        already-in  — add instruction, code already present at t-1 and still at t
        ignored     — add instruction, code absent at t
        -removed    — remove instruction, code present at t-1 and absent at t
        already-out — remove instruction, code absent at both t-1 and t
        ignored     — remove instruction, code still present at t
    """
    if not gt_set:
        return ""
    norm_targets = {normalize_icd(c) for c in (instr.target_codes or []) if normalize_icd(c)}
    if not norm_targets:
        return ""

    action = (instr.action or "").lower()

    # Correctness
    if action == "add":
        correct = bool(norm_targets & gt_set)
    else:  # remove / default
        correct = bool(norm_targets - gt_set)

    tick = "✓" if correct else "✗"

    # Adoption
    if action == "add":
        in_t = norm_targets & pred_t_codes
        in_prev = norm_targets & pred_prev_codes
        if in_t and not (in_t & in_prev):
            status = "+added"
        elif in_t:
            status = "already-in"
        else:
            status = "ignored"
    else:  # remove
        was_predicted = norm_targets & pred_prev_codes
        still_predicted = norm_targets & pred_t_codes
        if was_predicted and not still_predicted:
            status = "-removed"
        elif not was_predicted:
            status = "already-out"
        else:
            status = "ignored"

    return f"  {tick}  {status}"
