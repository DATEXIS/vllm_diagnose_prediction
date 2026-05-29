"""Instruction-level retrieval quality evaluation.

Three orthogonal quality dimensions for each instruction:

  1. Retrieval precision  (correct trigger rate)
     Does the instruction fire for the *right* cases?
     TP: retrieved + action aligns with ground truth
     FP: retrieved + action contradicts ground truth

  2. Retrieval recall  (description quality)
     When a case is relevant, does the instruction actually get retrieved?
     FN: case is relevant for this instruction, but it was never retrieved.
     Low recall → description / embedding doesn't match the notes it should.

     Relevance definition:
       action="add"    → relevant when any target_code ∈ ground_truth
       action="remove" → relevant when any target_code was predicted
                         (any iteration) AND ∉ ground_truth

  3. Adoption  (instruction text quality)
     When retrieved, does the model actually follow the instruction?
     Two adoption signals:
       * adopted          — model's prediction at iteration t already aligns
                            with the instruction's recommendation
       * changed_correctly — prediction concretely moved in the right direction
                             between t-1 and t (stronger causal signal; only
                             meaningful for instructions freshly retrieved at t).

     tp_adoption_rate  = adoption among TP-retrievals only
                         (ignores instructions that fired incorrectly)
     change_rate       = changed_correctly / n_retrieved

Combining (1) and (2):
  precision = n_tp / (n_tp + n_fp)
  recall    = n_tp / (n_tp + n_fn)
  f1        = harmonic mean

These replace the batch-level efficacy_score as a per-instruction quality
signal.  The existing efficacy_score is still included for comparison.

Public API
----------
compute_retrieval_correctness(results, ground_truth, instructions)
    → long-format events DataFrame (one row per retrieval event × target_code)
      columns: ... correct_retrieval, adopted, changed_correctly

compute_retrieval_fn(results, ground_truth, instructions)
    → long-format FN DataFrame (one row per missed-retrieval opportunity)

instruction_confusion_matrix(correctness_df, fn_df)
    → per-instruction aggregate with all metrics

aggregate_by_group(correctness_df, fn_df, group_col)
    → breakdown by type / action / section / path

compute_all(results, ground_truth, instructions)
    → dict with all DataFrames
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import pandas as pd

from src.data.evaluate import normalize_icd
from src.merlin2.pipeline import PipelineCaseResult
from src.merlin2.retriever import THRESHOLD_FPR
from src.meta_verifier.schemas import Instruction


# ------------------------------------------------------------------ helpers

def _codes_set(model) -> Set[str]:
    """Extract 3-digit normalised codes from an ICDsModel."""
    return {normalize_icd(d.icd_code) for d in model.diagnoses if normalize_icd(d.icd_code)}


def _instruction_meta(instr: Optional[Instruction], ev_path: str) -> Optional[dict]:
    """Return the metadata dict for an instruction, or None if it should be skipped."""
    if instr is not None:
        return {
            "action":        instr.action,
            "type":          instr.type,
            "section":       instr.section,
            "description":   (instr.description or "")[:80],
            "efficacy_score": instr.efficacy_score,
        }
    if ev_path == THRESHOLD_FPR:
        return {
            "action":        "remove",
            "type":          "threshold_fpr",
            "section":       "",
            "description":   "",
            "efficacy_score": 0.0,
        }
    return None


# ------------------------------------------------------------------ events (TP / FP + adoption)

def compute_retrieval_correctness(
    results: List[PipelineCaseResult],
    ground_truth: List[List[str]],
    instructions: List[Instruction],
) -> pd.DataFrame:
    """Classify every retrieval event as correct (TP) or incorrect (FP).

    Also computes adoption signals per event:
      * adopted           — model's prediction at iteration t aligns with
                            the instruction's recommended action.
      * changed_correctly — prediction moved in the right direction from
                            t-1 → t (only meaningful for t ≥ 1).

    Returns one row per (hadm_id, iteration, instruction_id, target_code).
    """
    instr_by_id: Dict[int, Instruction] = {i.instruction_id: i for i in instructions}
    rows = []

    for result, gt in zip(results, ground_truth):
        gt_set: Set[str] = {normalize_icd(c) for c in gt if normalize_icd(c)}
        preds = result.history.predictions  # List[ICDsModel]

        for t, events in enumerate(result.history.retrieval_events):
            pred_t    = _codes_set(preds[t])    if t < len(preds) else set()
            pred_prev = _codes_set(preds[t - 1]) if t > 0 and (t - 1) < len(preds) else set()

            for ev in events:
                meta = _instruction_meta(instr_by_id.get(ev.instruction_id), ev.path)
                if meta is None:
                    continue

                action = meta["action"]
                for code in (ev.target_codes or []):
                    norm = normalize_icd(code) or code

                    correct = (norm in gt_set) if action == "add" else (norm not in gt_set)

                    # adopted: does prediction at t already agree with instruction?
                    if action == "add":
                        adopted           = norm in pred_t
                        changed_correctly = (norm not in pred_prev) and (norm in pred_t)
                    else:
                        adopted           = norm not in pred_t
                        changed_correctly = (norm in pred_prev) and (norm not in pred_t)

                    rows.append({
                        "instruction_id":    ev.instruction_id,
                        "type":              meta["type"],
                        "action":            action,
                        "section":           meta["section"],
                        "description":       meta["description"],
                        "efficacy_score":    meta["efficacy_score"],
                        "hadm_id":           result.hadm_id,
                        "iteration":         t,
                        "path":              ev.path,
                        "trigger_value":     ev.trigger_value,
                        "target_code":       norm,
                        "correct_retrieval": correct,
                        "adopted":           adopted,
                        "changed_correctly": changed_correctly,
                    })

    return pd.DataFrame(rows)


# ------------------------------------------------------------------ FN events (retrieval recall)

def compute_retrieval_fn(
    results: List[PipelineCaseResult],
    ground_truth: List[List[str]],
    instructions: List[Instruction],
) -> pd.DataFrame:
    """Find cases where an instruction was relevant but never retrieved.

    This is the *description quality* signal: low recall on an instruction
    means its description / embedding fails to match the notes it should.

    Relevance definition (oracle, ignoring prediction outcome):
      action="add"    → relevant when any target_code ∈ ground_truth
      action="remove" → relevant when any target_code was predicted across
                        any iteration AND ∉ ground_truth

    One FN row per (instruction, hadm_id) pair where:
      instruction is relevant for the case  AND  was never retrieved.
    """
    # Inverted index: normalised code → list of relevant instructions
    add_idx:    Dict[str, List[Instruction]] = {}
    remove_idx: Dict[str, List[Instruction]] = {}
    for instr in instructions:
        idx = add_idx if instr.action == "add" else remove_idx
        for code in instr.target_codes:
            norm = normalize_icd(code) or code
            idx.setdefault(norm, []).append(instr)

    rows = []
    for result, gt in zip(results, ground_truth):
        gt_set: Set[str] = {normalize_icd(c) for c in gt if normalize_icd(c)}

        # All codes ever predicted across all iterations (for remove relevance).
        all_predicted: Set[str] = set()
        for pred in result.history.predictions:
            all_predicted.update(_codes_set(pred))

        hallucinated = all_predicted - gt_set

        # All instructions that fired for this case.
        retrieved_ids: Set[int] = {
            ev.instruction_id
            for events in result.history.retrieval_events
            for ev in events
        }

        # --- add FNs: instruction relevant when target code ∈ ground truth ---
        for code in gt_set:
            for instr in add_idx.get(code, []):
                if instr.instruction_id not in retrieved_ids:
                    rows.append({
                        "instruction_id":  instr.instruction_id,
                        "type":            instr.type,
                        "action":          instr.action,
                        "section":         instr.section,
                        "description":     (instr.description or "")[:80],
                        "efficacy_score":  instr.efficacy_score,
                        "hadm_id":         result.hadm_id,
                        "target_code":     code,
                        "fn_reason":       "add_code_in_gt",
                    })

        # --- remove FNs: instruction relevant when target code hallucinated ---
        for code in hallucinated:
            for instr in remove_idx.get(code, []):
                if instr.instruction_id not in retrieved_ids:
                    rows.append({
                        "instruction_id":  instr.instruction_id,
                        "type":            instr.type,
                        "action":          instr.action,
                        "section":         instr.section,
                        "description":     (instr.description or "")[:80],
                        "efficacy_score":  instr.efficacy_score,
                        "hadm_id":         result.hadm_id,
                        "target_code":     code,
                        "fn_reason":       "remove_code_hallucinated",
                    })

    return pd.DataFrame(rows)


# ------------------------------------------------------------------ aggregation

def instruction_confusion_matrix(
    correctness_df: pd.DataFrame,
    fn_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Per-instruction confusion matrix with all quality metrics.

    Columns:
      n_retrieved, n_tp, n_fp, n_fn
      precision   = n_tp / (n_tp + n_fp)          ← retrieval precision
      recall      = n_tp / (n_tp + n_fn)           ← description quality
      f1          = harmonic mean
      adoption_rate    = n_adopted / n_retrieved    ← overall adoption
      tp_adoption_rate = n_tp_adopted / n_tp        ← text quality (correct instructions only)
      change_rate      = n_changed / n_retrieved    ← causal instruction impact
      efficacy_score   = current running score (for comparison)
    """
    if correctness_df.empty:
        return pd.DataFrame()

    grp_cols = ["instruction_id", "type", "action", "section", "description", "efficacy_score"]

    agg = (
        correctness_df
        .groupby(grp_cols, sort=False)
        .agg(
            n_retrieved       = ("correct_retrieval", "count"),
            n_tp              = ("correct_retrieval", "sum"),
            n_adopted         = ("adopted", "sum"),
            n_tp_adopted      = (
                "adopted",
                # count adopted only among TP rows
                lambda s: int((s & correctness_df.loc[s.index, "correct_retrieval"]).sum()),
            ),
            n_changed         = ("changed_correctly", "sum"),
        )
        .reset_index()
    )

    agg["n_fp"]           = agg["n_retrieved"] - agg["n_tp"]
    agg["precision"]      = agg["n_tp"]       / agg["n_retrieved"]
    agg["adoption_rate"]  = agg["n_adopted"]  / agg["n_retrieved"]
    agg["change_rate"]    = agg["n_changed"]  / agg["n_retrieved"]
    agg["tp_adoption_rate"] = (
        agg["n_tp_adopted"] / agg["n_tp"].replace(0, float("nan"))
    )

    # Merge FN counts for recall
    if fn_df is not None and not fn_df.empty:
        fn_counts = (
            fn_df.groupby("instruction_id")
            .agg(n_fn=("hadm_id", "count"))
            .reset_index()
        )
        agg = agg.merge(fn_counts, on="instruction_id", how="left")
        agg["n_fn"] = agg["n_fn"].fillna(0).astype(int)
    else:
        agg["n_fn"] = 0

    denom_recall = agg["n_tp"] + agg["n_fn"]
    agg["recall"] = (agg["n_tp"] / denom_recall.replace(0, float("nan")))
    p, r = agg["precision"], agg["recall"]
    agg["f1"] = (2 * p * r / (p + r)).where(p + r > 0)

    col_order = [
        "instruction_id", "type", "action", "section", "description",
        "efficacy_score",
        "n_retrieved", "n_tp", "n_fp", "n_fn",
        "precision", "recall", "f1",
        "adoption_rate", "tp_adoption_rate", "change_rate",
    ]
    return (
        agg[[c for c in col_order if c in agg.columns]]
        .sort_values("n_retrieved", ascending=False)
        .reset_index(drop=True)
    )


def aggregate_by_group(
    correctness_df: pd.DataFrame,
    fn_df: Optional[pd.DataFrame],
    group_col: str,
) -> pd.DataFrame:
    """Aggregate all quality metrics by a grouping column.

    Works for: type, action, section, path.
    """
    if correctness_df.empty or group_col not in correctness_df.columns:
        return pd.DataFrame()

    agg = (
        correctness_df
        .groupby(group_col, sort=False)
        .agg(
            n_retrieved      = ("correct_retrieval", "count"),
            n_tp             = ("correct_retrieval", "sum"),
            n_adopted        = ("adopted", "sum"),
            n_changed        = ("changed_correctly", "sum"),
        )
        .reset_index()
    )
    agg["n_fp"]          = agg["n_retrieved"] - agg["n_tp"]
    agg["precision"]     = agg["n_tp"]      / agg["n_retrieved"]
    agg["adoption_rate"] = agg["n_adopted"] / agg["n_retrieved"]
    agg["change_rate"]   = agg["n_changed"] / agg["n_retrieved"]

    if fn_df is not None and not fn_df.empty and group_col in fn_df.columns:
        fn_counts = fn_df.groupby(group_col).agg(n_fn=("hadm_id", "count")).reset_index()
        agg = agg.merge(fn_counts, on=group_col, how="left")
        agg["n_fn"] = agg["n_fn"].fillna(0).astype(int)
    else:
        agg["n_fn"] = 0

    denom = agg["n_tp"] + agg["n_fn"]
    agg["recall"] = agg["n_tp"] / denom.replace(0, float("nan"))
    p, r = agg["precision"], agg["recall"]
    agg["f1"] = (2 * p * r / (p + r)).where(p + r > 0)

    return agg.sort_values("precision", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ convenience wrapper

def compute_all(
    results: List[PipelineCaseResult],
    ground_truth: List[List[str]],
    instructions: List[Instruction],
) -> dict:
    """Run all three evaluation passes and return the full results dict.

    Keys:
      'correctness'      – event-level TP/FP + adoption DataFrame
      'fn_events'        – FN (missed retrieval) DataFrame
      'per_instruction'  – per-instruction confusion matrix with all metrics
      'by_type'          – grouped by instruction type
      'by_action'        – grouped by action (add / remove)
      'by_section'       – grouped by admission-note section
    """
    correctness_df = compute_retrieval_correctness(results, ground_truth, instructions)
    fn_df          = compute_retrieval_fn(results, ground_truth, instructions)

    return {
        "correctness":     correctness_df,
        "fn_events":       fn_df,
        "per_instruction": instruction_confusion_matrix(correctness_df, fn_df),
        "by_type":         aggregate_by_group(correctness_df, fn_df, "type"),
        "by_action":       aggregate_by_group(correctness_df, fn_df, "action"),
        "by_section":      aggregate_by_group(correctness_df, fn_df, "section"),
    }
