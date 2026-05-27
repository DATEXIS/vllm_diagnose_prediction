"""Component builders for the MERLIN 2 Pipeline.

Construct the Generator / Retriever / Verifier from a config dict. Kept
out of the Pipeline class so the orchestration file stays free of config
parsing noise.
"""

from __future__ import annotations

from typing import Any, Dict

from src.merlin2.generator import Generator
from src.merlin2.retriever import Retriever
from src.merlin2.verifier import Verifier
from src.meta_verifier.code_stats import load_code_stats
from src.utils.admission_note_parser import load_section_config


def build_generator(config: Dict[str, Any]) -> Generator:
    model_cfg = config.get("model", {})
    inference_cfg = config.get("inference", {})
    job_name = config.get("job_name", "default")
    namespace = config.get("k8s", {}).get("namespace", "default")
    return Generator(
        api_base=model_cfg.get(
            "api_base",
            f"http://vllm-server-{job_name}.{namespace}.svc.cluster.local/v1",
        ),
        model=model_cfg.get("name", "Qwen/Qwen3-8B"),
        temperature=inference_cfg.get("temperature", 0.0),
        max_tokens=inference_cfg.get("max_tokens", 1024),
        config=config,
    )


def build_retriever(config: Dict[str, Any]) -> Retriever:
    m2_cfg = config.get("merlin2", {})
    code_stats = load_code_stats(m2_cfg.get("code_stats_path", "data/code_stats.parquet"))
    section_cfg = load_section_config(
        m2_cfg.get("admission_note_sections_path", "configs/admission_note_sections.yaml")
    )
    return Retriever(
        sim_note_threshold=m2_cfg.get("sim_note_threshold", 0.8),
        sim_icd_threshold=m2_cfg.get("sim_icd_threshold", 0.8),
        fpr_threshold=m2_cfg.get("fpr_threshold", 0.5),
        dedup_cluster_threshold=m2_cfg.get("dedup_cluster_threshold", 1.0),
        max_instructions_per_code=m2_cfg.get("max_instructions_per_code"),
        max_fp_warnings=m2_cfg.get("max_fp_warnings"),
        max_sem_instructions=m2_cfg.get("max_sem_instructions"),
        max_instructions_total=m2_cfg.get("max_instructions_total"),
        code_stats=code_stats,
        section_names=section_cfg.get("sections", []),
        ignore_phrases=section_cfg.get("ignore_phrases", []),
    )


def build_verifier(config: Dict[str, Any]) -> Verifier:
    m2_cfg = config.get("merlin2", {})
    return Verifier(
        max_iterations=m2_cfg.get("max_iterations", 5),
        convergence_threshold=m2_cfg.get("convergence_threshold", 0.9),
        min_prediction_size=m2_cfg.get("min_prediction_size", 3),
    )
