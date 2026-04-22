"""MERLIN 2 Meta-Verifier: Generates contrastive instructions from experiment results."""

import asyncio
import json
import logging
import re
from typing import List, Optional, Dict, Any

import pandas as pd

from src.meta_verifier.schemas import Instruction, RichErrorInstruction
from src.meta_verifier.prompts import ANALYSE_PROMPT, SUMMARIZE_PROMPT, ANAL_JSON, SUMM_JSON
from src.inference import run_inference_with_system

logger = logging.getLogger(__name__)


class MetaVerifier:
    """Generates contrastive instructions from experiment results using LLM-based RCA."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        embedding_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        min_error_count: int = 3,
        efficacy_threshold: float = 0.1,
    ):
        cfg = config.get("meta_verifier", {}) if config else {}
        self.embedding_model = cfg.get("embedding_model", embedding_model)
        self.min_error_count = cfg.get("min_error_count", min_error_count)
        self.efficacy_threshold = cfg.get("efficacy_threshold", efficacy_threshold)
        self._instruction_counter = 0
        self._api_base = None
        self._model_name = None

    def _configure_api(self, config: Dict[str, Any]) -> None:
        job_name = config.get("job_name", "default")
        namespace = config.get("k8s", {}).get("namespace", "default")
        self._api_base = config["model"].get(
            "api_base",
            f"http://vllm-server-{job_name}.{namespace}.svc.cluster.local/v1"
        )
        self._model_name = config["model"].get("name", "Qwen/Qwen3-8B")

    def _build_analyse_prompt(self, row: pd.Series) -> str:
        prompt = row.get("prompt", "")
        discharge_note = row.get("discharge_note", row.get("admission_note", ""))
        predicted_output = row.get("response", "")
        labels = row.get("ICD_CODES", "")

        return ANALYSE_PROMPT.format(
            prompt=prompt,
            discharge_note=discharge_note,
            predicted_output=predicted_output,
            labels=labels,
            json_example=ANAL_JSON,
        )

    def _build_summarize_prompt(self, analysis_results: List[List[Dict]]) -> str:
        raw_list = []
        for item in analysis_results:
            if isinstance(item, list):
                raw_list.extend(item)
            else:
                raw_list.append(item)

        cases_json = json.dumps(raw_list, indent=2)
        return SUMMARIZE_PROMPT.format(
            case_data=cases_json,
            json_example=SUMM_JSON,
        )

    def _extract_json(self, response_text: str) -> List[Dict]:
        if not response_text:
            return []

        try:
            json_match = re.search(r"(\[.*\])", response_text, re.DOTALL)
            if not json_match:
                json_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
                if not json_match:
                    return []
                raw_json = [json.loads(json_match.group(1))]
            else:
                raw_json = json.loads(json_match.group(1))

            validated = [RichErrorInstruction.model_validate(item).model_dump() for item in raw_json]
            return validated

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to extract JSON: {e}")
            return []

    async def generate_instructions_async(
        self,
        results_df: pd.DataFrame,
        config: Dict[str, Any],
    ) -> List[Instruction]:
        if results_df is None or len(results_df) == 0:
            logger.warning("No results provided to MetaVerifier")
            return []

        self._configure_api(config)

        logger.info(f"Stage 1: Analyzing {len(results_df)} cases for error patterns...")

        analyse_prompts = [self._build_analyse_prompt(row) for _, row in results_df.iterrows()]
        analyse_responses = await run_inference_with_system(
            config, analyse_prompts, temperature=0.4, max_tokens=2000
        )

        analysis_results = [self._extract_json(resp) for resp in analyse_responses]
        total_cases = len(analysis_results)
        failed_cases = sum(1 for r in analysis_results if not r)
        logger.info(f"Analysis complete: {total_cases - failed_cases}/{total_cases} succeeded")

        logger.info("Stage 2: Summarizing error patterns across cases...")
        summarize_prompt = self._build_summarize_prompt(analysis_results)
        summarize_responses = await run_inference_with_system(
            config, [summarize_prompt], temperature=0.4, max_tokens=3000
        )
        summarize_response = summarize_responses[0] if summarize_responses else None

        if not summarize_response:
            logger.warning("Summarization failed, falling back to flat instructions")
            return self._create_instructions_from_analysis(analysis_results, results_df)

        summary_results = self._extract_json(summarize_response)
        if not summary_results:
            logger.warning("Failed to parse summary, falling back to flat instructions")
            return self._create_instructions_from_analysis(analysis_results, results_df)

        logger.info(f"Generated {len(summary_results)} instruction patterns")
        return self._create_instructions_from_summary(summary_results, results_df)

    def generate_instructions(
        self,
        results_df: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[Instruction]:
        """Synchronous wrapper - runs async method with new event loop."""
        if config is None:
            config = {}

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("Cannot run async from within async context")
            return loop.run_until_complete(self.generate_instructions_async(results_df, config))
        except RuntimeError:
            return asyncio.run(self.generate_instructions_async(results_df, config))

    def _create_instructions_from_summary(
        self,
        summary_results: List[Dict],
        results_df: pd.DataFrame,
    ) -> List[Instruction]:
        instructions = []
        fpr_by_code, fnr_by_code = self._calculate_rates(results_df)

        for item in summary_results:
            self._instruction_counter += 1
            related_codes = item.get("related_icd_codes", [])

            for code in related_codes:
                code_fpr = fpr_by_code.get(code, 0.0)
                code_fnr = fnr_by_code.get(code, 0.0)

                efficacy = max(code_fpr, code_fnr, self.efficacy_threshold)

                try:
                    from src.utils.embeddings import encode_single_text
                    embedding = encode_single_text(item.get("description", ""))
                except Exception:
                    embedding = None

                instructions.append(Instruction(
                    id=self._instruction_counter,
                    target_code=code,
                    contrastive_rule=item.get("instructions", ""),
                    error_type=item.get("error_type", "unknown"),
                    quote=item.get("description", ""),
                    fpr=code_fpr,
                    fnr=code_fnr,
                    efficacy_score=efficacy,
                    semantic_embedding=embedding,
                ))

        return instructions

    def _create_instructions_from_analysis(
        self,
        analysis_results: List[List[Dict]],
        results_df: pd.DataFrame,
    ) -> List[Instruction]:
        instructions = []
        fpr_by_code, fnr_by_code = self._calculate_rates(results_df)

        for case_idx, case_errors in enumerate(analysis_results):
            if not case_errors:
                continue

            for error in case_errors:
                self._instruction_counter += 1
                related_codes = error.get("related_icd_codes", [])

                for code in related_codes:
                    code_fpr = fpr_by_code.get(code, 0.0)
                    code_fnr = fnr_by_code.get(code, 0.0)
                    efficacy = max(code_fpr, code_fnr, self.efficacy_threshold)

                    try:
                        from src.utils.embeddings import encode_single_text
                        embedding = encode_single_text(error.get("description", ""))
                    except Exception:
                        embedding = None

                    instructions.append(Instruction(
                        id=self._instruction_counter,
                        target_code=code,
                        contrastive_rule=error.get("instructions", ""),
                        error_type=error.get("error_type", "unknown"),
                        quote=error.get("description", ""),
                        fpr=code_fpr,
                        fnr=code_fnr,
                        efficacy_score=efficacy,
                        semantic_embedding=embedding,
                    ))

        return instructions

    def _calculate_rates(
        self,
        df: pd.DataFrame,
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        fpr_counts = {}
        fnr_counts = {}
        pred_counts = {}
        true_counts = {}

        for _, row in df.iterrows():
            pred_codes = row.get("pred_codes", [])
            true_codes = row.get("true_codes", [])

            pred_set = set(pred_codes) if isinstance(pred_codes, list) else set()
            true_set = set(true_codes) if isinstance(true_codes, list) else set()

            fp = pred_set - true_set
            fn = true_set - pred_set

            for code in fp:
                fpr_counts[code] = fpr_counts.get(code, 0) + 1
            for code in fn:
                fnr_counts[code] = fnr_counts.get(code, 0) + 1
            for code in pred_set:
                pred_counts[code] = pred_counts.get(code, 0) + 1
            for code in true_set:
                true_counts[code] = true_counts.get(code, 0) + 1

        fpr_rates = {
            code: fpr_counts.get(code, 0) / max(pred_counts.get(code, 1), 1)
            for code in set(fpr_counts) | set(pred_counts)
        }
        fnr_rates = {
            code: fnr_counts.get(code, 0) / max(true_counts.get(code, 1), 1)
            for code in set(fnr_counts) | set(true_counts)
        }

        return fpr_rates, fnr_rates