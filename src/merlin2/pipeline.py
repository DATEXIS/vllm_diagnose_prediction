"""MERLIN 2 inference pipeline orchestration."""

from typing import Any, Dict, List, Optional

from src.merlin2.generator import Generator
from src.prompter import ICDsModel
from src.merlin2.retriever import Retriever
from src.meta_verifier.schemas import Instruction
from src.merlin2.verifier import Verifier


class MERLINPipeline:
    """Orchestrates the dual-loop MERLIN 2 inference pipeline (Loop A: Synchronous Inference)."""

    def __init__(
        self,
        config: Dict[str, Any],
        generator: Optional[Generator] = None,
        retriever: Optional[Retriever] = None,
        verifier: Optional[Verifier] = None,
    ) -> None:
        """Initialize the MERLIN pipeline.

        Args:
            config: Full configuration dictionary.
            generator: Generator component for ICD code prediction. If None, creates from config.
            retriever: Retriever component for instruction retrieval. If None, creates from config.
            verifier: Verifier component for halting condition checking. If None, creates from config.
        """
        self.config = config

        # Initialize generator from config if not provided
        if generator is None:
            model_cfg = config.get('model', {})
            inference_cfg = config.get('inference', {})
            job_name = config.get('job_name', 'default')
            namespace = config.get('k8s', {}).get('namespace', 'default')

            self.generator = Generator(
                api_base=model_cfg.get('api_base', f"http://vllm-server-{job_name}.{namespace}.svc.cluster.local/v1"),
                model=model_cfg.get('name', 'Qwen/Qwen3-8B'),
                temperature=inference_cfg.get('temperature', 0.0),
                max_tokens=inference_cfg.get('max_tokens', 1024),
                config=config,
            )
        else:
            self.generator = generator

        # Initialize retriever from config if not provided
        if retriever is None:
            merlin2_cfg = config.get('merlin2', {})
            self.retriever = Retriever(
                sim_threshold=merlin2_cfg.get('sim_threshold', 0.7),
                fpr_threshold=merlin2_cfg.get('fpr_threshold', 0.1),
                fnr_threshold=merlin2_cfg.get('fnr_threshold', 0.1),
                max_tokens_budget=merlin2_cfg.get('max_tokens_budget', 512),
            )

            # Load instructions from wandb if configured
            meta_verifier_cfg = config.get('meta_verifier', {})
            download_run_name = meta_verifier_cfg.get('download_run_name')
            if download_run_name:
                self._load_instructions_from_wandb(download_run_name)
        else:
            self.retriever = retriever

        # Initialize verifier from config if not provided
        if verifier is None:
            merlin2_cfg = config.get('merlin2', {})
            self.verifier = Verifier(
                max_iterations=merlin2_cfg.get('max_iterations', 5),
                convergence_threshold=merlin2_cfg.get('convergence_threshold', 0.95),
                max_tokens_budget=merlin2_cfg.get('max_tokens_budget', 512),
            )
        else:
            self.verifier = verifier

    def _load_instructions_from_wandb(self, run_name: str) -> None:
        """Load instructions from wandb for retrieval.

        Args:
            run_name: Name of the run to download instructions from.
        """
        try:
            import src.utils.wandb_logger as wandb_logger

            wandb_cfg = wandb_logger.load_wandb_config(self.config)
            if not wandb_cfg:
                return

            project = wandb_cfg.get('project', 'ICD-prediction')
            entity = wandb_cfg.get('entity')

            instructions = wandb_logger.download_instructions(run_name, project, entity)
            if instructions:
                for instr in instructions:
                    self.retriever.add_instruction(instr)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to load instructions from wandb: {e}")

    def run(
        self,
        admission_note: str,
        ground_truth_codes: List[str] = None,
    ) -> Dict:
        """Run the MERLIN 2 inference pipeline on an admission note.

        Implements Loop A from spec:
            1. t=0: Initial pass - generator predicts with no instructions
            2. Retrieve instructions based on predicted codes
            3. Refinement passes with retrieved instructions (CoT integration)
            4. Check halting condition via verifier
            5. Repeat until halt

        Args:
            admission_note: The clinical admission note text.
            ground_truth_codes: Optional list of ground truth ICD codes for evaluation.

        Returns:
            Dictionary containing:
                - predictions: Final ICDsModel predictions
                - iterations: Number of iterations run
                - halt_reason: Reason for halting (from Verifier)
                - metadata: Additional metadata (all_predictions, instructions_retrieved, tokens_used)
        """
        current_predictions: Optional[ICDsModel] = None
        previous_predictions: Optional[ICDsModel] = None
        iterations = 0
        all_predictions: List[ICDsModel] = []
        instructions_retrieved = 0
        tokens_used = 0
        halt_reason = ""

        while True:
            iterations += 1

            if iterations == 1:
                current_predictions = self._run_single_iteration(
                    admission_note, [], iterations
                )
            else:
                predicted_codes = self._extract_codes(current_predictions)
                retrieved_instructions = self.retriever.retrieve(
                    admission_note, predicted_codes
                )
                instructions_retrieved = len(retrieved_instructions)

                current_predictions = self._run_single_iteration(
                    admission_note, retrieved_instructions, iterations
                )

            all_predictions.append(current_predictions)

            predicted_codes = self._extract_codes(current_predictions)
            previous_codes = (
                self._extract_codes(previous_predictions)
                if previous_predictions
                else None
            )

            should_halt, halt_reason = self.verifier.should_halt(
                iteration=iterations,
                current_predictions=predicted_codes,
                previous_predictions=previous_codes,
                instructions_retrieved=instructions_retrieved,
                tokens_used=tokens_used,
            )

            if should_halt:
                break

            previous_predictions = current_predictions

        return {
            "predictions": current_predictions,
            "iterations": iterations,
            "halt_reason": halt_reason,
            "metadata": {
                "all_predictions": all_predictions,
                "instructions_retrieved": instructions_retrieved,
                "tokens_used": tokens_used,
                "ground_truth_codes": ground_truth_codes,
            },
        }

    def _extract_codes(self, icd_model: ICDsModel) -> List[str]:
        """Extract ICD code list from ICDsModel.

        Args:
            icd_model: The ICD model containing diagnosis predictions.

        Returns:
            List of ICD code strings.
        """
        return [diagnosis.icd_code for diagnosis in icd_model.diagnoses]

    def _run_single_iteration(
        self,
        admission_note: str,
        instructions: List[Instruction],
        iteration: int,
    ) -> ICDsModel:
        """Run a single inference iteration.

        Args:
            admission_note: The clinical admission note text.
            instructions: List of retrieved instructions for refinement.
            iteration: Current iteration number.

        Returns:
            ICDsModel with predicted diagnoses.
        """
        return self.generator.generate(admission_note, instructions)


def run_inference(
    admission_note: str,
    generator: Optional[Generator] = None,
    retriever: Optional[Retriever] = None,
    verifier: Optional[Verifier] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict:
    """Convenience function to run the MERLIN pipeline.

    Args:
        admission_note: The clinical admission note text.
        generator: Generator instance. If None, creates default.
        retriever: Retriever instance. If None, creates default.
        verifier: Verifier instance. If None, creates default.
        config: Configuration dictionary for pipeline components.

    Returns:
        Dictionary with predictions, iterations, halt_reason, and metadata.
    """
    if config is None:
        config = {}

    pipeline = MERLINPipeline(config, generator, retriever, verifier)
    return pipeline.run(admission_note)