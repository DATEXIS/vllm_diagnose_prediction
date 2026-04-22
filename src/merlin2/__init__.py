"""MERLIN2 - Dual-loop multi-agent ICD coding framework."""
from src.meta_verifier.schemas import Instruction, ErrorType
from src.merlin2.generator import Generator
from src.merlin2.retriever import Retriever
from src.merlin2.verifier import Verifier, HaltReason
from src.meta_verifier import MetaVerifier
from src.merlin2.pipeline import MERLINPipeline, run_inference

__all__ = [
    "Instruction",
    "ErrorType",
    "Generator",
    "Retriever",
    "Verifier",
    "HaltReason",
    "MetaVerifier",
    "MERLINPipeline",
    "run_inference",
]