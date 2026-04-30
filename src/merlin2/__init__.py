"""MERLIN 2 package.

Lightweight init by design — submodules import the heavy stack
(`vLLM`, `wandb`, etc) only when actually used. Consumers should import
the specific submodule:

    from src.merlin2.pipeline import MERLINPipeline
    from src.merlin2.retriever import Retriever
"""
