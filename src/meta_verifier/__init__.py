"""Meta-Verifier package.

Lightweight init: importing the package no longer pulls in the LLM /
embedding stack. Consumers should import the specific submodule they
need (`from src.meta_verifier.schemas import Instruction`,
`from src.meta_verifier.meta_verifier import MetaVerifier`, etc).
"""
