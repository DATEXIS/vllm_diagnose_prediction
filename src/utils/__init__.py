"""Utility modules for MERLIN 2.

Submodules are imported on demand by their consumers; this `__init__`
intentionally does NOT eagerly import wandb / torch so unit tests that
only need `parsing_utils` or `prompt_loader` don't pay for those deps.
"""
