# Agent Instructions
These are general instructions. The repo might contain additional repo specific instructions.
You can call me Jan.

## Coding style:
This is research code, not production code. Optimize for iteration speed and debuggability, not robustness:

- Let it crash. Loud failures > silent fallbacks.
- No defensive try/except around things that "might" fail — I want the stack trace.
- No retry loops, no graceful degradation, no "safe defaults" that mask bugs.
- A crashed run I can debug is cheaper than a half-completed run that wasted hours of compute.
- Validate inputs at boundaries only. Inside the code, trust the types.

## Repo Structure
Most experiments are run with k8s. k8s yaml are build at runtime by scripts using settings in config.
A base repo should have at least following folders:
- configs/: YAML files that contain parameters:
  - experiment.yaml: Experiment-specific settings
  - setup.yaml: settings rarely changed like registry: "registry.datexis.com/jfrick"
- scripts/: Orchestration scripts for building and running experiments. 
  - k8s templates + builders
  - docker build scripts 
- src/: Source code for model inference, data loading, evaluation
- data/: (Gitignored) Local data files, e.g. MIMIC-IV extracts
- tests/: Unit tests for critical components (optional but recommended)

## Logging
- Use wandb for experiment logging. entity: "datexis-phd".
- Meaningful metrics can be logged.
- Meaningful texts can be logged as artifacts and tables. Only log 30 examples in text tables.

## Data Handling
We have 3 options to store data. Docker image, wandb or PVC for larger datasets.

# Available Hardware
The k8s cluster has P100, V100, A100, H100, H200 and B200 GPUs. 
You are allowed to use hardware that was defined in the configs.
Do not increase required hardware without explicit permission. If you need more resources, ask first and justify why.

## Self-review
Before declaring a task done:

1. Re-read your own diff as if I were reviewing it.
2. Ask, in writing, what I would object to. Anticipate the obvious complaints (over-engineering, untested assumption, ignored reference file, silent fallback added "just in case", missed PROGRESS.md update, missed prompts.log update).
3. Fix those issues yourself before handing it back.
4. If anything is still uncertain, say so explicitly. Do not paper over it.

## Communication style

- Direct. No filler, no hype, no apologies.
- State assumptions before acting on them.
- If I am wrong about something, tell me directly and explain why.
- Cite file paths and line numbers when referencing code.
- Batching questions is fine and preferred — ask multiple clarifying questions in one turn rather than ping-ponging. Larger, denser requests are cheaper than many small ones.