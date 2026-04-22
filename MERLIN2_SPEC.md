# MERLIN 2: System Architecture Specification

**Project Goal:** A dual-loop, multi-agent framework to iteratively improve ICD coding from clinical admission notes (using MIMIC-IV). The system uses an active inference loop for real-time prediction and an asynchronous memory loop to build a global, contrastive error taxonomy, addressing long-tail coding errors without constant fine-tuning.

---

## 1. Core Agents & Components

### Generator (G)
* **Role:** The primary clinical coding LLM.
* **Infrastructure:** vLLM with Guided Decoding.
* **Output Schema:** Strict JSON list (Fields: `icd_code`, `quote`, `reason`).
* **Execution:** Processes the admission note and iterates sequentially through retrieved instructions via a simulated `<think>` block before finalizing its JSON prediction.

### Meta-Verifier (M)
* **Role:** Asynchronous clinical auditor and taxonomy author.
* **Task:** Analyzes historical predictions against MIMIC-IV ground truth and discharge notes to find root causes of errors.
* **Instruction Generation:** Writes highly specific, **contrastive** rules (e.g., *"If the note mentions ketoacidosis, predict E11.1 INSTEAD OF E11.9"*).
* **Metrics:** Calculates and updates historical False Positive (FPR) and False Negative (FNR) rates for specific codes.

### Hybrid-Retriever (R)
* **Role:** The memory fetcher during active inference.
* **Logic:** Uses a discrete, threshold-based boolean selection. Fetches an instruction if it meets criteria for **either** semantic relevance (note text) **or** historical error rates (predicted codes).
* **Budgeting:** Caps retrieval at a maximum token limit, prioritizing instructions with the highest historical efficacy score.

### Verifier (V)
* **Role:** Synchronous traffic controller.
* **Task:** Halts the Generator's inference loop if any condition is met: Token budget exhausted, no new instructions retrieved from the database, predictions stabilize (convergence), or maximum iteration steps reached.

---

## 2. The Dual-Loop Pipeline

### Loop A: Synchronous Inference (Real-Time)
1. **Initial Pass (t=0):** Generator reads the Admission Note. Retriever fetches baseline instructions based purely on semantic text similarity. Generator outputs initial JSON predictions.
2. **Refinement Passes (t>0):** Retriever fetches specific instructions based on the exact codes predicted in the previous step that violate FPR/FNR thresholds.
3. **CoT Integration:** The Generator processes these warnings sequentially in a pre-filled internal monologue:

    ```text
    <think>
    Reviewing previous prediction: E11.9.
    Instruction Check: FPR warning triggered. Rule: Predict E11.4 INSTEAD OF E11.9 if diabetic neuropathy is present.
    Action: Note mentions neuropathy. Dropping E11.9, predicting E11.4.
    </think>
    [JSON Output]
    ```

4. **Halting:** Loop repeats until the Verifier (V) signals convergence.

### Loop B: Asynchronous Knowledge Acquisition (Global Memory)
1. **Audit:** Meta-Verifier (M) reviews closed cases, generating new contrastive instructions for missed codes or hallucinated codes.
2. **Database Maintenance:** A background clustering routine groups instructions by target ICD code and error type. Redundant instructions are merged and summarized via LLM, combining their efficacy scores to prevent database bloat.

---

## 3. Experimental Execution Strategy

To build a robust memory bank and avoid penalizing successful rules, the experiment runs in two phases:

* **Phase 1: Knowledge Bootstrapping:** Run the pipeline on a training subset of MIMIC-IV. The Meta-Verifier populates the empty database with the initial error taxonomy and establishes baseline FPR/FNR metrics.
* **Phase 2: Residual Error Mining:** Restart inference using the populated database. 
    * *Crucial Constraint:* Historical FPR/FNR metrics from Phase 1 are **frozen**. Because Phase 1 instructions prevent errors, actual error rates will drop. Freezing the metrics ensures these successful instructions continue to trigger the Retriever's thresholds. The Meta-Verifier only writes new instructions for novel errors.

---

## 4. Future Dataset Generation (Side Note)

While not the primary focus of the active inference architecture, the MERLIN 2 framework naturally produces a highly curated dataset of Admission Notes, sequential `<think>` reasoning traces, and correct JSON outputs. To finalize a "gold standard" dataset for future Supervised Fine-Tuning (SFT), Rejection Sampling will be applied: the Generator will produce Best-of-N candidate traces at higher temperatures, and only the trace yielding the highest F1 score against the MIMIC-IV ground truth will be saved.