from typing import List, Optional

import numpy as np
from numpy.linalg import norm

from src.meta_verifier.schemas import Instruction


class Retriever:
    def __init__(
        self,
        sim_threshold: float = 0.7,
        fpr_threshold: float = 0.1,
        fnr_threshold: float = 0.1,
        max_tokens_budget: int = 512,
    ):
        self.sim_threshold = sim_threshold
        self.fpr_threshold = fpr_threshold
        self.fnr_threshold = fnr_threshold
        self.max_tokens_budget = max_tokens_budget
        self._instructions: List[Instruction] = []

    def add_instruction(self, instruction: Instruction) -> None:
        self._instructions.append(instruction)

    def retrieve(
        self, note_text: str, predicted_codes: List[str]
    ) -> List[Instruction]:
        candidates = []
        for instr in self._instructions:
            if instr.target_code not in predicted_codes:
                continue

            similarity = self._compute_similarity(note_text, instr)

            if (
                similarity >= self.sim_threshold
                or instr.fpr >= self.fpr_threshold
                or instr.fnr >= self.fnr_threshold
            ):
                candidates.append(instr)

        candidates.sort(key=lambda x: x.efficacy_score, reverse=True)

        result = []
        total_tokens = 0
        for instr in candidates:
            est_tokens = self._estimate_tokens(instr)
            if total_tokens + est_tokens > self.max_tokens_budget:
                continue
            result.append(instr)
            total_tokens += est_tokens

        return result

    def _compute_similarity(self, note_text: str, instruction: Instruction) -> float:
        if instruction.semantic_embedding is None:
            return 0.0

        query_embedding = self._text_to_embedding(note_text)
        if query_embedding is None:
            return 0.0

        emb = np.array(instruction.semantic_embedding)
        query = np.array(query_embedding)

        dot_product = np.dot(emb, query)
        norm_emb = norm(emb)
        norm_query = norm(query)

        if norm_emb == 0 or norm_query == 0:
            return 0.0

        return float(dot_product / (norm_emb * norm_query))

    def _text_to_embedding(self, text: str) -> Optional[List[float]]:
        words = set(text.lower().split())
        if not words:
            return None

        embedding = [1.0 if i % 2 == 0 else 0.5 for i in range(len(words) * 4)]
        return embedding[:128]

    def _estimate_tokens(self, instruction: Instruction) -> int:
        words = len(instruction.contrastive_rule.split())
        return max(1, int(words * 1.3))