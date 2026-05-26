import tempfile
from pathlib import Path

from src.meta_verifier.schemas import Instruction, InstructionType
from src.meta_verifier.store import load_instructions, persist_efficacy_updates, save_instructions


def _instr(iid: int, score: float) -> Instruction:
    return Instruction(
        instruction_id=iid,
        type=InstructionType.SEMANTIC,
        instruction_text="t",
        description="d",
        target_codes=["E11"],
        source_hadm_ids=["1"],
        efficacy_score=score,
        semantic_embedding=[1.0, 0.0],
    )


def test_persist_efficacy_updates():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "instructions.parquet"
        save_instructions([_instr(1, 0.0), _instr(2, 0.0)], path)
        n = persist_efficacy_updates({1: 3.5, 2: -1.0}, path)
        assert n == 2
        loaded = load_instructions(path)
        by_id = {i.instruction_id: i.efficacy_score for i in loaded}
        assert by_id[1] == 3.5
        assert by_id[2] == -1.0
