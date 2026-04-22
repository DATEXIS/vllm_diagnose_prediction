BASE_PROMPT = """Analyze the following clinical note and predict ICD diagnoses.

Clinical Note:
{admission_note}

Provide your diagnosis predictions in JSON format:
"""

THINKING_PROMPT = """<thinking>
Instruction {instruction_id}: {contrastive_rule}
{previous_prediction}Applying to prediction...
</thinking>
"""

FINAL_PROMPT = """{thinking_blocks}

Provide your final diagnosis predictions in JSON format:
"""