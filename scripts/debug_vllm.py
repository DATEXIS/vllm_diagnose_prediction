"""Local vLLM debug script — works in terminal, IPython, or Jupyter.

Edit the CONFIG block below, then run:
    python scripts/debug_vllm.py
or paste into an IPython/Jupyter cell.
"""

import json
import sys
import textwrap

import requests

# ===========================================================================
# CONFIG — edit these
# ===========================================================================
HOST     = "localhost"
PORT     = 8000
MODEL    = None        # None = auto-detect from /v1/models
THINKING = False        # enable_thinking chat_template_kwarg
BUDGET   = 4096        # reasoning budget tokens (None = server default)
GUIDED   = False       # guided decoding with JSON schema
TEMP     = 0.6
MAX_TOK  = 4096
SHORT    = True        # True = short dummy note, False = longer note
# ===========================================================================

SCHEMA = {
    "name": "ICDsModel",
    "schema": {
        "type": "object",
        "properties": {
            "diagnoses": {
                "type": "array",
                "minItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "icd_code": {"type": "string"},
                        "reason":   {"type": "string"},
                    },
                    "required": ["icd_code", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["diagnoses"],
        "additionalProperties": False,
    },
    "strict": True,
}

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert medical coder. Extract ALL relevant diagnoses from the
    admission note below and assign the most appropriate ICD-10 code for each.
    Output a single JSON object with a "diagnoses" key containing a list of
    {icd_code, reason} objects. No prose outside the JSON.
""")

SHORT_NOTE = textwrap.dedent("""\
    68F admitted with chest pain and shortness of breath.
    PMH: T2DM, HTN, HLD. On metformin, lisinopril, atorvastatin.
    Troponin mildly elevated. EKG: normal sinus rhythm.
    Impression: NSTEMI ruled out. Likely demand ischemia.
""")

LONG_NOTE = textwrap.dedent("""\
    ADMISSION NOTE
    Patient: 72-year-old male presenting with progressive dyspnea on exertion x3 weeks.

    HISTORY OF PRESENT ILLNESS:
    Mr. X is a 72-year-old male with a significant past medical history of congestive
    heart failure (EF 35%), type 2 diabetes mellitus, chronic kidney disease stage 3,
    and atrial fibrillation on anticoagulation who presents with worsening dyspnea
    on exertion over the past 3 weeks. He reports 2-pillow orthopnea and bilateral
    leg swelling. He denies chest pain or fever. He had a similar episode 6 months
    ago that responded to diuresis.

    PAST MEDICAL HISTORY:
    - Congestive heart failure, systolic, EF 35%
    - Type 2 diabetes mellitus, insulin-dependent
    - Chronic kidney disease, stage 3 (baseline Cr 1.8)
    - Atrial fibrillation, on apixaban
    - Hypertension
    - Hyperlipidemia
    - GERD

    MEDICATIONS: furosemide 40mg daily, carvedilol 12.5mg BID, lisinopril 5mg,
    apixaban 5mg BID, insulin glargine 20 units QHS, atorvastatin 40mg,
    omeprazole 20mg.

    PHYSICAL EXAM: BP 158/92, HR 88 irregular, RR 22, SpO2 93% RA.
    JVP elevated at 14cm. Bilateral crackles lower lung fields. 2+ pitting edema bilaterally.

    LABS: BNP 1840, Cr 2.1 (up from 1.8), K 4.2, Na 138. Troponin neg x2.
    HbA1c 8.4%.

    IMPRESSION: Acute decompensated heart failure, likely dietary indiscretion.
    Plan: IV furosemide, fluid restriction, optimize guideline-directed therapy.
""")


# ---------------------------------------------------------------------------
def get_model_name(base_url):
    resp = requests.get(f"{base_url}/models", timeout=10)
    resp.raise_for_status()
    return resp.json()["data"][0]["id"]


def build_payload(model, messages):
    payload = {
        "model":       model,
        "messages":    messages,
        "temperature": TEMP,
        "max_tokens":  MAX_TOK,
    }
    if THINKING:
        kt = {"enable_thinking": True}
        if BUDGET is not None:
            kt["thinking_budget"] = BUDGET
        payload["chat_template_kwargs"] = kt
    if GUIDED:
        payload["response_format"] = {"type": "json_schema", "json_schema": SCHEMA}
    return payload


def call_streaming(base_url, payload):
    content_chunks   = []
    reasoning_chunks = []
    finish_reason    = None
    usage            = {}

    print("[streaming] reasoning=dots, content=text\n", flush=True)
    with requests.post(
        f"{base_url}/chat/completions",
        json={**payload, "stream": True},
        stream=True,
        timeout=600,
    ) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            usage  = chunk.get("usage") or usage
            choice = chunk.get("choices", [{}])[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            delta  = choice.get("delta", {})

            r = delta.get("reasoning") or ""
            if r:
                reasoning_chunks.append(r)
                print(".", end="", flush=True)


            c = delta.get("content") or ""
            if c:
                content_chunks.append(c)
                print(c, end="", flush=True)

    print("\n", flush=True)
    return {
        "choices": [{
            "message": {
                "content":           "".join(content_chunks),
                "reasoning": "".join(reasoning_chunks),
            },
            "finish_reason": finish_reason,
        }],
        "usage": usage,
    }


def sep(title, width=80):
    print(f"\n{'='*width}\n  {title}\n{'='*width}")


# ---------------------------------------------------------------------------
base_url = f"http://{HOST}:{PORT}/v1"
note     = SHORT_NOTE if SHORT else LONG_NOTE

model = MODEL
if model is None:
    model = get_model_name(base_url)
    print(f"[model] {model}")

# messages = [
#     {"role": "system", "content": SYSTEM_PROMPT},
#     {"role": "user",   "content": f"### Admission Note\n{note}\n\n### Output"},
# ]

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user",   "content": f"### Admission Note\n{note}\n\n### Output\n/think"},
]

payload = build_payload(model, messages)

print(f"[config] thinking={THINKING} budget={BUDGET} guided={GUIDED} temp={TEMP} max_tok={MAX_TOK}")
print(f"[payload]\n{json.dumps(payload, indent=2)}\n")

raw = call_streaming(base_url, payload)

msg          = raw["choices"][0]["message"]
content      = msg.get("content") or ""
reasoning    = msg.get("reasoning") or ""
finish       = raw["choices"][0].get("finish_reason", "?")
usage        = raw.get("usage", {})

sep("THINKING (reasoning)")

sep("THINKING DEBUG")
print(f"reasoning length : {len(reasoning)}")
print(f"reasoning repr   : {repr(reasoning[:200])}")
print(f"content length   : {len(content)}")

print(reasoning or "(empty)")

sep("CONTENT (raw)")
print(content or "(empty)")

sep("FINISH / USAGE")
print(f"finish_reason  : {finish}")
print(f"prompt_tokens  : {usage.get('prompt_tokens', '?')}")
print(f"completion_tok : {usage.get('completion_tokens', '?')}")
print(f"msg keys       : {list(msg.keys())}")

sep("PARSED JSON")
if content.strip():
    try:
        print(json.dumps(json.loads(content), indent=2))
    except json.JSONDecodeError as e:
        print(f"[PARSE ERROR] {e}")
        print("content[:300]:", repr(content[:300]))
else:
    print("(content empty — see reasoning above)")
