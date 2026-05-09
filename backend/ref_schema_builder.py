import os
import re
import json
import inspect
import pdfplumber
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig
from huggingface_hub import snapshot_download

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME     = "asky777/qwen-normabench-grpo-pass2-step76"  # LoRA adapter
BASE_MODEL     = "Qwen/Qwen2.5-7B-Instruct"
TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Exact system prompt from Kaggle notebook
SYSTEM_PROMPT = """You are a database normalization expert. Given a relation with attributes, multivalued attributes, and functional dependencies, normalize it step by step through 1NF, 2NF, and 3NF.

Return ONLY a JSON object with this exact structure:
{
  "attribute": [...],
  "multivalued": [...],
  "fd_set": [...],
  "1nf": {"tables": [{"name": "...", "attributes": [...], "primary_key": [...]}]},
  "anomalies_1nf": {"insertion": bool, "update": bool, "deletion": bool},
  "2nf": {"tables": [{"name": "...", "attributes": [...], "primary_key": [...]}]},
  "anomalies_2nf": {"insertion": bool, "update": bool, "deletion": bool},
  "3nf": {"tables": [{"name": "...", "attributes": [...], "primary_key": [...]}]},
  "anomalies_3nf": {"insertion": bool, "update": bool, "deletion": bool},
  "final": {"tables": [{"name": "...", "attributes": [...], "primary_key": [...]}]}
}

### 1NF: ONLY separate multivalued. DO NOT remove partial/transitive deps. They MUST still exist.
### 2NF: ONLY remove partial deps. DO NOT remove transitive. They MUST still exist.
### 3NF: ONLY remove transitive deps.
Rules:
- "bool" means the value must be either true or false.
- Determine the anomaly values based on the schema produced at each stage.
- Do NOT hardcode anomaly values.
- Analyze the tables at each normalization stage and decide whether insertion, update, or deletion anomalies exist.

"""

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(hf_token: str = None):
    token = hf_token or os.environ.get("HF_TOKEN")

    # Download adapter locally so we can patch its config
    adapter_path = snapshot_download(MODEL_NAME, token=token)

    # Strip unknown LoraConfig keys (adapter was saved with newer PEFT)
    config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(config_path) as f:
        adapter_cfg = json.load(f)
    valid_keys = set(inspect.signature(LoraConfig.__init__).parameters.keys()) - {"self"}
    for key in set(adapter_cfg.keys()) - valid_keys:
        adapter_cfg.pop(key)
    with open(config_path, "w") as f:
        json.dump(adapter_cfg, f)

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        token=token,
        trust_remote_code=True,
        use_fast=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quantization — fits a 7B model on a single 16GB GPU
    # Kaggle used 2x T4 (31GB total) with float16; we use 4-bit to match on 1 GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        token=token,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        token=token,
        low_cpu_mem_usage=False,  # avoids lm_head KeyError with device_map="auto"
    )
    model.eval()
    return tokenizer, model


# ─────────────────────────────────────────────────────────────────────────────
# FD NORMALIZER
# ─────────────────────────────────────────────────────────────────────────────

def normalize_fds(fds_raw: list) -> list:
    """
    Accepts fds in ANY of these formats and always returns list-of-(lhs_list, rhs_list):
      - [["guest_id"], ["guest_name", "guest_email"]]   <- JSON list-of-lists
      - "guest_id->guest_name,guest_email"              <- string arrow format
      - {"lhs": ["guest_id"], "rhs": ["guest_name"]}    <- dict format
    """
    normalized = []
    for fd in fds_raw:
        if isinstance(fd, str):
            if "->" not in fd:
                continue
            lhs_str, rhs_str = fd.split("->", 1)
            lhs = [a.strip() for a in lhs_str.split(",") if a.strip()]
            rhs = [a.strip() for a in rhs_str.split(",") if a.strip()]
            normalized.append((lhs, rhs))

        elif isinstance(fd, dict):
            lhs = fd.get("lhs", [])
            rhs = fd.get("rhs", [])
            if isinstance(lhs, str): lhs = [lhs]
            if isinstance(rhs, str): rhs = [rhs]
            normalized.append((lhs, rhs))

        elif isinstance(fd, (list, tuple)) and len(fd) == 2:
            lhs, rhs = fd[0], fd[1]
            if isinstance(lhs, str): lhs = [lhs]
            if isinstance(rhs, str): rhs = [rhs]
            normalized.append((lhs, rhs))

    return normalized


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER — exact Kaggle format
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(schema: dict, tokenizer) -> str:
    fds_normalized = normalize_fds(schema.get('fds', []))
    fd_lines = "\n".join(
        f"{', '.join(lhs)} -> {', '.join(rhs)}"
        for lhs, rhs in fds_normalized
    )

    # Exact user prompt format used in Kaggle training loop
    user_prompt = (
        f"Relation attributes: {schema['attributes']}\n"
        f"Multivalued attributes: {schema.get('multivalued', [])}\n"
        f"Functional Dependencies:\n{fd_lines}\n\n"
        f"Return the normalization in JSON format."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSON EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_json_robust(text: str):
    # Try fenced code blocks first
    for pattern in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```']:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

    # Fall back to brace-matching
    start = text.find('{')
    if start < 0:
        return None

    depth, in_string, escape = 0, False, False
    for i, c in enumerate(text[start:], start=start):
        if escape:
            escape = False
            continue
        if c == '\\':
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if not in_string:
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT NORMALIZER
# Adapts model output keys to what the Streamlit app expects
# Model returns: "1nf": {"tables": [...]} and "primary_key"
# App expects:   "1nf": [...] and "pk"
# ─────────────────────────────────────────────────────────────────────────────

def normalize_model_output(parsed: dict) -> dict:
    if parsed is None:
        return None

    # Flatten "1nf": {"tables": [...]} -> "1nf": [...]
    for stage in ["1nf", "2nf", "3nf"]:
        if stage in parsed and isinstance(parsed[stage], dict):
            parsed[stage] = parsed[stage].get("tables", [])

    # Map "final" -> "final_tables"
    if "final" in parsed and "final_tables" not in parsed:
        final = parsed.pop("final")
        parsed["final_tables"] = final.get("tables", []) if isinstance(final, dict) else final

    # Map "primary_key" -> "pk" in every table
    for stage in ["1nf", "2nf", "3nf", "final_tables"]:
        for table in parsed.get(stage, []):
            if "primary_key" in table and "pk" not in table:
                table["pk"] = table.pop("primary_key")

    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION — exact Kaggle params
# ─────────────────────────────────────────────────────────────────────────────

def generate_from_model(prompt: str, hf_token: str = None, max_new_tokens: int = 2048):
    tokenizer, model = load_model(hf_token)

    inputs = tokenizer(prompt, return_tensors="pt")
    # Use model.device — works correctly with 4-bit + device_map="auto"
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    print(f"Input tokens: {input_len}, generating...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
        )

    print("Generation done.")
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# def preprocess(text: str) -> str:
    text = text.lower()
    text = text.replace("→", "->").replace("=>", "->")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# def extract_text(file_like, filename: str) -> str:
#     text = ""
#     if filename.lower().endswith('.pdf'):
#         with pdfplumber.open(file_like) as pdf:
#             for page in pdf.pages:
#                 extracted = page.extract_text()
#                 if extracted:
#                     text += extracted + "\n"
#     else:
#         text = file_like.getvalue().decode("utf-8")
#     return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def llm_generate_reference(schema: dict, hf_token=None, max_new_tokens: int = 2048):
    tokenizer, _ = load_model(hf_token)
    prompt = build_prompt(schema, tokenizer)

    print("=== PROMPT (last 400 chars) ===")
    print(prompt[-400:])
    print("=== END PROMPT ===")

    raw_response = generate_from_model(prompt, hf_token=hf_token, max_new_tokens=max_new_tokens)

    print("=== RAW RESPONSE ===")
    print(raw_response)
    print("=== END RAW RESPONSE ===")

    parsed = extract_json_robust(raw_response)
    parsed = normalize_model_output(parsed)

    print("=== PARSED ===")
    print(parsed)
    print("=== END PARSED ===")

    return parsed, raw_response


# def llm_extract_schema(text: str, hf_token=None, max_new_tokens: int = 2048):
    tokenizer, _ = load_model(hf_token)
    clean_text = preprocess(text)

    system = "You are extracting database normalization answers from student text. Output ONLY valid JSON — no markdown, no explanation."
    user = f"""REQUIRED JSON FORMAT:
{{
  "attribute": [], "multivalued": [], "compositeattributes": [],
  "fds": [], "1nf": [], "2nf": [], "3nf": [], "final_tables": []
}}

Student Answer:
<<<{clean_text}>>>

Output JSON:"""

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print("=== EXTRACT SCHEMA PROMPT (last 300 chars) ===")
    print(prompt[-300:])

    raw_response = generate_from_model(prompt, hf_token=hf_token, max_new_tokens=max_new_tokens)

    print("=== EXTRACT SCHEMA RAW RESPONSE ===")
    print(raw_response)

    parsed = extract_json_robust(raw_response)
    return parsed, raw_response