import os
import re
import json
import pdfplumber
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL (same as Kaggle)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "asky777/rl-normalizer-qwen-7b"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    return tokenizer, model


tokenizer, model = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

REFERENCE_PROMPT_SYSTEM = """You are a database normalization expert.
You must show the COMPLETE normalization process: 1NF → 2NF → 3NF.
Each stage must show tables, even if they don't change from the previous stage.
Output ONLY valid JSON — no markdown fences, no explanation."""

REFERENCE_PROMPT_USER = """Normalize this schema step-by-step, showing 1NF, 2NF, and 3NF stages.

REQUIRED JSON FORMAT:
{{
  "attribute": ["all", "attributes", "here"],
  "multivalued": [],
  "compositeattributes": [],
  "fds": ["attr1->attr2", "attr2->attr3"],
  "1nf": [
    {{"name": "TableName", "attributes": ["attr1","attr2","attr3"], "pk": ["attr1"]}}
  ],
  "anomalies_1nf": {{"insertion": false, "update": false, "deletion": false}},
  "2nf": [
    {{"name": "Table1", "attributes": ["attr1","attr2"], "pk": ["attr1"]}},
    {{"name": "Table2", "attributes": ["attr2","attr3"], "pk": ["attr2"]}}
  ],
  "anomalies_2nf": {{"insertion": false, "update": false, "deletion": false}},
  "3nf": [
    {{"name": "Table1", "attributes": ["attr1","attr2"], "pk": ["attr1"]}},
    {{"name": "Table2", "attributes": ["attr2","attr3"], "pk": ["attr2"]}}
  ],
  "anomalies_3nf": {{"insertion": false, "update": false, "deletion": false}},
  "final_tables": [
    {{"name": "Table1", "attributes": ["attr1","attr2"], "pk": ["attr1"]}},
    {{"name": "Table2", "attributes": ["attr2","attr3"], "pk": ["attr2"]}}
  ]
}}

Input Schema:
Relation: {relation_name}
Attributes: {attrs_str}
Multivalued Attributes: {mv_str}
Functional Dependencies:
{fds_str}

Output JSON:"""

STUDENT_EXTRACT_PROMPT = """You are extracting database normalization answers from student text.

STRICT RULES:
- Output ONLY valid JSON — no markdown, no explanation
- Do NOT invent attributes, tables, or dependencies
- If information is missing, use empty lists []
- Attribute names must be lowercase
- Use '->' for functional dependencies

REQUIRED JSON FORMAT:
{{
  "attribute": [],
  "multivalued": [],
  "compositeattributes": [],
  "fds": [],
  "1nf": [],
  "2nf": [],
  "3nf": [],
  "final_tables": []
}}

Student Answer:
<<<{text}>>>

Output JSON:"""

# ─────────────────────────────────────────────────────────────────────────────
# JSON EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_json_robust(text: str):

    for pattern in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```']:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

    start = text.find('{')
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False

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
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(text: str):
    text = text.lower()
    text = text.replace("→", "->").replace("=>", "->").replace(":", "->")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ─────────────────────────────────────────────────────────────────────────────
# TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_text(file_like, filename):

    text = ""
    if filename.lower().endswith('.pdf'):
        with pdfplumber.open(file_like) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    else:
        text = file_like.getvalue().decode("utf-8")

    return text.strip()

# ─────────────────────────────────────────────────────────────────────────────
# MODEL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_from_model(prompt, max_new_tokens=2048):

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

def llm_generate_reference(schema: dict, hf_token=None, max_new_tokens: int = 2048):

    attrs_str = ', '.join(schema['attributes'])
    mv_str = json.dumps(schema.get('multivalued', []))

    fds_display = []
    for lhs, rhs in schema.get('fds', []):
        lhs_str = ','.join(lhs)
        rhs_str = ','.join(rhs)
        fds_display.append(f"{lhs_str} -> {rhs_str}")

    fds_str = "\n".join(fds_display)

    user_content = REFERENCE_PROMPT_USER.format(
        relation_name=schema.get('relation_name', 'RELATION'),
        attrs_str=attrs_str,
        mv_str=mv_str,
        fds_str=fds_str
    )

    prompt = f"{REFERENCE_PROMPT_SYSTEM}\n\n{user_content}"

    raw_response = generate_from_model(prompt, max_new_tokens)

    parsed = extract_json_robust(raw_response)

    return parsed, raw_response

# ─────────────────────────────────────────────────────────────────────────────
# STUDENT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def llm_extract_schema(text, hf_token=None, max_new_tokens=2048):

    clean_text = preprocess(text)

    prompt = STUDENT_EXTRACT_PROMPT.format(text=clean_text)

    raw_response = generate_from_model(prompt, max_new_tokens)

    parsed = extract_json_robust(raw_response)

    return parsed, raw_response