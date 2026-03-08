import os
import re
import json
import pdfplumber
from huggingface_hub import InferenceClient

LLM_PROMPT = """
You are extracting database normalization answers.

STRICT RULES:
- Output ONLY valid JSON
- Do NOT include markdown, comments, or explanations
- Do NOT invent attributes, tables, or dependencies
- If information is missing, use empty lists []
- Attribute names must be lowercase
- Use '->' for functional dependencies
- Follow the JSON format EXACTLY

REQUIRED JSON FORMAT:
{
  "attribute": [],
  "multivalued": [],
  "compositeattributes": [],
  "fds": [],
  "1nf": [],
  "2nf": [],
  "3nf": [],
  "final_tables": []
}

Database Schema Description:
<<<{text}>>>
"""

def extract_text(file_like, filename: str):
    """
    Extract text. We receive a Streamlit UploadedFile object.
    """
    text = ""
    if filename.lower().endswith('.pdf'):
        with pdfplumber.open(file_like) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted: text += extracted + "\\n"
    else:
        text = file_like.getvalue().decode("utf-8")
    return text.strip()

def preprocess(text: str) -> str:
    text = text.lower()
    text = text.replace("→", "->").replace("=>", "->").replace(":", "->")
    text = re.sub(r"\\s+", " ", text)
    fillers = ["the table is", "we have", "let us consider", "primary key is", "pk is"]
    for f in fillers:
        text = text.replace(f, "")
    return text.strip()

def extract_json_from_response(text):
    match = re.search(r"\\{.*\\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in response")
    return match.group(0)

def llm_extract_schema(text, hf_token=None, is_student=False):
    if not hf_token:
        raise ValueError("Hugging Face API token is required.")
        
    client = InferenceClient(api_key=hf_token)
    clean_text = preprocess(text)
    
    prompt = LLM_PROMPT.format(text=clean_text)
    if is_student:
        prompt = prompt.replace("Database Schema Description:", "Student Answer:")
        
    messages = [{"role": "user", "content": prompt}]
    
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct", 
        messages=messages, 
        max_tokens=4096,
        temperature=0.0
    )
    
    raw_response = response.choices[0].message.content
    try:
        return json.loads(extract_json_from_response(raw_response))
    except Exception as e:
        print(f"Extraction Error: {e}")
        print(f"Raw Output: {raw_response}")
        return None
