import streamlit as st
import time
import json
import os
import pandas as pd
from dotenv import load_dotenv
from backend.model_call import extract_text, llm_extract_schema
from backend.ref_schema_builder import extract_text, llm_extract_schema, llm_generate_reference
# ── Page title ────────────────────────────────────────────────────────────────
st.title("NormaScore: Upload Assignment Details")

# ── Token: auto-load from .env, no sidebar UI (Code 2) ───────────────────────
load_dotenv()
if 'hf_token' not in st.session_state:
    st.session_state['hf_token'] = os.getenv("HF_TOKEN")

# ── Section 1: Question input (Code 2 — json added to accepted types) ─────────
st.markdown("### 1) Upload Question")
q_input_method = st.radio(
    "Question Input Format",
    ["Paste JSON", "Upload Document"],
    horizontal=True,
    label_visibility="collapsed"
)

question_text = ""
question_files = []

if q_input_method == "Paste JSON":
    question_text = st.text_area(
        "Paste Question/Schema JSON",
        height=200,
        help='E.g. {"relation_name": ... }'
    )
else:
    question_file = st.file_uploader(
        "Upload Question Document",
        type=['txt', 'md', 'pdf', 'docx', 'json'],
        accept_multiple_files=True  # Code 2: json included
    )

# ── Section 2: Rubrics (unchanged in both) ────────────────────────────────────
fd_score_per_item    = 1.0
final_score_per_item = 1.0

st.markdown("### 2) Input Rubrics")
st.markdown("Set up the custom scoring rubric table.")

if 'rubric_df' not in st.session_state:
    default_rubric_data = {
        "Functional Dependencies": {"Score": fd_score_per_item,    "Scoring Rule": " marks per FD"},
        "1NF (Composite)":         {"Score": 1.0,                  "Scoring Rule": "Handling of composite attributes"},
        "1NF (Multivalued)":       {"Score": 1.0,                  "Scoring Rule": "Handling of multivalued attributes"},
        "2NF (Partial)":           {"Score": 1.0,                  "Scoring Rule": "Identify and remove partial dependencies"},
        "2NF (Lossless)":          {"Score": 0.5,                  "Scoring Rule": "Ensure lossless decomposition"},
        "2NF (Key)":               {"Score": 1.0,                  "Scoring Rule": "Correct primary keys assigned"},
        "3NF (Transitive)":        {"Score": 1.0,                  "Scoring Rule": "Identify and remove transitive dependencies"},
        "3NF (Lossless)":          {"Score": 0.5,                  "Scoring Rule": "Ensure lossless decomposition"},
        "3NF (Key)":               {"Score": 1.0,                  "Scoring Rule": "Correct primary keys assigned"},
        "Final Relations":         {"Score": final_score_per_item, "Scoring Rule": " marks per Relation"},
    }
    st.session_state['rubric_df'] = pd.DataFrame.from_dict(default_rubric_data, orient='index')

edited_df = st.data_editor(
    st.session_state['rubric_df'],
    use_container_width=True,
    column_config={
        "Score":        st.column_config.NumberColumn("Score", min_value=0.0, step=0.5, format="%.1f"),
        "Scoring Rule": st.column_config.TextColumn("Scoring Rule"),
    }
)

# ── Section 3: Reference Schema upload (Code 1 description text) ──────────────
st.markdown("### 3) Reference Schema")
st.markdown(
    "If you already have a fully evaluated reference schema, upload it here. "
    "Otherwise we generate it from the question above."
)
schema_file = st.file_uploader("Upload Reference Schema (JSON)", type=['json'])

# ── Generate & Preview button (Code 1 flow) ───────────────────────────────────
if st.button("🔍 Generate & Preview Reference Schema", use_container_width=True):
    with st.spinner("Running model — this may take a minute..."):
        preview_schema = None
        preview_raw    = None

        # ── Path A: uploaded reference JSON ──────────────────────────────────
        if schema_file:
            try:
                text           = extract_text(schema_file, schema_file.name)
                preview_schema = json.loads(text)
                preview_raw    = json.dumps(preview_schema, indent=2)
            except Exception as e:
                st.error(f"Failed to parse uploaded JSON: {e}")
                st.stop()

        # ── Path B: pasted JSON question ──────────────────────────────────────
        elif question_text:
            if not st.session_state.get('hf_token'):
                st.error("No Hugging Face token found. Add HF_TOKEN to env/.env.")
                st.stop()
            try:
                q_data = json.loads(question_text)
            except json.JSONDecodeError:
                st.error("Invalid JSON pasted — please check the format.")
                st.stop()

            # Code 2 shortcut: if pasted JSON already IS a full schema, skip LLM
            if all(k in q_data for k in ['1nf', '2nf', '3nf', 'final_tables']):
                preview_schema = q_data
                preview_raw    = json.dumps(preview_schema, indent=2)
            else:
                # Code 1: use llm_generate_reference for structured JSON questions
                preview_schema, preview_raw = llm_generate_reference(
                    q_data, hf_token=st.session_state['hf_token']
                )

        # ── Path C: uploaded question document ───────────────────────────────
        elif question_files:
            if not st.session_state.get('hf_token'):
                st.error("No Hugging Face token found. Add HF_TOKEN to env/.env.")
                st.stop()
            all_schemas = []

            for question_file in question_files:
                text = extract_text(question_file, question_file.name)

                try:
                    q_data = json.loads(text)

                    # Already full schema
                    if all(k in q_data for k in ['1nf', '2nf', '3nf', 'final_tables']):
                        preview_schema = q_data

                    else:
                        preview_schema, preview_raw = llm_generate_reference(
                            q_data,
                            hf_token=st.session_state['hf_token']
                        )

                except json.JSONDecodeError:
                    preview_schema = llm_extract_schema(
                        text,
                        hf_token=st.session_state['hf_token']
                    )

                all_schemas.append({
                    "filename": question_file.name,
                    "schema": preview_schema
                })

            preview_schema = all_schemas
            preview_raw = json.dumps(all_schemas, indent=2)
        else:
            st.error("Please provide a question or upload a reference schema first.")
            st.stop()

        st.session_state['preview_schema'] = preview_schema
        st.session_state['preview_raw']    = preview_raw

# ── Preview output (Code 1 two-tab UI) ───────────────────────────────────────
if st.session_state.get('preview_raw') or st.session_state.get('preview_schema'):
    st.markdown("---")
    st.markdown("### 🤖 Model Output")

    tab1, tab2 = st.tabs(["📄 Raw Model Output", "🗂️ Parsed JSON"])

    with tab1:
        raw = st.session_state.get('preview_raw', '')
        st.markdown("This is the **exact text** your fine-tuned model returned:")
        st.text_area(
            label="raw_output",
            value=raw if raw else "(empty — model returned nothing)",
            height=500,
            disabled=True,
            label_visibility="collapsed"
        )
        if st.session_state.get('preview_schema'):
            st.success("✅ JSON parsing succeeded — schema extracted cleanly from model output.")
        else:
            st.error("❌ JSON parsing failed — the model output could not be parsed into a schema. Check the raw text above.")

    with tab2:
        schema = st.session_state.get('preview_schema')
        if schema:
            st.markdown("This is the **parsed schema** that will be used as the reference answer:")
            st.json(schema)
        else:
            st.warning("No parsed schema available — see Raw Model Output tab for details.")

    st.markdown("---")

    if st.session_state.get('preview_schema'):
        if st.button("✅ Looks good — Save & Proceed to Submissions", type="primary", use_container_width=True):
            st.session_state['reference_schema']  = st.session_state['preview_schema']
            st.session_state['custom_rubric']     = edited_df
            st.session_state['question_uploaded'] = True
            st.success("Saved! Redirecting...")
            time.sleep(1)
            st.switch_page("views/upload.py")
    else:
        st.error("Fix the model output before proceeding — the schema could not be parsed.")