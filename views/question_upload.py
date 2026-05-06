import streamlit as st
import time
import json
import pandas as pd
from backend.model_call import extract_text, llm_extract_schema, llm_generate_reference

st.title("NormaScore: Upload Assignment Details")

with st.sidebar:
    st.markdown("### ⚙️ API Configuration")
    hf_token = st.text_input("Hugging Face API Token", type="password", help="Required to run the AI evaluator.")
    if hf_token:
        st.session_state['hf_token'] = hf_token

st.markdown("### 1) Upload Question")
q_input_method = st.radio("Question Input Format", ["Paste JSON", "Upload Document"], horizontal=True, label_visibility="collapsed")

question_text = ""
question_file = None

if q_input_method == "Paste JSON":
    question_text = st.text_area("Paste Question/Schema JSON", height=200, help="E.g. {\"relation_name\": ... }")
else:
    question_file = st.file_uploader("Upload Question Document", type=['txt', 'md', 'pdf', 'docx'])

fd_score_per_item = 1.0
final_score_per_item = 1.0

st.markdown("### 2) Input Rubrics")
st.markdown("Set up the custom scoring rubric table.")

if 'rubric_df' not in st.session_state:
    default_rubric_data = {
        "Functional Dependencies": {"Score": fd_score_per_item, "Scoring Rule": " marks per FD"},
        "1NF (Composite)":         {"Score": 1.0, "Scoring Rule": "Handling of composite attributes"},
        "1NF (Multivalued)":       {"Score": 1.0, "Scoring Rule": "Handling of multivalued attributes"},
        "2NF (Partial)":           {"Score": 1.0, "Scoring Rule": "Identify and remove partial dependencies"},
        "2NF (Lossless)":          {"Score": 2.0, "Scoring Rule": "Ensure lossless decomposition"},
        "2NF (Key)":               {"Score": 1.0, "Scoring Rule": "Correct primary keys assigned"},
        "3NF (Transitive)":        {"Score": 1.0, "Scoring Rule": "Identify and remove transitive dependencies"},
        "3NF (Lossless)":          {"Score": 2.0, "Scoring Rule": "Ensure lossless decomposition"},
        "3NF (Key)":               {"Score": 1.0, "Scoring Rule": "Correct primary keys assigned"},
        "Final Relations":         {"Score": final_score_per_item, "Scoring Rule": " marks per Relation"}
    }
    st.session_state['rubric_df'] = pd.DataFrame.from_dict(default_rubric_data, orient='index')

edited_df = st.data_editor(
    st.session_state['rubric_df'],
    use_container_width=True,
    column_config={
        "Score": st.column_config.NumberColumn("Score", min_value=0.0, step=0.5, format="%.1f"),
        "Scoring Rule": st.column_config.TextColumn("Scoring Rule")
    }
)

st.markdown("### 3) Reference Schema")
st.markdown("If you already have a fully evaluated reference schema, upload it here. Otherwise we generate it from the question above.")
schema_file = st.file_uploader("Upload Reference Schema (JSON)", type=['json'])

# ── Generate & Preview button ─────────────────────────────────────────────────
if st.button("🔍 Generate & Preview Reference Schema", use_container_width=True):
    with st.spinner("Running model — this may take a minute..."):
        preview_schema = None
        preview_raw    = None

        if schema_file:
            try:
                text = extract_text(schema_file, schema_file.name)
                preview_schema = json.loads(text)
                preview_raw    = json.dumps(preview_schema, indent=2)
            except Exception as e:
                st.error(f"Failed to parse uploaded JSON: {e}")
                st.stop()

        elif question_text:
            if not st.session_state.get('hf_token'):
                st.error("Add your Hugging Face token in the sidebar first.")
                st.stop()
            try:
                q_data = json.loads(question_text)
            except json.JSONDecodeError:
                st.error("Invalid JSON pasted — please check the format.")
                st.stop()
            preview_schema, preview_raw = llm_generate_reference(
                q_data, hf_token=st.session_state['hf_token']
            )

        elif question_file:
            if not st.session_state.get('hf_token'):
                st.error("Add your Hugging Face token in the sidebar first.")
                st.stop()
            text = extract_text(question_file, question_file.name)
            try:
                q_data = json.loads(text)
                preview_schema, preview_raw = llm_generate_reference(
                    q_data, hf_token=st.session_state['hf_token']
                )
            except json.JSONDecodeError:
                preview_schema, preview_raw = llm_extract_schema(
                    text, hf_token=st.session_state['hf_token']
                )

        else:
            st.error("Please provide a question or upload a reference schema first.")
            st.stop()

        st.session_state['preview_schema'] = preview_schema
        st.session_state['preview_raw']    = preview_raw

# ── Full output preview ───────────────────────────────────────────────────────
if st.session_state.get('preview_raw') or st.session_state.get('preview_schema'):
    st.markdown("---")
    st.markdown("### 🤖 Model Output")

    tab1, tab2 = st.tabs(["📄 Raw Model Output", "🗂️ Parsed JSON"])

    # Tab 1 — exactly what the model returned, nothing hidden
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
        # Also show whether JSON parsing succeeded or failed
        if st.session_state.get('preview_schema'):
            st.success("✅ JSON parsing succeeded — schema extracted cleanly from model output.")
        else:
            st.error("❌ JSON parsing failed — the model output could not be parsed into a schema. Check the raw text above.")

    # Tab 2 — the parsed schema as pretty JSON, so you can inspect structure
    with tab2:
        schema = st.session_state.get('preview_schema')
        if schema:
            st.markdown("This is the **parsed schema** that will be used as the reference answer:")
            st.json(schema)   # st.json renders collapsible, syntax-highlighted JSON
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