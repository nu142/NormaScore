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

# Default multiplier values
fd_score_per_item = 1.0
final_score_per_item = 1.0

st.markdown("### 2) Input Rubrics")
st.markdown("Set up the custom scoring rubric table.")

# Rebuild default rubric if missing
if 'rubric_df' not in st.session_state:
    default_rubric_data = {
        "Functional Dependencies": {"Score": fd_score_per_item, "Scoring Rule": " marks per FD"},
        "1NF (Composite)": {"Score": 1.0, "Scoring Rule": "Handling of composite attributes"},
        "1NF (Multivalued)": {"Score": 1.0, "Scoring Rule": "Handling of multivalued attributes"},
        "2NF (Partial)": {"Score": 1.0, "Scoring Rule": "Identify and remove partial dependencies"},
        "2NF (Lossless)": {"Score": 2.0, "Scoring Rule": "Ensure lossless decomposition"},
        "2NF (Key)": {"Score": 1.0, "Scoring Rule": "Correct primary keys assigned"},
        "3NF (Transitive)": {"Score": 1.0, "Scoring Rule": "Identify and remove transitive dependencies"},
        "3NF (Lossless)": {"Score": 2.0, "Scoring Rule": "Ensure lossless decomposition"},
        "3NF (Key)": {"Score": 1.0, "Scoring Rule": "Correct primary keys assigned"},
        "Final Relations": {"Score": final_score_per_item, "Scoring Rule": " marks per Relation"}
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
st.markdown("If you already have a fully evaluated reference schema, you can upload it here directly. Otherwise, skip this and we will generate it from the question above.")
schema_file = st.file_uploader("Upload Reference Schema (JSON)", type=['json'])

if st.button("Save & Proceed to Submissions", type="primary", use_container_width=True):
    with st.spinner("Processing..."):
        ref_schema = None
        needs_generation = False
        
        if schema_file:
            try:
                text = extract_text(schema_file, schema_file.name)
                ref_schema = json.loads(text)
            except Exception as e:
                st.error(f"Failed to parse Reference Schema JSON: {e}")
                st.stop()
        elif question_text:
            try:
                q_data = json.loads(question_text)
                if all(k in q_data for k in ['1nf', '2nf', '3nf', 'final_tables']):
                    ref_schema = q_data
                else:
                    needs_generation = True
            except:
                needs_generation = True
        elif question_file:
            needs_generation = True
        else:
            st.error("Please provide a Question or a Reference Schema.")
            st.stop()

        if needs_generation:
            if not st.session_state.get('hf_token'):
                st.error("Please provide a Hugging Face API Token in the sidebar to generate the schema.")
                st.stop()
            if question_text:
                try:
                    q_data = json.loads(question_text)
                    ref_schema, _ = llm_generate_reference(q_data, hf_token=st.session_state['hf_token'])
                except json.JSONDecodeError:
                    ref_schema, _ = llm_extract_schema(question_text, hf_token=st.session_state['hf_token'])
            elif question_file:
                text = extract_text(question_file, question_file.name)
                try:
                    q_data = json.loads(text)
                    ref_schema, _ = llm_generate_reference(q_data, hf_token=st.session_state['hf_token'])
                except json.JSONDecodeError:
                    ref_schema, _ = llm_extract_schema(text, hf_token=st.session_state['hf_token'])
        
        if ref_schema:
            st.session_state['reference_schema'] = ref_schema
            st.session_state['custom_rubric'] = edited_df
            st.session_state['question_uploaded'] = True
            st.success("Assignment details and rubric saved! Redirecting...")
            time.sleep(1)
            st.switch_page("views/upload.py")

