import streamlit as st
import time
from backend.model_call import extract_text, llm_extract_schema
import json

st.title("NormaScore: Upload Assignment Details")
st.markdown("### Step 1: Provide the Question or Rubric")

with st.sidebar:
    st.markdown("### ⚙️ API Configuration")
    hf_token = st.text_input("Hugging Face API Token", type="password", help="Required to run the AI evaluator.")
    if hf_token:
        st.session_state['hf_token'] = hf_token

st.markdown("""
To accurately evaluate student submissions, NormaScore first needs to understand the assignment. 
Please upload the assignment prompt, grading rubric, or reference solution below.
""")

# File uploader
question_file = st.file_uploader(
    "Upload Question/Rubric", 
    accept_multiple_files=False,
    type=['txt', 'md', 'pdf', 'docx', 'json']
)

# Upload action
if question_file:
    if st.button("Proceed to Submissions", type="primary", use_container_width=True):
        if not st.session_state.get('hf_token'):
            st.error("Please provide a Hugging Face API Token in the sidebar.")
        else:
            with st.spinner("Analyzing assignment rubric... Generating reference schema..."):
                text = extract_text(question_file, question_file.name)
                
                # Check if it's already a json schema
                if question_file.name.endswith('.json'):
                    try:
                        ref_schema = json.loads(text)
                    except:
                        ref_schema = llm_extract_schema(text, hf_token=st.session_state['hf_token'])
                else:
                    ref_schema = llm_extract_schema(text, hf_token=st.session_state['hf_token'])
                
                if ref_schema:
                    st.session_state['reference_schema'] = ref_schema
                    st.session_state['question_uploaded'] = True
                    st.success("Assignment details saved! Redirecting to student submissions upload...")
                    time.sleep(1)
                    st.switch_page("views/upload.py")
                else:
                    st.error("Failed to generate reference schema from the document. Please ensure it's a valid Database Normalization problem.")
