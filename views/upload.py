import streamlit as st
import time
from backend.evaluation import evaluate_all

st.title("NormaScore: Upload Student Submissions")
st.markdown("### Step 2: Provide the Student Work")

if not st.session_state.get('reference_schema'):
    st.warning("⚠️ No assignment context found. Please upload the question/rubric first.")
    if st.button("Go to Step 1"):
        st.switch_page("views/question_upload.py")
    st.stop()

st.markdown("""
Now that the assignment details are loaded, please upload the student submissions.
NormaScore will evaluate these against the provided rubric and generate individual feedback.
""")

st.info("💡 **Tip**: You can drag and drop an entire folder of student files (.txt, .md, .pdf, .docx, .json) directly into the uploader below, or select multiple files at once.")

# File uploader
uploaded_files = st.file_uploader(
    "Upload Student Files or Folder", 
    accept_multiple_files=True,
    type=['txt', 'md', 'pdf', 'docx', 'py', 'json']
)

# Upload action
if uploaded_files:
    if st.button("Process Submissions", type="primary", use_container_width=True):
        if not st.session_state.get('hf_token'):
            st.error("Missing Hugging Face API Token. Please go back to Step 1 and provide the token.")
        else:
            with st.spinner("Analyzing student submissions against reference schema using AI... This may take a moment."):
                df_results = evaluate_all(
                    uploaded_files, 
                    st.session_state['reference_schema'], 
                    hf_token=st.session_state['hf_token'],
                    custom_rubric=st.session_state.get('custom_rubric')
                )
                
                # Save state
                st.session_state['evaluation_df'] = df_results
                st.session_state['submissions_processed'] = True
                
                st.success("Analysis complete! Redirecting to the evaluation dashboard...")
                time.sleep(1)
                
                # Redirect to evaluation page
                st.switch_page("views/evaluation.py")
