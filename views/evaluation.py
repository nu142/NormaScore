import streamlit as st
import pandas as pd

st.title("NormaScore: Evaluation Dashboard")

# Check if we landed here without evaluations
if 'evaluation_df' not in st.session_state:
    st.warning("⚠️ No evaluations found. Please upload submissions first.")
    if st.button("Go to Upload"):
        st.switch_page("views/question_upload.py")
    st.stop()

df = st.session_state['evaluation_df']
num_students = len(df)

# Calculate Summary Metrics
avg_score = df["Score"].mean()
highest_score = df["Score"].max()
lowest_score = df["Score"].min()

st.markdown("### Class Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Submissions", f"{num_students}")
with col2:
    st.metric("Average Score", f"{avg_score:.1f}%")
with col3:
    st.metric("Highest Score", f"{highest_score:.1f}%")
with col4:
    st.metric("Lowest Score", f"{lowest_score:.1f}%")

st.markdown("---")

st.markdown("### 🤖 Generalized AI Feedback")

if 'class_summary' not in st.session_state:
    from backend.model_call import generate_class_summary
    hf_token = st.session_state.get('hf_token')
    with st.spinner("Generating AI Analysis of Class Performance..."):
        st.session_state['class_summary'] = generate_class_summary(df, hf_token)

st.info(f"**Overall Class Performance Summary:**\n\n{st.session_state['class_summary']}")

st.markdown("---")

st.markdown("### 📝 Individual Student Submissions")

# Format dataframe for display
display_df = df.copy()
display_df['Score (Raw)'] = display_df.apply(lambda row: f"{row['RawScore']:.1f} / {row['MaxScore']:.1f}", axis=1)
display_df['Score'] = display_df['Score'].apply(lambda x: float(f"{x:.1f}"))

st.dataframe(
    display_df[['Student Name', 'Score', 'Score (Raw)', 'Feedback']],
    column_config={
        "Student Name": st.column_config.TextColumn("Student Name", width="medium"),
        "Score": st.column_config.ProgressColumn(
            "Score (%)", 
            format="%.1f%%", 
            min_value=0, 
            max_value=100
        ),
        "Score (Raw)": st.column_config.TextColumn("Score (Raw)", width="small"),
        "Feedback": st.column_config.TextColumn("Detailed AI Feedback", width="large"),
    },
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# Download Action
st.markdown("### Export Results")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Complete Report (CSV)",
    data=csv,
    file_name='normascore_evaluation_report.csv',
    mime='text/csv',
    type="primary"
)
