import streamlit as st

st.set_page_config(
    page_title="NormaScore",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

def local_css():
    st.markdown("""
    <style>
    .stApp {
        background-color: #f8fafc;
    }
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    h1, h2, h3 {
        color: #0f172a;
        font-family: 'Inter', sans-serif;
    }
    h1 {
        font-weight: 800;
        letter-spacing: -0.025em;
    }
    [data-testid="stMetricValue"] {
        color: #4f46e5;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #64748b;
        font-weight: 500;
    }
    .stButton>button[data-baseweb="button"] {
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stAlert {
        border-radius: 0.5rem;
        border: none;
        box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebarNav"] span {
        font-weight: 500;
        color: #334155;
    }
    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-color: #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# Define Navigation — use default=True for the landing page, no url_path="/"
question_page  = st.Page("views/question_upload.py", title="Upload Question",    icon="📄", default=True)
upload_page    = st.Page("views/upload.py",           title="Upload Submissions", icon="📁", url_path="submissions")
evaluation_page = st.Page("views/evaluation.py",      title="Evaluation View",   icon="📊", url_path="evaluation")

pg = st.navigation([question_page, upload_page, evaluation_page])
pg.run()