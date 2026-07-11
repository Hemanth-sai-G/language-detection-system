"""
Home page for the Language Detection System.
"""

from __future__ import annotations

import streamlit as st

from src.app_utils import is_model_available, load_metrics


st.set_page_config(
    page_title="Home | Language Detection System",
    page_icon=":house:",
    layout="wide",
)

st.title("Home")
st.caption("Welcome to the Language Detection System dashboard.")

status_col, metrics_col = st.columns(2)

with status_col:
    st.subheader("System Readiness")

    if is_model_available():
        st.success("The trained prediction pipeline is available.")
    else:
        st.warning("The trained prediction pipeline is not available yet.")

    st.markdown(
        """
        Use the sidebar to navigate across:
        - text-based prediction
        - document-based prediction
        - batch CSV prediction
        - model performance review
        """
    )

with metrics_col:
    st.subheader("Saved Evaluation Snapshot")

    metrics = load_metrics()

    if metrics:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0.0):.4f}")
        st.metric("F1 Macro", f"{metrics.get('f1_macro', 0.0):.4f}")
        st.metric(
            "Cross Validation Mean",
            f"{metrics.get('cross_validation_mean', 0.0):.4f}",
        )
    else:
        st.info("No saved metrics found yet. Train the model to populate this section.")

st.markdown("---")

st.subheader("Project Workflow")
st.markdown(
    """
    1. Load and validate the language dataset
    2. Preprocess the text without removing language-specific signals
    3. Build combined character and word TF-IDF features
    4. Train and tune the Linear SVC pipeline
    5. Save model artifacts
    6. Use this Streamlit app for inference and analysis
    """
)

st.subheader("Supported Inputs")
support_col1, support_col2 = st.columns(2)

with support_col1:
    st.info("Manual text snippets for instant language prediction.")
    st.info("TXT, PDF, and DOCX files for full-document classification.")

with support_col2:
    st.info("CSV files for batch prediction workflows.")
    st.info("Saved metrics for performance dashboard review.")
