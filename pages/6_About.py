"""
About page for the Language Detection System.
"""

from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="About | Language Detection System",
    page_icon=":information_source:",
    layout="wide",
)

st.title("About")
st.caption("Project summary, architecture, and technology stack.")

st.subheader("Project Objective")
st.write(
    """
    The Language Detection System is a production-style machine learning and
    Streamlit project designed to identify the language of input text with
    high accuracy across 17 supported languages.
    """
)

st.subheader("Architecture")
st.markdown(
    """
    - raw text input
    - combined character and word TF-IDF feature extraction
    - scikit-learn pipeline
    - Linear SVC classification
    - modular training, evaluation, serialization, and prediction layers
    """
)

st.subheader("Technology Stack")
st.markdown(
    """
    - Python 3.11
    - Streamlit
    - scikit-learn
    - pandas
    - NumPy
    - matplotlib
    - plotly
    - joblib
    - PyPDF2
    - python-docx
    """
)

st.subheader("Supported Workflows")
st.markdown(
    """
    - text detection
    - file detection for TXT, PDF, and DOCX
    - CSV-based batch prediction
    - saved metrics inspection through a performance dashboard
    """
)
