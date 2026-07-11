"""
Text detection page.
"""

from __future__ import annotations

import streamlit as st

from src.app_utils import format_confidence, load_predictor, render_missing_model_message


st.set_page_config(
    page_title="Text Detection | Language Detection System",
    page_icon=":memo:",
    layout="wide",
)

st.title("Text Detection")
st.caption("Predict the language of manually entered text.")

try:
    predictor = load_predictor()
except Exception:
    predictor = None

if predictor is None:
    render_missing_model_message()
else:
    sample_text = st.text_area(
        "Enter text",
        height=220,
        placeholder="Type or paste text here...",
    )

    predict_clicked = st.button("Detect Language", type="primary", use_container_width=True)

    if predict_clicked:
        try:
            result = predictor.predict_text(sample_text)

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.success(f"Predicted Language: {result['predicted_language']}")

            with result_col2:
                st.info(f"Confidence Score: {format_confidence(result['confidence'])}")

            st.subheader("Normalized Text Used for Prediction")
            st.code(result["text"], language="text")
        except Exception as error:
            st.error(f"Prediction failed: {error}")
