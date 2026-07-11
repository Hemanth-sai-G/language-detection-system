"""
File detection page.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.app_utils import (
    format_confidence,
    load_predictor,
    render_missing_model_message,
    save_uploaded_file,
)


st.set_page_config(
    page_title="File Detection | Language Detection System",
    page_icon=":page_facing_up:",
    layout="wide",
)

st.title("File Detection")
st.caption("Upload a TXT, PDF, or DOCX file and detect its language.")

try:
    predictor = load_predictor()
except Exception:
    predictor = None

if predictor is None:
    render_missing_model_message()
else:
    uploaded_file = st.file_uploader(
        "Upload a file",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        st.write(f"Selected file: `{uploaded_file.name}`")

        if st.button("Detect File Language", type="primary", use_container_width=True):
            temp_path: Path | None = None

            try:
                temp_path = save_uploaded_file(uploaded_file)
                output = predictor.predict_file(temp_path)
                result = output["result"]

                col1, col2 = st.columns(2)

                with col1:
                    st.success(f"Predicted Language: {result['predicted_language']}")

                with col2:
                    st.info(f"Confidence Score: {format_confidence(result['confidence'])}")

                st.subheader("Extracted and Normalized Text")
                st.code(result["text"], language="text")
            except Exception as error:
                st.error(f"File prediction failed: {error}")
            finally:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink()
