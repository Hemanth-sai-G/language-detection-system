"""
Batch prediction page.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.app_utils import load_predictor, render_missing_model_message, save_uploaded_file


st.set_page_config(
    page_title="Batch Prediction | Language Detection System",
    page_icon=":bar_chart:",
    layout="wide",
)

st.title("Batch Prediction")
st.caption("Upload a CSV file and run language detection for multiple rows.")

try:
    predictor = load_predictor()
except Exception:
    predictor = None

if predictor is None:
    render_missing_model_message()
else:
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        st.write(f"Selected file: `{uploaded_file.name}`")

        if st.button("Run Batch Prediction", type="primary", use_container_width=True):
            temp_path: Path | None = None

            try:
                temp_path = save_uploaded_file(uploaded_file)
                output = predictor.predict_file(temp_path)
                results = output["results"]

                result_frame = pd.DataFrame(results)

                st.success(f"Batch prediction completed for {len(result_frame)} rows.")
                st.dataframe(result_frame, use_container_width=True)
                st.download_button(
                    label="Download Results as CSV",
                    data=result_frame.to_csv(index=False).encode("utf-8"),
                    file_name="batch_language_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            except Exception as error:
                st.error(f"Batch prediction failed: {error}")
            finally:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink()
