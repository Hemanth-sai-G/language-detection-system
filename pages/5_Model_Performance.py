"""
Model performance dashboard page.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.app_utils import load_metrics


st.set_page_config(
    page_title="Model Performance | Language Detection System",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

st.title("Model Performance")
st.caption("Review saved evaluation metrics and validation results.")

metrics = load_metrics()

if not metrics:
    st.warning("No `metrics.json` file is available yet. Run model training first.")
else:
    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0.0):.4f}")
        st.metric("Precision Macro", f"{metrics.get('precision_macro', 0.0):.4f}")

    with summary_col2:
        st.metric("Recall Macro", f"{metrics.get('recall_macro', 0.0):.4f}")
        st.metric("F1 Macro", f"{metrics.get('f1_macro', 0.0):.4f}")

    with summary_col3:
        st.metric(
            "CV Mean",
            f"{metrics.get('cross_validation_mean', 0.0):.4f}",
        )
        st.metric(
            "CV Std",
            f"{metrics.get('cross_validation_std', 0.0):.4f}",
        )

    st.markdown("---")

    st.subheader("Baseline Model Comparison")
    baseline_results = metrics.get("baseline_results", {})

    if baseline_results:
        baseline_frame = pd.DataFrame(
            [
                {"Model": name, "Accuracy": info.get("accuracy", 0.0)}
                for name, info in baseline_results.items()
            ]
        )
        st.dataframe(baseline_frame, use_container_width=True)

    st.subheader("Classification Report")
    report = metrics.get("classification_report", {})

    if report:
        report_frame = pd.DataFrame(report).transpose()
        st.dataframe(report_frame, use_container_width=True)

    st.subheader("Confusion Matrix")
    confusion_matrix = metrics.get("confusion_matrix", [])
    confusion_labels = metrics.get("confusion_matrix_labels", [])

    if confusion_matrix and confusion_labels:
        confusion_frame = pd.DataFrame(
            confusion_matrix,
            index=confusion_labels,
            columns=confusion_labels,
        )
        st.dataframe(confusion_frame, use_container_width=True)

    st.subheader("Selected Model")
    st.write(f"Final model: `{metrics.get('best_model_name', 'Unavailable')}`")
    st.write(
        f"Best baseline model observed: `{metrics.get('best_baseline_model_name', 'Unavailable')}`"
    )
