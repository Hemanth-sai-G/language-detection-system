"""
Main Streamlit entry point for the Language Detection System.
"""

from __future__ import annotations

import streamlit as st

from src.app_utils import is_model_available, load_predictor


PROJECT_TITLE = "Language Detection System"
PROJECT_SUBTITLE = "Machine Learning + Streamlit application for multilingual text classification"


def render_sidebar() -> None:
    """
    Render the application sidebar.
    """

    st.sidebar.title("Navigation")
    st.sidebar.caption("Use the pages below to explore the application workflow.")

    st.sidebar.markdown(
        """
        ### Available Pages
        - `Home`
        - `Text Detection`
        - `File Detection`
        - `Batch Prediction`
        - `Model Performance`
        - `About`
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        ### Workflow
        1. Train the model with `train_model.py`
        2. Ensure `language_detector.pkl` exists
        3. Use the prediction pages for inference
        """
    )


def render_status_section() -> None:
    """
    Render model availability and readiness status.
    """

    st.subheader("System Status")

    col1, col2 = st.columns(2)

    with col1:
        if is_model_available():
            st.success("Trained pipeline detected")
            st.caption("Model artifact is available in the `models/` directory.")
        else:
            st.error("Trained pipeline not found")
            st.caption("Run `python train_model.py` before using the prediction pages.")

    with col2:
        try:
            load_predictor()
            st.success("Prediction service ready")
            st.caption("The model pipeline loaded successfully.")
        except Exception as error:
            st.warning("Prediction service unavailable")
            st.caption(f"Reason: `{error}`")


def render_overview_section() -> None:
    """
    Render the project overview section.
    """

    st.subheader("Project Overview")

    st.write(
        """
        This application predicts the language of user-provided text using a
        production-style machine learning pipeline built with scikit-learn.
        The backend combines character-level and word-level TF-IDF features
        and performs classification using a trained Linear SVC pipeline.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Languages", "17")

    with col2:
        st.metric("Model Family", "LinearSVC")

    with col3:
        st.metric("Feature Stack", "Char + Word TF-IDF")


def render_capabilities_section() -> None:
    """
    Render application capability cards.
    """

    st.subheader("Application Capabilities")

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            """
            **Text Detection**

            Predict the language of a manually entered text snippet instantly.
            """
        )

        st.info(
            """
            **File Detection**

            Upload `.txt`, `.pdf`, or `.docx` files and classify their content.
            """
        )

    with col2:
        st.info(
            """
            **Batch Prediction**

            Run language detection on multiple rows from a CSV file.
            """
        )

        st.info(
            """
            **Performance Dashboard**

            Review saved metrics, cross-validation results, and model performance.
            """
        )


def render_quick_start_section() -> None:
    """
    Render onboarding instructions for using the app.
    """

    st.subheader("Quick Start")

    st.markdown(
        """
        1. Train the model pipeline by running `python train_model.py`
        2. Confirm the file `models/language_detector.pkl` exists
        3. Open the detection pages from the sidebar
        4. Test text, documents, or CSV inputs
        """
    )


def main() -> None:
    """
    Run the Streamlit home application.
    """

    st.set_page_config(
        page_title=PROJECT_TITLE,
        page_icon=":globe_with_meridians:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_sidebar()

    st.title(PROJECT_TITLE)
    st.caption(PROJECT_SUBTITLE)

    st.markdown("---")

    render_status_section()
    st.markdown("---")
    render_overview_section()
    st.markdown("---")
    render_capabilities_section()
    st.markdown("---")
    render_quick_start_section()


if __name__ == "__main__":
    main()
