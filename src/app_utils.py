"""
Shared Streamlit application helpers.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import MODEL_DIR
from src.predictor import Predictor
from src.serializer import ModelSerializer


MODEL_FILE = MODEL_DIR / ModelSerializer.PIPELINE_FILENAME
METRICS_FILE = MODEL_DIR / ModelSerializer.METRICS_FILENAME


@st.cache_resource(show_spinner=False)
def load_predictor() -> Predictor:
    """
    Load and cache the predictor instance.

    Returns:
        Predictor: Initialized predictor.
    """

    return Predictor()


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, Any] | None:
    """
    Load persisted evaluation metrics if available.

    Returns:
        dict[str, Any] | None: Metrics dictionary or None when unavailable.
    """

    if not METRICS_FILE.exists():
        return None

    return ModelSerializer.load_metrics(METRICS_FILE)


def is_model_available() -> bool:
    """
    Check whether the trained model exists.

    Returns:
        bool: True when the model file exists.
    """

    return MODEL_FILE.exists()


def format_confidence(confidence: float | None) -> str:
    """
    Format the model confidence-like score for UI display.

    Args:
        confidence: Raw confidence-like score.

    Returns:
        str: Human-readable confidence string.
    """

    if confidence is None:
        return "Not available"

    return f"{confidence:.4f}"


def save_uploaded_file(uploaded_file: Any) -> Path:
    """
    Persist an uploaded Streamlit file to a temporary path.

    Args:
        uploaded_file: Streamlit uploaded file object.

    Returns:
        Path: Temporary file path.
    """

    suffix = Path(uploaded_file.name).suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def render_missing_model_message() -> None:
    """
    Render a standard message when model artifacts are unavailable.
    """

    st.error("Trained model not found.")
    st.info("Run `python train_model.py` to generate `language_detector.pkl` and `metrics.json`.")
