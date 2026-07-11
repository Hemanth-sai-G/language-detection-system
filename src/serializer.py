"""
Serialization helpers for trained artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from src.config import MODEL_DIR
from utils.logger import logger


class ModelSerializer:
    """
    Save and load trained model artifacts.
    """

    PIPELINE_FILENAME = "language_detector.pkl"
    METRICS_FILENAME = "metrics.json"

    @classmethod
    def save_pipeline(
        cls,
        pipeline: Pipeline,
        output_path: str | Path | None = None,
    ) -> Path:
        """
        Save the trained pipeline to disk.

        Args:
            pipeline: Trained sklearn pipeline.
            output_path: Optional custom output path.

        Returns:
            Path: Path to the saved pipeline file.
        """

        save_path = Path(output_path) if output_path else MODEL_DIR / cls.PIPELINE_FILENAME
        save_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving trained pipeline to %s", save_path)
        joblib.dump(pipeline, save_path)

        return save_path

    @classmethod
    def load_pipeline(cls, input_path: str | Path | None = None) -> Pipeline:
        """
        Load a trained pipeline from disk.

        Args:
            input_path: Optional custom pipeline path.

        Returns:
            Pipeline: Loaded sklearn pipeline.

        Raises:
            FileNotFoundError: If the saved pipeline does not exist.
        """

        load_path = Path(input_path) if input_path else MODEL_DIR / cls.PIPELINE_FILENAME

        if not load_path.exists():
            raise FileNotFoundError(f"Saved pipeline not found: {load_path}")

        logger.info("Loading trained pipeline from %s", load_path)

        pipeline = joblib.load(load_path)

        return pipeline

    @classmethod
    def save_metrics(
        cls,
        metrics: dict[str, Any],
        output_path: str | Path | None = None,
    ) -> Path:
        """
        Save evaluation metrics to disk.

        Args:
            metrics: Metrics dictionary.
            output_path: Optional custom output path.

        Returns:
            Path: Path to the saved metrics file.
        """

        save_path = Path(output_path) if output_path else MODEL_DIR / cls.METRICS_FILENAME
        save_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving evaluation metrics to %s", save_path)

        with save_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4, ensure_ascii=False)

        return save_path
