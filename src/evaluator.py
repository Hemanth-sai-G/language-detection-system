"""
Evaluation module for the language detection system.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report as sklearn_classification_report,
    confusion_matrix as sklearn_confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.serializer import ModelSerializer
from utils.logger import logger


class Evaluator:
    """
    Evaluate a trained language detection pipeline.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        X_test: pd.Series,
        y_test: pd.Series,
        training_summary: dict[str, Any] | None = None,
    ):
        """
        Initialize the evaluator.

        Args:
            pipeline: Trained sklearn pipeline.
            X_test: Test features.
            y_test: Test labels.
            training_summary: Optional training metadata to include in metrics.
        """

        self.pipeline = pipeline
        self.X_test = X_test
        self.y_test = y_test
        self.training_summary = training_summary or {}

        self.predictions: list[str] | None = None
        self.metrics: dict[str, Any] = {}

    def classification_report(self) -> dict[str, Any]:
        """
        Generate the classification report as a dictionary.

        Returns:
            dict[str, Any]: Classification report.
        """

        if self.predictions is None:
            raise RuntimeError("Predictions are not available. Call evaluate() first.")

        return sklearn_classification_report(
            self.y_test,
            self.predictions,
            output_dict=True,
            zero_division=0,
        )

    def confusion_matrix(self) -> list[list[int]]:
        """
        Generate the confusion matrix.

        Returns:
            list[list[int]]: Confusion matrix as a serializable nested list.
        """

        if self.predictions is None:
            raise RuntimeError("Predictions are not available. Call evaluate() first.")

        labels = sorted(self.y_test.unique().tolist())

        return sklearn_confusion_matrix(
            self.y_test,
            self.predictions,
            labels=labels,
        ).tolist()

    def evaluate(self) -> dict[str, Any]:
        """
        Compute evaluation metrics for the test set.

        Returns:
            dict[str, Any]: Evaluation metrics and reports.
        """

        logger.info("Evaluating trained pipeline on the test split.")

        self.predictions = self.pipeline.predict(self.X_test).tolist()
        labels = sorted(self.y_test.unique().tolist())

        self.metrics = {
            "best_model_name": self.training_summary.get("best_model_name"),
            "selected_model_baseline_accuracy": self.training_summary.get("best_accuracy"),
            "best_baseline_model_name": self.training_summary.get(
                "best_baseline_model_name"
            ),
            "best_baseline_accuracy": self.training_summary.get(
                "best_baseline_accuracy"
            ),
            "baseline_results": self.training_summary.get("baseline_results", {}),
            "accuracy": float(accuracy_score(self.y_test, self.predictions)),
            "precision_macro": float(
                precision_score(
                    self.y_test,
                    self.predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            "precision_weighted": float(
                precision_score(
                    self.y_test,
                    self.predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "recall_macro": float(
                recall_score(
                    self.y_test,
                    self.predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            "recall_weighted": float(
                recall_score(
                    self.y_test,
                    self.predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "f1_macro": float(
                f1_score(
                    self.y_test,
                    self.predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            "f1_weighted": float(
                f1_score(
                    self.y_test,
                    self.predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "classification_report": self.classification_report(),
            "confusion_matrix_labels": labels,
            "confusion_matrix": self.confusion_matrix(),
            "cross_validation_scores": self.training_summary.get(
                "cross_validation_scores",
                [],
            ),
            "cross_validation_mean": self.training_summary.get("cross_validation_mean"),
            "cross_validation_std": self.training_summary.get("cross_validation_std"),
        }

        logger.info(
            "Evaluation completed. Accuracy=%.4f, F1 Macro=%.4f",
            self.metrics["accuracy"],
            self.metrics["f1_macro"],
        )

        return self.metrics

    def save_metrics(self) -> str:
        """
        Save evaluation metrics to disk.

        Returns:
            str: Saved metrics file path.
        """

        if not self.metrics:
            raise RuntimeError("Metrics are not available. Call evaluate() first.")

        saved_path = ModelSerializer.save_metrics(self.metrics)
        logger.info("Evaluation metrics saved to %s", saved_path)

        return str(saved_path)
