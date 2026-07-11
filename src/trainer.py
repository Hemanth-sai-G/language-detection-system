"""
Training module for the language detection system.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE, TEST_SIZE
from src.model_selection import ModelSelection
from utils.logger import logger


class Trainer:
    """
    Handle model training, selection, and validation.
    """

    REQUIRED_COLUMNS = ["Text", "Language"]

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the trainer with a prepared dataset.

        Args:
            dataframe: Cleaned dataframe containing text and labels.
        """

        self.df = dataframe.copy()
        self._validate_input_dataframe()

        self.X_train: pd.Series | None = None
        self.X_test: pd.Series | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None

        self.models = ModelSelection.get_models()
        self.results: dict[str, dict[str, Any]] = {}

        self.best_baseline_model_name: str | None = None
        self.best_baseline_accuracy: float = 0.0
        self.best_model_name: str | None = None
        self.best_pipeline: Pipeline | None = None
        self.best_accuracy: float = 0.0

        self.cross_validation_scores: list[float] = []
        self.cross_validation_mean: float | None = None
        self.cross_validation_std: float | None = None

    def _validate_input_dataframe(self) -> None:
        """
        Validate the trainer input dataframe.

        Raises:
            ValueError: If required columns are missing or data is empty.
        """

        if self.df.empty:
            raise ValueError("Training dataframe is empty.")

        missing_columns = [
            column for column in self.REQUIRED_COLUMNS if column not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Training dataframe is missing required columns: {missing_columns}"
            )

    def split_dataset(self) -> None:
        """
        Split the dataframe into train and test sets.
        """

        logger.info("Splitting dataset into training and testing sets.")

        features = self.df["Text"]
        target = self.df["Language"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features,
            target,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=target,
        )

        logger.info(
            "Train-test split completed. Training samples=%s, Testing samples=%s",
            len(self.X_train),
            len(self.X_test),
        )

    @staticmethod
    def evaluate_accuracy(actual: pd.Series, predicted: list[str] | pd.Series) -> float:
        """
        Calculate classification accuracy.

        Args:
            actual: Ground truth labels.
            predicted: Predicted labels.

        Returns:
            float: Accuracy score.
        """

        return float(accuracy_score(actual, predicted))

    def compare_models(self) -> dict[str, dict[str, Any]]:
        """
        Train and compare all baseline models.

        Returns:
            dict[str, dict[str, Any]]: Baseline model results.

        Raises:
            RuntimeError: If dataset split has not been performed.
        """

        if any(
            value is None
            for value in [self.X_train, self.X_test, self.y_train, self.y_test]
        ):
            raise RuntimeError("Dataset must be split before model comparison.")

        logger.info("Starting baseline model comparison.")

        for model_name, pipeline in self.models.items():
            logger.info("Training baseline model: %s", model_name)

            pipeline.fit(self.X_train, self.y_train)
            predictions = pipeline.predict(self.X_test)
            accuracy = self.evaluate_accuracy(self.y_test, predictions)

            self.results[model_name] = {
                "pipeline": pipeline,
                "accuracy": accuracy,
            }

            logger.info("%s accuracy: %.4f", model_name, accuracy)

            if accuracy > self.best_baseline_accuracy:
                self.best_baseline_accuracy = accuracy
                self.best_baseline_model_name = model_name

        logger.info(
            "Baseline comparison completed. Best baseline model=%s, Best baseline accuracy=%.4f",
            self.best_baseline_model_name,
            self.best_baseline_accuracy,
        )

        linear_svc_result = self.results.get("Linear SVC")

        if linear_svc_result is None:
            raise RuntimeError("Linear SVC baseline result is unavailable.")

        self.best_model_name = "Linear SVC"
        self.best_pipeline = linear_svc_result["pipeline"]
        self.best_accuracy = float(linear_svc_result["accuracy"])

        logger.info(
            "Selected final model according to project architecture: %s (baseline accuracy=%.4f)",
            self.best_model_name,
            self.best_accuracy,
        )

        return self.results

    def display_results(self) -> None:
        """
        Log model comparison results.
        """

        logger.info("Model comparison summary:")

        for model_name, metrics in self.results.items():
            logger.info("%s -> Accuracy: %.4f", model_name, metrics["accuracy"])

        if self.best_model_name is not None:
            logger.info(
                "Best baseline model: %s (accuracy=%.4f)",
                self.best_baseline_model_name,
                self.best_baseline_accuracy,
            )
            logger.info(
                "Final selected model for tuning and serialization: %s (baseline accuracy=%.4f)",
                self.best_model_name,
                self.best_accuracy,
            )

    def tune_best_model(self) -> Pipeline:
        """
        Tune the best model using GridSearchCV.

        Returns:
            Pipeline: Tuned best pipeline.

        Raises:
            RuntimeError: If no best pipeline has been selected yet.
        """

        if self.best_pipeline is None or self.best_model_name is None:
            raise RuntimeError("Final model must be selected before tuning.")

        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Training data is unavailable for grid search.")

        logger.info("Starting GridSearchCV for Linear SVC.")

        param_grid = {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__loss": ["hinge", "squared_hinge"],
        }

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        grid_search = GridSearchCV(
            estimator=self.best_pipeline,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )

        grid_search.fit(self.X_train, self.y_train)

        self.best_pipeline = grid_search.best_estimator_

        logger.info("Grid search completed. Best params=%s", grid_search.best_params_)
        logger.info("Best cross-validated f1_macro: %.4f", grid_search.best_score_)

        return self.best_pipeline

    def cross_validate_best_model(self) -> tuple[float, float]:
        """
        Cross-validate the best pipeline on the full dataset.

        Returns:
            tuple[float, float]: Mean and standard deviation of CV scores.

        Raises:
            RuntimeError: If best pipeline is unavailable.
        """

        if self.best_pipeline is None:
            raise RuntimeError("Best pipeline must be available before cross validation.")

        logger.info("Running cross validation on the best pipeline.")

        scores = cross_val_score(
            self.best_pipeline,
            self.df["Text"],
            self.df["Language"],
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        )

        self.cross_validation_scores = scores.tolist()
        self.cross_validation_mean = float(scores.mean())
        self.cross_validation_std = float(scores.std())

        logger.info("Cross validation fold scores: %s", self.cross_validation_scores)
        logger.info(
            "Cross validation summary. Mean=%.4f, Std=%.4f",
            self.cross_validation_mean,
            self.cross_validation_std,
        )

        return self.cross_validation_mean, self.cross_validation_std

    def train(self) -> dict[str, Any]:
        """
        Execute the end-to-end training workflow.

        Returns:
            dict[str, Any]: Training artifacts and metadata.
        """

        self.split_dataset()
        self.compare_models()
        self.display_results()
        self.tune_best_model()

        if self.best_pipeline is None or self.X_train is None or self.y_train is None:
            raise RuntimeError(
                "Training could not complete because the best pipeline is unavailable."
            )

        logger.info("Fitting the selected pipeline on the training split.")
        self.best_pipeline.fit(self.X_train, self.y_train)

        self.cross_validate_best_model()

        return {
            "best_pipeline": self.best_pipeline,
            "best_model_name": self.best_model_name,
            "best_accuracy": self.best_accuracy,
            "best_baseline_model_name": self.best_baseline_model_name,
            "best_baseline_accuracy": self.best_baseline_accuracy,
            "baseline_results": {
                name: {"accuracy": float(info["accuracy"])}
                for name, info in self.results.items()
            },
            "cross_validation_scores": self.cross_validation_scores,
            "cross_validation_mean": self.cross_validation_mean,
            "cross_validation_std": self.cross_validation_std,
            "X_test": self.X_test,
            "y_test": self.y_test,
        }
