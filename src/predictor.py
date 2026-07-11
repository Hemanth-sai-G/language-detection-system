"""
Prediction module for the language detection system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sklearn.pipeline import Pipeline

from src.preprocess import Preprocessor
from src.serializer import ModelSerializer
from utils.docx_reader import DOCXReader
from utils.helpers import read_csv_texts, validate_file_path
from utils.logger import logger
from utils.pdf_reader import PDFReader
from utils.txt_reader import TXTReader


class Predictor:
    """
    Load a trained pipeline and generate predictions.
    """

    def __init__(self, pipeline: Pipeline | None = None):
        """
        Initialize the predictor.

        Args:
            pipeline: Optional preloaded trained pipeline.
        """

        self.pipeline = pipeline or ModelSerializer.load_pipeline()
        self.preprocessor = Preprocessor()

    def _prepare_text(self, text: str) -> str:
        """
        Normalize text before inference.

        Args:
            text: Raw input text.

        Returns:
            str: Cleaned text.
        """

        cleaned_text = self.preprocessor.clean_text(text)

        if not cleaned_text:
            raise ValueError("Input text is empty after preprocessing.")

        return cleaned_text

    def _extract_confidence(self, cleaned_text: str, predicted_label: str) -> float | None:
        """
        Extract a confidence-like score when supported by the classifier.

        Args:
            cleaned_text: Preprocessed text.
            predicted_label: Predicted label.

        Returns:
            float | None: Maximum decision score for the predicted class, if available.
        """

        classifier = self.pipeline.named_steps.get("classifier")

        if classifier is None or not hasattr(classifier, "decision_function"):
            return None

        decision_scores = classifier.decision_function(
            self.pipeline.named_steps["features"].transform([cleaned_text])
        )

        if getattr(decision_scores, "ndim", 1) == 1:
            return float(abs(decision_scores[0]))

        class_labels = classifier.classes_.tolist()
        predicted_index = class_labels.index(predicted_label)

        return float(decision_scores[0][predicted_index])

    def predict_text(self, text: str) -> dict[str, Any]:
        """
        Predict the language for a single text input.

        Args:
            text: Raw input text.

        Returns:
            dict[str, Any]: Prediction result.
        """

        cleaned_text = self._prepare_text(text)
        predicted_label = str(self.pipeline.predict([cleaned_text])[0])
        confidence = self._extract_confidence(cleaned_text, predicted_label)

        logger.info("Prediction generated for single text input.")

        return {
            "text": cleaned_text,
            "predicted_language": predicted_label,
            "confidence": confidence,
        }

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """
        Predict languages for multiple text inputs.

        Args:
            texts: List of raw texts.

        Returns:
            list[dict[str, Any]]: Batch prediction results.
        """

        if not texts:
            raise ValueError("Batch prediction input is empty.")

        return [self.predict_text(text) for text in texts]

    def predict_file(self, file_path: str | Path) -> dict[str, Any]:
        """
        Predict language from a supported file type.

        TXT, PDF, and DOCX return a single prediction.
        CSV returns batch predictions.

        Args:
            file_path: Path to the input file.

        Returns:
            dict[str, Any]: File prediction output.
        """

        path = validate_file_path(file_path)

        suffix = path.suffix.lower()
        logger.info("Running prediction for file: %s", path)

        if suffix == ".txt":
            return {
                "file_path": str(path),
                "file_type": "txt",
                "result": self.predict_text(TXTReader.read(path)),
            }

        if suffix == ".pdf":
            return {
                "file_path": str(path),
                "file_type": "pdf",
                "result": self.predict_text(PDFReader.read(path)),
            }

        if suffix == ".docx":
            return {
                "file_path": str(path),
                "file_type": "docx",
                "result": self.predict_text(DOCXReader.read(path)),
            }

        if suffix == ".csv":
            texts = read_csv_texts(path)
            return {
                "file_path": str(path),
                "file_type": "csv",
                "results": self.predict_batch(texts),
            }

        raise ValueError(
            "Unsupported file format. Supported formats are: .txt, .pdf, .docx, .csv"
        )
