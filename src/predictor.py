"""
Prediction module for the language detection system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from docx import Document
from PyPDF2 import PdfReader
from sklearn.pipeline import Pipeline

from src.preprocess import Preprocessor
from src.serializer import ModelSerializer
from utils.logger import logger


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

    @staticmethod
    def _read_txt_file(file_path: Path) -> str:
        """
        Read plain text content from a TXT file.

        Args:
            file_path: Path to the TXT file.

        Returns:
            str: Extracted text.
        """

        return file_path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _read_pdf_file(file_path: Path) -> str:
        """
        Read text content from a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            str: Extracted text.
        """

        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]

        return "\n".join(pages).strip()

    @staticmethod
    def _read_docx_file(file_path: Path) -> str:
        """
        Read text content from a DOCX file.

        Args:
            file_path: Path to the DOCX file.

        Returns:
            str: Extracted text.
        """

        document = Document(str(file_path))

        return "\n".join(paragraph.text for paragraph in document.paragraphs).strip()

    @staticmethod
    def _read_csv_file(file_path: Path) -> list[str]:
        """
        Read text entries from a CSV file.

        Args:
            file_path: Path to the CSV file.

        Returns:
            list[str]: Text entries for batch prediction.
        """

        dataframe = pd.read_csv(file_path)

        if "Text" in dataframe.columns:
            series = dataframe["Text"]
        else:
            series = dataframe.iloc[:, 0]

        return series.dropna().astype(str).tolist()

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

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        suffix = path.suffix.lower()
        logger.info("Running prediction for file: %s", path)

        if suffix == ".txt":
            return {
                "file_path": str(path),
                "file_type": "txt",
                "result": self.predict_text(self._read_txt_file(path)),
            }

        if suffix == ".pdf":
            return {
                "file_path": str(path),
                "file_type": "pdf",
                "result": self.predict_text(self._read_pdf_file(path)),
            }

        if suffix == ".docx":
            return {
                "file_path": str(path),
                "file_type": "docx",
                "result": self.predict_text(self._read_docx_file(path)),
            }

        if suffix == ".csv":
            texts = self._read_csv_file(path)
            return {
                "file_path": str(path),
                "file_type": "csv",
                "results": self.predict_batch(texts),
            }

        raise ValueError(
            "Unsupported file format. Supported formats are: .txt, .pdf, .docx, .csv"
        )
