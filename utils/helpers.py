"""
Shared helper utilities for file-based prediction workflows.
"""

from pathlib import Path

import pandas as pd

from utils.logger import logger


def validate_file_path(file_path: str | Path) -> Path:
    """
    Validate that a file exists and return a normalized path.

    Args:
        file_path: Input file path.

    Returns:
        Path: Validated path object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    return path


def read_csv_texts(file_path: str | Path) -> list[str]:
    """
    Read text entries from a CSV file for batch prediction.

    The function prefers a column named ``Text`` and falls back to
    the first column when that column is unavailable.

    Args:
        file_path: Path to the CSV file.

    Returns:
        list[str]: Non-empty text entries.

    Raises:
        ValueError: If the file extension is invalid or no usable rows exist.
    """

    path = validate_file_path(file_path)

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Invalid CSV file path: {path}")

    logger.info("Reading CSV file: %s", path)

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"CSV file is empty: {path}")

    series = dataframe["Text"] if "Text" in dataframe.columns else dataframe.iloc[:, 0]
    texts = series.dropna().astype(str).str.strip()
    values = [text for text in texts.tolist() if text]

    if not values:
        raise ValueError(f"CSV file contains no valid text rows: {path}")

    logger.info("CSV file read successfully with %s text rows.", len(values))

    return values
