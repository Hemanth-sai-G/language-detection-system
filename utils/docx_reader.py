"""
DOCX file reader utility.
"""

from pathlib import Path

from docx import Document

from utils.logger import logger


class DOCXReader:
    """
    Read text content from DOCX files.
    """

    @staticmethod
    def read(file_path: str | Path) -> str:
        """
        Read and validate text content from a DOCX file.

        Args:
            file_path: Path to the DOCX file.

        Returns:
            str: Extracted text content.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not .docx or content is empty.
        """

        path = Path(file_path)

        logger.info("Reading DOCX file: %s", path)

        if not path.exists():
            raise FileNotFoundError(f"DOCX file not found: {path}")

        if path.suffix.lower() != ".docx":
            raise ValueError(f"Invalid DOCX file path: {path}")

        document = Document(str(path))
        paragraphs = [paragraph.text for paragraph in document.paragraphs]
        content = "\n".join(paragraphs).strip()

        if not content:
            raise ValueError(f"DOCX file is empty: {path}")

        logger.info("DOCX file read successfully.")

        return content
