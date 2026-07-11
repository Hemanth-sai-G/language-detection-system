"""
PDF file reader utility.
"""

from pathlib import Path

from PyPDF2 import PdfReader

from utils.logger import logger


class PDFReader:
    """
    Read text content from PDF files.
    """

    @staticmethod
    def read(file_path: str | Path) -> str:
        """
        Read and validate text content from a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            str: Extracted text content.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not .pdf or content is empty.
        """

        path = Path(file_path)

        logger.info("Reading PDF file: %s", path)

        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Invalid PDF file path: {path}")

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        content = "\n".join(pages).strip()

        if not content:
            raise ValueError(f"PDF file contains no extractable text: {path}")

        logger.info("PDF file read successfully.")

        return content
