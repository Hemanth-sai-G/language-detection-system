"""
TXT file reader utility.
"""

from pathlib import Path

from utils.logger import logger


class TXTReader:
    """
    Read plain text content from TXT files.
    """

    @staticmethod
    def read(file_path: str | Path) -> str:
        """
        Read and validate text content from a TXT file.

        Args:
            file_path: Path to the TXT file.

        Returns:
            str: Extracted text content.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not .txt or content is empty.
        """

        path = Path(file_path)

        logger.info("Reading TXT file: %s", path)

        if not path.exists():
            raise FileNotFoundError(f"TXT file not found: {path}")

        if path.suffix.lower() != ".txt":
            raise ValueError(f"Invalid TXT file path: {path}")

        content = path.read_text(encoding="utf-8").strip()

        if not content:
            raise ValueError(f"TXT file is empty: {path}")

        logger.info("TXT file read successfully.")

        return content
