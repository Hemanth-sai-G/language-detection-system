"""
Loads and validates the dataset.
"""

from pathlib import Path

import pandas as pd

from src.config import RAW_DATASET

from utils.logger import logger


class DataLoader:
    """
    Handles loading of the dataset.
    """

    REQUIRED_COLUMNS = ["Text", "Language"]

    def __init__(self):

        self.dataset_path = Path(RAW_DATASET)

    def load_dataset(self) -> pd.DataFrame:
        """
        Load dataset.

        Returns
        -------
        pd.DataFrame
        """

        logger.info("Loading dataset...")

        if not self.dataset_path.exists():

            raise FileNotFoundError(

                f"Dataset not found:\n{self.dataset_path}"

            )

        df = pd.read_csv(self.dataset_path)

        logger.info("Dataset loaded successfully.")

        self.validate_dataset(df)

        return df

    def validate_dataset(self, df: pd.DataFrame) -> None:
        """
        Validate dataset schema.
        """

        logger.info("Validating dataset...")

        missing = [

            column

            for column in self.REQUIRED_COLUMNS

            if column not in df.columns

        ]

        if missing:

            raise ValueError(

                f"Missing columns: {missing}"

            )

        logger.info("Dataset validation completed.")