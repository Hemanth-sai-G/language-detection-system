"""
Preprocessing module.
"""

import re

import pandas as pd

from src.config import CLEAN_DATASET

from utils.logger import logger


class Preprocessor:

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text.

        Parameters
        ----------
        text : str

        Returns
        -------
        str
        """

        text = str(text)

        text = text.strip()

        text = re.sub(r"\s+", " ", text)

        return text

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean complete dataframe.
        """

        logger.info("Removing duplicates...")

        df = df.drop_duplicates()

        logger.info("Removing null values...")

        df = df.dropna()

        logger.info("Cleaning text...")

        df["Text"] = df["Text"].apply(self.clean_text)

        CLEAN_DATASET.parent.mkdir(

            parents=True,

            exist_ok=True

        )

        df.to_csv(

            CLEAN_DATASET,

            index=False,

            encoding="utf-8"

        )

        logger.info("Clean dataset saved.")

        return df