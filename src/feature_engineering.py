"""
Feature engineering module.

Creates reusable TF-IDF vectorizers.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from src.config import (
    CHAR_NGRAM_RANGE,
    WORD_NGRAM_RANGE
)


class FeatureEngineering:
    """
    Creates TF-IDF feature extractors.
    """

    @staticmethod
    def get_character_vectorizer():
        """
        Character-level TF-IDF.
        """

        return TfidfVectorizer(
            analyzer="char",
            ngram_range=CHAR_NGRAM_RANGE,
            lowercase=True
        )

    @staticmethod
    def get_word_vectorizer():
        """
        Word-level TF-IDF.
        """

        return TfidfVectorizer(
            analyzer="word",
            ngram_range=WORD_NGRAM_RANGE,
            lowercase=True
        )

    @classmethod
    def get_combined_features(cls):
        """
        Combine character and word TF-IDF.
        """

        return FeatureUnion([
            ("char", cls.get_character_vectorizer()),
            ("word", cls.get_word_vectorizer())
        ])