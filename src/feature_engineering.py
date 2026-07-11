"""
Feature engineering module for language detection.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from src.config import CHAR_NGRAM_RANGE, WORD_NGRAM_RANGE


class FeatureEngineering:
    """
    Build reusable TF-IDF feature extractors for text data.
    """

    @staticmethod
    def get_character_vectorizer() -> TfidfVectorizer:
        """
        Create the character-level TF-IDF vectorizer.

        Returns:
            TfidfVectorizer: Configured character n-gram vectorizer.
        """

        return TfidfVectorizer(
            analyzer="char",
            ngram_range=CHAR_NGRAM_RANGE,
            lowercase=True,
        )

    @staticmethod
    def get_word_vectorizer() -> TfidfVectorizer:
        """
        Create the word-level TF-IDF vectorizer.

        Returns:
            TfidfVectorizer: Configured word n-gram vectorizer.
        """

        return TfidfVectorizer(
            analyzer="word",
            ngram_range=WORD_NGRAM_RANGE,
            lowercase=True,
        )

    @classmethod
    def get_combined_features(cls) -> FeatureUnion:
        """
        Combine character and word TF-IDF extractors.

        Returns:
            FeatureUnion: Combined feature extraction block.
        """

        return FeatureUnion(
            transformer_list=[
                ("char_tfidf", cls.get_character_vectorizer()),
                ("word_tfidf", cls.get_word_vectorizer()),
            ]
        )
