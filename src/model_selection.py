"""
Model selection module.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import RANDOM_STATE
from src.feature_engineering import FeatureEngineering


class ModelSelection:
    """
    Build candidate model pipelines for comparison.
    """

    @staticmethod
    def get_models() -> dict[str, Pipeline]:
        """
        Create all baseline model pipelines.

        Returns:
            dict[str, Pipeline]: Mapping of model names to pipelines.
        """

        return {
            "Naive Bayes": Pipeline(
                steps=[
                    ("features", FeatureEngineering.get_combined_features()),
                    ("classifier", MultinomialNB()),
                ]
            ),
            "Logistic Regression": Pipeline(
                steps=[
                    ("features", FeatureEngineering.get_combined_features()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=1000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "Linear SVC": Pipeline(
                steps=[
                    ("features", FeatureEngineering.get_combined_features()),
                    ("classifier", LinearSVC()),
                ]
            ),
        }
