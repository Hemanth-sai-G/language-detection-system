"""
Creates machine learning pipelines.
"""

from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from src.feature_engineering import FeatureEngineering


class ModelSelection:

    @staticmethod
    def get_models():
        """
        Returns all candidate models.
        """

        return {

            "Naive Bayes": Pipeline([

                ("features",
                 FeatureEngineering.get_combined_features()),

                ("classifier",
                 MultinomialNB())

            ]),

            "Logistic Regression": Pipeline([

                ("features",
                 FeatureEngineering.get_combined_features()),

                ("classifier",
                 LogisticRegression(max_iter=1000))

            ]),

            "Linear SVC": Pipeline([

                ("features",
                 FeatureEngineering.get_combined_features()),

                ("classifier",
                 LinearSVC())

            ])

        }