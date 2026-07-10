"""
Training module.
"""

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from src.config import TEST_SIZE
from src.config import RANDOM_STATE

from src.model_selection import ModelSelection


class Trainer:

    def __init__(self, dataframe):

        self.df = dataframe

    def train(self):

        X = self.df["Text"]

        y = self.df["Language"]

        X_train, X_test, y_train, y_test = train_test_split(

            X,

            y,

            test_size=TEST_SIZE,

            random_state=RANDOM_STATE,

            stratify=y

        )

        models = ModelSelection.get_models()

        results = {}

        for name, pipeline in models.items():

            print(f"\nTraining {name}...")

            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)

            accuracy = accuracy_score(

                y_test,

                predictions

            )

            results[name] = accuracy

            print(

                f"{name} Accuracy : {accuracy:.4f}"

            )

        return results