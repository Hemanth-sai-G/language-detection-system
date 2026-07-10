"""
Entry point of the ML pipeline.

Phase 1:
Load dataset
Preprocess dataset
Save cleaned dataset
"""

from src.data_loader import DataLoader

from src.preprocess import Preprocessor


def main():

    loader = DataLoader()

    dataframe = loader.load_dataset()

    print("\nDataset Loaded Successfully\n")

    print(dataframe.head())

    print()

    processor = Preprocessor()

    cleaned_dataframe = processor.preprocess(dataframe)

    print("\nPreprocessing Completed\n")

    print(cleaned_dataframe.head())

    print()

    print("Clean dataset saved successfully.")


if __name__ == "__main__":

    main()