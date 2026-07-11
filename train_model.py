"""
Entry point for training the language detection pipeline.
"""

from src.data_loader import DataLoader
from src.evaluator import Evaluator
from src.preprocess import Preprocessor
from src.serializer import ModelSerializer
from src.trainer import Trainer
from utils.logger import logger


def main() -> None:
    """
    Execute the complete training workflow.
    """

    logger.info("Language detection training pipeline started.")

    loader = DataLoader()
    dataframe = loader.load_dataset()

    preprocessor = Preprocessor()
    clean_dataframe = preprocessor.preprocess(dataframe)

    trainer = Trainer(clean_dataframe)
    training_summary = trainer.train()

    evaluator = Evaluator(
        pipeline=training_summary["best_pipeline"],
        X_test=training_summary["X_test"],
        y_test=training_summary["y_test"],
        training_summary=training_summary,
    )
    metrics = evaluator.evaluate()

    pipeline_path = ModelSerializer.save_pipeline(training_summary["best_pipeline"])
    metrics_path = ModelSerializer.save_metrics(metrics)

    logger.info("Training workflow completed successfully.")
    logger.info("Saved pipeline: %s", pipeline_path)
    logger.info("Saved metrics: %s", metrics_path)


if __name__ == "__main__":
    main()
