"""
Project configuration file.

Contains all configurable paths and constants used throughout
the Language Detection System.
"""

from pathlib import Path

# ======================================================
# Project Paths
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models"

ASSETS_DIR = BASE_DIR / "assets"

# ======================================================
# Dataset Paths
# ======================================================

RAW_DATASET = RAW_DATA_DIR / "Language Detection.csv"

CLEAN_DATASET = PROCESSED_DATA_DIR / "cleaned_dataset.csv"

# ======================================================
# Training Configuration
# ======================================================

TEST_SIZE = 0.20

RANDOM_STATE = 42

# ======================================================
# Feature Engineering
# ======================================================

CHAR_NGRAM_RANGE = (2, 5)

WORD_NGRAM_RANGE = (1, 2)