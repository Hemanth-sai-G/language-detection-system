# Language Detection System

A production-style Machine Learning + Streamlit project that predicts the language of input text using a modular scikit-learn pipeline.

## Project Objective

The goal of this project is to detect the language of user-provided text with high accuracy across 17 languages using classical machine learning, clean modular architecture, and a user-friendly Streamlit interface.

## Dataset

- Source: Kaggle Language Detection Dataset
- Records: approximately 10,337
- Languages: 17
- File: `data/raw/Language Detection.csv`

## Tech Stack

- Python 3.11
- Streamlit
- scikit-learn
- pandas
- NumPy
- matplotlib
- plotly
- joblib
- PyPDF2
- python-docx
- tqdm

## Project Structure

```text
Language-Detection-System/
|
+-- data/
|   +-- raw/
|   |   +-- Language Detection.csv
|   +-- processed/
|       +-- cleaned_dataset.csv
|
+-- notebooks/
|   +-- 01_EDA.ipynb
|   +-- 02_Model_Comparison.ipynb
|   +-- 03_Error_Analysis.ipynb
|
+-- src/
|   +-- __init__.py
|   +-- app_utils.py
|   +-- config.py
|   +-- data_loader.py
|   +-- preprocess.py
|   +-- feature_engineering.py
|   +-- model_selection.py
|   +-- trainer.py
|   +-- evaluator.py
|   +-- serializer.py
|   +-- predictor.py
|
+-- utils/
|   +-- __init__.py
|   +-- logger.py
|   +-- helpers.py
|   +-- txt_reader.py
|   +-- pdf_reader.py
|   +-- docx_reader.py
|
+-- models/
+-- pages/
|   +-- 1_Home.py
|   +-- 2_Text_Detection.py
|   +-- 3_File_Detection.py
|   +-- 4_Batch_Prediction.py
|   +-- 5_Model_Performance.py
|   +-- 6_About.py
|
+-- configs/
|   +-- config.yaml
|
+-- app.py
+-- train_model.py
+-- requirements.txt
+-- README.md
```

## Machine Learning Workflow

```text
Raw Text
   |
   v
FeatureUnion
|             |
v             v
Character     Word
TF-IDF        TF-IDF
      |
      v
Pipeline
      |
      v
LinearSVC
      |
      v
Prediction
```

### Training Flow

1. Load dataset
2. Validate required columns
3. Remove duplicates and missing values
4. Normalize whitespace and trim spaces
5. Create combined character and word TF-IDF features
6. Compare baseline models
7. Tune the final `LinearSVC` model with `GridSearchCV`
8. Evaluate on the test split
9. Save:
   - `models/language_detector.pkl`
   - `models/metrics.json`

## Implemented Model Architecture

- Character TF-IDF
  - `analyzer="char"`
  - `ngram_range=(2, 5)`
- Word TF-IDF
  - `analyzer="word"`
  - `ngram_range=(1, 2)`
- Combined using `FeatureUnion`
- Final classifier: `LinearSVC`

## Baseline Models Compared

- Multinomial Naive Bayes
- Logistic Regression
- Linear SVC

## Evaluation Metrics

The project tracks:

- Accuracy
- Precision Macro
- Precision Weighted
- Recall Macro
- Recall Weighted
- F1 Macro
- F1 Weighted
- Classification Report
- Confusion Matrix
- Cross Validation Mean
- Cross Validation Standard Deviation

## Streamlit Application Features

- Home
- Text Detection
- File Detection
- Batch Prediction
- Model Performance Dashboard
- About

## Supported Input Types

- Manual text input
- TXT files
- PDF files
- DOCX files
- CSV files for batch prediction

## Installation

### 1. Clone the Repository

```powershell
git clone <your-repository-url>
cd Language-Detection-System
```

### 2. Create Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

## Training the Model

Run the training pipeline:

```powershell
python train_model.py
```

This should generate:

- `models/language_detector.pkl`
- `models/metrics.json`

## Running the Streamlit App

```powershell
streamlit run app.py
```

## How Prediction Works

### Text Detection

Enter a text snippet in the Streamlit app and the model predicts the language.

### File Detection

Upload a:

- `.txt`
- `.pdf`
- `.docx`

The system extracts the text and predicts the language.

### Batch Prediction

Upload a `.csv` file containing a `Text` column. If `Text` is not present, the first column is used.

## Important Project Rules Followed

- No TensorFlow
- No PyTorch
- No Transformers
- No FastText
- No spaCy
- No NLTK
- No paid APIs

## Current Project Status

Completed:

- project setup
- configuration and logging
- data loading and preprocessing
- feature engineering
- model selection and training pipeline
- evaluation and serialization
- predictor and file readers
- Streamlit application layer
- repository documentation

Pending or future refinement:

- successful full training artifact verification
- notebook completion and analysis polishing
- final UI refinement if needed

## Authoring Notes

This project is being developed phase by phase with clean commit checkpoints after each completed stage.
