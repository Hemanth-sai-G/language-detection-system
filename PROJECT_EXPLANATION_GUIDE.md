# Language Detection System: Complete Project Explanation Guide

## Purpose of This Document

This document explains the entire project from start to finish in a way that helps you:

- understand what every file does
- explain the workflow confidently in an interview
- describe how the machine learning pipeline is built
- explain how training and inference are separated
- explain how the Streamlit app works
- explain why certain models and preprocessing choices were used

This is not just a feature list. It is the actual system explanation from A to Z.

---

## 1. High-Level Project Summary

This project is a **machine learning based language detection system** built using:

- scikit-learn for ML
- pandas for dataset handling
- Streamlit for the web interface
- joblib for model saving/loading
- PyPDF2 and python-docx for reading document files

The project detects the language of input text using a classical ML pipeline.

At a high level, the project has **two major flows**:

1. **Training flow**
   - read dataset
   - clean dataset
   - build features
   - compare models
   - tune final model
   - evaluate performance
   - save model and metrics

2. **Inference flow**
   - load saved trained model
   - accept user input from text or files
   - preprocess input
   - predict language
   - show result in Streamlit

So the important interview point is:

> The project separates training time logic from prediction time logic.

That is a strong architectural decision.

---

## 2. Real Problem Statement

The goal is to identify the language of a text sample among 17 supported languages.

Examples:

- English sentence -> English
- French sentence -> French
- Hindi sentence -> Hindi
- Arabic sentence -> Arabic

This is a text classification problem where:

- **input** = text
- **output** = language label

So mathematically:

- `X` = text samples
- `y` = language classes

---

## 3. Why This Problem Can Be Solved Using Classical ML

Language detection is strongly based on:

- character patterns
- script patterns
- word patterns
- frequency of n-grams

For example:

- French often contains patterns like `que`, `est`, `tion`
- Spanish contains `ción`, `que`, `los`
- German contains `sch`, `der`, `und`
- Hindi/Arabic/Russian use very distinctive scripts

That means we do not necessarily need deep learning for this scale of problem.

A good classical pipeline using:

- character TF-IDF
- word TF-IDF
- linear classifier

is already very strong for language detection.

That is why this project uses scikit-learn instead of Transformers or neural networks.

---

## 4. Overall Architecture

The architecture of the ML system is:

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

This means:

1. raw text goes into two parallel feature extraction branches
2. one branch extracts character-level TF-IDF features
3. one branch extracts word-level TF-IDF features
4. both feature sets are combined using `FeatureUnion`
5. the combined matrix is passed to a classifier
6. the final classifier is `LinearSVC`

This full structure is saved as **one pipeline object**.

That is important:

> The project does not save vectorizer and model separately. It saves the complete trained pipeline as one artifact.

This avoids mismatch problems during inference.

---

## 5. Folder Structure and What Each Part Means

### `data/`

- stores raw and processed data

#### `data/raw/Language Detection.csv`
- original dataset used for training

#### `data/processed/cleaned_dataset.csv`
- cleaned version saved after preprocessing

---

### `src/`

Contains the main machine learning and application backend logic.

#### `src/config.py`
- central configuration file
- defines:
  - project paths
  - dataset paths
  - model directory
  - feature settings
  - train/test settings

This file prevents hardcoding paths in multiple places.

#### `src/data_loader.py`
- loads the CSV dataset
- checks whether file exists
- validates required columns

Required columns:

- `Text`
- `Language`

#### `src/preprocess.py`
- cleans the dataset
- removes duplicates
- removes missing values
- normalizes whitespace
- trims spaces
- saves cleaned dataset

Important design decision:

It does **not** remove punctuation, accents, or script-specific characters.

Why?

Because those are actually useful for identifying languages.

#### `src/feature_engineering.py`
- creates the TF-IDF vectorizers
- contains:
  - character TF-IDF vectorizer
  - word TF-IDF vectorizer
  - `FeatureUnion` to combine both

#### `src/model_selection.py`
- builds candidate model pipelines
- each candidate pipeline includes:
  - combined features
  - classifier

The three baseline models are:

- Multinomial Naive Bayes
- Logistic Regression
- Linear SVC

#### `src/trainer.py`
- handles training workflow
- split data
- compare baseline models
- choose final model according to project architecture
- tune final model using grid search
- run cross validation

This is one of the most important files in the project.

#### `src/evaluator.py`
- evaluates final trained model on test set
- calculates all required metrics
- builds classification report
- builds confusion matrix

#### `src/serializer.py`
- saves and loads:
  - trained pipeline
  - metrics JSON

#### `src/predictor.py`
- used during inference
- loads saved pipeline
- predicts from:
  - single text
  - file input
  - batch CSV input

#### `src/app_utils.py`
- shared helper functions for Streamlit
- load predictor with caching
- load metrics with caching
- save uploaded files temporarily
- show standard missing-model message

---

### `utils/`

Contains utility/support code.

#### `utils/logger.py`
- defines a global logger
- used across the project for standardized logs

#### `utils/txt_reader.py`
- reads plain text files

#### `utils/pdf_reader.py`
- extracts text from PDF

#### `utils/docx_reader.py`
- extracts text from DOCX

#### `utils/helpers.py`
- validates file paths
- reads CSV text rows for batch prediction

---

### `pages/`

Contains Streamlit multipage UI files.

#### `pages/1_Home.py`
- overview dashboard
- shows whether model exists
- shows stored metrics summary

#### `pages/2_Text_Detection.py`
- user enters manual text
- app predicts language

#### `pages/3_File_Detection.py`
- upload TXT, PDF, DOCX
- app extracts content and predicts language

#### `pages/4_Batch_Prediction.py`
- upload CSV
- app predicts row by row
- app allows CSV download of results

#### `pages/5_Model_Performance.py`
- reads metrics from `metrics.json`
- displays evaluation results

#### `pages/6_About.py`
- explains project summary and stack

---

### Root Files

#### `train_model.py`
- entry point for training
- orchestrates the whole ML training workflow
- does not contain actual ML logic itself

#### `app.py`
- main Streamlit shell entry point
- shows overall status and navigation guidance

#### `README.md`
- GitHub/project documentation

#### `PROJECT_EXPLANATION_GUIDE.md`
- this interview explanation document

---

## 6. Training Workflow: End-to-End

Now let us walk through the training process exactly in order.

### Step 1: Run Training Command

You start training using:

```cmd
python train_model.py
```

If using the project environment:

```cmd
.\.venv\Scripts\python.exe train_model.py
```

---

### Step 2: `train_model.py` Starts the Workflow

`train_model.py` is the **orchestrator**.

It performs these steps in sequence:

1. create `DataLoader`
2. load dataset
3. create `Preprocessor`
4. preprocess dataset
5. create `Trainer`
6. train model
7. create `Evaluator`
8. evaluate best model
9. save pipeline
10. save metrics

This file keeps the training process clean and readable.

Interview line:

> I used `train_model.py` as a pure orchestration layer, while all business logic stayed inside dedicated modules.

---

### Step 3: Dataset Loading

Handled by `src/data_loader.py`.

What happens:

1. it reads `RAW_DATASET` path from `src/config.py`
2. checks whether the file exists
3. loads CSV using pandas
4. validates schema

Validation ensures:

- `Text` column exists
- `Language` column exists

If missing, training stops early with a meaningful error.

Why this matters:

> It prevents silent failures due to bad dataset structure.

---

### Step 4: Preprocessing

Handled by `src/preprocess.py`.

What happens:

1. duplicate rows are removed
2. null rows are removed
3. each text is cleaned using `clean_text()`
4. cleaned dataset is saved to `data/processed/cleaned_dataset.csv`

What `clean_text()` actually does:

- converts input to string
- strips leading and trailing spaces
- normalizes repeated whitespace into single spaces

What it intentionally does not do:

- remove punctuation
- remove accents
- remove script-specific symbols
- remove stopwords

Why?

Because language detection depends on:

- punctuation patterns
- special characters
- diacritics
- script forms

For example:

- `é`, `à`, `ç` help in French
- `ñ` helps in Spanish
- Arabic script directly identifies Arabic
- Devanagari script helps identify Hindi

So over-cleaning would actually damage the model.

---

## 7. Feature Engineering: How Text Becomes Numbers

Machine learning models cannot read raw text directly.

So the text must be converted into numerical features.

This is done in `src/feature_engineering.py`.

### 7.1 Character TF-IDF

Configuration:

- `analyzer="char"`
- `ngram_range=(2, 5)`

Meaning:

The model looks at character chunks of length 2 to 5.

Example for `"bonjour"`:

- 2-grams: `bo`, `on`, `nj`, `jo`, `ou`, `ur`
- 3-grams: `bon`, `onj`, `njo`, `jou`, `our`

Why character features are useful:

- very strong for language detection
- captures spelling patterns
- captures script patterns
- works even on short text

### 7.2 Word TF-IDF

Configuration:

- `analyzer="word"`
- `ngram_range=(1, 2)`

Meaning:

The model looks at:

- single words
- two-word combinations

Why word features are useful:

- captures frequent language-specific words
- captures local context
- helps on longer sentences

### 7.3 Why Use Both Together

Character features are excellent for:

- short text
- morphology
- script and spelling clues

Word features are excellent for:

- vocabulary patterns
- phrase-level information

By combining both, the model becomes more robust.

### 7.4 How Combination Happens

`FeatureUnion` runs both vectorizers in parallel and concatenates the output feature matrices.

That means the final classifier sees:

- all character-level TF-IDF features
- all word-level TF-IDF features

in one combined feature space.

Interview line:

> I used a hybrid feature strategy: character n-grams capture script and morphology, while word n-grams capture lexical patterns.

---

## 8. Model Pipelines

Built in `src/model_selection.py`.

Each model is wrapped in a scikit-learn `Pipeline`.

This means every candidate pipeline includes:

1. feature extraction
2. classification

So each model is represented like:

```text
Pipeline(
    features = FeatureUnion(char_tfidf + word_tfidf),
    classifier = model
)
```

### Candidate Models

#### 1. Multinomial Naive Bayes

Why it was included:

- strong baseline for text classification
- fast to train
- common classical benchmark

#### 2. Logistic Regression

Why it was included:

- linear model
- strong performance on sparse text features
- interpretable baseline

#### 3. Linear SVC

Why it was included:

- very strong for high-dimensional sparse text data
- commonly performs extremely well on text classification
- efficient compared to nonlinear SVM

### Actual LinearSVC Settings

In the final version:

- `dual="auto"`
- `max_iter=5000`
- `random_state=RANDOM_STATE`

Why:

- `max_iter=5000` helps convergence
- `dual="auto"` lets scikit-learn choose a suitable optimization strategy
- random state keeps behavior reproducible where applicable

---

## 9. How Model Comparison Happens

Handled in `src/trainer.py`.

### Step 1: Train-Test Split

The cleaned dataset is split into:

- training set
- testing set

Using:

- `test_size = 0.20`
- `random_state = 42`
- `stratify = y`

Why stratify?

Because the class distribution of languages should remain balanced between train and test sets.

Without stratification, some languages could be underrepresented in the test set.

### Step 2: Baseline Training

For each candidate model:

1. fit pipeline on `X_train`, `y_train`
2. predict on `X_test`
3. compute test accuracy
4. store pipeline and accuracy

This gives baseline comparison results.

### Step 3: Best Baseline vs Final Selected Model

This project has an important design detail:

- it records the **best baseline model by accuracy**
- but it still forces the **final production model to be Linear SVC**

Why?

Because the project architecture explicitly defined `LinearSVC` as the final model.

In this actual project run:

- Logistic Regression had slightly higher baseline accuracy
- but final model was still set to `Linear SVC`

Interview explanation:

> I compared multiple baselines honestly, but the project specification fixed `LinearSVC` as the final production model, so I preserved that architecture and then tuned the Linear SVC pipeline.

That is a very strong and honest answer.

---

## 10. Hyperparameter Tuning with GridSearchCV

After selecting the final model, the project tunes **only Linear SVC**.

This happens in `Trainer.tune_best_model()`.

### Hyperparameters Tuned

- `classifier__C`
- `classifier__loss`

Search grid:

- `C = [0.01, 0.1, 1, 10, 100]`
- `loss = ["hinge", "squared_hinge"]`

### Why `classifier__...` Syntax?

Because the classifier is inside a scikit-learn `Pipeline`.

So scikit-learn accesses nested parameters using:

```text
step_name__parameter_name
```

Here:

- step name = `classifier`
- parameter = `C` or `loss`

So:

- `classifier__C`
- `classifier__loss`

### Cross-Validation Strategy for Tuning

Used:

- `StratifiedKFold`
- `n_splits = 5`
- `shuffle = True`
- `random_state = 42`

Why stratified k-fold?

Because class distribution should remain balanced across folds.

### Optimization Metric

Used:

- `scoring="f1_macro"`

Why macro F1?

Because this is a multiclass classification problem and we want balanced performance across all languages, not just overall majority-class accuracy.

That is a strong interview point:

> I tuned using macro F1 because it treats all classes equally, which is important in multiclass language detection.

---

## 11. Final Training After Tuning

After `GridSearchCV` finds the best parameter combination:

1. `best_estimator_` is stored as the final pipeline
2. that tuned pipeline is fitted again on the training split

This final tuned and fitted pipeline becomes the actual production model.

---

## 12. Cross Validation of Final Model

After tuning, the project performs another validation step:

- `cross_val_score`
- 5 folds
- scoring = `accuracy`

This is done on the full cleaned dataset.

It stores:

- fold scores
- mean accuracy
- standard deviation

Why is this useful?

Because it tells us:

- average generalization performance
- how stable the model is across folds

In this project run:

- cross-validation mean was around `0.9880`
- standard deviation was around `0.0081`

That means the model performs strongly and fairly consistently.

---

## 13. Evaluation on Test Set

Handled by `src/evaluator.py`.

### Inputs to Evaluator

The evaluator receives:

- trained pipeline
- `X_test`
- `y_test`
- training summary metadata

### What Happens

1. final model predicts on `X_test`
2. predictions are stored
3. metrics are computed

### Metrics Computed

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
- Cross Validation Std

### Why Both Macro and Weighted Metrics?

#### Macro

- gives equal importance to each language
- useful to evaluate fairness across classes

#### Weighted

- weights metrics by class support
- useful for overall practical performance

So together they give a balanced picture.

### Classification Report

This contains for each class:

- precision
- recall
- f1-score
- support

This helps explain which languages are performing better or worse.

### Confusion Matrix

This shows:

- true language vs predicted language
- which languages get confused with each other

This is useful for error analysis.

---

## 14. Model Serialization

Handled by `src/serializer.py`.

### Why Save the Full Pipeline?

The project saves the whole trained pipeline as:

- `models/language_detector.pkl`

This includes:

- character vectorizer
- word vectorizer
- feature union
- trained classifier

This is better than saving vectorizer and model separately because:

- no risk of mismatch
- simpler inference
- easier deployment

### Metrics Saving

Metrics are saved as:

- `models/metrics.json`

This is used by the Streamlit dashboard to show performance.

### Why JSON for Metrics?

Because:

- easy to inspect
- easy to load in Streamlit
- easy to version
- human-readable

---

## 15. Inference Workflow

Now let us switch from training flow to prediction flow.

The prediction side starts from the saved `.pkl` model.

### Main Rule

The app does **not** train the model.

It only:

- loads the saved trained pipeline
- accepts user inputs
- predicts

That is a good production design because training is expensive and should not happen at runtime.

---

## 16. How `Predictor` Works

Handled by `src/predictor.py`.

### Initialization

When `Predictor()` is created:

1. it loads the saved pipeline using `ModelSerializer.load_pipeline()`
2. it also creates a `Preprocessor` instance

### Why Reuse `Preprocessor` During Inference?

Because the text seen during prediction should be cleaned in the same way as training data.

This is critical.

Interview line:

> I reused the same preprocessing behavior in inference to maintain train-inference consistency.

### `predict_text()`

Used for single text prediction.

Flow:

1. clean text
2. ensure text is not empty
3. call pipeline `.predict([cleaned_text])`
4. return:
   - cleaned text
   - predicted language
   - confidence-like score

### Confidence Score

For `LinearSVC`, true probabilities are not available by default.

So this project uses `decision_function()` as a confidence-like score.

That means:

- it is not a calibrated probability
- it is the classifier margin/decision score

Important interview honesty:

> The app shows a confidence-like score derived from `decision_function`, not a true probability.

That is a smart and accurate answer.

### `predict_batch()`

Used for multiple text strings.

It simply loops through texts and applies `predict_text()` to each one.

### `predict_file()`

Used for file-based prediction.

Depending on extension:

- `.txt` -> single text prediction
- `.pdf` -> single text prediction
- `.docx` -> single text prediction
- `.csv` -> batch prediction

---

## 17. File Reading Utilities

The project keeps file reading outside the predictor logic.

That is good separation of concerns.

### `TXTReader`

- validates `.txt`
- reads UTF-8 content
- rejects empty file

### `PDFReader`

- validates `.pdf`
- uses `PyPDF2.PdfReader`
- extracts text page by page
- joins text

### `DOCXReader`

- validates `.docx`
- uses `python-docx`
- reads paragraph text

### `read_csv_texts()`

Defined in `utils/helpers.py`.

Behavior:

1. validate file exists
2. ensure `.csv`
3. load with pandas
4. if `Text` column exists, use it
5. otherwise use first column
6. drop nulls
7. strip whitespace
8. return list of non-empty strings

This is the backbone of batch prediction.

---

## 18. Streamlit App Workflow

The UI has two layers:

1. `app.py`
2. page files inside `pages/`

### `app.py`

This is the application shell.

It:

- sets page configuration
- shows project title
- shows system status
- shows capability overview
- gives quick-start instructions

It also checks:

- whether model file exists
- whether predictor can be loaded

### `src/app_utils.py`

This file contains reusable helpers used by the Streamlit pages.

#### `load_predictor()`

- loads predictor once
- caches it with `st.cache_resource`

Why caching?

Because loading a 100MB pipeline every page interaction would be slow.

So caching avoids repeated expensive loads.

#### `load_metrics()`

- loads metrics JSON
- caches it with `st.cache_data`

#### `save_uploaded_file()`

- temporarily stores uploaded files so the backend can process them through normal file paths

#### `render_missing_model_message()`

- standard error UI shown when model artifacts are missing

This avoids duplicating the same logic across pages.

---

## 19. Page-by-Page Runtime Flow

### Home Page

File: `pages/1_Home.py`

Purpose:

- show readiness
- show saved metrics snapshot
- explain workflow

It is a dashboard/introduction page.

### Text Detection Page

File: `pages/2_Text_Detection.py`

Flow:

1. load predictor
2. user enters text in `st.text_area`
3. user clicks button
4. `predictor.predict_text()` runs
5. UI shows:
   - predicted language
   - confidence-like score
   - normalized text used

### File Detection Page

File: `pages/3_File_Detection.py`

Flow:

1. load predictor
2. user uploads TXT, PDF, or DOCX
3. Streamlit file is saved temporarily
4. predictor reads file via utility class
5. extracted text is cleaned and classified
6. UI shows:
   - predicted language
   - confidence-like score
   - normalized extracted text
7. temporary file is deleted

### Batch Prediction Page

File: `pages/4_Batch_Prediction.py`

Flow:

1. user uploads CSV
2. temporary file created
3. `predictor.predict_file()` sees `.csv`
4. `read_csv_texts()` extracts rows
5. `predict_batch()` predicts each row
6. results converted to DataFrame
7. shown in Streamlit table
8. downloadable as CSV
9. temporary file removed

### Model Performance Page

File: `pages/5_Model_Performance.py`

Flow:

1. load `metrics.json`
2. show:
   - accuracy
   - macro metrics
   - CV stats
   - baseline comparison
   - classification report
   - confusion matrix
   - selected model info

### About Page

File: `pages/6_About.py`

Purpose:

- explain project objective
- explain architecture
- list tech stack
- list supported workflows

---

## 20. Exact File-to-File Dependency Flow

This is how the files depend on each other.

### Training Side

`train_model.py`

-> `src.data_loader.DataLoader`  
-> `src.preprocess.Preprocessor`  
-> `src.trainer.Trainer`  
-> `src.evaluator.Evaluator`  
-> `src.serializer.ModelSerializer`

### Trainer Internals

`src.trainer.Trainer`

-> `src.model_selection.ModelSelection`  
-> `src.feature_engineering.FeatureEngineering`  
-> `sklearn Pipeline / GridSearchCV / cross_val_score`

### Inference Side

`app.py` and `pages/*`

-> `src.app_utils`  
-> `src.predictor.Predictor`  
-> `src.serializer.ModelSerializer`

### Predictor Internals

`src.predictor.Predictor`

-> `src.preprocess.Preprocessor`  
-> `utils.txt_reader.TXTReader`  
-> `utils.pdf_reader.PDFReader`  
-> `utils.docx_reader.DOCXReader`  
-> `utils.helpers.read_csv_texts`

That is the real interaction map.

---

## 21. Why the Project Uses a Pipeline

This is a very likely interview question.

### Reason 1: Prevent train-inference mismatch

If feature extraction and classifier are saved separately, there is a risk that:

- training uses one vectorizer
- inference accidentally uses another

Pipeline prevents that.

### Reason 2: Cleaner code

Training, tuning, prediction all happen through one object.

### Reason 3: Easy hyperparameter search

Grid search can directly access classifier parameters inside the pipeline.

### Reason 4: Easy deployment

One `.pkl` file contains the full end-to-end inference logic.

Good interview line:

> I used scikit-learn pipelines so that preprocessing, feature extraction, and classification stay coupled and reproducible across training and inference.

---

## 22. Why LinearSVC Was Chosen as Final Model

Reasons:

- strong performance on sparse high-dimensional text data
- efficient for text classification
- good generalization
- fits classical ML approach very well

Even though Logistic Regression had slightly better baseline accuracy in this run, the project architecture fixed `LinearSVC` as the final production classifier, and then that model was tuned.

This is okay as long as you explain it honestly.

---

## 23. Key Results of the Project

From the final training run:

- final model = `Linear SVC`
- test accuracy = about `0.9912`
- macro F1 = about `0.9928`
- cross-validation mean = about `0.9880`

These are strong results for a classical ML language detection system.

---

## 24. Important Engineering Decisions

### Decision 1: Keep project modular

Instead of writing everything inside one notebook or one script, the logic is split into:

- data loading
- preprocessing
- feature engineering
- training
- evaluation
- serialization
- inference
- UI

This improves maintainability and makes the project interview-friendly.

### Decision 2: Separate training from app runtime

The Streamlit app never retrains the model.

It only loads saved artifacts.

That is better for:

- performance
- user experience
- reproducibility

### Decision 3: Preserve language-specific symbols

This was crucial.

Over-cleaning would hurt language detection.

### Decision 4: Use both char and word features

This hybrid strategy gives better robustness than only one type of feature.

### Decision 5: Save metrics separately

This allows the dashboard to work without retraining.

---

## 25. Limitations of the Project

A strong interview answer also includes limitations.

Possible limitations:

1. confidence is not a true probability
   - it is based on `decision_function`

2. PDF extraction quality depends on extractable text
   - scanned PDFs without text layer may fail

3. performance may drop on:
   - very short text
   - mixed-language text
   - slang/noisy text

4. the system is dataset-dependent
   - if the real-world distribution differs a lot, performance may change

5. no OCR support
   - image-only documents are not handled

Being able to say these points in an interview makes you sound much stronger.

---

## 26. Possible Future Improvements

If asked “what would you improve next?”, say:

1. add OCR support for scanned PDFs/images
2. calibrate confidence scores
3. add unit tests and integration tests
4. add sample files and demo datasets
5. deploy to Streamlit Cloud or another platform
6. add language probability ranking instead of only top-1 prediction
7. add model monitoring or logging for real usage
8. compare with deep learning models for research purposes

---

## 27. How to Explain the Entire Project in 60 Seconds

Use this answer:

> I built a modular language detection system using scikit-learn and Streamlit. The training pipeline loads and validates the dataset, preprocesses text while preserving language-specific characters, and converts text into a hybrid feature space using both character-level and word-level TF-IDF. I compared Naive Bayes, Logistic Regression, and Linear SVC as baselines, then tuned the final Linear SVC model using GridSearchCV with macro F1 scoring and stratified 5-fold cross-validation. The final trained pipeline is saved as a single `.pkl` artifact, and evaluation metrics are saved to JSON. The Streamlit app loads the saved model and supports manual text prediction, TXT/PDF/DOCX file prediction, CSV batch prediction, and a model performance dashboard.

---

## 28. How to Explain the Entire Project in 2 to 3 Minutes

Use this answer:

> This project is a production-style machine learning application for language detection across 17 languages. I structured it into separate modules for data loading, preprocessing, feature engineering, training, evaluation, serialization, prediction, and the Streamlit frontend. For preprocessing, I removed duplicates and null values and normalized whitespace, but intentionally did not remove punctuation, accents, or script-specific symbols because those carry important language-identification signals. For feature engineering, I used a hybrid approach: character-level TF-IDF with 2 to 5 n-grams and word-level TF-IDF with 1 to 2 n-grams, then combined them with FeatureUnion. I built scikit-learn pipelines for three models: Multinomial Naive Bayes, Logistic Regression, and Linear SVC. After comparing baseline accuracy, I kept Linear SVC as the final architecture and tuned it with GridSearchCV over `C` and `loss`, using 5-fold stratified cross-validation and macro F1 scoring. I then evaluated the final model with accuracy, macro and weighted precision/recall/F1, classification report, confusion matrix, and additional cross-validation statistics. The trained pipeline is saved as one `language_detector.pkl` file, and metrics are saved in `metrics.json`. On the frontend side, the Streamlit app loads the saved pipeline and supports direct text prediction, document-based prediction for TXT/PDF/DOCX, CSV-based batch prediction, and a performance dashboard backed by the saved metrics.

---

## 29. Interview Questions You Should Be Ready For

### Q1. Why did you use TF-IDF?

Because TF-IDF converts text into sparse numerical features that capture important patterns while reducing the effect of overly common tokens.

### Q2. Why did you use character n-grams?

Because language detection depends heavily on spelling and script patterns, and character n-grams capture those extremely well.

### Q3. Why not remove punctuation and accents?

Because they are useful signals for identifying languages.

### Q4. Why compare multiple models?

To establish baseline performance and justify the final model choice.

### Q5. Why use LinearSVC finally?

Because it performs strongly on sparse high-dimensional text classification and fits the intended architecture.

### Q6. Why use macro F1 in tuning?

Because it treats all language classes equally and is more suitable for multiclass fairness than plain accuracy alone.

### Q7. Why save the whole pipeline?

To avoid train-inference mismatch and keep deployment simple.

### Q8. Why use Streamlit?

Because it provides a fast and clean way to turn the model into an interactive application.

### Q9. How does batch prediction work?

The app reads the CSV, extracts the `Text` column or first column, predicts row by row using the same saved pipeline, then displays and exports results.

### Q10. What happens if the model file is missing?

The app shows a standard message telling the user to run training first.

---

## 30. Final One-Line Mental Model

If you want one sentence to remember the whole project, remember this:

> This project trains a hybrid TF-IDF plus LinearSVC language classifier offline, saves it as one reusable pipeline artifact, and serves it through a modular Streamlit app for text, file, batch, and metrics-based workflows.

---

## 31. Files You Should Study First Before an Interview

If you are short on time, focus on these in this order:

1. `train_model.py`
2. `src/trainer.py`
3. `src/feature_engineering.py`
4. `src/model_selection.py`
5. `src/evaluator.py`
6. `src/predictor.py`
7. `src/app_utils.py`
8. `app.py`
9. `pages/2_Text_Detection.py`
10. `pages/3_File_Detection.py`
11. `pages/4_Batch_Prediction.py`

That gives you most of the important explanation flow.

---

## 32. Final Advice for Explaining It as Your Project

Do not say:

- “ChatGPT made it”
- “I don’t know exactly what happened”
- “I just copied code”

Instead say:

> I designed the project in phases, kept the architecture modular, and built a classical NLP pipeline for language detection using hybrid TF-IDF features, model comparison, tuning, evaluation, artifact serialization, and a Streamlit-based inference UI.

That is accurate, strong, and professional.
