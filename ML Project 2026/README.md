# Fake Review Detection System

## Overview

This project detects whether a product review is likely original or computer-generated. It combines a classical machine learning classifier for fast, reproducible scoring with an optional instruction-tuned language model for qualitative comparison.

The system is designed around two goals:

- strong baseline performance on the provided dataset
- an inference flow that can be extended into an application or browser extension

## Problem Definition

Given a review text, predict whether the review belongs to:

- `OR`: original / genuine review
- `CG`: computer-generated / fake review

This is a binary text classification problem.

## Dataset

File:

- `fake_reviews_dataset.csv`

Schema:

- `category`: product category
- `rating`: review rating
- `label`: class label (`CG` or `OR`)
- `text_`: review text

Dataset summary used by the training pipeline:

- Total rows: `40,432`
- Balanced classes: `20,216` genuine and `20,216` fake

## System Architecture

The project has two inference layers.

### 1. ML Classification Layer

Implemented in [train_model.py](</c:/Users/Harshal/OneDrive/ドキュメント/ML Project 2026/train_model.py>) and [predict_review.py](</c:/Users/Harshal/OneDrive/ドキュメント/ML Project 2026/predict_review.py>).

Pipeline:

1. Load and clean the dataset
2. Normalize column names
3. Remove missing or empty review text
4. Map labels to binary targets
5. Split data into train, validation, and test sets
6. Convert text into TF-IDF features
7. Train a Logistic Regression classifier
8. Evaluate on validation and test data
9. Save the trained model and evaluation outputs

### 2. LLM Comparison Layer

Implemented in [predict_review.py](</c:/Users/Harshal/OneDrive/ドキュメント/ML Project 2026/predict_review.py>).

This layer sends the custom review to an open-source instruction model:

- `Qwen/Qwen2.5-1.5B-Instruct`

The LLM is not used for model training. It is used only at inference time as a second opinion for a custom review. The script then compares:

- ML model label
- LLM label
- fake probability gap
- final agreement summary

## ML Model

Model:

- `TfidfVectorizer`
- `LogisticRegression`

Reason for selection:

- strong baseline for text classification
- efficient training and inference
- easy to interpret
- robust on medium-sized labeled datasets
- much simpler and more reproducible than a heavier ensemble

### Feature Extraction

The review text is transformed using TF-IDF with:

- word unigrams and bigrams
- `max_features=8000`
- `min_df=3`
- `sublinear_tf=True`

This allows the model to capture both common lexical patterns and meaningful short phrases such as repeated promotional wording.

### Classifier

The classifier is Logistic Regression with:

- `max_iter=1000`
- `random_state=42`

The output probability is used for:

- final class prediction
- confidence band generation
- ML vs LLM comparison

## Data Split Strategy

The dataset is split using stratified sampling to preserve class balance.

Final split sizes:

- Train: `28,318`
- Validation: `6,049`
- Test: `6,065`

The test set is held out from training and used only for final evaluation.

## Training and Evaluation

Training script:

- [train_model.py](</c:/Users/Harshal/OneDrive/ドキュメント/ML Project 2026/train_model.py>)

Saved outputs:

- model: `models/fake_review_detector.joblib`
- metrics: `outputs/metrics.json`
- test predictions: `outputs/test_predictions.csv`
- confusion matrix: `outputs/confusion_matrix.png`
- ROC curve: `outputs/roc_curve.png`

### Performance

Current saved test results:

- Accuracy: `0.9355`
- F1-score: `0.9350`
- ROC-AUC: `0.9834`

Class-wise performance from the test set:

- Genuine precision: `0.9292`
- Genuine recall: `0.9430`
- Fake precision: `0.9421`
- Fake recall: `0.9281`

These results indicate strong and balanced performance on the provided dataset.

## Inference Flow

Custom review inference is handled in [predict_review.py](</c:/Users/Harshal/OneDrive/ドキュメント/ML Project 2026/predict_review.py>).

Default flow:

1. Accept review text from the command line or interactive input
2. Run the saved ML model
3. Extract influential TF-IDF terms
4. Generate simple suspicious-writing flags
5. Run the LLM comparison layer unless skipped
6. Print both results side by side
7. Save the latest report to `outputs/last_prediction_report.json`

### Suspicious-Writing Heuristics

The script adds lightweight rule-based indicators for interpretability:

- heavy exclamation use
- multiple all-caps words
- promotional superlatives
- very short review length

These are not the primary classifier. They are used to highlight patterns that may be useful during manual review.

## Project Structure

- `fake_reviews_dataset.csv`: source dataset
- `train_model.py`: training and evaluation pipeline
- `predict_review.py`: custom review scoring and ML vs LLM comparison
- `build_notebook.py`: notebook generator
- `fake_review_detection_improved.ipynb`: generated notebook version
- `models/`: saved trained model artifacts
- `outputs/`: metrics, charts, and prediction reports
- `requirements.txt`: dependencies

## Installation

```powershell
python -m pip install -r requirements.txt
```

For the current LLM comparison path, `transformers`, `torch`, `accelerate`, and related runtime dependencies must be available locally.

## Usage

### Train the model

```powershell
python train_model.py
```

### Run interactive custom testing

```powershell
python predict_review.py
```

This runs both the ML model and the default LLM comparison.

### Run custom testing with direct text input

```powershell
python predict_review.py --text "This product arrived on time and works well."
```

This also runs both the ML model and the default LLM comparison.

### Run ML-only inference

```powershell
python predict_review.py --skip-llm
```

### Build the notebook

```powershell
python build_notebook.py
```

## Notebook

The notebook is generated by [build_notebook.py](</c:/Users/Harshal/OneDrive/ドキュメント/ML Project 2026/build_notebook.py>) and saved as [fake_review_detection_improved.ipynb](</c:/Users/Harshal/OneDrive/ドキュメント/ML Project 2026/fake_review_detection_improved.ipynb>). It presents:

- dataset loading
- exploratory summaries
- model training
- evaluation
- visualization
- sample review testing

## Limitations

- The ML model is trained only on the provided dataset, so generalization depends on how closely new reviews match the dataset distribution.
- Promotional or exaggerated text is not always fake, so suspicious language alone should not be treated as proof.
- The LLM comparison is advisory and may disagree with the trained classifier.
- Local LLM inference depends on installed dependencies and available system resources.

## Extension Roadmap

This project can be extended into a browser extension or web application.

Recommended next architecture:

1. Keep the trained ML model as the fast scoring engine
2. Expose inference through a small backend API
3. Use the LLM as an optional secondary review layer
4. Build a browser extension UI that sends review text to the backend
5. Return:
   ML score
   LLM score
   agreement summary
   suspicious flags

This separation is more practical than embedding a local LLM directly inside an extension.
