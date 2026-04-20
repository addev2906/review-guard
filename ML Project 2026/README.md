# Fake Review Detection System

## Overview

This project detects whether a product review is likely original or computer-generated. It trains and compares multiple classical ML candidates:

- `TF-IDF + Logistic Regression`
- `TF-IDF + XGBoost`
- `Weighted Ensemble Fusion (0.60 XGBoost + 0.40 LogReg)` — fixed weights
- `Weighted Ensemble Fusion (Validation-Optimized)` — grid-searched weights

The training script evaluates all candidates on the same validation split, saves metrics for each candidate, and stores the best-performing option as the active classifier used by inference.

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

## Project Structure

```
ML Project 2026/
├── train_model.py                         # Model training and evaluation pipeline
├── predict_review.py                      # Inference script with ML + LLM comparison
├── build_notebook.py                      # Generates the Jupyter notebook
├── requirements.txt                       # Python dependencies
├── fake_reviews_dataset.csv               # Dataset
├── fake_review_detection_improved.ipynb   # Auto-generated notebook
├── models/
│   └── fake_review_detector.joblib        # Saved model artifact (format v3)
├── outputs/
│   ├── metrics.json                       # Evaluation metrics for all candidates
│   ├── test_predictions.csv               # Test set predictions from the selected model
│   ├── confusion_matrix.png               # Confusion matrix plot
│   ├── roc_curve.png                      # ROC curve plot
│   └── last_prediction_report.json        # Most recent inference report
└── README.md
```

## System Architecture

The project has two inference layers.

### 1. ML Classification Layer

Implemented in `train_model.py`.

Training flow:

1. Load and clean the dataset
2. Normalize column names
3. Remove missing or empty review text
4. Map labels to binary targets (`CG` → 1, `OR` → 0)
5. Split data into train, validation, and test sets
6. Train `TF-IDF + Logistic Regression`
7. Train `TF-IDF + XGBoost`
8. Build weighted ensemble candidates from both model outputs
9. Grid-search the best ensemble weights on validation data (in 0.05 steps)
10. Evaluate all candidates on validation and test data
11. Select the best validation model and save it with evaluation outputs

### 2. LLM Comparison Layer

Implemented in `predict_review.py`.

This layer sends the custom review to an open-source instruction model:

- `Qwen/Qwen2.5-1.5B-Instruct`

The LLM is not used for model training. It is used only at inference time as a second opinion for a custom review. The script then compares:

- ML model label
- LLM label
- fake probability gap
- final agreement summary
- final decision source (shared, LLM override, or manual review)

## ML Models

### TF-IDF Features

Both models start with TF-IDF text features built from the `text_` column.

Logistic Regression pipeline:

- `ngram_range=(1, 2)`
- `max_features=8000`
- `min_df=3`
- `sublinear_tf=True`

XGBoost pipeline:

- `ngram_range=(1, 2)`
- `max_features=12000`
- `min_df=2`
- `sublinear_tf=True`

### Logistic Regression

The linear baseline uses:

- `max_iter=1000`
- `random_state=42`

This remains the most interpretable model in the project and powers term-level explanations through coefficient weights.

### XGBoost

The boosted tree model uses:

- `n_estimators=250`
- `max_depth=6`
- `learning_rate=0.08`
- `subsample=0.9`
- `colsample_bytree=0.9`
- `reg_lambda=1.0`
- `eval_metric=logloss`

This gives the project a stronger non-linear alternative while still working directly on sparse TF-IDF features.

### Weighted Ensemble Fusion

The project also supports score-level fusion between the two base models:

- **Fixed fusion**: `0.60 × XGBoost + 0.40 × Logistic Regression`
- **Optimized fusion**: validation-searched weights in `0.05` steps across the full `[0, 1]` range

All four candidates are evaluated side by side on the same validation and test sets. The candidate with the best validation AUC (then F1, then accuracy as tiebreakers) is selected automatically.

## Model Performance

The current saved model is **Weighted Ensemble Fusion (Validation-Optimized)** with weights `0.25 × XGBoost + 0.75 × Logistic Regression`.

### Selected Model — Test Set Results

| Metric | Value |
|---|---|
| **Accuracy** | 93.55% |
| **F1-score** | 93.49% |
| **ROC-AUC** | 98.33% |
| Genuine precision / recall | 92.70% / 94.56% |
| Fake precision / recall | 94.45% / 92.55% |

### All Candidate Comparison — Validation Set

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| TF-IDF + Logistic Regression | 92.96% | 92.99% | 97.98% |
| TF-IDF + XGBoost | 89.87% | 89.71% | 96.76% |
| Weighted Ensemble Fixed (0.60/0.40) | 92.03% | 91.95% | 97.80% |
| **Weighted Ensemble Optimized (0.25/0.75)** | **92.94%** | **92.93%** | **98.05%** |

### All Candidate Comparison — Test Set

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| TF-IDF + Logistic Regression | 93.55% | 93.50% | 98.34% |
| TF-IDF + XGBoost | 90.06% | 89.85% | 96.75% |
| Weighted Ensemble Fixed (0.60/0.40) | 92.73% | 92.62% | 97.98% |
| **Weighted Ensemble Optimized (0.25/0.75)** | **93.55%** | **93.49%** | **98.33%** |

> **Note:** The optimized ensemble converged to weights that heavily favor Logistic Regression (0.75) over XGBoost (0.25) on this dataset. Logistic Regression alone performs nearly identically, confirming it is the stronger base model for this text-only classification task.

## Data Split Strategy

The dataset is split using stratified sampling to preserve class balance.

Final split sizes:

- Train: `28,318`
- Validation: `6,049`
- Test: `6,065`

The test set is held out from training and used only for final evaluation.

## Training and Evaluation

Training script:

- `train_model.py`

Saved outputs:

- model artifact: `models/fake_review_detector.joblib`
- metrics: `outputs/metrics.json`
- test predictions: `outputs/test_predictions.csv`
- confusion matrix: `outputs/confusion_matrix.png`
- ROC curve: `outputs/roc_curve.png`

### Model Artifact Format (v3)

The saved `.joblib` artifact contains:

```python
{
    "format_version": 3,
    "selected_model_name": "weighted_ensemble_optimized",
    "models": {
        "tfidf_logreg": <trained Pipeline>,
        "tfidf_xgboost": <trained Pipeline>,
    },
    "ensembles": {
        "weighted_ensemble_fixed": {
            "enabled": True,
            "weights": {"tfidf_xgboost": 0.60, "tfidf_logreg": 0.40},
            "base_models": ["tfidf_xgboost", "tfidf_logreg"],
        },
        "weighted_ensemble_optimized": {
            "enabled": True,
            "weights": {"tfidf_xgboost": 0.25, "tfidf_logreg": 0.75},
            "base_models": ["tfidf_xgboost", "tfidf_logreg"],
        },
    },
}
```

### Metrics JSON

`outputs/metrics.json` includes:

- dataset split sizes
- available candidate models
- selected saved model
- validation and test metrics for each candidate
- classification report for each candidate
- ensemble weights for ensemble candidates

## Inference Flow

Default flow:

1. Accept review text from the command line or interactive input
2. Load the saved model artifact
3. Run the selected ML model
4. If the selected model is an ensemble, combine component probabilities with the saved weights
5. Extract influential TF-IDF terms from the Logistic Regression branch for interpretability
6. Generate simple suspicious-writing flags (exclamation use, all-caps, promotional words, short reviews)
7. Run the LLM comparison layer unless skipped
8. Compare ML and LLM outputs and decide whether the result is shared, LLM-overridden, or manual-review
9. Save the latest report to `outputs/last_prediction_report.json`

### Inference Output Example

```
ML model prediction
  Model: Weighted Ensemble Fusion (Validation-Optimized)
  Label: Likely original / genuine
  Fake probability: 0.2261
  Confidence: moderate confidence
  Component model scores:
    TF-IDF + Logistic Regression: 0.1606
    TF-IDF + XGBoost: 0.4228
  Ensemble weights:
    TF-IDF + XGBoost: 0.25
    TF-IDF + Logistic Regression: 0.75
  Top contributing terms:
    at (-0.357)
    after (-0.283)
    is bit (+0.248)
```

### Backward Compatibility

`predict_review.py` is backward-compatible with older single-pipeline model files (format v1) and also supports the current multi-model artifact format (format v3).

## Installation

```powershell
python -m pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `scikit-learn` | TF-IDF, Logistic Regression, metrics, pipelines |
| `xgboost` | XGBoost classifier |
| `matplotlib` | Plotting (confusion matrix, ROC) |
| `seaborn` | Plot styling |
| `joblib` | Model serialization |
| `nbformat` | Notebook generation |
| `transformers` | LLM inference (Qwen model) |
| `torch` | LLM runtime |
| `accelerate` | LLM device management |
| `protobuf` | Serialization support |

For the current LLM comparison path, `transformers`, `torch`, `accelerate`, and related runtime dependencies must be available locally. For the boosted baseline, `xgboost` is also required.

## Usage

### Train the model

```powershell
python train_model.py
```

This trains the two base models, evaluates the fixed and optimized weighted ensembles, then saves the best validation candidate.

### Run interactive custom testing

```powershell
python predict_review.py
```

### Run custom testing with direct text input

```powershell
python predict_review.py --text "This product arrived on time and works well."
```

### Run ML-only inference (skip LLM)

```powershell
python predict_review.py --skip-llm
```

### Use a different LLM model

```powershell
python predict_review.py --llm-model "Qwen/Qwen2.5-3B-Instruct"
```

### Build the notebook

```powershell
python build_notebook.py
```

## Notebook

The notebook is generated by `build_notebook.py`. It presents:

- dataset loading and cleaning
- exploratory data analysis (label distribution, review length histogram)
- training both base models (Logistic Regression and XGBoost)
- model comparison table with validation and test metrics
- classification report
- confusion matrix and ROC curve visualization
- saving the model artifact and metrics
- sample review predictions

## Suspicious Writing Flags

The inference script applies rule-based heuristic flags independent of the ML model:

| Flag | Trigger |
|---|---|
| Heavy exclamation use | ≥ 3 exclamation marks |
| Multiple all-caps words | ≥ 2 words that are fully uppercase (length > 2) |
| Contains promotional superlatives | Matches terms like "best", "amazing", "must buy", etc. |
| Very short review | ≤ 4 words |

These flags supplement the ML prediction and are displayed as warnings in the inference output.

## ML + LLM Agreement Logic

When both the ML model and LLM produce predictions:

| Condition | Final Decision |
|---|---|
| Both agree, probability gap ≤ 0.20 | Shared label (strong agreement) |
| Both agree, probability gap > 0.20 | Shared label (same direction, different confidence) |
| Disagree, gap ≥ 0.40 | LLM overrides the final verdict |
| Disagree, gap < 0.40 | Manual review recommended |

The override gap threshold is `0.40` (configurable in `predict_review.py` as `LLM_OVERRIDE_GAP`).

## Limitations

- The ML models are trained only on the provided dataset, so generalization depends on how closely new reviews match the dataset distribution.
- Promotional or exaggerated text is not always fake, so suspicious language alone should not be treated as proof.
- The LLM comparison depends on `transformers`, `torch`, and access to a cached or downloadable Hugging Face model.
- If `xgboost` is not installed, the training script falls back to training the Logistic Regression pipeline only.
- If the LLM cannot start, the script falls back gracefully by reporting the LLM error and still saving the ML result report.

## Extension Roadmap

This project can be extended into a browser extension or web application.

Recommended next architecture:

1. Keep the trained ML model as the fast scoring engine
2. Expose inference through a small backend API (e.g., Flask)
3. Use the LLM as an optional secondary review layer
4. Build a browser extension UI that sends review text to the backend
5. Return the ML score, model name, LLM score, agreement summary, and suspicious flags
6. Add batch scanning for all reviews on a product page
7. Add company-wide review aggregation and trust scoring
