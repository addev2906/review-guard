from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "fake_reviews_dataset.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = MODEL_DIR / "fake_review_detector.joblib"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
MODEL_LABELS = {
    "tfidf_logreg": "TF-IDF + Logistic Regression",
    "tfidf_xgboost": "TF-IDF + XGBoost",
    "weighted_ensemble_fixed": "Weighted Ensemble Fusion (0.60 XGBoost + 0.40 LogReg)",
    "weighted_ensemble_optimized": "Weighted Ensemble Fusion (Validation-Optimized)",
}
ENSEMBLE_WEIGHTS = {
    "tfidf_xgboost": 0.60,
    "tfidf_logreg": 0.40,
}


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    df = df.dropna(subset=["text_", "label"]).copy()
    df["text_"] = df["text_"].astype(str).str.strip()
    df = df[df["text_"] != ""].copy()
    df["target"] = df["label"].astype(str).str.strip().map({"CG": 1, "OR": 0})
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)
    return df


def build_logreg_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=8000,
                    min_df=3,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def build_xgboost_pipeline() -> Pipeline:
    if XGBClassifier is None:
        raise RuntimeError(
            "xgboost is not installed. Run `python -m pip install -r requirements.txt` first."
        )

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=12000,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                XGBClassifier(
                    n_estimators=250,
                    max_depth=6,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=4,
                ),
            ),
        ]
    )


def metrics_from_scores(
    y_true: pd.Series, predictions: Any, scores: Any
) -> dict[str, Any]:
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "f1": round(float(f1_score(y_true, predictions)), 4),
        "auc": round(float(roc_auc_score(y_true, scores)), 4),
        "classification_report": classification_report(
            y_true,
            predictions,
            target_names=["Original / Genuine", "Computer-Generated / Fake"],
            digits=4,
            output_dict=True,
        ),
    }


def evaluate_model(
    model: Pipeline,
    X_val: pd.Series,
    y_val: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    val_pred = model.predict(X_val)
    val_score = model.predict_proba(X_val)[:, 1]
    test_pred = model.predict(X_test)
    test_score = model.predict_proba(X_test)[:, 1]

    return {
        "validation": metrics_from_scores(y_val, val_pred, val_score),
        "test": metrics_from_scores(y_test, test_pred, test_score),
        "test_predictions": test_pred,
        "test_scores": test_score,
        "validation_scores": val_score,
    }


def weighted_ensemble_scores(
    logreg_scores: Any, xgboost_scores: Any
) -> Any:
    return weighted_ensemble_scores_with_weights(logreg_scores, xgboost_scores, ENSEMBLE_WEIGHTS)


def weighted_ensemble_scores_with_weights(
    logreg_scores: Any, xgboost_scores: Any, weights: dict[str, float]
) -> Any:
    return (
        weights["tfidf_xgboost"] * xgboost_scores
        + weights["tfidf_logreg"] * logreg_scores
    )


def labels_from_scores(scores: Any, threshold: float = 0.5) -> Any:
    return (scores >= threshold).astype(int)


def evaluate_ensemble(
    logreg_metrics: dict[str, Any],
    xgboost_metrics: dict[str, Any],
    y_val: pd.Series,
    y_test: pd.Series,
    weights: dict[str, float],
) -> dict[str, Any]:
    val_scores = weighted_ensemble_scores_with_weights(
        logreg_metrics["validation_scores"], xgboost_metrics["validation_scores"], weights
    )
    test_scores = weighted_ensemble_scores_with_weights(
        logreg_metrics["test_scores"], xgboost_metrics["test_scores"], weights
    )
    val_predictions = labels_from_scores(val_scores)
    test_predictions = labels_from_scores(test_scores)

    return {
        "validation": metrics_from_scores(y_val, val_predictions, val_scores),
        "test": metrics_from_scores(y_test, test_predictions, test_scores),
        "test_predictions": test_predictions,
        "test_scores": test_scores,
        "validation_scores": val_scores,
        "weights": weights,
    }


def choose_best_model(metrics_by_model: dict[str, dict[str, Any]]) -> str:
    return max(
        metrics_by_model,
        key=lambda name: (
            metrics_by_model[name]["validation"]["auc"],
            metrics_by_model[name]["validation"]["f1"],
            metrics_by_model[name]["validation"]["accuracy"],
        ),
    )


def find_best_ensemble_weights(
    logreg_metrics: dict[str, Any],
    xgboost_metrics: dict[str, Any],
    y_val: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    best_result: dict[str, Any] | None = None

    for xgb_weight_pct in range(0, 101, 5):
        xgb_weight = xgb_weight_pct / 100
        weights = {
            "tfidf_xgboost": xgb_weight,
            "tfidf_logreg": round(1 - xgb_weight, 2),
        }
        result = evaluate_ensemble(
            logreg_metrics=logreg_metrics,
            xgboost_metrics=xgboost_metrics,
            y_val=y_val,
            y_test=y_test,
            weights=weights,
        )
        if best_result is None:
            best_result = result
            continue

        candidate_key = (
            result["validation"]["auc"],
            result["validation"]["f1"],
            result["validation"]["accuracy"],
        )
        best_key = (
            best_result["validation"]["auc"],
            best_result["validation"]["f1"],
            best_result["validation"]["accuracy"],
        )
        if candidate_key > best_key:
            best_result = result

    if best_result is None:
        raise RuntimeError("Could not compute optimized ensemble weights.")

    return best_result


def save_plots(
    y_true: pd.Series, y_pred: pd.Series, y_score: pd.Series, model_label: str
) -> None:
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Original / Genuine", "Computer-Generated / Fake"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix ({model_label})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_true, y_score):.4f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(f"ROC Curve ({model_label})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=160)
    plt.close(fig)


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_dataset(DATA_PATH)
    X = df["text_"]
    y = df["target"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.176,
        random_state=42,
        stratify=y_train_val,
    )

    logreg_model = build_logreg_pipeline()
    logreg_model.fit(X_train, y_train)
    logreg_metrics = evaluate_model(logreg_model, X_val, y_val, X_test, y_test)

    trained_models: dict[str, Any] = {"tfidf_logreg": logreg_model}
    metrics_by_model: dict[str, dict[str, Any]] = {
        "tfidf_logreg": {
            "validation": logreg_metrics["validation"],
            "test": logreg_metrics["test"],
        }
    }
    metrics: dict[str, Any] = {
        "dataset_rows": int(len(df)),
        "train_size": int(len(X_train)),
        "validation_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "available_models": {"tfidf_logreg": MODEL_LABELS["tfidf_logreg"]},
        "ensemble_weights": ENSEMBLE_WEIGHTS,
        "selected_model": "tfidf_logreg",
        "models": metrics_by_model,
    }

    selected_metrics = logreg_metrics

    if XGBClassifier is not None:
        xgboost_model = build_xgboost_pipeline()
        xgboost_model.fit(X_train, y_train)
        xgboost_metrics = evaluate_model(xgboost_model, X_val, y_val, X_test, y_test)

        fixed_ensemble_metrics = evaluate_ensemble(
            logreg_metrics=logreg_metrics,
            xgboost_metrics=xgboost_metrics,
            y_val=y_val,
            y_test=y_test,
            weights=ENSEMBLE_WEIGHTS,
        )
        optimized_ensemble_metrics = find_best_ensemble_weights(
            logreg_metrics=logreg_metrics,
            xgboost_metrics=xgboost_metrics,
            y_val=y_val,
            y_test=y_test,
        )

        trained_models["tfidf_xgboost"] = xgboost_model
        metrics["available_models"]["tfidf_xgboost"] = MODEL_LABELS["tfidf_xgboost"]
        metrics["available_models"]["weighted_ensemble_fixed"] = MODEL_LABELS["weighted_ensemble_fixed"]
        metrics["available_models"]["weighted_ensemble_optimized"] = MODEL_LABELS["weighted_ensemble_optimized"]
        metrics_by_model["tfidf_xgboost"] = {
            "validation": xgboost_metrics["validation"],
            "test": xgboost_metrics["test"],
        }
        metrics_by_model["weighted_ensemble_fixed"] = {
            "validation": fixed_ensemble_metrics["validation"],
            "test": fixed_ensemble_metrics["test"],
            "weights": fixed_ensemble_metrics["weights"],
        }
        metrics_by_model["weighted_ensemble_optimized"] = {
            "validation": optimized_ensemble_metrics["validation"],
            "test": optimized_ensemble_metrics["test"],
            "weights": optimized_ensemble_metrics["weights"],
        }
        metrics["selected_model"] = choose_best_model(metrics_by_model)
        if metrics["selected_model"] == "tfidf_xgboost":
            selected_metrics = xgboost_metrics
        elif metrics["selected_model"] == "weighted_ensemble_fixed":
            selected_metrics = fixed_ensemble_metrics
        elif metrics["selected_model"] == "weighted_ensemble_optimized":
            selected_metrics = optimized_ensemble_metrics
        else:
            selected_metrics = logreg_metrics
    else:
        metrics["xgboost_status"] = "xgboost not installed, ensemble disabled"

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(
        {
            "text": X_test.reset_index(drop=True),
            "true_label": y_test.reset_index(drop=True),
            "selected_model": metrics["selected_model"],
            "predicted_label": pd.Series(selected_metrics["test_predictions"]),
            "fake_probability": pd.Series(selected_metrics["test_scores"]).round(6),
        }
    ).to_csv(PREDICTIONS_PATH, index=False)

    artifact = {
        "format_version": 3,
        "selected_model_name": metrics["selected_model"],
        "models": trained_models,
        "ensembles": {
            "weighted_ensemble_fixed": {
                "enabled": "tfidf_xgboost" in trained_models,
                "weights": ENSEMBLE_WEIGHTS,
                "base_models": ["tfidf_xgboost", "tfidf_logreg"],
            },
            "weighted_ensemble_optimized": {
                "enabled": "tfidf_xgboost" in trained_models,
                "weights": (
                    metrics_by_model.get("weighted_ensemble_optimized", {}).get("weights", {})
                ),
                "base_models": ["tfidf_xgboost", "tfidf_logreg"],
            },
        },
    }
    joblib.dump(artifact, MODEL_PATH)
    save_plots(
        y_test,
        pd.Series(selected_metrics["test_predictions"]),
        pd.Series(selected_metrics["test_scores"]),
        MODEL_LABELS[metrics["selected_model"]],
    )

    print("Training complete.")
    print(f"Dataset rows: {len(df):,}")
    print(f"Train / Val / Test: {len(X_train):,} / {len(X_val):,} / {len(X_test):,}")
    print()
    for model_name, model_metrics in metrics["models"].items():
        print(MODEL_LABELS[model_name])
        print(f"  Validation Accuracy: {model_metrics['validation']['accuracy']:.4f}")
        print(f"  Validation F1-score: {model_metrics['validation']['f1']:.4f}")
        print(f"  Validation ROC-AUC : {model_metrics['validation']['auc']:.4f}")
        print(f"  Test Accuracy      : {model_metrics['test']['accuracy']:.4f}")
        print(f"  Test F1-score      : {model_metrics['test']['f1']:.4f}")
        print(f"  Test ROC-AUC       : {model_metrics['test']['auc']:.4f}")
        print()
    print(f"Selected model: {MODEL_LABELS[metrics['selected_model']]}")
    print(f"Saved model to: {MODEL_PATH.name}")
    print(f"Saved metrics to: {METRICS_PATH.name}")
    print(f"Saved plots to: {OUTPUT_DIR.name}/")


if __name__ == "__main__":
    main()
