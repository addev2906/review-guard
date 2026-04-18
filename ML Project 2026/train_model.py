from __future__ import annotations

import json
from pathlib import Path

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


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "fake_reviews_dataset.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = MODEL_DIR / "fake_review_detector.joblib"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"


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


def build_pipeline() -> Pipeline:
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


def save_plots(y_true: pd.Series, y_pred: pd.Series, y_score: pd.Series) -> None:
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Original / Genuine", "Computer-Generated / Fake"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_true, y_score):.4f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curve")
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

    model = build_pipeline()
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_score = model.predict_proba(X_val)[:, 1]

    test_pred = model.predict(X_test)
    test_score = model.predict_proba(X_test)[:, 1]

    metrics = {
        "dataset_rows": int(len(df)),
        "train_size": int(len(X_train)),
        "validation_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "validation_accuracy": round(float(accuracy_score(y_val, val_pred)), 4),
        "validation_f1": round(float(f1_score(y_val, val_pred)), 4),
        "validation_auc": round(float(roc_auc_score(y_val, val_score)), 4),
        "test_accuracy": round(float(accuracy_score(y_test, test_pred)), 4),
        "test_f1": round(float(f1_score(y_test, test_pred)), 4),
        "test_auc": round(float(roc_auc_score(y_test, test_score)), 4),
        "classification_report": classification_report(
            y_test,
            test_pred,
            target_names=["Original / Genuine", "Computer-Generated / Fake"],
            digits=4,
            output_dict=True,
        ),
    }

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(
        {
            "text": X_test.reset_index(drop=True),
            "true_label": y_test.reset_index(drop=True),
            "predicted_label": pd.Series(test_pred),
            "fake_probability": pd.Series(test_score).round(6),
        }
    ).to_csv(PREDICTIONS_PATH, index=False)

    joblib.dump(model, MODEL_PATH)
    save_plots(y_test, test_pred, test_score)

    print("Training complete.")
    print(f"Dataset rows: {len(df):,}")
    print(f"Train / Val / Test: {len(X_train):,} / {len(X_val):,} / {len(X_test):,}")
    print()
    print("Validation metrics")
    print(f"  Accuracy: {metrics['validation_accuracy']:.4f}")
    print(f"  F1-score: {metrics['validation_f1']:.4f}")
    print(f"  ROC-AUC : {metrics['validation_auc']:.4f}")
    print()
    print("Test metrics")
    print(f"  Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"  F1-score: {metrics['test_f1']:.4f}")
    print(f"  ROC-AUC : {metrics['test_auc']:.4f}")
    print()
    print(f"Saved model to: {MODEL_PATH.name}")
    print(f"Saved metrics to: {METRICS_PATH.name}")
    print(f"Saved plots to: {OUTPUT_DIR.name}/")


if __name__ == "__main__":
    main()
