from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "fake_review_detection_improved.ipynb"


def main() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# Fake Review Detection\n"
            "\n"
            "This notebook is a cleaned-up and reproducible version of the original project. "
            "It compares `TF-IDF + Logistic Regression`, `TF-IDF + XGBoost`, and "
            "`Weighted Ensemble Fusion` (with both fixed and validation-optimized weights), "
            "then keeps the strongest validation model as the saved classifier. "
            "It also includes an optional `Qwen/Qwen2.5-1.5B-Instruct` comparison for custom reviews."
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## Why this version is stronger\n"
            "\n"
            "- Compares two strong text-classification baselines instead of only one\n"
            "- Adds weighted ensemble fusion with automatic weight optimization\n"
            "- Adds a proper train/validation/test split\n"
            "- Saves metrics and charts for reporting\n"
            "- Produces a reusable model artifact (format v3) with all trained models and ensemble metadata\n"
            "- Lets you compare the ML prediction with `Qwen/Qwen2.5-1.5B-Instruct`"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "\n"
            "import joblib\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n"
            "import seaborn as sns\n"
            "from sklearn.feature_extraction.text import TfidfVectorizer\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.metrics import (\n"
            "    ConfusionMatrixDisplay,\n"
            "    accuracy_score,\n"
            "    classification_report,\n"
            "    f1_score,\n"
            "    roc_auc_score,\n"
            "    roc_curve,\n"
            ")\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.pipeline import Pipeline\n"
            "from xgboost import XGBClassifier\n"
            "\n"
            "ROOT = Path.cwd()\n"
            "DATA_PATH = ROOT / 'fake_reviews_dataset.csv'\n"
            "MODEL_DIR = ROOT / 'models'\n"
            "OUTPUT_DIR = ROOT / 'outputs'\n"
            "MODEL_DIR.mkdir(exist_ok=True)\n"
            "OUTPUT_DIR.mkdir(exist_ok=True)\n"
            "sns.set_theme(style='whitegrid')"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "df = pd.read_csv(DATA_PATH)\n"
            "df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]\n"
            "df = df.dropna(subset=['text_', 'label']).copy()\n"
            "df['text_'] = df['text_'].astype(str).str.strip()\n"
            "df = df[df['text_'] != ''].copy()\n"
            "df['target'] = df['label'].astype(str).str.strip().map({'CG': 1, 'OR': 0})\n"
            "df = df.dropna(subset=['target']).copy()\n"
            "df['target'] = df['target'].astype(int)\n"
            "df.head()"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "summary = pd.DataFrame({\n"
            "    'rows': [len(df)],\n"
            "    'unique_categories': [df['category'].nunique()],\n"
            "    'avg_review_length': [round(df['text_'].str.len().mean(), 2)],\n"
            "    'fake_count': [int(df['target'].sum())],\n"
            "    'genuine_count': [int((1 - df['target']).sum())],\n"
            "})\n"
            "summary"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "df['label'].value_counts().plot(kind='bar', ax=axes[0], color=['#5B8FF9', '#61DDAA'])\n"
            "axes[0].set_title('Label Distribution')\n"
            "axes[0].set_xlabel('Label')\n"
            "axes[0].set_ylabel('Count')\n"
            "\n"
            "df['text_'].str.len().plot(kind='hist', bins=40, ax=axes[1], color='#F6BD16')\n"
            "axes[1].set_title('Review Length Distribution')\n"
            "axes[1].set_xlabel('Characters')\n"
            "plt.tight_layout()"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "X = df['text_']\n"
            "y = df['target']\n"
            "\n"
            "X_train_val, X_test, y_train_val, y_test = train_test_split(\n"
            "    X, y, test_size=0.15, random_state=42, stratify=y\n"
            ")\n"
            "X_train, X_val, y_train, y_val = train_test_split(\n"
            "    X_train_val, y_train_val, test_size=0.176, random_state=42, stratify=y_train_val\n"
            ")\n"
            "\n"
            "print(f'Train size: {len(X_train):,}')\n"
            "print(f'Validation size: {len(X_val):,}')\n"
            "print(f'Test size: {len(X_test):,}')"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "models = {\n"
            "    'tfidf_logreg': Pipeline([\n"
            "        ('tfidf', TfidfVectorizer(\n"
            "            ngram_range=(1, 2),\n"
            "            max_features=8000,\n"
            "            min_df=3,\n"
            "            sublinear_tf=True,\n"
            "        )),\n"
            "        ('clf', LogisticRegression(max_iter=1000, random_state=42)),\n"
            "    ]),\n"
            "    'tfidf_xgboost': Pipeline([\n"
            "        ('tfidf', TfidfVectorizer(\n"
            "            ngram_range=(1, 2),\n"
            "            max_features=12000,\n"
            "            min_df=2,\n"
            "            sublinear_tf=True,\n"
            "        )),\n"
            "        ('clf', XGBClassifier(\n"
            "            n_estimators=250,\n"
            "            max_depth=6,\n"
            "            learning_rate=0.08,\n"
            "            subsample=0.9,\n"
            "            colsample_bytree=0.9,\n"
            "            reg_lambda=1.0,\n"
            "            objective='binary:logistic',\n"
            "            eval_metric='logloss',\n"
            "            random_state=42,\n"
            "            n_jobs=4,\n"
            "        )),\n"
            "    ]),\n"
            "}\n"
            "\n"
            "results = {}\n"
            "for name, model in models.items():\n"
            "    model.fit(X_train, y_train)\n"
            "    val_pred = model.predict(X_val)\n"
            "    val_score = model.predict_proba(X_val)[:, 1]\n"
            "    test_pred = model.predict(X_test)\n"
            "    test_score = model.predict_proba(X_test)[:, 1]\n"
            "    results[name] = {\n"
            "        'model': model,\n"
            "        'val_pred': val_pred,\n"
            "        'val_score': val_score,\n"
            "        'test_pred': test_pred,\n"
            "        'test_score': test_score,\n"
            "        'validation_accuracy': round(accuracy_score(y_val, val_pred), 4),\n"
            "        'validation_f1': round(f1_score(y_val, val_pred), 4),\n"
            "        'validation_auc': round(roc_auc_score(y_val, val_score), 4),\n"
            "        'test_accuracy': round(accuracy_score(y_test, test_pred), 4),\n"
            "        'test_f1': round(f1_score(y_test, test_pred), 4),\n"
            "        'test_auc': round(roc_auc_score(y_test, test_score), 4),\n"
            "    }\n"
            "\n"
            "results_df = pd.DataFrame(results).T[[\n"
            "    'validation_accuracy', 'validation_f1', 'validation_auc',\n"
            "    'test_accuracy', 'test_f1', 'test_auc'\n"
            "]]\n"
            "selected_model_name = results_df.sort_values(\n"
            "    by=['validation_auc', 'validation_f1'], ascending=False\n"
            ").index[0]\n"
            "selected = results[selected_model_name]\n"
            "results_df"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## Weighted Ensemble Fusion\n"
            "\n"
            "Combine both models with score-level fusion. The fixed ensemble uses "
            "`0.60 × XGBoost + 0.40 × LogReg`. The optimized version grid-searches weights "
            "in `0.05` steps and picks the combination with the best validation AUC."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "import numpy as np\n"
            "\n"
            "FIXED_WEIGHTS = {'tfidf_xgboost': 0.60, 'tfidf_logreg': 0.40}\n"
            "\n"
            "def ensemble_scores(logreg_scores, xgb_scores, weights):\n"
            "    return weights['tfidf_xgboost'] * xgb_scores + weights['tfidf_logreg'] * logreg_scores\n"
            "\n"
            "# Fixed ensemble\n"
            "fixed_val_scores = ensemble_scores(\n"
            "    results['tfidf_logreg']['val_score'],\n"
            "    results['tfidf_xgboost']['val_score'],\n"
            "    FIXED_WEIGHTS,\n"
            ")\n"
            "fixed_test_scores = ensemble_scores(\n"
            "    results['tfidf_logreg']['test_score'],\n"
            "    results['tfidf_xgboost']['test_score'],\n"
            "    FIXED_WEIGHTS,\n"
            ")\n"
            "\n"
            "# Optimized ensemble (grid search)\n"
            "best_auc, best_weights = 0, FIXED_WEIGHTS\n"
            "for xgb_pct in range(0, 101, 5):\n"
            "    w = {'tfidf_xgboost': xgb_pct / 100, 'tfidf_logreg': round(1 - xgb_pct / 100, 2)}\n"
            "    scores = ensemble_scores(\n"
            "        results['tfidf_logreg']['val_score'],\n"
            "        results['tfidf_xgboost']['val_score'],\n"
            "        w,\n"
            "    )\n"
            "    auc = roc_auc_score(y_val, scores)\n"
            "    if auc > best_auc:\n"
            "        best_auc, best_weights = auc, w\n"
            "\n"
            "opt_val_scores = ensemble_scores(\n"
            "    results['tfidf_logreg']['val_score'],\n"
            "    results['tfidf_xgboost']['val_score'],\n"
            "    best_weights,\n"
            ")\n"
            "opt_test_scores = ensemble_scores(\n"
            "    results['tfidf_logreg']['test_score'],\n"
            "    results['tfidf_xgboost']['test_score'],\n"
            "    best_weights,\n"
            ")\n"
            "\n"
            "ensemble_results = {\n"
            "    'weighted_ensemble_fixed': {\n"
            "        'validation_accuracy': round(accuracy_score(y_val, (fixed_val_scores >= 0.5).astype(int)), 4),\n"
            "        'validation_f1': round(f1_score(y_val, (fixed_val_scores >= 0.5).astype(int)), 4),\n"
            "        'validation_auc': round(roc_auc_score(y_val, fixed_val_scores), 4),\n"
            "        'test_accuracy': round(accuracy_score(y_test, (fixed_test_scores >= 0.5).astype(int)), 4),\n"
            "        'test_f1': round(f1_score(y_test, (fixed_test_scores >= 0.5).astype(int)), 4),\n"
            "        'test_auc': round(roc_auc_score(y_test, fixed_test_scores), 4),\n"
            "        'weights': FIXED_WEIGHTS,\n"
            "    },\n"
            "    'weighted_ensemble_optimized': {\n"
            "        'validation_accuracy': round(accuracy_score(y_val, (opt_val_scores >= 0.5).astype(int)), 4),\n"
            "        'validation_f1': round(f1_score(y_val, (opt_val_scores >= 0.5).astype(int)), 4),\n"
            "        'validation_auc': round(roc_auc_score(y_val, opt_val_scores), 4),\n"
            "        'test_accuracy': round(accuracy_score(y_test, (opt_test_scores >= 0.5).astype(int)), 4),\n"
            "        'test_f1': round(f1_score(y_test, (opt_test_scores >= 0.5).astype(int)), 4),\n"
            "        'test_auc': round(roc_auc_score(y_test, opt_test_scores), 4),\n"
            "        'weights': best_weights,\n"
            "    },\n"
            "}\n"
            "\n"
            "# Add ensembles to comparison\n"
            "for name, metrics in ensemble_results.items():\n"
            "    results_df.loc[name] = {\n"
            "        k: v for k, v in metrics.items() if k != 'weights'\n"
            "    }\n"
            "\n"
            "# Pick the best overall model\n"
            "selected_model_name = results_df.sort_values(\n"
            "    by=['validation_auc', 'validation_f1'], ascending=False\n"
            ").index[0]\n"
            "\n"
            "print(f'Optimized ensemble weights: {best_weights}')\n"
            "print(f'Selected model: {selected_model_name}')\n"
            "results_df"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "print(classification_report(\n"
            "    y_test,\n"
            "    selected['test_pred'],\n"
            "    target_names=['Original / Genuine', 'Computer-Generated / Fake'],\n"
            "    digits=4,\n"
            "))"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
            "\n"
            "ConfusionMatrixDisplay.from_predictions(\n"
            "    y_test,\n"
            "    selected['test_pred'],\n"
            "    display_labels=['Original / Genuine', 'Computer-Generated / Fake'],\n"
            "    cmap='Blues',\n"
            "    colorbar=False,\n"
            "    ax=axes[0],\n"
            ")\n"
            "axes[0].set_title(f'Confusion Matrix ({selected_model_name})')\n"
            "\n"
            "fpr, tpr, _ = roc_curve(y_test, selected['test_score'])\n"
            "axes[1].plot(fpr, tpr, linewidth=2)\n"
            "axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray')\n"
            "axes[1].set_title(\n"
            "    f\"ROC Curve ({selected_model_name}, AUC = {roc_auc_score(y_test, selected['test_score']):.4f})\"\n"
            ")\n"
            "axes[1].set_xlabel('False Positive Rate')\n"
            "axes[1].set_ylabel('True Positive Rate')\n"
            "\n"
            "plt.tight_layout()"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "artifact = {\n"
            "    'format_version': 3,\n"
            "    'selected_model_name': selected_model_name,\n"
            "    'models': {name: payload['model'] for name, payload in results.items() if 'model' in payload},\n"
            "    'ensembles': {\n"
            "        'weighted_ensemble_fixed': {\n"
            "            'enabled': True,\n"
            "            'weights': FIXED_WEIGHTS,\n"
            "            'base_models': ['tfidf_xgboost', 'tfidf_logreg'],\n"
            "        },\n"
            "        'weighted_ensemble_optimized': {\n"
            "            'enabled': True,\n"
            "            'weights': best_weights,\n"
            "            'base_models': ['tfidf_xgboost', 'tfidf_logreg'],\n"
            "        },\n"
            "    },\n"
            "}\n"
            "joblib.dump(artifact, MODEL_DIR / 'fake_review_detector.joblib')\n"
            "with open(OUTPUT_DIR / 'metrics.json', 'w', encoding='utf-8') as f:\n"
            "    json.dump({\n"
            "        'selected_model': selected_model_name,\n"
            "        'models': results_df.to_dict(orient='index'),\n"
            "    }, f, indent=2)\n"
            "print(f'Saved model and metrics. Selected model: {selected_model_name}')"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "examples = [\n"
            "    'BEST PRODUCT EVER!!! AMAZING QUALITY!!! BUY NOW!!!',\n"
            "    'The item arrived in two days and matches the photos. The zipper feels solid.',\n"
            "    'Good product. Recommended.',\n"
            "]\n"
            "\n"
            "for text in examples:\n"
            "    probability = selected['model'].predict_proba([text])[0, 1]\n"
            "    if probability >= 0.6:\n"
            "        label = 'Likely computer-generated / fake'\n"
            "    elif probability <= 0.4:\n"
            "        label = 'Likely original / genuine'\n"
            "    else:\n"
            "        label = 'Borderline / uncertain'\n"
            "    print(f'\\nText: {text}')\n"
            "    print(f'Fake probability: {probability:.4f}')\n"
            "    print(f'Prediction: {label}')"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## Optional LLM comparison\n"
            "\n"
            "For a custom review, you can also compare the saved ML model with `Qwen/Qwen2.5-1.5B-Instruct`. "
            "The command below prints both results side by side.\n"
            "\n"
            "```powershell\n"
            "python predict_review.py --text \"BEST PRODUCT EVER!!! AMAZING QUALITY!!! BUY NOW!!!\"\n"
            "```"
        )
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
    }

    with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Notebook written to {NOTEBOOK_PATH.name}")


if __name__ == "__main__":
    main()
