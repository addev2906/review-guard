from __future__ import annotations

import argparse
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fake_review_detector.joblib"
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROMO_PATTERN = re.compile(
    r"\b(best|amazing|perfect|must buy|buy now|highly recommend|life[- ]changing)\b",
    re.IGNORECASE,
)


def confidence_band(probability: float) -> str:
    if probability >= 0.85 or probability <= 0.15:
        return "high confidence"
    if probability >= 0.65 or probability <= 0.35:
        return "moderate confidence"
    return "uncertain"


def human_label(probability: float) -> str:
    if probability >= 0.6:
        return "Likely computer-generated / fake"
    if probability <= 0.4:
        return "Likely original / genuine"
    return "Borderline / uncertain"


def explain_text(model, text: str, top_n: int = 8) -> list[str]:
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["clf"]
    features = vectorizer.transform([text])
    coefs = classifier.coef_[0]
    nz = features.nonzero()[1]
    if len(nz) == 0:
        return []

    names = vectorizer.get_feature_names_out()
    scored = sorted(
        ((names[i], features[0, i] * coefs[i]) for i in nz),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    return [f"{term} ({value:+.3f})" for term, value in scored[:top_n]]


def suspicious_flags(text: str) -> list[str]:
    flags = []
    words = text.split()
    upper_words = [word for word in words if len(word) > 2 and word.isupper()]

    if text.count("!") >= 3:
        flags.append("heavy exclamation use")
    if len(upper_words) >= 2:
        flags.append("multiple all-caps words")
    if PROMO_PATTERN.search(text):
        flags.append("contains promotional superlatives")
    if len(words) <= 4:
        flags.append("very short review")

    return flags


def ml_prediction(model, text: str) -> dict[str, Any]:
    probability = float(model.predict_proba(pd.Series([text]))[0, 1])
    return {
        "label": human_label(probability),
        "fake_probability": round(probability, 4),
        "confidence_band": confidence_band(probability),
        "top_terms": explain_text(model, text),
        "flags": suspicious_flags(text),
    }


@lru_cache(maxsize=2)
def load_llm(model_name: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing LLM dependencies. Run `python -m pip install -r requirements.txt` first."
        ) from exc

    token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if torch.cuda.is_available() else None
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        dtype=dtype,
        device_map=device_map,
    )
    return tokenizer, model


def parse_llm_output(output_text: str) -> dict[str, Any]:
    cleaned = output_text.strip()

    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            verdict = str(parsed.get("verdict", "")).strip().upper()
            confidence = float(parsed.get("confidence", 0.5))
            reason = str(parsed.get("reason", "")).strip()
        except (json.JSONDecodeError, ValueError, TypeError):
            verdict = ""
            confidence = 0.5
            reason = cleaned
    else:
        verdict = ""
        confidence = 0.5
        reason = cleaned

    if verdict not in {"FAKE", "GENUINE", "UNCERTAIN"}:
        upper = cleaned.upper()
        if "FAKE" in upper and "GENUINE" not in upper:
            verdict = "FAKE"
        elif "GENUINE" in upper and "FAKE" not in upper:
            verdict = "GENUINE"
        else:
            verdict = "UNCERTAIN"

    confidence = max(0.0, min(1.0, confidence))
    if verdict == "FAKE":
        fake_probability = confidence
        label = "Likely computer-generated / fake"
    elif verdict == "GENUINE":
        fake_probability = 1 - confidence
        label = "Likely original / genuine"
    else:
        fake_probability = 0.5
        label = "Borderline / uncertain"

    return {
        "label": label,
        "fake_probability": round(fake_probability, 4),
        "confidence_band": confidence_band(fake_probability),
        "reason": reason if reason else "No reason returned by the LLM.",
        "raw_output": cleaned,
    }


def score_with_llm(text: str, model_name: str) -> dict[str, Any]:
    tokenizer, model = load_llm(model_name)

    messages = [
        {
            "role": "system",
            "content": (
                "You classify product reviews as FAKE, GENUINE, or UNCERTAIN. "
                "Return only valid JSON with keys: verdict, confidence, reason. "
                "Use confidence as a decimal between 0 and 1."
            ),
        },
        {
            "role": "user",
            "content": (
                "Review the product review below and decide whether it looks fake.\n\n"
                f"Review:\n{text}\n\n"
                "Output example:\n"
                '{"verdict":"FAKE","confidence":0.78,"reason":"Promotional language with little detail."}'
            ),
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")

    if hasattr(model, "device"):
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

    output = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        temperature=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = output[0][inputs["input_ids"].shape[-1] :]
    text_output = tokenizer.decode(generated, skip_special_tokens=True)

    parsed = parse_llm_output(text_output)
    parsed["model_name"] = model_name
    return parsed


def compare_models(ml_result: dict[str, Any], llm_result: dict[str, Any]) -> dict[str, Any]:
    difference = abs(ml_result["fake_probability"] - llm_result["fake_probability"])
    agree = ml_result["label"] == llm_result["label"]

    if agree and difference <= 0.2:
        summary = "Both models agree strongly."
        final_label = ml_result["label"]
    elif agree:
        summary = "Both models point in the same direction, but with different confidence."
        final_label = ml_result["label"]
    else:
        summary = "The ML model and LLM disagree. Treat this review as a manual-review case."
        final_label = "Manual review recommended"

    return {
        "agreement": agree,
        "probability_gap": round(difference, 4),
        "final_label": final_label,
        "summary": summary,
    }


def save_report(payload: dict[str, Any]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    report_path = OUTPUT_DIR / "last_prediction_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return report_path


def print_side_by_side_comparison(
    ml_result: dict[str, Any], llm_result: dict[str, Any], comparison: dict[str, Any]
) -> None:
    rows = [
        ("Label", ml_result["label"], llm_result["label"]),
        (
            "Fake probability",
            f"{ml_result['fake_probability']:.4f}",
            f"{llm_result['fake_probability']:.4f}",
        ),
        ("Confidence", ml_result["confidence_band"], llm_result["confidence_band"]),
    ]

    left_title = "ML model"
    right_title = "LLM"
    metric_width = max(len(metric) for metric, _, _ in rows + [("Metric", "", "")])
    left_width = max(len(left_title), *(len(left) for _, left, _ in rows))
    right_width = max(len(right_title), *(len(right) for _, _, right in rows))

    divider = (
        f"+-{'-' * metric_width}-+-{'-' * left_width}-+-{'-' * right_width}-+"
    )
    print("Side-by-side comparison")
    print(divider)
    print(
        f"| {'Metric'.ljust(metric_width)} | {left_title.ljust(left_width)} | {right_title.ljust(right_width)} |"
    )
    print(divider)
    for metric, left, right in rows:
        print(
            f"| {metric.ljust(metric_width)} | {left.ljust(left_width)} | {right.ljust(right_width)} |"
        )
    print(divider)
    print(f"Final label: {comparison['final_label']}")
    print(f"Agreement: {comparison['agreement']}")
    print(f"Probability gap: {comparison['probability_gap']:.4f}")
    print(f"Summary: {comparison['summary']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict whether a review appears fake using the ML model and an optional open-source LLM comparison."
    )
    parser.add_argument("--text", help="Review text to score")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip the open-source LLM and run only the ML model",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help=f"Hugging Face model to use for the LLM scorer (default: {DEFAULT_LLM_MODEL})",
    )
    args = parser.parse_args()
    review_text = args.text

    if not review_text:
        print("Enter the review text below, then press Enter.")
        review_text = input("Review: ").strip()
        if not review_text:
            raise SystemExit("No review text was provided.")

    model = joblib.load(MODEL_PATH)
    ml_result = ml_prediction(model, review_text)

    print("ML model prediction")
    print(f"  Label: {ml_result['label']}")
    print(f"  Fake probability: {ml_result['fake_probability']:.4f}")
    print(f"  Confidence: {ml_result['confidence_band']}")
    if ml_result["top_terms"]:
        print("  Top contributing terms:")
        for item in ml_result["top_terms"]:
            print(f"    {item}")
    if ml_result["flags"]:
        print("  Suspicious writing flags:")
        for item in ml_result["flags"]:
            print(f"    {item}")

    report: dict[str, Any] = {
        "review_text": review_text,
        "ml_model": ml_result,
    }

    if not args.skip_llm:
        try:
            llm_result = score_with_llm(review_text, args.llm_model)
            comparison = compare_models(ml_result, llm_result)
            report["llm_model"] = llm_result
            report["comparison"] = comparison

            print("LLM prediction")
            print(f"  Model: {llm_result['model_name']}")
            print(f"  Reason: {llm_result['reason']}")
            print_side_by_side_comparison(ml_result, llm_result, comparison)
        except Exception as exc:
            report["llm_error"] = str(exc)
            print("LLM prediction")
            print("  The LLM comparison could not run.")
            print(f"  Reason: {exc}")
            print("  Tip: ensure dependencies are installed and that the model is either cached locally or can be downloaded from Hugging Face on first run.")

    report_path = save_report(report)
    print(f"Saved report to: {report_path.name}")


if __name__ == "__main__":
    main()
