"""Compare DistilBERT with the TF-IDF baseline and calibrate DistilBERT's confidence.

    python scripts/analyze_classifier.py

Per task (queue, priority):
1. Run the DistilBERT model locally on val + test (cached logits); check agreement with Colab's predictions.
2. Paired bootstrap (1,000 resamples of the same test tickets): 95% CI of the macro-F1 gain over the baseline.
3. Per-class F1 for both models and DistilBERT's most frequent confusions.
4. Calibration: expected calibration error (ECE) before/after temperature scaling fitted on val.
   The temperature is saved to models/distilbert/<task>/calibration.json and used by predict.py.
5. The "needs review" rule (confidence < threshold): how many tickets get flagged, and accuracy of the rest.

Writes evals/reports/classifier/analysis.json and analysis.md.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from supportpilot.classifier.data import TASKS
from supportpilot.config import get_settings

DATA = Path("data/processed/tickets")
REPORTS = Path("evals/reports/classifier")
MODELS = Path(get_settings().classifier_model_dir)
BOOTSTRAP = 1000
SEED = 7


def logits_for(task: str, split: str, texts: list[str]) -> np.ndarray:
    cache = MODELS / task / f"logits_{split}.npy"
    if cache.exists():
        return np.load(cache)
    tok = AutoTokenizer.from_pretrained(MODELS / task / "model")
    model = AutoModelForSequenceClassification.from_pretrained(MODELS / task / "model").eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tok(texts[i : i + 32], truncation=True, max_length=256, padding=True, return_tensors="pt")
            out.append(model(**enc).logits.numpy())
    arr = np.concatenate(out)
    np.save(cache, arr)
    return arr


def softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = z / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in itertools.pairwise(edges):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature minimizing negative log-likelihood on validation (grid search, no gradients needed)."""
    best_t, best_nll = 1.0, np.inf
    for t in np.arange(0.5, 10.01, 0.05):
        p = softmax(logits, t)[np.arange(len(y)), y]
        nll = -np.log(np.clip(p, 1e-12, None)).mean()
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return round(best_t, 2)


def bootstrap_gain(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    """Paired bootstrap of macro-F1(b) - macro-F1(a) on the same resampled tickets."""
    rng = np.random.default_rng(SEED)
    n, diffs = len(y), []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, n, n)
        diffs.append(f1_score(y[idx], b[idx], average="macro") - f1_score(y[idx], a[idx], average="macro"))
    d = np.array(diffs)
    return {
        "gain": round(f1_score(y, b, average="macro") - f1_score(y, a, average="macro"), 4),
        "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
        "p_no_gain": round(float((d <= 0).mean()), 4),
    }


def review_rule(probs: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    conf, pred = probs.max(1), probs.argmax(1)
    flagged = conf < threshold
    return {
        "flagged_share": round(float(flagged.mean()), 3),
        "accuracy_auto_routed": round(float((pred[~flagged] == y[~flagged]).mean()), 3),
        "accuracy_flagged": round(float((pred[flagged] == y[flagged]).mean()), 3) if flagged.any() else None,
    }


def main() -> None:
    val = pd.read_parquet(DATA / "val.parquet")
    test = pd.read_parquet(DATA / "test.parquet")
    threshold = get_settings().review_confidence_threshold
    results, md = {}, ["# Ticket classifier analysis (DistilBERT run 2 vs TF-IDF baseline)\n"]

    for task in TASKS:
        report = json.loads((MODELS / task / "report.json").read_text(encoding="utf-8"))
        labels = report["labels"]
        lid = {label: i for i, label in enumerate(labels)}
        y_val = val[task].map(lid).to_numpy()
        y = test[task].map(lid).to_numpy()

        lv = logits_for(task, "val", val["text"].tolist())
        lt = logits_for(task, "test", test["text"].tolist())
        bert = lt.argmax(1)
        colab = pd.read_parquet(MODELS / task / "test_predictions.parquet").set_index("ticket_id")
        agree = float((colab.loc[test["ticket_id"], "pred"].map(lid).to_numpy() == bert).mean())

        base = pd.read_parquet(REPORTS / f"baseline_{task}_test_predictions.parquet").set_index("ticket_id")
        base_pred = base.loc[test["ticket_id"], "pred"].map(lid).to_numpy()
        boot = bootstrap_gain(y, base_pred, bert)

        f1_bert = f1_score(y, bert, average=None, labels=range(len(labels)))
        f1_base = f1_score(y, base_pred, average=None, labels=range(len(labels)))
        support = np.bincount(y, minlength=len(labels))
        per_class = [
            {"label": lab, "n": int(support[i]), "baseline_f1": round(float(f1_base[i]), 3),
             "distilbert_f1": round(float(f1_bert[i]), 3), "delta": round(float(f1_bert[i] - f1_base[i]), 3)}
            for i, lab in enumerate(labels)
        ]
        cm = pd.crosstab(pd.Series(y, name="true"), pd.Series(bert, name="pred"))
        confusions = sorted(
            ((labels[t], labels[p], int(cm.loc[t, p]), round(cm.loc[t, p] / support[t], 3))
             for t in cm.index for p in cm.columns if t != p and cm.loc[t, p] > 0),
            key=lambda x: -x[2],
        )[:6]

        temp = fit_temperature(lv, y_val)
        p_raw, p_cal = softmax(lt), softmax(lt, temp)
        calib = {
            "temperature": temp,
            "ece_before": round(ece(p_raw, y), 4),
            "ece_after": round(ece(p_cal, y), 4),
            "mean_confidence_before": round(float(p_raw.max(1).mean()), 3),
            "mean_confidence_after": round(float(p_cal.max(1).mean()), 3),
            "accuracy": round(float((bert == y).mean()), 3),
        }
        (MODELS / task / "calibration.json").write_text(json.dumps({"temperature": temp}), encoding="utf-8")
        review = {"before": review_rule(p_raw, y, threshold), "after": review_rule(p_cal, y, threshold)}

        results[task] = {"colab_local_agreement": round(agree, 4), "bootstrap": boot, "per_class": per_class,
                         "top_confusions": confusions, "calibration": calib, "review_rule": review}

        md += [
            f"\n## {task}\n",
            f"- Local CPU predictions agree with Colab GPU predictions on {agree:.1%} of test tickets.",
            (f"- Macro-F1 gain over TF-IDF: **{boot['gain']:+.3f}**, 95% CI [{boot['ci95'][0]:+.3f}, {boot['ci95'][1]:+.3f}], "
            f"P(no gain) = {boot['p_no_gain']:.3f} ({BOOTSTRAP} paired bootstrap resamples)."),
            (f"- Calibration: temperature {temp}; ECE {calib['ece_before']:.3f} → {calib['ece_after']:.3f}; "
            f"mean confidence {calib['mean_confidence_before']:.2f} → {calib['mean_confidence_after']:.2f} "
            f"(accuracy {calib['accuracy']:.2f})."),
            (f"- Needs-review rule (confidence < {threshold}): before calibration flags {review['before']['flagged_share']:.1%} "
            f"(auto-routed accuracy {review['before']['accuracy_auto_routed']:.2f}); after calibration flags "
            f"{review['after']['flagged_share']:.1%} (auto-routed accuracy {review['after']['accuracy_auto_routed']:.2f}, "
            f"flagged accuracy {review['after']['accuracy_flagged']}).\n"),
            "| Class | n | TF-IDF F1 | DistilBERT F1 | Δ |",
            "|---|---|---|---|---|",
            *[f"| {c['label']} | {c['n']} | {c['baseline_f1']:.2f} | {c['distilbert_f1']:.2f} | {c['delta']:+.2f} |"
              for c in sorted(per_class, key=lambda c: -c["n"])],
            "\nMost frequent DistilBERT confusions (true → predicted, count, share of true class):\n",
            *[f"- {t} → {p}: {n} ({s:.0%})" for t, p, n, s in confusions],
        ]

    (REPORTS / "analysis.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (REPORTS / "analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
