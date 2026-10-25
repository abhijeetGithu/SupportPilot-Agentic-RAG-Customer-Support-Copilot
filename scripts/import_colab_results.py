"""Import the DistilBERT results trained on Colab into local MLflow and compare with the baseline.

    python scripts/import_colab_results.py [--src models/distilbert]

Expects, per task (queue, priority):  <src>/<task>/model/, report.json, test_predictions.parquet
(the folder the Colab notebook saved to Google Drive).

- checks the split fingerprints match the local baseline
- logs one MLflow run per task (params, metrics, confusion matrix, report)
- measures CPU inference latency locally (what production will see)
- writes evals/reports/classifier/comparison.md
"""

import argparse
import json
import time
from pathlib import Path

import mlflow
import pandas as pd

from supportpilot.classifier.baseline import EXPERIMENT, REPORT_DIR, save_confusion
from supportpilot.classifier.data import TASKS
from supportpilot.config import get_settings


def cpu_latency_ms(model_dir: Path, texts: list[str]) -> float:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    with torch.no_grad():
        model(**tok(texts[0], truncation=True, max_length=256, return_tensors="pt"))  # warm-up
        t0 = time.perf_counter()
        for t in texts:
            model(**tok(t, truncation=True, max_length=256, return_tensors="pt"))
    return (time.perf_counter() - t0) * 1000 / len(texts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path("models/distilbert"))
    p.add_argument("--latency-samples", type=int, default=100)
    args = p.parse_args()

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    test = pd.read_parquet("data/processed/tickets/test.parquet")
    rows = []
    for task in TASKS:
        task_dir = args.src / task
        report = json.loads((task_dir / "report.json").read_text(encoding="utf-8"))
        base = json.loads((REPORT_DIR / f"baseline_{task}.json").read_text(encoding="utf-8"))
        if report["fingerprints"] != base["fingerprints"]:
            raise SystemExit(f"{task}: Colab split {report['fingerprints']} != baseline {base['fingerprints']}")

        preds = pd.read_parquet(task_dir / "test_predictions.parquet")
        cm_path = REPORT_DIR / f"distilbert_{task}_confusion.png"
        save_confusion(preds[task], preds["pred"], report["labels"], cm_path, f"DistilBERT: {task} (test, row-normalized)")
        latency = cpu_latency_ms(task_dir / "model", test["text"].head(args.latency_samples).tolist())

        with mlflow.start_run(run_name=f"distilbert-{task}"):
            mlflow.set_tags({"task": task, "model": "distilbert", "gpu": report["gpu"],
                             "data_fingerprint_test": report["fingerprints"]["test"]})
            mlflow.log_params({**report["hparams"], "base_model": report["model"], "max_len": report["max_len"],
                               "epochs_run": report["epochs_run"]})
            mlflow.log_metrics({
                "val_macro_f1": report["val_macro_f1"],
                "test_macro_f1": report["test"]["macro_f1"],
                "test_accuracy": report["test"]["accuracy"],
                "test_weighted_f1": report["test"]["weighted_f1"],
                "train_seconds": report["train_seconds"],
                "inference_ms_per_ticket_cpu": round(latency, 2),
            })
            for step, entry in enumerate(e for e in report["log_history"] if "eval_macro_f1" in e):
                mlflow.log_metric("val_macro_f1_by_epoch", entry["eval_macro_f1"], step=step + 1)
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(task_dir / "report.json"))

        rows.append({
            "task": task,
            "baseline_macro_f1": base["test"]["macro_f1"],
            "distilbert_macro_f1": report["test"]["macro_f1"],
            "baseline_acc": base["test"]["accuracy"],
            "distilbert_acc": report["test"]["accuracy"],
            "distilbert_cpu_ms": round(latency, 1),
            "train_min_gpu": round(report["train_seconds"] / 60, 1),
        })

    lines = [
        "| Task | TF-IDF macro-F1 | DistilBERT macro-F1 | Δ | TF-IDF acc | DistilBERT acc | DistilBERT CPU ms/ticket | GPU train min |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        delta = r["distilbert_macro_f1"] - r["baseline_macro_f1"]
        lines.append(
            f"| {r['task']} | {r['baseline_macro_f1']:.3f} | **{r['distilbert_macro_f1']:.3f}** | {delta:+.3f} | "
            f"{r['baseline_acc']:.3f} | {r['distilbert_acc']:.3f} | {r['distilbert_cpu_ms']} | {r['train_min_gpu']} |"
        )
    (REPORT_DIR / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
