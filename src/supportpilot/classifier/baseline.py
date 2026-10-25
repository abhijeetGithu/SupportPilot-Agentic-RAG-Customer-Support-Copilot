"""TF-IDF + Logistic Regression baseline for queue and priority, logged to MLflow.

    python -m supportpilot.classifier.baseline

Saves splits to data/processed/tickets/, the models to models/baseline/, and per-task
reports (metrics, confusion matrix, test predictions) to evals/reports/classifier/.
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import Pipeline

from supportpilot.classifier.data import TASKS, fingerprint, load_tickets, split
from supportpilot.config import get_settings

DATA_DIR = Path("data/processed/tickets")
MODEL_DIR = Path("models/baseline")
REPORT_DIR = Path("evals/reports/classifier")
EXPERIMENT = "ticket-classifier"

PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_features": 200_000,
    "sublinear_tf": True,
    "C": 4.0,
    "class_weight": "balanced",
}


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=PARAMS["ngram_range"],
                    min_df=PARAMS["min_df"],
                    max_features=PARAMS["max_features"],
                    sublinear_tf=PARAMS["sublinear_tf"],
                ),
            ),
            ("clf", LogisticRegression(C=PARAMS["C"], class_weight=PARAMS["class_weight"], max_iter=2000)),
        ]
    )


def evaluate(y_true, y_pred, labels: list[str]) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
    }


def save_confusion(y_true, y_pred, labels: list[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, normalize="true", values_format=".2f", ax=ax, xticks_rotation=45, colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    df = load_tickets()
    train, val, test = split(df)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_parquet(DATA_DIR / f"{name}.parquet", index=False)
    fps = {name: fingerprint(part) for name, part in (("train", train), ("val", val), ("test", test))}
    print(f"rows after dedup {len(df)} | train {len(train)} val {len(val)} test {len(test)} | fingerprints {fps}")

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for task in TASKS:
        labels = sorted(train[task].unique())
        with mlflow.start_run(run_name=f"tfidf-logreg-{task}"):
            mlflow.set_tags({"task": task, "model": "tfidf-logreg", "data_fingerprint_test": fps["test"]})
            mlflow.log_params({**{k: str(v) for k, v in PARAMS.items()}, "n_train": len(train)})
            pipe = build_pipeline()
            t0 = time.perf_counter()
            pipe.fit(train["text"], train[task])
            train_s = time.perf_counter() - t0

            val_m = evaluate(val[task], pipe.predict(val["text"]), labels)
            t0 = time.perf_counter()
            test_pred = pipe.predict(test["text"])
            ms_per_ticket = (time.perf_counter() - t0) * 1000 / len(test)
            test_m = evaluate(test[task], test_pred, labels)

            mlflow.log_metrics(
                {
                    "val_macro_f1": val_m["macro_f1"],
                    "test_macro_f1": test_m["macro_f1"],
                    "test_accuracy": test_m["accuracy"],
                    "test_weighted_f1": test_m["weighted_f1"],
                    "train_seconds": round(train_s, 1),
                    "inference_ms_per_ticket": round(ms_per_ticket, 3),
                }
            )
            cm_path = REPORT_DIR / f"baseline_{task}_confusion.png"
            save_confusion(test[task], test_pred, labels, cm_path, f"TF-IDF + LogReg: {task} (test, row-normalized)")
            report = {"task": task, "model": "tfidf-logreg", "fingerprints": fps, "val": val_m, "test": test_m}
            report_path = REPORT_DIR / f"baseline_{task}.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            test.assign(pred=test_pred)[["ticket_id", "text", task, "pred"]].to_parquet(
                REPORT_DIR / f"baseline_{task}_test_predictions.parquet", index=False
            )
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(report_path))
            joblib.dump(pipe, MODEL_DIR / f"{task}.joblib")
            print(
                f"{task:9s} val macro-F1 {val_m['macro_f1']:.3f} | test macro-F1 {test_m['macro_f1']:.3f} "
                f"acc {test_m['accuracy']:.3f} | train {train_s:.0f}s | {ms_per_ticket:.2f} ms/ticket"
            )


if __name__ == "__main__":
    main()
