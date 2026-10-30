"""Export the fine-tuned DistilBERT ticket classifiers to ONNX, quantize them to int8, and check that
quality holds on the held-out test split.

Why: the production host (Render free tier) has 512 MB RAM. PyTorch + two fp32 DistilBERT models need
well over 1 GB; onnxruntime (already used by fastembed) + two int8 models fit.

    pip install -e ".[export,test]"
    python scripts/export_onnx.py          # writes models/onnx/{queue,priority}/ and a report

Output per task: model.int8.onnx, tokenizer.json, meta.json (labels, temperature, max_length).
"""

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from supportpilot.classifier.onnx_model import OnnxClassifier

SRC = ROOT / "models" / "distilbert"
OUT = ROOT / "models" / "onnx"
TEST = ROOT / "data" / "processed" / "tickets" / "test.parquet"
REPORT = ROOT / "docs" / "results" / "onnx_export.json"
MAX_LEN = 256


def export(task: str) -> Path:
    src, out = SRC / task / "model", OUT / task
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(src).eval()
    sample = tok(["Our crawl fails with error 401"], return_tensors="pt")
    fp32 = out / "model.fp32.onnx"
    torch.onnx.export(
        model, (sample["input_ids"], sample["attention_mask"]), fp32,
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    # QUInt8 weights: queue macro-F1 0.639 vs 0.628 with QInt8 (fp32 0.652). Per-channel variants broke the
    # model (F1 ~0.01) and keeping the head in fp32 changed nothing; see docs/results/phase9_deploy.md
    quantize_dynamic(fp32, out / "model.int8.onnx", weight_type=QuantType.QUInt8)
    tok.save_pretrained(out / "_tok")
    shutil.copy(out / "_tok" / "tokenizer.json", out / "tokenizer.json")
    shutil.rmtree(out / "_tok")
    calib = SRC / task / "calibration.json"
    temperature = json.loads(calib.read_text())["temperature"] if calib.exists() else 1.0
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    (out / "meta.json").write_text(json.dumps({"labels": labels, "temperature": temperature, "max_length": MAX_LEN},
                                              indent=2))
    return out


def evaluate(task: str) -> dict:
    test = pd.read_parquet(TEST)
    ref = pd.read_parquet(SRC / task / "test_predictions.parquet")
    ref_logits = np.load(SRC / task / "logits_test.npy")
    ref = ref.assign(row=range(len(ref))).set_index("ticket_id")
    test = test[test["ticket_id"].isin(ref.index)]
    out = {"task": task, "test_rows": len(test)}
    for variant in ("fp32", "int8"):
        clf = OnnxClassifier(OUT / task, model_file=f"model.{variant}.onnx")
        t0 = time.perf_counter()
        logits = np.vstack([clf.logits(t) for t in test["text"]])
        per_ticket_ms = (time.perf_counter() - t0) / len(test) * 1000
        pred = [clf.labels[i] for i in logits.argmax(1)]
        gold = test[task].tolist()
        torch_pred = ref.loc[test["ticket_id"], "pred"].tolist()  # the original PyTorch model's predictions
        out[variant] = {
            "size_mb": round((OUT / task / f"model.{variant}.onnx").stat().st_size / 1e6, 1),
            "accuracy": round(accuracy_score(gold, pred), 4),
            "macro_f1": round(f1_score(gold, pred, average="macro"), 4),
            "agrees_with_pytorch": round(float(np.mean(np.array(pred) == np.array(torch_pred))), 4),
            "max_logit_diff_vs_pytorch": round(float(np.abs(
                logits - ref_logits[ref.loc[test["ticket_id"], "row"].to_numpy()]).max()), 3),
            "cpu_ms_per_ticket": round(per_ticket_ms, 1),
        }
    out["pytorch_reference"] = {"macro_f1": round(f1_score(test[task], ref.loc[test["ticket_id"], "pred"],
                                                           average="macro"), 4)}
    return out


def main():
    results = []
    for task in ("queue", "priority"):
        export(task)
        r = evaluate(task)
        (OUT / task / "model.fp32.onnx").unlink()  # only the int8 model ships
        results.append(r)
        print(json.dumps(r, indent=2))
    REPORT.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
