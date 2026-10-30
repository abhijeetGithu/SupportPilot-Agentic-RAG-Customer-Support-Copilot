"""DistilBERT ticket classifier on onnxruntime (int8): the production path, no PyTorch needed.

Files per task (scripts/export_onnx.py): model.int8.onnx, tokenizer.json, meta.json
(labels, temperature from validation-set temperature scaling, max_length).
"""

import json
from pathlib import Path

import numpy as np


class OnnxClassifier:
    def __init__(self, folder: Path, model_file: str = "model.int8.onnx"):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        meta = json.loads((folder / "meta.json").read_text())
        self.labels: list[str] = meta["labels"]
        self.temperature: float = meta.get("temperature", 1.0)
        self.tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tok.enable_truncation(max_length=meta.get("max_length", 256))
        self.tok.no_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # small, predictable memory on a shared CPU
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = False  # one request at a time: the arena only costs memory (~26 MB/model)
        opts.enable_mem_pattern = False
        self.session = ort.InferenceSession(str(folder / model_file), opts, providers=["CPUExecutionProvider"])

    def logits(self, text: str) -> np.ndarray:
        enc = self.tok.encode(text)
        feeds = {"input_ids": np.array([enc.ids], dtype=np.int64),
                 "attention_mask": np.array([enc.attention_mask], dtype=np.int64)}
        return self.session.run(["logits"], feeds)[0][0]

    def predict(self, text: str) -> tuple[str, float]:
        z = self.logits(text) / self.temperature
        p = np.exp(z - z.max())
        p /= p.sum()
        i = int(p.argmax())
        return self.labels[i], float(p[i])
