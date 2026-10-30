"""Zip the int8 ONNX ticket classifiers for a GitHub release (the Docker build downloads it).

    python scripts/package_models.py      # -> dist/classifiers-onnx-int8.zip
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models" / "onnx"
OUT = ROOT / "dist" / "classifiers-onnx-int8.zip"
FILES = ("model.int8.onnx", "tokenizer.json", "meta.json")

OUT.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for task in ("queue", "priority"):
        for name in FILES:
            z.write(SRC / task / name, f"{task}/{name}")
print(f"{OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")
