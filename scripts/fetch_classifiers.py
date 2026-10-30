"""Download the int8 ONNX ticket classifiers (a GitHub release asset) into models/onnx. Used by the Dockerfile.

    python scripts/fetch_classifiers.py <zip url>
"""

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

url = sys.argv[1]
target = Path("models/onnx")
data = urllib.request.urlopen(url, timeout=300).read()
zipfile.ZipFile(io.BytesIO(data)).extractall(target)
found = sorted(p.parent.name for p in target.glob("*/model.int8.onnx"))
if found != ["priority", "queue"]:
    sys.exit(f"expected queue/ and priority/ int8 models in the zip, found {found}")
print(f"classifiers ready: {found} ({len(data) / 1e6:.0f} MB)")
