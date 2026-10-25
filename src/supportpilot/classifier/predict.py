"""Ticket classification used by TicketService.

Loads the fine-tuned DistilBERT models from CLASSIFIER_MODEL_DIR (one per task) if present,
otherwise falls back to the TF-IDF baseline. Models are loaded once and reused.

Priority: the dataset has low / medium / high. "urgent" is set by rules on top, because a missed
security or outage ticket costs far more than a false alarm.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

from supportpilot.config import get_settings

URGENT = re.compile(
    r"\b(security (breach|incident|vulnerability)|data (breach|leak)|hacked|ransomware|"
    r"(complete|total|full) outage|(system|service|site|platform) (is )?down|legal action|lawsuit)\b",
    re.IGNORECASE,
)


class TicketPrediction(BaseModel):
    category: str
    priority: str  # low | medium | high | urgent
    confidence: float  # category confidence
    priority_confidence: float
    model: str  # distilbert | tfidf-logreg
    urgent_rule: bool = False


class _Task:
    def __init__(self, task: str):
        model_dir = Path(get_settings().classifier_model_dir) / task / "model"
        if model_dir.exists():
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.kind = "distilbert"
            self.tok = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
            self._torch = torch
            # temperature scaling fitted on validation (scripts/analyze_classifier.py); 1.0 = uncalibrated
            calib = model_dir.parent / "calibration.json"
            self.temperature = json.loads(calib.read_text())["temperature"] if calib.exists() else 1.0
        else:
            import joblib

            self.kind = "tfidf-logreg"
            self.model = joblib.load(Path("models/baseline") / f"{task}.joblib")

    def predict(self, text: str) -> tuple[str, float]:
        if self.kind == "distilbert":
            with self._torch.no_grad():
                logits = self.model(**self.tok(text, truncation=True, max_length=256, return_tensors="pt")).logits
            probs = self._torch.softmax(logits / self.temperature, -1)[0]
            i = int(probs.argmax())
            return self.model.config.id2label[i], float(probs[i])
        probs = self.model.predict_proba([text])[0]
        i = int(probs.argmax())
        return str(self.model.classes_[i]), float(probs[i])


@lru_cache
def _task(task: str) -> _Task:
    return _Task(task)


def classify_ticket(text: str) -> TicketPrediction:
    category, conf = _task("queue").predict(text)
    priority, p_conf = _task("priority").predict(text)
    urgent = bool(URGENT.search(text))
    return TicketPrediction(
        category=category,
        priority="urgent" if urgent else priority,
        confidence=round(conf, 4),
        priority_confidence=round(p_conf, 4),
        model=_task("queue").kind,
        urgent_rule=urgent,
    )
