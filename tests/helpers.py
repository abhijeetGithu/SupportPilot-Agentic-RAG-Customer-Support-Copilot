"""Test doubles shared across tests."""

import re

from supportpilot.classifier.predict import TicketPrediction


def fake_classifier(text: str) -> TicketPrediction:
    """Deterministic stand-in for DistilBERT: keyword rules, confidence low for vague text."""
    t = text.lower()
    if "charged" in t or "invoice" in t:
        return TicketPrediction(category="Billing and Payments", priority="high", confidence=0.9,
                                priority_confidence=0.8, model="fake")
    if "outage" in t:
        return TicketPrediction(category="Service Outages and Maintenance", priority="urgent", confidence=0.8,
                                priority_confidence=0.7, model="fake", urgent_rule=True)
    return TicketPrediction(category="Technical Support", priority="medium", confidence=0.3,
                            priority_confidence=0.6, model="fake")


def word_overlap_similarity(a: str, b: str) -> float:
    """Jaccard over words (punctuation ignored); stands in for embedding cosine similarity."""
    wa, wb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(wa & wb) / len(wa | wb)
