"""Ticket dataset: load, clean, deduplicate, split. Single source of truth for the local baseline
AND the Colab notebook (scripts/build_colab_notebook.py embeds this file), so both are trained and
tested on exactly the same rows. `fingerprint()` lets the notebook verify that.

Dataset: Tobi-Bueck/customer-support-tickets (Hugging Face, CC BY-NC 4.0), English rows only.
Labels: `queue` (10 routing categories) and `priority` (low / medium / high).

Leakage guard: 16% of tickets are exact copies of another ticket. Duplicates are removed by
normalized body BEFORE splitting, so no test ticket has a copy in training.
"""

import hashlib
import re

import pandas as pd
from sklearn.model_selection import train_test_split

DATASET = "Tobi-Bueck/customer-support-tickets"
SEED = 42
TASKS = ("queue", "priority")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_tickets() -> pd.DataFrame:
    from datasets import load_dataset

    raw = load_dataset(DATASET, split="train").to_pandas()
    df = raw[raw["language"] == "en"].copy()
    df["ticket_id"] = df.index.astype(int)  # row index in the source dataset, stable across runs
    df["subject"] = df["subject"].fillna("").str.strip()
    df["body"] = df["body"].fillna("").str.strip()
    df = df[df["body"].str.len() > 0]
    df["text"] = (df["subject"] + "\n\n" + df["body"]).str.strip()
    df = df[~df["body"].map(_norm).duplicated(keep="first")]
    return df[["ticket_id", "text", "queue", "priority"]].reset_index(drop=True)


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """80 / 10 / 10, stratified by queue (the more imbalanced label)."""
    train, rest = train_test_split(df, test_size=0.2, stratify=df["queue"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["queue"], random_state=SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def fingerprint(df: pd.DataFrame) -> str:
    """Short hash of the ticket ids in a split, to check two environments use identical splits."""
    ids = ",".join(map(str, sorted(df["ticket_id"])))
    return hashlib.sha256(ids.encode()).hexdigest()[:12]
