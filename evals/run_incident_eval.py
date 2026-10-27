"""Replay synthetic ticket traffic through the incident detector and score it.

Traffic (evals/datasets/incident_sim.json + golden questions), over 3 days per seed:
  - background: 150 real SearchUnify questions, uniformly spread, each from a different customer
  - slow topic: a popular how-to question asked 6 times across the 3 days (must NOT open an incident)
  - near-misses: 3 same-area but different questions within 45 minutes (must NOT open an incident)
  - incidents: 7 reports of the same outage, exponential inter-arrival (mean 8 min), different customers

Scores per seed, then averaged: incidents detected, false incidents, reports needed / minutes until
detection, and membership precision / recall. No LLM calls (embeddings only).

    python evals/run_incident_eval.py [--seeds 5] [--sweep]
"""

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sqlalchemy import select

from supportpilot.config import get_settings
from supportpilot.db import base
from supportpilot.db.models import Incident, Ticket
from supportpilot.incidents import detector

DATA = ROOT / "evals" / "datasets" / "incident_sim.json"
REPORT = ROOT / "evals" / "reports" / "incident_eval.json"
START = datetime(2026, 10, 20)
DAYS = 3


def traffic(seed: int, data: Path = DATA) -> list[dict]:
    rng = random.Random(seed)
    d = json.loads(data.read_text())
    golden = [json.loads(line)["question"] for f in ("golden.jsonl", "golden_hard.jsonl")
              for line in (ROOT / "evals" / "datasets" / f).read_text(encoding="utf-8").splitlines()]
    span = DAYS * 24 * 60
    at = lambda minute: START + timedelta(minutes=minute)
    items = [{"text": q, "label": "background", "t": at(rng.uniform(0, span))} for q in rng.sample(golden, 150)]
    items += [{"text": q, "label": "slow_topic", "t": at(rng.uniform(0, span))} for q in d["slow_topic"]["reports"]]
    for g in d["near_misses"]:
        t0 = rng.uniform(0, span - 60)
        items += [{"text": q, "label": f"near:{g['id']}", "t": at(t0 + rng.uniform(0, 45))} for q in g["reports"]]
    for inc in d["incidents"]:
        t = rng.uniform(6 * 60, span - 180)  # leave some history before each incident
        reports = inc["reports"][:]
        rng.shuffle(reports)
        for q in reports:
            items.append({"text": q, "label": inc["id"], "t": at(t)})
            t += rng.expovariate(1 / 8)
    items.sort(key=lambda x: x["t"])
    for i, it in enumerate(items):
        it["email"] = f"customer{i}@example{i % 97}.com"
    return items


def run(items: list[dict], vectors) -> dict:
    base.configure("sqlite://")
    base.init_db()
    truth = {lab for lab in (it["label"] for it in items) if not lab.startswith(("background", "near:", "slow"))}
    first_report = {}
    with base.session_scope() as db:
        for it, v in zip(items, vectors):
            first_report.setdefault(it["label"], it["t"])
            t = Ticket(customer_name="C", customer_email=it["email"], subject=it["text"][:120], description=it["text"],
                       category="Technical Support", priority="medium", status="open", source="form",
                       created_at=it["t"], embedding=v.tolist())
            db.add(t)
            db.flush()
            it["ticket_id"] = t.id
            detector.on_ticket_created(db, t, now=it["t"])
        label_of = {it["ticket_id"]: it["label"] for it in items}
        incidents = db.scalars(select(Incident)).all()
        found, false_incidents, delays, precisions = {}, [], [], []
        for inc in incidents:
            members = db.scalars(select(Ticket).where(Ticket.incident_id == inc.id)).all()
            labels = Counter(label_of[m.id] for m in members)
            top, n = labels.most_common(1)[0]
            precisions.append(n / len(members))
            if top in truth and top not in found:
                found[top] = inc
                opened = datetime.fromisoformat(inc.detection["opened_at"])
                delays.append({"incident": top, "reports_before_detection": inc.detection["customers"],
                               "minutes": round((opened - first_report[top]).total_seconds() / 60, 1)})
            else:
                false_incidents.append({"majority": top, "labels": dict(labels)})
        recall = []
        for lab, inc in found.items():
            reports = [it for it in items if it["label"] == lab]
            linked = sum(1 for it in reports if db.get(Ticket, it["ticket_id"]).incident_id == inc.id)
            recall.append(linked / len(reports))
        return {"detected": len(found), "true": len(truth), "false_incidents": false_incidents,
                "delays": delays, "member_precision": precisions, "member_recall": recall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sweep", action="store_true", help="also try other similarity thresholds")
    ap.add_argument("--data", default=str(DATA), help="dataset (e.g. evals/datasets/incident_holdout.json)")
    args = ap.parse_args()
    s = get_settings()
    thresholds = [0.70, 0.72, 0.74, 0.76, 0.78] if args.sweep else [s.incident_similarity]
    cache: dict[int, tuple] = {}
    out = {}
    for th in thresholds:
        s.incident_similarity = th
        runs = []
        for seed in range(args.seeds):
            if seed not in cache:
                items = traffic(seed, Path(args.data))
                cache[seed] = (items, detector.embed([it["text"] for it in items]))
            items, vectors = cache[seed]
            runs.append(run([dict(it) for it in items], vectors))
        n_true = sum(r["true"] for r in runs)
        delays = [d for r in runs for d in r["delays"]]
        summary = {
            "incidents_detected": f"{sum(r['detected'] for r in runs)}/{n_true}",
            "false_incidents": sum(len(r["false_incidents"]) for r in runs),
            "false_incident_sources": dict(Counter(f["majority"] for r in runs for f in r["false_incidents"])),
            "median_reports_until_detected": sorted(d["reports_before_detection"] for d in delays)[len(delays) // 2]
            if delays else None,
            "median_minutes_until_detected": sorted(d["minutes"] for d in delays)[len(delays) // 2] if delays else None,
            "member_precision": round(sum(p for r in runs for p in r["member_precision"])
                                      / max(1, sum(len(r["member_precision"]) for r in runs)), 3),
            "member_recall": round(sum(x for r in runs for x in r["member_recall"])
                                   / max(1, sum(len(r["member_recall"]) for r in runs)), 3),
        }
        out[f"{th:.2f}"] = summary
        print(f"similarity {th:.2f}: {json.dumps(summary)}")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report = REPORT if Path(args.data) == DATA else REPORT.with_name("incident_eval_holdout.json")
    report.write_text(json.dumps({"data": Path(args.data).name, "seeds": args.seeds, "days_per_seed": DAYS,
                                  "by_threshold": out}, indent=2))


if __name__ == "__main__":
    main()
