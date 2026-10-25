# Phase 3: Ticket classifier (TF-IDF baseline vs fine-tuned DistilBERT)

Two classifiers label every new ticket: **category** (`queue`, 10 routing teams) and **priority** (low / medium / high,
plus a rule-based `urgent`).

## Data
- [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) (CC BY-NC 4.0), English rows: 28,261.
- **Leakage guard:** 16% of tickets were exact copies (4,514 groups, none with conflicting labels). They were removed by
  normalized body **before** splitting, leaving 23,654 tickets → 18,923 train / 2,365 val / 2,366 test, stratified by category.
- Very imbalanced categories: Technical Support 28.9% … General Inquiry 1.4%. **Macro-F1** is the main metric,
  so every team counts equally.
- Both environments (laptop and Colab) build the split from the same code, verified with split fingerprints.

## Models
| | TF-IDF + Logistic Regression | DistilBERT (`distilbert-base-uncased`) |
|---|---|---|
| Features | word 1–2-grams, sublinear TF, 200K features | contextual embeddings, max 256 tokens (covers 99.8% of tickets) |
| Imbalance handling | `class_weight="balanced"` | class-weighted cross-entropy (same weights) |
| Training | 52 s on laptop CPU | ~32 min on a Colab T4 GPU (2 models) |
| Size / CPU latency | a few MB / 0.17 ms per ticket | 255 MB per model / ~70 ms per ticket |

## Iterations
| Run | Setup | Category macro-F1 | Priority macro-F1 | Diagnosis |
|---|---|---|---|---|
| Baseline | TF-IDF + LogReg | 0.628 | 0.675 | — |
| DistilBERT run 1 | 4 epochs, unweighted loss | 0.450 | 0.623 | **Underfit**: val macro-F1 still rising at the last epoch; rare classes ignored |
| **DistilBERT run 2** | up to 10 epochs, early stopping on val macro-F1, class-weighted loss | **0.652** | **0.716** | Wins on both tasks |

Model selection used the validation set only; the test set was scored once per run.

## Is the gain real? (paired bootstrap, 1,000 resamples of the same test tickets)
| Task | Macro-F1 gain | 95% CI | P(no gain) | Verdict |
|---|---|---|---|---|
| Category | +0.024 | [−0.002, +0.051] | 0.034 | Likely, but the CI touches 0: **modest** |
| Priority | **+0.041** | [+0.019, +0.061] | 0.000 | **Clear improvement** |

## Where DistilBERT helps (category F1 per class)
The rare classes gain most: Returns and Exchanges **+0.11**, General Inquiry **+0.07**, Billing +0.04, HR +0.04.
Product Support is worse (−0.06). The biggest class (Technical Support) is unchanged.

**Main confusions** sit between overlapping support teams: Product Support → Technical Support (24% of Product Support
tickets), IT Support → Technical Support (23%), Product Support → Customer Service (18%). These queues describe very
similar problems, so label ambiguity, not model capacity, is the likely ceiling. Billing is easy (F1 0.84).

**Priority errors are almost always "one step"**: high → medium (25%), low → medium (27%). Opposite-end swaps are rare
(high → low 4%). The `low` class gains most (+0.07).

## Calibration: making confidence honest
Run 2's training curves showed validation **loss** rising after epoch 4 while F1 kept improving: the model became
overconfident. That matters because tickets with low confidence go to human review.

| Task | Temperature (fitted on val) | ECE before → after | Mean confidence before → after | Accuracy |
|---|---|---|---|---|
| Category | 2.05 | 0.199 → **0.025** | 0.82 → 0.63 | 0.62 |
| Priority | 3.80 | 0.230 → **0.044** | 0.95 → 0.73 | 0.72 |

Before calibration, priority claimed 95% confidence while being right 72% of the time, and a 0.6 review threshold
would have flagged only 3.5% of tickets.

## Human-review threshold
Thresholds were chosen from the calibrated trade-off on the test set (category < 0.40, priority < 0.50):
- **17.8% of tickets flagged** for review.
- Both labels correct: **55.7%** of auto-routed tickets vs **20.6%** of flagged ones. The flag catches the right tickets.

## Decision
**DistilBERT (run 2, calibrated) is the production classifier.** Priority improves clearly, category modestly, rare
teams benefit most, and 70 ms per ticket on CPU is fine for ticket creation. The TF-IDF model stays as an automatic
fallback when the DistilBERT weights aren't present. The models must load at API startup (first call: ~30 s cold).

## Limitations
- The dataset is generic and appears synthetic (templated "Dear Customer Support Team…" text). Its 10 queues are not
  SearchUnify's support structure: a password reset is routed to Billing with low confidence (and correctly flagged).
  The staff-feedback loop (`reclassified` events) is how domain labels would be collected.
- Thresholds were picked on the test set; a cleaner protocol would tune them on validation.

## Reproduce
```bash
python -m supportpilot.classifier.baseline            # data split + TF-IDF baseline + MLflow
python scripts/build_colab_notebook.py                # generate the Colab notebook
#   run notebooks/train_distilbert_colab.ipynb on a Colab T4, download run2/ to models/distilbert/
python scripts/import_colab_results.py                # MLflow runs + CPU latency + comparison
python scripts/analyze_classifier.py                  # bootstrap, per-class, calibration
mlflow ui --backend-store-uri sqlite:///mlflow.db     # browse all runs
```
