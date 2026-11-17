"""Quality evaluation on the labelled test set (data/labels.csv).

Every clause of the three sample contracts was labelled by the author (none/low/medium/high) from the
point of view of the party that signs. The system runs with an empty memory (no learning) and its
clause ratings are compared with the labels; clauses Attention did not select count as "none".

Metrics:
- flag detection (flag = medium or high, i.e. the user should act): precision, recall, F1;
- exact level accuracy and within-one-level accuracy;
- confusion matrix;
- document level: is the order dangerous > medium > balanced preserved?

Run:  python -m cognitive.evaluation
"""
from __future__ import annotations

import csv
import tempfile
from collections import Counter
from pathlib import Path

from .config import DATA_DIR, LEVEL_NAMES, LEVELS
from .memory import MemoryStore
from .pipeline import run_pipeline

FLAG = {"medium", "high"}


def load_labels(path: Path = DATA_DIR / "labels.csv") -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.setdefault(row["document"], {})[row["clause"]] = row["gold_risk"]
    return labels


def predict(document: Path, store: MemoryStore) -> tuple[dict[str, str], object]:
    cycle = run_pipeline(document, store=store, remember=False)
    pred = {}
    for clause in cycle.perception.clauses:
        key = clause.features.number or "preamble"
        a = cycle.reasoning.assessments.get(clause.id)
        pred[key] = a.risk if a else "none"
    return pred, cycle


def evaluate(data_dir: Path = DATA_DIR) -> dict:
    labels = load_labels(data_dir / "labels.csv")
    rows, docs = [], []
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp) / "eval.db")   # empty memory: measures the system before learning
        for doc_name, gold in labels.items():
            pred, cycle = predict(data_dir / doc_name, store)
            docs.append({"document": doc_name, "score": cycle.decision.score, "level": cycle.decision.level,
                         "recommendation": cycle.decision.dss.recommended})
            for clause, g in gold.items():
                rows.append({"document": doc_name, "clause": clause, "gold": g, "pred": pred.get(clause, "none")})

    tp = sum(r["gold"] in FLAG and r["pred"] in FLAG for r in rows)
    fp = sum(r["gold"] not in FLAG and r["pred"] in FLAG for r in rows)
    fn = sum(r["gold"] in FLAG and r["pred"] not in FLAG for r in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact = sum(r["gold"] == r["pred"] for r in rows) / len(rows)
    within_one = sum(abs(LEVELS[r["gold"]] - LEVELS[r["pred"]]) <= 1 for r in rows) / len(rows)
    confusion = Counter((r["gold"], r["pred"]) for r in rows)
    scores = [d["score"] for d in docs]
    return {
        "n_clauses": len(rows), "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
        "exact_accuracy": round(exact, 3), "within_one_accuracy": round(within_one, 3),
        "confusion": {g: {p: confusion.get((g, p), 0) for p in LEVEL_NAMES} for g in LEVEL_NAMES},
        "documents": docs, "document_order_ok": scores == sorted(scores, reverse=True),
        "rows": rows,
    }


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    r = evaluate()
    print(f"Clauses: {r['n_clauses']}  |  flag = medium/high")
    print(f"Precision {r['precision']:.2f}  Recall {r['recall']:.2f}  F1 {r['f1']:.2f}  "
          f"(TP {r['tp']}, FP {r['fp']}, FN {r['fn']})")
    print(f"Exact level accuracy {r['exact_accuracy']:.0%}, within one level {r['within_one_accuracy']:.0%}")
    print("Confusion (rows = gold, cols = predicted):")
    print("        " + " ".join(f"{p:>7}" for p in LEVEL_NAMES))
    for g in LEVEL_NAMES:
        print(f"{g:>7} " + " ".join(f"{r['confusion'][g][p]:>7}" for p in LEVEL_NAMES))
    for d in r["documents"]:
        print(f"{d['document']}: score {d['score']}, level {d['level']}, DSS → {d['recommendation']}")
    print("Document order preserved:", r["document_order_ok"])
    misses = [x for x in r["rows"] if (x["gold"] in FLAG) != (x["pred"] in FLAG)]
    if misses:
        print("Disagreements on flagging:")
        for x in misses:
            print(f"  {x['document']} {x['clause']}: gold {x['gold']}, predicted {x['pred']}")
