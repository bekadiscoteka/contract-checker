"""T3 Memory: long-term knowledge and experience of the system.

Long-term memory (SQLite, `cognitive_memory.db`):
- knowledge base: risk patterns from `risk_patterns.json` + category weights learned from feedback;
- episodic memory: every analysed document (to compare with a previous run of the same document);
- case memory: clauses the user rated (case-based reasoning: a very similar clause reuses the user's rating);
- decision log: recommendation vs. the decision the human actually made.
Short-term memory is the Streamlit session state (current document, DSS weights).

Input:  AttentionResult (+ document hash).
Output: MemoryContext — patterns with learned weights, similar past cases per clause, previous run.
"""
from __future__ import annotations

import copy
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from . import config
from .schemas import AttentionResult, MemoryContext, SimilarCase

SIMILARITY_THRESHOLD = 0.85  # embedding cosine; text ratio uses the same threshold

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY, name TEXT, doc_hash TEXT, analyzed_at TEXT, provider TEXT,
    level TEXT, score INTEGER, highs INTEGER, n_findings INTEGER, findings_json TEXT);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY, created_at TEXT, document_name TEXT, clause_text TEXT,
    categories_json TEXT, system_risk TEXT, user_risk TEXT);
CREATE TABLE IF NOT EXISTS category_weights (
    category TEXT PRIMARY KEY, weight REAL, updated_at TEXT);
CREATE TABLE IF NOT EXISTS learning_log (
    id INTEGER PRIMARY KEY, created_at TEXT, category TEXT, old_weight REAL, new_weight REAL, reason TEXT);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY, created_at TEXT, document_name TEXT, recommended TEXT, chosen TEXT,
    weights_json TEXT, params_json TEXT, comment TEXT);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class MemoryStore:
    """Thin wrapper around the SQLite database."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or config.MEMORY_DB)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _rows(self, sql: str, args=()) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, args).fetchall()]

    # --- knowledge ---
    def learned_weights(self) -> dict[str, float]:
        return {r["category"]: r["weight"] for r in self._rows("SELECT category, weight FROM category_weights")}

    def set_weight(self, category: str, old: float, new: float, reason: str):
        with self._conn() as c:
            c.execute("INSERT INTO category_weights(category, weight, updated_at) VALUES (?,?,?) "
                      "ON CONFLICT(category) DO UPDATE SET weight=excluded.weight, updated_at=excluded.updated_at",
                      (category, new, _now()))
            c.execute("INSERT INTO learning_log(created_at, category, old_weight, new_weight, reason) "
                      "VALUES (?,?,?,?,?)", (_now(), category, old, new, reason))

    def learning_log(self, limit: int = 100) -> list[dict]:
        return self._rows("SELECT * FROM learning_log ORDER BY id DESC LIMIT ?", (limit,))

    # --- cases (feedback) ---
    def add_feedback(self, document_name, clause_text, categories, system_risk, user_risk):
        with self._conn() as c:
            c.execute("INSERT INTO feedback(created_at, document_name, clause_text, categories_json, "
                      "system_risk, user_risk) VALUES (?,?,?,?,?,?)",
                      (_now(), document_name, clause_text, json.dumps(categories, ensure_ascii=False),
                       system_risk, user_risk))

    def feedback(self) -> list[dict]:
        return self._rows("SELECT * FROM feedback ORDER BY id DESC")

    # --- episodes ---
    def save_run(self, name, doc_hash, provider, level, score, highs, findings: dict[int, str]):
        with self._conn() as c:
            c.execute("INSERT INTO documents(name, doc_hash, analyzed_at, provider, level, score, highs, "
                      "n_findings, findings_json) VALUES (?,?,?,?,?,?,?,?,?)",
                      (name, doc_hash, _now(), provider, level, score, highs,
                       sum(r != "none" for r in findings.values()), json.dumps(findings)))

    def previous_run(self, doc_hash: str) -> Optional[dict]:
        rows = self._rows("SELECT * FROM documents WHERE doc_hash=? ORDER BY id DESC LIMIT 1", (doc_hash,))
        if not rows:
            return None
        row = rows[0]
        row["findings"] = {int(k): v for k, v in json.loads(row.pop("findings_json") or "{}").items()}
        return row

    def documents(self, limit: int = 50) -> list[dict]:
        return self._rows("SELECT id, name, analyzed_at, provider, level, score, highs, n_findings "
                          "FROM documents ORDER BY id DESC LIMIT ?", (limit,))

    def count(self, table: str) -> int:
        assert table in {"documents", "feedback", "decisions"}
        return self._rows(f"SELECT COUNT(*) AS n FROM {table}")[0]["n"]

    # --- decisions ---
    def log_decision(self, document_name, recommended, chosen, weights, params, comment="") -> int:
        with self._conn() as c:
            cur = c.execute("INSERT INTO decisions(created_at, document_name, recommended, chosen, "
                            "weights_json, params_json, comment) VALUES (?,?,?,?,?,?,?)",
                            (_now(), document_name, recommended, chosen, json.dumps(weights),
                             json.dumps(params), comment))
            return cur.lastrowid

    def decisions(self, limit: int = 50) -> list[dict]:
        return self._rows("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,))

    def reset(self):
        with self._conn() as c:
            for t in ("documents", "feedback", "category_weights", "learning_log", "decisions"):
                c.execute(f"DELETE FROM {t}")


# ---------- knowledge ----------
def load_patterns() -> list[dict]:
    return json.loads(config.PATTERNS_PATH.read_text(encoding="utf-8"))


def load_knowledge(store: MemoryStore) -> tuple[list[dict], dict[str, float]]:
    """Returns patterns with learned weights applied, and the original (base) weights."""
    base = load_patterns()
    learned = store.learned_weights()
    patterns = copy.deepcopy(base)
    for p in patterns:
        p["weight"] = float(learned.get(p["category"], p["weight"]))
    return patterns, {p["category"]: float(p["weight"]) for p in base}


# ---------- retrieval ----------
def _text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_similar(texts: list[str], cases: list[dict], embedder=None,
                 threshold: float = SIMILARITY_THRESHOLD) -> list[list[tuple[float, dict]]]:
    """For each text, the stored cases with similarity ≥ threshold, most similar first."""
    if not cases:
        return [[] for _ in texts]
    if embedder is not None:
        A = embedder.encode(texts, normalize_embeddings=True)
        B = embedder.encode([c["clause_text"] for c in cases], normalize_embeddings=True)
        sims = (A @ B.T).tolist()
    else:
        sims = [[_text_similarity(t, c["clause_text"]) for c in cases] for t in texts]
    out = []
    for row in sims:
        matches = [(float(s), cases[j]) for j, s in enumerate(row) if s >= threshold]
        out.append(sorted(matches, key=lambda m: -m[0]))
    return out


def recall(attention: AttentionResult, patterns: list[dict], store: MemoryStore,
           doc_hash: str = "", base_weights: Optional[dict] = None, embedder=None) -> MemoryContext:
    """Retrieves what the system remembers about the selected clauses and this document."""
    cases = store.feedback()
    # newest rating of each distinct clause text wins
    unique, seen = [], set()
    for c in cases:
        if c["clause_text"] not in seen:
            seen.add(c["clause_text"])
            unique.append(c)
    matches = find_similar([i.text for i in attention.items], unique, embedder)
    similar = {
        item.id: [SimilarCase(clause_text=c["clause_text"], user_risk=c["user_risk"],
                              system_risk=c["system_risk"], similarity=round(s, 3),
                              document_name=c["document_name"]) for s, c in m[:3]]
        for item, m in zip(attention.items, matches)
    }
    return MemoryContext(
        patterns=patterns,
        base_weights=base_weights or {},
        similar_cases=similar,
        previous_run=store.previous_run(doc_hash) if doc_hash else None,
        documents_seen=store.count("documents"),
        feedback_count=len(cases),
    )
