"""T2 Attention: select the clauses that matter.

Input:  PerceptionResult + risk patterns (with learned weights) from Memory.
Output: AttentionResult — clauses with risk categories and an attention score, best first.

Method: keyword match + cosine similarity between the clause embedding
(multilingual MiniLM) and example risky clauses of each category.
match strength = similarity + 0.5 if a keyword matched; a category is a candidate if a keyword
matched or similarity ≥ threshold. Only categories within 0.08 of the best match are kept (max 3);
attention score = weight of the primary category × (1 + its match strength). The top-N clauses are kept.
Evidence: 1.0 for a keyword match, otherwise (similarity − threshold) / (0.80 − threshold) — the
embedding model gives ~0.6 to almost any contract text, so similarity alone is weak evidence.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from .config import EMBEDDING_MODEL
from .schemas import AttentionItem, AttentionResult, Clause, PerceptionResult

LOGGER = logging.getLogger(__name__)
DEFAULT_THRESHOLD = 0.55
KEYWORD_BONUS = 0.5      # an exact keyword is stronger evidence than similarity alone
CATEGORY_MARGIN = 0.08   # keep categories whose match is within this margin of the best one
MAX_CATEGORIES = 3
SIM_FULL = 0.80          # similarity at which a match without keywords counts as full evidence


def get_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception:
        LOGGER.warning("sentence-transformers недоступен: используются только ключевые слова.")
        return None


@lru_cache(maxsize=1)
def cached_embedder():
    return get_embedder()


def select_clauses(clauses: list[Clause], patterns: list[dict], embedder,
                   threshold: float = DEFAULT_THRESHOLD) -> list[AttentionItem]:
    sims = None
    if embedder is not None:
        import numpy as np
        ex = [(i, e) for i, p in enumerate(patterns) for e in p["examples"]]
        E = embedder.encode([e for _, e in ex], normalize_embeddings=True)
        C = embedder.encode([c.text for c in clauses], normalize_embeddings=True)
        S = C @ E.T
        sims = np.zeros((len(clauses), len(patterns)))
        for col, (pi, _) in enumerate(ex):
            sims[:, pi] = np.maximum(sims[:, pi], S[:, col])
    selected = []
    for ci, clause in enumerate(clauses):
        low = clause.text.lower()
        candidates = []  # (match strength, pattern, similarity, keyword hits)
        for pi, p in enumerate(patterns):
            kw = [k for k in p["keywords"] if k in low]
            sm = float(sims[ci, pi]) if sims is not None else 0.0
            if kw or sm >= threshold:
                candidates.append((sm + (KEYWORD_BONUS if kw else 0.0), p, sm, kw))
        if not candidates:
            continue
        # keep only categories close to the best match: one clause ≈ one or two risk types
        candidates.sort(key=lambda c: -c[0])
        best = candidates[0][0]
        kept = [c for c in candidates if c[0] >= best - CATEGORY_MARGIN][:MAX_CATEGORIES]
        strength, primary, primary_sim, primary_kw = kept[0]
        evidence = 1.0 if primary_kw else min(1.0, max(0.0, (primary_sim - threshold) / (SIM_FULL - threshold)))
        score = primary["weight"] * (1 + strength)
        selected.append(AttentionItem(
            clause=clause, categories=[c[1]["category"] for c in kept], score=round(score, 3),
            weight=primary["weight"], similarities={c[1]["category"]: round(c[2], 3) for c in kept},
            keyword_hits=[k for c in kept for k in c[3]], evidence=round(evidence, 3)))
    return sorted(selected, key=lambda x: -x.score)


def attend(perception: PerceptionResult, patterns: list[dict], top: int = 25,
           threshold: float = DEFAULT_THRESHOLD) -> AttentionResult:
    """Keeps at most `top` most important clauses (prioritization by attention score)."""
    if top < 1:
        raise ValueError("top must be at least 1.")
    embedder = cached_embedder()
    relevant = select_clauses(perception.clauses, patterns, embedder, threshold)
    items = relevant[:top]
    return AttentionResult(items=items, total_clauses=len(perception.clauses), threshold=threshold,
                           embeddings_used=embedder is not None, dropped=len(relevant) - len(items))
