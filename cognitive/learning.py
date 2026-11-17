"""T7 Learning: improve behaviour from the user's feedback.

Input:  FeedbackItem list (system risk vs. the risk the user says is correct) + MemoryStore.
Output: LearningResult — saved cases and category weight updates.

Two mechanisms:
1. Instance-based: every rating is stored as a case; Memory/Reasoning reuse it for very similar
   clauses (the same clause in the next run gets the user's rating immediately).
2. Weight update (generalisation): for each category of the clause
       w_new = clip(w_old + η · (user_level − system_level), W_MIN, W_MAX)
   with η = 0.3. The weight sets the base risk level and the attention score, so after a few
   corrections the system rates *other* clauses of that category differently too.
"""
from __future__ import annotations

from .config import LEVELS
from .memory import MemoryStore, load_knowledge
from .schemas import FeedbackItem, LearningResult

LEARNING_RATE = 0.3
W_MIN, W_MAX = 0.5, 3.5


def learn(store: MemoryStore, feedback: list[FeedbackItem]) -> LearningResult:
    patterns, _ = load_knowledge(store)
    weights = {p["category"]: p["weight"] for p in patterns}
    updates: dict[str, dict] = {}
    for fb in feedback:
        store.add_feedback(fb.document_name, fb.clause_text, fb.categories, fb.system_risk, fb.user_risk)
        delta = LEVELS[fb.user_risk] - LEVELS[fb.system_risk]
        if delta == 0:
            continue
        for cat in fb.categories:
            if cat not in weights:
                continue
            old = weights[cat]
            new = round(min(W_MAX, max(W_MIN, old + LEARNING_RATE * delta)), 3)
            if new == old:
                continue
            weights[cat] = new
            store.set_weight(cat, old, new, f"user: {fb.system_risk} → {fb.user_risk}")
            entry = updates.setdefault(cat, {"category": cat, "old": old})
            entry["new"] = new
    return LearningResult(feedback_saved=len(feedback), weight_updates=list(updates.values()))
