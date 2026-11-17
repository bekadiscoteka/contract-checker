"""The single cognitive cycle: the output of each stage is the input of the next one.

Perception → Attention → Memory → Reasoning → Decision (DSS) → Action, and Learning
(triggered by user feedback) writes back into Memory, changing the next cycle.
"""
from __future__ import annotations

import logging
from typing import Optional

from . import action, attention, decision, memory, perception, reasoning
from .schemas import CycleResult, DSSParams

LOGGER = logging.getLogger(__name__)


def run_pipeline(file_path, provider: str = "rules", top: int = 25, source_name: Optional[str] = None,
                 output_path=None, weights: Optional[dict] = None, params: Optional[DSSParams] = None,
                 store: Optional[memory.MemoryStore] = None, remember: bool = True) -> CycleResult:
    LOGGER.info("Starting analysis; requested provider (запрошенный провайдер): %s", provider)
    store = store or memory.MemoryStore()
    patterns, base_weights = memory.load_knowledge(store)                     # long-term knowledge

    perceived = perception.perceive(file_path, source_name)                   # T1
    attended = attention.attend(perceived, patterns, top=top)                 # T2
    context = memory.recall(attended, patterns, store, perceived.doc_hash,    # T3
                            base_weights, attention.cached_embedder())
    reasoned = reasoning.reason(attended, context, provider)                  # T4
    decided = decision.decide(attended, reasoned, weights, params)            # T5
    acted = action.act(perceived.source_name, attended.total_clauses,         # T6
                       attended.items, reasoned.assessments, decided,
                       reasoned.provider_name, output_path)
    if remember:
        store.save_run(perceived.source_name, perceived.doc_hash, reasoned.provider_name,
                       decided.level, decided.score, decided.highs,
                       {i: a.risk for i, a in reasoned.assessments.items()})
    # T7 Learning runs when the user gives feedback (cognitive.learning.learn) and changes Memory.
    return CycleResult(perception=perceived, attention=attended, memory=context,
                       reasoning=reasoned, decision=decided, action=acted)
