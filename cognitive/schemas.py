"""Data passed between the modules of the cognitive cycle.

The output of each stage is the input of the next one:
PerceptionResult → AttentionResult → MemoryContext → ReasoningResult → DecisionResult → ActionResult,
and LearningResult feeds updated knowledge back into Memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------- T1 Perception ----------
@dataclass
class ClauseFeatures:
    """Features extracted from the text of a clause."""
    number: Optional[str] = None              # clause number, e.g. "4.1"
    percents: List[float] = field(default_factory=list)
    percent_per_day: bool = False             # the percent accrues for every day
    money: List[Tuple[float, str]] = field(default_factory=list)  # (amount, currency: KZT/MRP/RUB/USD/EUR)
    durations_days: List[float] = field(default_factory=list)  # periods converted to days
    word_count: int = 0

    @property
    def max_percent(self) -> Optional[float]:
        return max(self.percents) if self.percents else None

    @property
    def min_duration_days(self) -> Optional[float]:
        return min(self.durations_days) if self.durations_days else None

    @property
    def max_duration_days(self) -> Optional[float]:
        return max(self.durations_days) if self.durations_days else None


@dataclass
class Clause:
    id: int
    text: str
    features: ClauseFeatures = field(default_factory=ClauseFeatures)


@dataclass
class PerceptionResult:
    source_name: str
    raw_text: str
    clauses: List[Clause]
    cleaned_text: str = ""
    cleaning_stats: Dict[str, int] = field(default_factory=dict)
    doc_hash: str = ""


# ---------- T2 Attention ----------
@dataclass
class AttentionItem:
    clause: Clause
    categories: List[str]
    score: float
    weight: float
    similarities: Dict[str, float] = field(default_factory=dict)
    keyword_hits: List[str] = field(default_factory=list)
    evidence: float = 1.0   # 1.0 = keyword match; similarity-only matches are weaker (0..1)

    @property
    def id(self) -> int:
        return self.clause.id

    @property
    def text(self) -> str:
        return self.clause.text


@dataclass
class AttentionResult:
    items: List[AttentionItem]
    total_clauses: int
    threshold: float
    embeddings_used: bool
    dropped: int = 0  # relevant clauses cut off by the top-N limit


# ---------- T3 Memory ----------
@dataclass
class SimilarCase:
    """A clause the user rated earlier that looks like the current one."""
    clause_text: str
    user_risk: str
    system_risk: str
    similarity: float
    document_name: str


@dataclass
class MemoryContext:
    patterns: List[dict]                      # long-term knowledge with learned weights
    base_weights: Dict[str, float] = field(default_factory=dict)
    similar_cases: Dict[int, List[SimilarCase]] = field(default_factory=dict)
    previous_run: Optional[dict] = None       # last analysis of the same document
    documents_seen: int = 0
    feedback_count: int = 0


# ---------- T4 Reasoning ----------
@dataclass
class Assessment:
    id: int
    risk: str
    issue: str = ""
    recommendation: str = ""
    trace: List[str] = field(default_factory=list)
    rules_fired: List[str] = field(default_factory=list)
    base_risk: str = ""
    memory_applied: bool = False


@dataclass
class ReasoningResult:
    assessments: Dict[int, Assessment]
    provider_name: str


# ---------- T5 Decision / DSS ----------
@dataclass
class DSSParams:
    """Scenario parameters set by the user (amounts in tenge)."""
    deal_value: float = 5_000_000
    lawyer_cost: float = 150_000
    negotiation_cost: float = 30_000
    p_accept: float = 0.6               # chance the counterparty accepts amendments
    lawyer_effectiveness: float = 0.8   # share of risk a lawyer removes
    walk_away: float = 0.15             # chance the counterparty walks away if you push back
    negotiation_days: float = 7
    lawyer_days: float = 14


@dataclass
class RiskProfile:
    total_severity: float
    fixable_severity: float
    counts: Dict[str, int]
    fixable_clauses: List[dict]


@dataclass
class DSSResult:
    alternatives: Dict[str, str]                 # key → display name
    criteria: List[dict]                         # key, name, type, unit
    matrix: Dict[str, Dict[str, float]]          # alternative → criterion → value
    weights: Dict[str, float]                    # normalized, sum = 1
    params: DSSParams
    profile: RiskProfile
    topsis: Dict[str, float]
    wsm: Dict[str, float]
    wsm_contributions: Dict[str, Dict[str, float]]
    ranking: List[str]                           # by TOPSIS, best first
    recommended: str
    runner_up: str
    models_agree: bool
    explanation: List[str]
    stability: Dict[str, float]                  # share of wins under perturbed weights
    sensitivity: Dict[str, dict]                 # criterion → curve and switch points


@dataclass
class DecisionResult:
    level: str
    score: int
    highs: int
    dss: Optional[DSSResult] = None


# ---------- T6 Action ----------
@dataclass
class ActionResult:
    report_html: str
    report_path: Optional[Path] = None
    logged_decision_id: Optional[int] = None


# ---------- T7 Learning ----------
@dataclass
class FeedbackItem:
    clause_text: str
    categories: List[str]
    system_risk: str
    user_risk: str
    document_name: str = ""


@dataclass
class LearningResult:
    feedback_saved: int
    weight_updates: List[dict]                 # category, old, new


@dataclass
class CycleResult:
    perception: PerceptionResult
    attention: AttentionResult
    memory: MemoryContext
    reasoning: ReasoningResult
    decision: DecisionResult
    action: ActionResult
