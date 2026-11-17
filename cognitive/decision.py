"""T5 Decision-making + DSS: what should the user do with this contract?

Input:  AttentionResult + ReasoningResult (+ criterion weights and scenario parameters from the user).
Output: DecisionResult — overall risk level and a DSSResult with ranking, recommendation,
        explanation and sensitivity analysis. The human makes the final decision.

Chain: data (assessed clauses) → model (scenario model) → alternatives → criteria → recommendation.

Alternatives: sign as is · sign with amendments · send to a lawyer · reject.
Criteria (type):
  residual risk, severity points (cost) · direct cost, ₸ (cost) · delay, days (cost) ·
  expected deal value kept, ₸ (benefit).
Scenario model (S = total severity, F = severity that can be negotiated away,
p = P(counterparty accepts amendments), e = share of risk a lawyer removes,
w = P(counterparty walks away if you push back), V = deal value):
  sign:    risk S,           cost 0,   delay 0,       value V
  amend:   risk S − p·F,     cost c_n, delay d_n,     value V·(1 − w)
  lawyer:  risk S·(1 − e),   cost L,   delay d_l,     value V·(1 − w)
  reject:  risk 0,           cost 0,   delay 0,       value 0
MCDA models: TOPSIS (main) and WSM (cross-check). Criteria are normalised against fixed
reference points (risk scale, deal value, 30 days) rather than min–max over the alternatives,
so a nearly safe contract is not made to look "risky" just because "reject" has zero risk.
Sensitivity: one-at-a-time weight sweep (switch points) + Monte-Carlo weight perturbation.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Optional

import numpy as np

from .config import NEGOTIABILITY, category_en
from .schemas import (AttentionResult, DecisionResult, DSSParams, DSSResult, ReasoningResult,
                      RiskProfile)

RISK_POINTS = {"low": 1, "medium": 2, "high": 4}       # overall score (report headline)
SEVERITY = {"none": 0, "low": 1, "medium": 3, "high": 6}  # DSS risk criterion, convex: one high > several lows
RISK_REF = 80      # severity points at which a contract counts as "fully risky" (≈13 high-risk clauses)
DELAY_REF = 30     # days

ALTERNATIVES = {
    "sign": "Sign as is",
    "amend": "Sign with amendments",
    "lawyer": "Send to a lawyer",
    "reject": "Reject the contract",
}
CRITERIA = [
    {"key": "risk", "name": "Residual risk", "type": "cost", "unit": "pts"},
    {"key": "cost", "name": "Direct cost", "type": "cost", "unit": "₸"},
    {"key": "delay", "name": "Delay", "type": "cost", "unit": "days"},
    {"key": "value", "name": "Expected deal value", "type": "benefit", "unit": "₸"},
]
CRITERION_NAMES = {c["key"]: c["name"] for c in CRITERIA}
DEFAULT_WEIGHTS = {"risk": 0.40, "cost": 0.15, "delay": 0.15, "value": 0.30}


# ---------- overall level ----------
def aggregate(attention: AttentionResult, reasoning: ReasoningResult) -> tuple[str, int, int]:
    risks = [reasoning.assessments[x.id].risk for x in attention.items if x.id in reasoning.assessments]
    score = sum(RISK_POINTS.get(r, 0) for r in risks)
    highs, mediums = risks.count("high"), risks.count("medium")
    if highs >= 2:
        level = "high"
    elif highs == 1 or mediums >= 2:
        level = "medium"
    elif score > 0:
        level = "low"
    else:
        level = "none"
    return level, score, highs


# ---------- data → model ----------
def risk_profile(attention: AttentionResult, reasoning: ReasoningResult) -> RiskProfile:
    total = fixable = 0.0
    counts = {"low": 0, "medium": 0, "high": 0}
    clauses = []
    for item in attention.items:
        a = reasoning.assessments.get(item.id)
        if a is None or a.risk == "none":
            continue
        s = SEVERITY[a.risk]
        n = max(NEGOTIABILITY.get(c, 0.5) for c in item.categories)
        total += s
        fixable += s * n
        counts[a.risk] += 1
        clauses.append({"id": item.id, "number": item.clause.features.number or f"#{item.id}",
                        "category": category_en(item.categories[0]), "risk": a.risk,
                        "negotiability": n, "gain": round(s * n, 2)})
    clauses.sort(key=lambda c: -c["gain"])
    return RiskProfile(total_severity=total, fixable_severity=round(fixable, 3), counts=counts,
                       fixable_clauses=clauses)


def build_matrix(profile: RiskProfile, p: DSSParams) -> dict[str, dict[str, float]]:
    S, F, V = profile.total_severity, profile.fixable_severity, p.deal_value
    return {
        "sign": {"risk": S, "cost": 0.0, "delay": 0.0, "value": V},
        "amend": {"risk": S - p.p_accept * F, "cost": p.negotiation_cost,
                  "delay": p.negotiation_days, "value": V * (1 - p.walk_away)},
        "lawyer": {"risk": S * (1 - p.lawyer_effectiveness), "cost": p.lawyer_cost,
                   "delay": p.lawyer_days, "value": V * (1 - p.walk_away)},
        "reject": {"risk": 0.0, "cost": 0.0, "delay": 0.0, "value": 0.0},
    }


def normalize_weights(weights: Optional[dict]) -> dict[str, float]:
    w = {k: max(0.0, float((weights or DEFAULT_WEIGHTS).get(k, 0.0))) for k in CRITERION_NAMES}
    total = sum(w.values())
    if total <= 0:
        return {k: 1 / len(w) for k in w}
    return {k: v / total for k, v in w.items()}


def normalize_matrix(matrix: dict, profile: RiskProfile, p: DSSParams) -> dict[str, dict[str, float]]:
    """Benefit-oriented scores in [0, 1] against fixed reference points (1 = best possible)."""
    risk_ref = RISK_REF
    money_ref = max(p.deal_value, 1.0)
    delay_ref = max(DELAY_REF, p.negotiation_days, p.lawyer_days)
    clip = lambda x: min(1.0, max(0.0, x))
    return {a: {"risk": clip(1 - row["risk"] / risk_ref),
                "cost": clip(1 - row["cost"] / money_ref),
                "delay": clip(1 - row["delay"] / delay_ref),
                "value": clip(row["value"] / money_ref)} for a, row in matrix.items()}


# ---------- MCDA models ----------
def wsm(norm: dict, w: dict) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    contrib = {a: {c: w[c] * r[c] for c in w} for a, r in norm.items()}
    return {a: sum(cs.values()) for a, cs in contrib.items()}, contrib


def topsis(norm: dict, w: dict) -> dict[str, float]:
    """Closeness to the ideal point (all criteria at their best) vs. the anti-ideal (all worst)."""
    out = {}
    for a, r in norm.items():
        d_plus = math.sqrt(sum((w[c] * (1 - r[c])) ** 2 for c in w))
        d_minus = math.sqrt(sum((w[c] * r[c]) ** 2 for c in w))
        out[a] = d_minus / (d_plus + d_minus) if d_plus + d_minus > 0 else 0.5
    return out


def _scores(profile, params, weights) -> dict[str, float]:
    norm = normalize_matrix(build_matrix(profile, params), profile, params)
    return topsis(norm, normalize_weights(weights))


def _winner(scores: dict) -> str:
    return max(scores, key=scores.get)


# ---------- what-if ----------
def sweep_weight(profile, params, weights, criterion, points: int = 51) -> dict:
    """Vary one weight 0…1 (others keep their proportions) and record the TOPSIS scores."""
    w = normalize_weights(weights)
    rest = 1 - w[criterion]
    grid = np.linspace(0, 1, points)
    curve, winners = [], []
    for t in grid:
        ww = {k: (t if k == criterion else (v * (1 - t) / rest if rest > 0 else (1 - t) / (len(w) - 1)))
              for k, v in w.items()}
        s = _scores(profile, params, ww)
        curve.append({"weight": round(float(t), 3), **{a: round(v, 4) for a, v in s.items()}})
        winners.append(_winner(s))
    switches = [{"weight": round(float(grid[i]), 3), "from": winners[i - 1], "to": winners[i]}
                for i in range(1, len(grid)) if winners[i] != winners[i - 1]]
    # stable interval around the current weight
    cur_idx = int(round(w[criterion] * (points - 1)))
    lo = hi = cur_idx
    while lo > 0 and winners[lo - 1] == winners[cur_idx]:
        lo -= 1
    while hi < points - 1 and winners[hi + 1] == winners[cur_idx]:
        hi += 1
    return {"curve": curve, "switches": switches, "current": round(w[criterion], 3),
            "stable_range": (round(float(grid[lo]), 3), round(float(grid[hi]), 3))}


def stability(profile, params, weights, n: int = 400, concentration: float = 40, seed: int = 7) -> dict[str, float]:
    """Share of wins per alternative when weights are randomly perturbed (Dirichlet around the user's)."""
    w = normalize_weights(weights)
    keys = list(w)
    rng = np.random.default_rng(seed)
    samples = rng.dirichlet([w[k] * concentration + 0.05 for k in keys], size=n)
    wins = {a: 0 for a in ALTERNATIVES}
    for row in samples:
        wins[_winner(_scores(profile, params, dict(zip(keys, row))))] += 1
    return {a: round(c / n, 3) for a, c in wins.items()}


# ---------- explanation ----------
def explain(rec, runner, matrix, contrib, topsis_s, wsm_s, agree, stab, sens, profile) -> list[str]:
    def fmt(c, v):
        unit = next(x["unit"] for x in CRITERIA if x["key"] == c)
        return f"{v:,.0f} {unit}" if unit == "₸" else f"{v:g} {unit}"

    lines = [f"Recommended: **{ALTERNATIVES[rec]}** — TOPSIS closeness {topsis_s[rec]:.3f} "
             f"vs {topsis_s[runner]:.3f} for the runner-up «{ALTERNATIVES[runner]}»."]
    diffs = sorted(((contrib[rec][c] - contrib[runner][c], c) for c in CRITERION_NAMES), reverse=True)
    pros = [(d, c) for d, c in diffs if d > 0.005]
    cons = [(d, c) for d, c in diffs if d < -0.005]
    if pros:
        lines.append("Why it beats the runner-up: " + "; ".join(
            f"{CRITERION_NAMES[c].lower()} {fmt(c, matrix[rec][c])} vs {fmt(c, matrix[runner][c])} "
            f"(+{d:.3f} weighted)" for d, c in pros[:2]) + ".")
    if cons:
        lines.append("Trade-off accepted: " + "; ".join(
            f"{CRITERION_NAMES[c].lower()} {fmt(c, matrix[rec][c])} vs {fmt(c, matrix[runner][c])} "
            f"({d:.3f})" for d, c in cons[:2]) + ".")
    lines.append(("WSM gives the same winner — the result does not depend on the model."
                  if agree else f"WSM would pick «{ALTERNATIVES[_winner(wsm_s)]}» — the choice is close, "
                                f"look at the sensitivity analysis."))
    lines.append(f"Robustness: wins in {stab[rec]:.0%} of random weight perturbations.")
    risk_sens = sens.get("risk")
    if risk_sens:
        lo, hi = risk_sens["stable_range"]
        lines.append(f"The recommendation holds while the risk weight is between {lo:.0%} and {hi:.0%} "
                     f"(now {risk_sens['current']:.0%}).")
    if rec in ("amend", "lawyer") and profile.fixable_clauses:
        top = ", ".join(f"{c['number']} ({c['category'].lower()})" for c in profile.fixable_clauses[:4])
        lines.append(f"Negotiate first: {top}.")
    if rec == "sign":
        lines.append(f"Total risk severity is low ({profile.total_severity:g} pts), extra steps cost more than they save.")
    if rec == "reject":
        lines.append(f"Residual risk ({profile.total_severity:g} pts) outweighs the value of the deal at these weights.")
    return lines


def run_dss(attention: AttentionResult, reasoning: ReasoningResult, weights: Optional[dict] = None,
            params: Optional[DSSParams] = None) -> DSSResult:
    params = params or DSSParams()
    w = normalize_weights(weights)
    profile = risk_profile(attention, reasoning)
    matrix = build_matrix(profile, params)
    norm = normalize_matrix(matrix, profile, params)
    t = topsis(norm, w)
    s, contrib = wsm(norm, w)
    ranking = sorted(t, key=t.get, reverse=True)
    rec, runner = ranking[0], ranking[1]
    agree = _winner(s) == rec
    sens = {c: sweep_weight(profile, params, w, c) for c in CRITERION_NAMES}
    stab = stability(profile, params, w)
    return DSSResult(
        alternatives=dict(ALTERNATIVES), criteria=[dict(c) for c in CRITERIA],
        matrix={a: {c: round(v, 2) for c, v in row.items()} for a, row in matrix.items()},
        weights={k: round(v, 4) for k, v in w.items()}, params=params, profile=profile,
        topsis={a: round(v, 4) for a, v in t.items()}, wsm={a: round(v, 4) for a, v in s.items()},
        wsm_contributions={a: {c: round(v, 4) for c, v in cs.items()} for a, cs in contrib.items()},
        ranking=ranking, recommended=rec, runner_up=runner, models_agree=agree,
        explanation=explain(rec, runner, matrix, contrib, t, s, agree, stab, sens, profile),
        stability=stab, sensitivity=sens,
    )


def decide(attention: AttentionResult, reasoning: ReasoningResult, weights: Optional[dict] = None,
           params: Optional[DSSParams] = None) -> DecisionResult:
    level, score, highs = aggregate(attention, reasoning)
    return DecisionResult(level=level, score=score, highs=highs,
                          dss=run_dss(attention, reasoning, weights, params))


def params_dict(p: DSSParams) -> dict:
    return asdict(p)
