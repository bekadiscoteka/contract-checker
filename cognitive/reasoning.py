"""T4 Reasoning: infer the risk level of each selected clause and show the inference steps.

Input:  AttentionResult (clauses + categories + learned weights) and MemoryContext (similar rated cases).
Output: ReasoningResult — an Assessment per clause with risk, explanation and trace.

Method — forward-chaining rule engine over the clause features:
  1. base level from the (learned) category weight;
  2. numeric and wording rules (penalty %/day, liability cap in tenge, too-short deadlines,
     long lock-in, 100% prepayment, one-sided rights, mutual rights …) raise or lower the level;
  3. case-based step: if the user already rated a very similar clause, that rating wins.
Optionally an LLM (local Ollama or Anthropic) gives its own assessment; the rule trace is kept
as a cross-check and memory still has the final word.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from . import config
from .config import FX_TO_KZT, LEVEL_NAMES, LEVELS, PROVIDERS, RECOMMENDATIONS_EN, category_en
from .schemas import AttentionItem, AttentionResult, Assessment, MemoryContext, ReasoningResult

LOGGER = logging.getLogger(__name__)

PROVIDER_NAMES = {
    "rules": "Built-in rules (no LLM)",
    "ollama": "Ollama (local LLM) + rules",
    "anthropic": "Anthropic Claude + rules",
}

PEN = "Штрафы и неустойка"
RENEW = "Автоматическое продление"
UNILAT = "Односторонний отказ или изменение условий"
LIAB = "Неограниченная ответственность"
PREPAY = "Невозвратная предоплата"
CONF = "Конфиденциальность"
TERM = "Расторжение и выход из договора"

LIABILITY_CAP_MIN_KZT = 500_000   # a cap below this is symbolic for a commercial contract
PENALTY_DAILY_HIGH = 0.5          # % per day ≈ 180% per year
PENALTY_DAILY_MODERATE = 0.1      # % per day ≈ 36% per year — common market practice
PENALTY_LUMP_HIGH = 30            # % of the contract value
SHORT_DEADLINE_DAYS = 3
LONG_LOCKIN_DAYS = 3 * 365


def level_from_weight(weight: float) -> str:
    if weight >= 2.5:
        return "high"
    if weight >= 1.5:
        return "medium"
    if weight >= 0.75:
        return "low"
    return "none"


def _raise(level: str, steps: int = 1) -> str:
    return LEVEL_NAMES[min(3, max(0, LEVELS[level] + steps))]


def _at_least(level: str, floor: str) -> str:
    return level if LEVELS[level] >= LEVELS[floor] else floor


@dataclass
class Rule:
    id: str
    description: str
    apply: Callable[[AttentionItem, str, str], Optional[tuple]]  # (item, text_lower, level) → (new_level, why)


def _kzt(amounts) -> list[float]:
    return [a * FX_TO_KZT.get(cur, 1.0) for a, cur in amounts]


def _r_penalty_daily(item, low, level):
    f = item.clause.features
    if PEN in item.categories and f.percent_per_day and f.max_percent is not None:
        p = f.max_percent
        yearly = round(p * 365)
        if p >= PENALTY_DAILY_HIGH:
            return "high", f"penalty {p:g}% per day ≈ {yearly}% per year ≥ {PENALTY_DAILY_HIGH}%/day"
        if p >= PENALTY_DAILY_MODERATE:
            return "medium", f"penalty {p:g}% per day ≈ {yearly}% per year (moderate)"
        return "low", f"penalty {p:g}% per day is within usual market practice"


def _r_penalty_lump(item, low, level):
    f = item.clause.features
    if PEN in item.categories and not f.percent_per_day and f.max_percent is not None \
            and f.max_percent >= PENALTY_LUMP_HIGH and re.search(r"штраф|неустойк|пен[яиюе]", low):
        return "high", f"one-off penalty of {f.max_percent:g}% ≥ {PENALTY_LUMP_HIGH}% of the value"


def _r_liability_cap(item, low, level):
    amounts = _kzt(item.clause.features.money)
    if amounts and "ограничен" in low and "ответственност" in low:
        cap = min(amounts)
        if cap < LIABILITY_CAP_MIN_KZT:
            return "high", f"liability capped at ≈{cap:,.0f} ₸ < {LIABILITY_CAP_MIN_KZT:,} ₸ (symbolic cap)"


def _r_short_deadline(item, low, level):
    d = item.clause.features.min_duration_days
    if d is not None and d < SHORT_DEADLINE_DAYS and re.search(r"претенз|уведом|оплат|предоплат|вносит", low):
        hours = d * 24
        shown = f"{hours:g} h" if hours < 24 else f"{d:g} day(s)"
        return _raise(level), f"deadline of {shown} < {SHORT_DEADLINE_DAYS} days leaves no time to react"


def _r_long_lockin(item, low, level):
    d = item.clause.features.max_duration_days
    if d is not None and d >= LONG_LOCKIN_DAYS and (
            {RENEW, TERM, CONF} & set(item.categories) or re.search(r"конкурент|продлева|не вправе", low)):
        return _raise(level), f"binding period of {d / 365:g} years ≥ {LONG_LOCKIN_DAYS // 365} years"


def _r_full_prepayment(item, low, level):
    if 100 in item.clause.features.percents and re.search(r"предоплат|аванс", low):
        floor = "high" if PREPAY in item.categories else "medium"
        return _at_least(level, floor), "100% prepayment: you carry the whole performance risk"


def _r_no_notice(item, low, level):
    if re.search(r"без (?:предварительного )?уведомлени|без объяснения причин|без согласия", low):
        return _raise(level), "a right can be used without notice / consent / reasons"


def _r_absolute_waiver(item, low, level):
    if re.search(r"ни при каких обстоятельствах|не несет никакой|не несёт никакой|отказывается от права|"
                 r"не рассматриваются|даже если они возникли по вине", low):
        return "high", "absolute wording removes your remedies"


def _r_mutual(item, low, level):
    if level in ("none", "low"):
        return None
    if re.search(r"кажд\w+ из сторон|любая из сторон|обе стороны|обеими сторонами|стороны обязуются|стороны вправе|взаимн|"
                 r"согласия другой стороны|согласованн\w+ сторонами|по месту нахождения ответчика|"
                 r"остаются (?:его |ее |её )?собственностью", low) \
            and not re.search(r"исполнитель вправе|поставщик вправе|заказчик не вправе", low):
        return _raise(level, -1), "the right/obligation is mutual or protects both parties"


def _r_penalty_cap(item, low, level):
    if PEN in item.categories and level not in ("none", "low") and re.search(r"не более \d|не превыша", low):
        return _raise(level, -1), "the total penalty is capped"


RULES = [
    Rule("R1", "Daily penalty rate", _r_penalty_daily),
    Rule("R2", "Large one-off penalty", _r_penalty_lump),
    Rule("R3", "Symbolic liability cap", _r_liability_cap),
    Rule("R4", "Too-short deadline", _r_short_deadline),
    Rule("R5", "Long lock-in period", _r_long_lockin),
    Rule("R6", "100% prepayment", _r_full_prepayment),
    Rule("R7", "No notice / consent", _r_no_notice),
    Rule("R8", "Absolute waiver wording", _r_absolute_waiver),
    Rule("R9", "Mutual / protective wording", _r_mutual),
    Rule("R10", "Penalty cap", _r_penalty_cap),
]


def infer(item: AttentionItem, memory: Optional[MemoryContext] = None) -> Assessment:
    """Runs the rule engine for one clause."""
    low = item.text.lower()
    cats = ", ".join(category_en(c) for c in item.categories)
    effective = item.weight * item.evidence
    base = level_from_weight(effective)
    level = base
    evidence = ("keyword " + ", ".join(f"«{k}»" for k in item.keyword_hits[:3]) if item.evidence >= 1
                else f"similarity {max(item.similarities.values(), default=0):.2f}, no keyword")
    trace = [f"Categories: {cats}",
             f"Base: category weight {item.weight:.2f} × evidence {item.evidence:.2f} ({evidence}) "
             f"= {effective:.2f} → {base}"]
    fired, reasons = [], []
    for rule in RULES:
        out = rule.apply(item, low, level)
        if out is None:
            continue
        new_level, why = out
        trace.append(f"{rule.id} {rule.description}: {why} → {level} ⇒ {new_level}")
        fired.append(rule.id)
        reasons.append(why)
        level = new_level
    assessment = Assessment(id=item.id, risk=level, base_risk=base, rules_fired=fired, trace=trace)
    if level == "none":
        assessment.issue = f"Resembles {cats}, but the evidence is weak and no rule fired — not flagged."
    else:
        assessment.issue = (f"{cats}: " + "; ".join(reasons) + ".") if reasons else f"Typical risky wording: {cats}."
    recs = [RECOMMENDATIONS_EN[c] for c in item.categories if c in RECOMMENDATIONS_EN]
    assessment.recommendation = " ".join(dict.fromkeys(recs)) or "Review this clause with a lawyer."
    return assessment


def apply_memory(assessment: Assessment, memory: Optional[MemoryContext]) -> Assessment:
    """Case-based step: reuse the user's rating of a very similar clause."""
    cases = (memory.similar_cases.get(assessment.id) if memory else None) or []
    if cases:
        best = cases[0]
        if best.user_risk != assessment.risk:
            assessment.trace.append(
                f"M Memory: you rated a similar clause (similarity {best.similarity:.2f}) as "
                f"'{best.user_risk}' → {assessment.risk} ⇒ {best.user_risk}")
            if assessment.risk == "none" or best.user_risk == "none":
                assessment.issue = (f"You rated a very similar clause as '{best.user_risk}' "
                                    f"(similarity {best.similarity:.2f}); the system now follows your rating.")
            assessment.risk = best.user_risk
            assessment.memory_applied = True
        else:
            assessment.trace.append(
                f"M Memory: similar clause (similarity {best.similarity:.2f}) was also rated "
                f"'{best.user_risk}' — confirmed")
    return assessment


def analyze_rules(items: list[AttentionItem], memory: Optional[MemoryContext] = None) -> dict[int, Assessment]:
    return {x.id: infer(x, memory) for x in items}


# ---------- optional LLM providers ----------
SYSTEM = ("You review contracts written in Russian for a party that is about to sign. For each clause "
          "rate the risk for that party. Answer only with a JSON array, no prose, no markdown. "
          "Element format: {\"id\": number, \"risk\": \"none|low|medium|high\", "
          "\"issue\": \"what is wrong, 1-2 sentences in English\", "
          "\"recommendation\": \"what to clarify or change, in English\"}. "
          "Do not give legal conclusions and never say the contract is safe to sign.")


def parse_llm_response(raw: str) -> dict[int, Assessment]:
    LOGGER.debug("Raw LLM response: %s", raw)
    raw_cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    match = re.search(r"(\[.*\]|\{.*\})", raw_cleaned, re.DOTALL)
    if match:
        raw_cleaned = match.group(0)
    try:
        parsed = json.loads(raw_cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"The model did not return JSON. Raw answer: {raw[:300]}") from exc
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError("The model answer must be a JSON array.")
    results: dict[int, Assessment] = {}
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("The model answer contains an element of the wrong format.")
        risk = item.get("risk")
        if type(item.get("id")) is not int or not isinstance(risk, str) or risk not in LEVELS:
            raise ValueError("The model answer has no valid id or risk level.")
        if item["id"] in results:
            raise ValueError("The model answer repeats a clause id.")
        results[item["id"]] = Assessment(id=item["id"], risk=risk, issue=str(item.get("issue", "")),
                                         recommendation=str(item.get("recommendation", "")))
    return results


def _prompt(chunk: list[AttentionItem]) -> str:
    return "Analyse these contract clauses:\n" + json.dumps(
        [{"id": x.id, "categories": [category_en(c) for c in x.categories], "text": x.text} for x in chunk],
        ensure_ascii=False)


def _check_ids(chunk_results: dict, chunk: list[AttentionItem], who: str):
    if set(chunk_results) != {item.id for item in chunk}:
        raise ValueError(f"{who} returned an incomplete or extra set of clauses.")


def analyze_anthropic(items: list[AttentionItem], batch: int = 10) -> dict[int, Assessment]:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY to use the Anthropic provider.")
    import anthropic
    client = anthropic.Anthropic()
    results = {}
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        r = client.messages.create(model=config.CLAUDE_MODEL, max_tokens=4000, system=SYSTEM,
                                   messages=[{"role": "user", "content": _prompt(chunk)}])
        text = next((b.text for b in r.content if getattr(b, "type", None) == "text"), "")
        chunk_results = parse_llm_response(text)
        _check_ids(chunk_results, chunk, "Claude")
        results.update(chunk_results)
    return results


def analyze_ollama(items: list[AttentionItem], batch: int = 1) -> dict[int, Assessment]:
    results = {}
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        payload = json.dumps({
            "model": config.OLLAMA_MODEL, "stream": False, "format": "json", "keep_alive": "5m",
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": _prompt(chunk)}],
            "options": {"temperature": 0},
        }).encode("utf-8")
        request = Request(config.OLLAMA_URL.rstrip("/") + "/api/chat", data=payload,
                          headers={"Content-Type": "application/json"}, method="POST")
        LOGGER.info("Sending request to Ollama: model %s, batch %d of %d",
                    config.OLLAMA_MODEL, i // batch + 1, (len(items) + batch - 1) // batch)
        try:
            with urlopen(request, timeout=600) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Ollama HTTP error {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"Cannot connect to Ollama ({config.OLLAMA_URL}). "
                               "Start Ollama and make sure the model is pulled.") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Ollama returned an invalid response.") from exc
        if not isinstance(data, dict):
            raise RuntimeError("Ollama returned a response of the wrong format.")
        if data.get("error"):
            raise RuntimeError(f"Ollama error: {data['error']}")
        message = data.get("message", {})
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise RuntimeError("Ollama returned a response without content.")
        chunk_results = parse_llm_response(message["content"])
        _check_ids(chunk_results, chunk, "Ollama")
        results.update(chunk_results)
    return results


def reason(attention: AttentionResult, memory: Optional[MemoryContext] = None,
           provider: str = "rules") -> ReasoningResult:
    """Assesses every selected clause; the rule trace is always produced."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")
    items = attention.items
    if not items:
        LOGGER.info("Provider actually used: none (no clauses selected)")
        return ReasoningResult(assessments={}, provider_name="LLM not called: no clauses selected")

    assessments = analyze_rules(items, memory)
    if provider in ("ollama", "anthropic"):
        llm = analyze_ollama(items) if provider == "ollama" else analyze_anthropic(items)
        for item_id, a in assessments.items():
            l = llm[item_id]
            a.trace.append(f"LLM: {l.risk} (rules said {a.risk}) → final = LLM")
            a.risk, a.issue = l.risk, l.issue or a.issue
            a.recommendation = l.recommendation or a.recommendation
    for a in assessments.values():
        apply_memory(a, memory)
    name = PROVIDER_NAMES[provider]
    LOGGER.info("Provider actually used: %s", name)
    return ReasoningResult(assessments=assessments, provider_name=name)
