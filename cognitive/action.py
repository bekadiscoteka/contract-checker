"""T6 Action: deliver the result to the outside world.

Input:  outputs of Perception, Attention, Reasoning and Decision (+ the human's final decision).
Output: ActionResult — a self-contained HTML report (downloadable / written to disk / sent by the
        Telegram bot) and a record of the human decision in the decision log (SQLite).
"""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import DISCLAIMER, category_en
from .decision import CRITERION_NAMES, params_dict
from .memory import MemoryStore
from .schemas import ActionResult, Assessment, AttentionItem, DecisionResult

COLORS = {"high": "#b3261e", "medium": "#b26a00", "low": "#2e7d32", "none": "#5d6b7a"}
VERDICTS = {"high": "High risk. Have a lawyer review the contract before signing.",
            "medium": "Medium risk. A lawyer's review is recommended.",
            "low": "Low risk, but the flagged clauses are worth clarifying.",
            "none": "No obvious red flags found. This is not a guarantee of safety."}


def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _md_bold(text: str) -> str:
    return _e(text).replace("**", "<b>", 1).replace("**", "</b>", 1)


def _dss_section(decision: DecisionResult, human_choice: Optional[str]) -> str:
    dss = decision.dss
    if dss is None:
        return ""
    rows = "".join(
        f"<tr{' class=rec' if a == dss.recommended else ''}><td>{i + 1}</td><td>{_e(dss.alternatives[a])}</td>"
        f"<td>{dss.matrix[a]['risk']:g}</td><td>{dss.matrix[a]['cost']:,.0f}</td>"
        f"<td>{dss.matrix[a]['delay']:g}</td><td>{dss.matrix[a]['value']:,.0f}</td>"
        f"<td><b>{dss.topsis[a]:.3f}</b></td><td>{dss.wsm[a]:.3f}</td></tr>"
        for i, a in enumerate(dss.ranking))
    weights = ", ".join(f"{_e(CRITERION_NAMES[k])} {v:.0%}" for k, v in dss.weights.items())
    expl = "".join(f"<li>{_md_bold(x)}</li>" for x in dss.explanation)
    human = (f'<p class="human"><b>Final decision made by the user:</b> {_e(dss.alternatives[human_choice])}'
             f'{" (differs from the recommendation)" if human_choice != dss.recommended else ""}</p>'
             if human_choice else '<p class="human">Final decision: <i>not made yet — it stays with the user.</i></p>')
    return f"""<section id="dss"><h2>Decision support: what to do with this contract</h2>
<p>Criteria weights: {weights}. Model: TOPSIS (WSM as a cross-check).</p>
<table><thead><tr><th>#</th><th>Alternative</th><th>Residual risk, pts</th><th>Cost, ₸</th>
<th>Delay, days</th><th>Expected value, ₸</th><th>TOPSIS</th><th>WSM</th></tr></thead><tbody>{rows}</tbody></table>
<ul>{expl}</ul>{human}</section>"""


def build_report(name: str, total: int, items: list[AttentionItem], assessments: dict[int, Assessment],
                 decision: DecisionResult, provider: str, human_choice: Optional[str] = None) -> str:
    analyzed_ids = {item.id for item in items if item.id in assessments}
    assessed_count = len(analyzed_ids)
    findings_count = sum(assessments[i].risk in {"low", "medium", "high"} for i in analyzed_ids)
    not_assessed_count = max(0, total - assessed_count)
    rows = []
    for x in items:
        a = assessments.get(x.id)
        risk = a.risk if a else None
        if risk in {"low", "medium", "high"}:
            status, color = f"Risk: {risk}", COLORS[risk]
        elif risk == "none":
            status, color = "No risk found", COLORS["none"]
        else:
            status, color = "Not assessed", COLORS["none"]
        details = ""
        if a and a.issue:
            details += f"<p><b>Issue:</b> {_e(a.issue)}</p>"
        if a and a.recommendation:
            details += f"<p><b>Recommendation:</b> {_e(a.recommendation)}</p>"
        if a and a.trace:
            details += "<details><summary>Inference steps</summary><ol>" + "".join(
                f"<li>{_e(t)}</li>" for t in a.trace) + "</ol></details>"
        number = x.clause.features.number
        label = f"Clause {number}" if number else f"Fragment #{x.id}"
        rows.append(
            f'<article class="it" style="border-left-color:{color}"><h3>{_e(label)}: '
            f'{_e(", ".join(category_en(c) for c in x.categories))} — {status}</h3>'
            f'<p class="q">{_e(x.text)}</p>{details}</article>')
    level = decision.level
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    findings_section = "".join(rows) or (
        '<p class="empty">No clauses were selected for review. '
        'This does not mean the document has no risks.</p>')
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Report: {_e(name)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:860px;margin:24px auto;padding:0 16px;color:#1c2530}}
section{{margin:18px 0}}.sum{{border:1px solid #d5dce3;border-radius:6px;padding:14px 18px}}
.it{{border:1px solid #d5dce3;border-left:5px solid;border-radius:6px;padding:4px 16px;margin-bottom:12px}}
.it h3{{font-size:1rem;margin:10px 0 4px}}.q{{color:#5d6b7a;font-style:italic}}
.w{{background:#fff4e0;border-left:4px solid #b26a00;padding:8px 12px;font-size:.9rem}}
.empty{{background:#f2f5f8;padding:12px;border-radius:5px}}dt{{font-weight:bold;margin-top:8px}}dd{{margin-left:0}}
table{{border-collapse:collapse;width:100%;font-size:.9rem}}th,td{{border:1px solid #d5dce3;padding:4px 8px;text-align:right}}
td:nth-child(2),th:nth-child(2){{text-align:left}}tr.rec{{background:#e8f3ea}}.human{{background:#eef3fb;padding:8px 12px;border-radius:5px}}
details{{font-size:.85rem;color:#3d4a57}}footer{{font-size:.85rem;color:#5d6b7a;margin-top:24px}}</style></head><body>
<h1>Contract report: {_e(name)}</h1>
<aside id="disclaimer" class="w">{DISCLAIMER}</aside>
<section id="summary"><div class="sum">
<h2 style="color:{COLORS[level]};margin-top:0">Result: {VERDICTS[level]}</h2>
<dl><dt>Fragments found by splitting</dt><dd>{total}</dd>
<dt>Fragments selected for analysis</dt><dd>{len(items)}</dd>
<dt>Fragments assessed</dt><dd>{assessed_count}</dd>
<dt>Fragments not assessed</dt><dd>{not_assessed_count}</dd>
<dt>Fragments with risk</dt><dd>{findings_count}</dd>
<dt>Risk score</dt><dd>{decision.score} (high-risk clauses: {decision.highs})</dd></dl>
</div></section>
{_dss_section(decision, human_choice)}
<section id="provider"><h2>Analysis provider</h2><p>{_e(provider)}</p></section>
<section id="findings"><h2>Clauses reviewed</h2>{findings_section}</section>
<footer>Generated: {_e(generated_at)}</footer>
</body></html>"""


def act(name: str, total: int, items: list[AttentionItem], assessments: dict[int, Assessment],
        decision: DecisionResult, provider: str, output_path=None,
        human_choice: Optional[str] = None) -> ActionResult:
    """Builds the report and writes it to disk if a path is given."""
    report = build_report(name, total, items, assessments, decision, provider, human_choice)
    path = None
    if output_path is not None:
        path = Path(output_path)
        path.write_text(report, encoding="utf-8")
    return ActionResult(report_html=report, report_path=path)


def record_decision(store: MemoryStore, name: str, decision: DecisionResult, chosen: str,
                    comment: str = "") -> int:
    """Writes the human's final decision next to the system's recommendation."""
    dss = decision.dss
    return store.log_decision(name, dss.recommended if dss else "", chosen,
                              dss.weights if dss else {}, params_dict(dss.params) if dss else {}, comment)
