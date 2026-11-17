"""Streamlit UI for the cognitive contract-risk system with a DSS element.

Run:  streamlit run app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from cognitive import action, decision, evaluation, learning
from cognitive.config import DATA_DIR, DISCLAIMER, PROVIDERS, category_en
from cognitive.decision import ALTERNATIVES, CRITERIA, CRITERION_NAMES, DEFAULT_WEIGHTS
from cognitive.memory import MemoryStore
from cognitive.pipeline import run_pipeline
from cognitive.schemas import DSSParams, FeedbackItem

st.set_page_config(page_title="Contract Risk DSS", page_icon="📄", layout="wide")

RISK_COLORS = {"high": "#c62828", "medium": "#ef8f00", "low": "#2e7d32", "none": "#90a4ae"}
RISK_ICON = {"high": "🔴", "medium": "🟠", "low": "🟢", "none": "⚪"}
ALT_COLORS = {"sign": "#90a4ae", "amend": "#1e88e5", "lawyer": "#43a047", "reject": "#e53935"}
SHORT = {"sign": "Sign", "amend": "Amend", "lawyer": "Lawyer", "reject": "Reject"}
PROVIDER_LABELS = {"rules": "Rules only (fast, offline)", "ollama": "Ollama local LLM + rules",
                   "anthropic": "Anthropic Claude + rules"}


@st.cache_resource
def get_store() -> MemoryStore:
    return MemoryStore()


store = get_store()
ss = st.session_state
for key, default in DEFAULT_WEIGHTS.items():
    ss.setdefault(f"w_{key}", int(default * 100))


def plot(fig, height=320):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")


def current_params() -> DSSParams:
    return DSSParams(
        deal_value=ss.get("p_value", 5_000_000), lawyer_cost=ss.get("p_lawyer", 150_000),
        negotiation_cost=ss.get("p_neg_cost", 30_000), p_accept=ss.get("p_accept", 0.6),
        lawyer_effectiveness=ss.get("p_eff", 0.8), walk_away=ss.get("p_walk", 0.15),
        negotiation_days=ss.get("p_neg_days", 7), lawyer_days=ss.get("p_law_days", 14))


def current_weights() -> dict:
    return {k: ss[f"w_{k}"] for k in CRITERION_NAMES}


def run_cycle(path: str, name: str):
    with st.spinner("Running the cognitive cycle… (first run loads the embedding model)"):
        ss.cycle = run_pipeline(path, provider=ss.provider, top=ss.top, source_name=name,
                                weights=current_weights(), params=current_params(), store=store)
    ss.file_path, ss.file_name = path, name
    ss.human_choice = None


# ---------------- sidebar: input ----------------
with st.sidebar:
    st.title("📄 Contract Risk Advisor")
    st.caption("Cognitive system with a decision-support element. Not legal advice.")
    source = st.radio("Contract source", ["Sample contract", "Upload my file"], horizontal=True)
    samples = sorted(DATA_DIR.glob("*.txt"))
    chosen_path, chosen_name = None, None
    if source == "Sample contract" and samples:
        pick = st.selectbox("Sample", samples, format_func=lambda p: p.stem.replace("_", " "))
        chosen_path, chosen_name = str(pick), pick.name
    else:
        up = st.file_uploader("PDF, DOCX or TXT", type=["pdf", "docx", "txt"])
        if up is not None:
            tmp = Path(tempfile.gettempdir()) / f"contract_upload{Path(up.name).suffix.lower()}"
            tmp.write_bytes(up.getvalue())
            chosen_path, chosen_name = str(tmp), up.name
    st.selectbox("Reasoning provider", PROVIDERS, key="provider", format_func=PROVIDER_LABELS.get,
                 help="Rules always run and produce the inference trace; an LLM can add its own assessment.")
    st.slider("Max clauses to analyse (attention budget)", 5, 40, 25, key="top")
    if st.button("▶ Run cognitive cycle", type="primary", disabled=chosen_path is None, width="stretch"):
        run_cycle(chosen_path, chosen_name)
    with st.expander("Memory"):
        st.write(f"Documents analysed: **{store.count('documents')}** · ratings: **{store.count('feedback')}**")
        if st.checkbox("I want to erase all memory") and st.button("Reset memory"):
            store.reset()
            st.success("Memory cleared.")

cycle = ss.get("cycle")
if cycle is None:
    st.header("Check my contract")
    st.write("Pick a sample or upload a contract in the sidebar and press **Run cognitive cycle**. "
             "The system goes through **Perception → Attention → Memory → Reasoning → Decision → Action → "
             "Learning** and helps you decide what to do with the contract.")
    st.info(DISCLAIMER)
    st.stop()

# live DSS: recomputed on every widget change (what-if)
dec = decision.decide(cycle.attention, cycle.reasoning, current_weights(), current_params())
dss = dec.dss
A = cycle.reasoning.assessments

# ---------------- header ----------------
st.subheader(f"{cycle.perception.source_name}")
m = st.columns(5)
m[0].metric("Overall risk", f"{RISK_ICON[dec.level]} {dec.level}")
m[1].metric("Risk score", dec.score)
m[2].metric("High-risk clauses", dec.highs)
m[3].metric("Clauses analysed", f"{len(cycle.attention.items)} / {cycle.attention.total_clauses}")
m[4].metric("DSS recommends", SHORT[dss.recommended], help=ALTERNATIVES[dss.recommended])

stage_out = [
    ("① Perception", f"{len(cycle.perception.clauses)} clauses"),
    ("② Attention", f"{len(cycle.attention.items)} selected"),
    ("③ Memory", f"{sum(bool(v) for v in cycle.memory.similar_cases.values())} similar cases"),
    ("④ Reasoning", f"{sum(a.risk != 'none' for a in A.values())} risks"),
    ("⑤ Decision", ALTERNATIVES[dss.recommended]),
    ("⑥ Action", "report ready" if not ss.get("human_choice") else "decision logged"),
    ("⑦ Learning", f"{cycle.memory.feedback_count} ratings"),
]
cols = st.columns(len(stage_out))
for c, (title, out) in zip(cols, stage_out):
    c.markdown(f"<div style='border:1px solid #d0d7de;border-radius:8px;padding:6px 8px;text-align:center'>"
               f"<b>{title}</b><br><span style='font-size:.85rem'>{out}</span></div>", unsafe_allow_html=True)
st.write("")

tabs = st.tabs(["① Perception", "② Attention", "③ Memory", "④ Reasoning", "⑤ Decision · DSS",
                "⑥ Action", "⑦ Learning"])

# ---------------- ① Perception ----------------
with tabs[0]:
    p = cycle.perception
    st.caption("Parse the file → clean text → split into clauses → extract numeric features.")
    c = st.columns(4)
    c[0].metric("Characters (raw → clean)", f"{p.cleaning_stats.get('chars_before', 0)} → {p.cleaning_stats.get('chars_after', 0)}")
    c[1].metric("Hyphenations fixed", p.cleaning_stats.get("hyphenations_fixed", 0))
    c[2].metric("Page numbers removed", p.cleaning_stats.get("page_numbers_removed", 0))
    c[3].metric("Clauses", len(p.clauses))
    rows = [{"#": cl.id, "Clause": cl.features.number or "", "Text": cl.text[:110] + ("…" if len(cl.text) > 110 else ""),
             "%": ", ".join(f"{x:g}" for x in cl.features.percents),
             "per day": "✓" if cl.features.percent_per_day else "",
             "Money": ", ".join(f"{a:,.0f} {cur}" for a, cur in cl.features.money),
             "Periods (days)": ", ".join(f"{d:g}" for d in cl.features.durations_days)} for cl in p.clauses]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    with st.expander("Cleaned text"):
        st.text(p.cleaned_text)

# ---------------- ② Attention ----------------
with tabs[1]:
    at = cycle.attention
    st.caption(f"Keyword match + embedding similarity to risky examples. Threshold {at.threshold}, "
               f"embeddings {'on' if at.embeddings_used else 'off (keywords only)'}; "
               f"{at.dropped} relevant clauses cut by the budget.")
    if at.items:
        df = pd.DataFrame([{"Clause": it.clause.features.number or f"#{it.id}", "Score": it.score,
                            "Category": category_en(it.categories[0]),
                            "All categories": ", ".join(category_en(c) for c in it.categories),
                            "Keywords": ", ".join(it.keyword_hits),
                            "Similarity": max(it.similarities.values()) if it.similarities else 0}
                           for it in at.items])
        fig = px.bar(df.iloc[::-1], x="Score", y="Clause", color="Category", orientation="h",
                     hover_data=["All categories", "Keywords", "Similarity"], title="Attention score by clause")
        fig.update_yaxes(type="category", title="Clause",   # "4.1" is a label, not a number
                         categoryorder="array", categoryarray=list(df["Clause"])[::-1])
        plot(fig, height=max(320, 24 * len(df)))
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.info("No clause looked risky enough to deserve attention.")

# ---------------- ③ Memory ----------------
with tabs[2]:
    mem = cycle.memory
    st.caption("Long-term memory (SQLite): learned category weights, rated cases, analysed documents, decisions.")
    c1, c2 = st.columns(2)
    with c1:
        wdf = pd.DataFrame([{"Category": category_en(pt["category"]), "Base": mem.base_weights.get(pt["category"]),
                             "Learned": pt["weight"]} for pt in mem.patterns])
        fig = go.Figure([go.Bar(name="Base weight", y=wdf["Category"], x=wdf["Base"], orientation="h", marker_color="#b0bec5"),
                         go.Bar(name="Current (learned)", y=wdf["Category"], x=wdf["Learned"], orientation="h", marker_color="#1e88e5")])
        fig.update_layout(barmode="group", title="Knowledge: category weights")
        plot(fig, height=420)
    with c2:
        prev = mem.previous_run
        if prev:
            st.markdown(f"**This document was analysed before** ({prev['analyzed_at']}).")
            cc = st.columns(2)
            cc[0].metric("Score then → now", dec.score, delta=dec.score - prev["score"], delta_color="inverse")
            cc[1].metric("Level then → now", dec.level, delta=None if prev["level"] == dec.level else f"was {prev['level']}",
                         delta_color="off")
            changed = [{"Clause": next((it.clause.features.number for it in cycle.attention.items if it.id == i), i),
                        "Before": r, "Now": A[i].risk} for i, r in prev["findings"].items() if i in A and A[i].risk != r]
            if changed:
                st.dataframe(pd.DataFrame(changed), hide_index=True, width="stretch")
        else:
            st.info("First time this document is seen.")
        sim_rows = [{"Clause": next(it.clause.features.number or f"#{it.id}" for it in cycle.attention.items if it.id == i),
                     "Similar past clause": cs[0].clause_text[:80] + "…", "Your rating": cs[0].user_risk,
                     "Similarity": cs[0].similarity} for i, cs in mem.similar_cases.items() if cs]
        st.markdown("**Similar clauses you rated before** (case-based memory)")
        if sim_rows:
            st.dataframe(pd.DataFrame(sim_rows), hide_index=True, width="stretch")
        else:
            st.caption("None yet — rate clauses in the Learning tab.")
    with st.expander(f"Episodic memory: analysed documents ({mem.documents_seen})"):
        st.dataframe(pd.DataFrame(store.documents()), hide_index=True, width="stretch")

# ---------------- ④ Reasoning ----------------
with tabs[3]:
    st.caption(f"Forward-chaining rules over clause features + case-based memory. Provider: {cycle.reasoning.provider_name}.")
    if A:
        c1, c2 = st.columns(2)
        dist = pd.Series([a.risk for a in A.values()]).value_counts().reindex(["high", "medium", "low", "none"]).fillna(0)
        fig = px.pie(values=dist.values, names=dist.index, hole=0.5, color=dist.index,
                     color_discrete_map=RISK_COLORS, title="Risk levels of selected clauses")
        with c1:
            plot(fig)
        fired = pd.Series([r for a in A.values() for r in a.rules_fired]).value_counts()
        with c2:
            if len(fired):
                plot(px.bar(x=fired.index, y=fired.values, labels={"x": "Rule", "y": "Times fired"},
                            title="Rules fired"))
        order = {"high": 0, "medium": 1, "low": 2, "none": 3}
        for it in sorted(cycle.attention.items, key=lambda i: order[A[i.id].risk] if i.id in A else 9):
            a = A.get(it.id)
            if a is None:
                continue
            label = it.clause.features.number or f"#{it.id}"
            mem_tag = " · 🧠 memory" if a.memory_applied else ""
            with st.expander(f"{RISK_ICON[a.risk]} {label} · {category_en(it.categories[0])} — **{a.risk}**{mem_tag}"):
                st.markdown(f"> {it.text}")
                st.markdown(f"**Issue:** {a.issue}\n\n**Recommendation:** {a.recommendation}")
                st.markdown("**Inference steps**\n" + "\n".join(f"{n}. {t}" for n, t in enumerate(a.trace, 1)))
    else:
        st.info("Nothing to reason about.")

# ---------------- ⑤ Decision · DSS ----------------
with tabs[4]:
    st.caption("data → scenario model → 4 alternatives → 4 weighted criteria → TOPSIS ranking → recommendation → "
               "**your decision**.")
    left, right = st.columns([1, 2])
    with left:
        st.markdown("**Criteria weights** (normalized to 100%)")
        for crit in CRITERIA:
            st.slider(f"{crit['name']} ({crit['type']}, {crit['unit']})", 0, 100, key=f"w_{crit['key']}")
        st.caption("Normalized: " + ", ".join(f"{CRITERION_NAMES[k]} {v:.0%}" for k, v in dss.weights.items()))
        st.markdown("**Scenario parameters**")
        st.number_input("Deal value, ₸", 0, 10_000_000_000, 5_000_000, 500_000, key="p_value")
        st.number_input("Lawyer cost, ₸", 0, 100_000_000, 150_000, 10_000, key="p_lawyer")
        st.number_input("Negotiation cost (your time), ₸", 0, 100_000_000, 30_000, 5_000, key="p_neg_cost")
        st.slider("P(counterparty accepts amendments)", 0.0, 1.0, 0.6, 0.05, key="p_accept")
        st.slider("Share of risk a lawyer removes", 0.0, 1.0, 0.8, 0.05, key="p_eff")
        st.slider("P(counterparty walks away if you push back)", 0.0, 1.0, 0.15, 0.05, key="p_walk")
        st.slider("Negotiation delay, days", 0, 60, 7, key="p_neg_days")
        st.slider("Lawyer delay, days", 0, 60, 14, key="p_law_days")
    with right:
        st.success(f"### Recommendation: {ALTERNATIVES[dss.recommended]}")
        for line in dss.explanation:
            st.markdown(f"- {line}")
        rank_df = pd.DataFrame([{"Alternative": ALTERNATIVES[a], "key": a, "TOPSIS": dss.topsis[a], "WSM": dss.wsm[a]}
                                for a in dss.ranking])
        fig = go.Figure([go.Bar(name="TOPSIS", x=rank_df["Alternative"], y=rank_df["TOPSIS"],
                                marker_color=[ALT_COLORS[k] for k in rank_df["key"]], text=rank_df["TOPSIS"].round(3)),
                         go.Bar(name="WSM", x=rank_df["Alternative"], y=rank_df["WSM"], marker_color="#cfd8dc",
                                text=rank_df["WSM"].round(3))])
        fig.update_layout(barmode="group", title="Ranking (higher is better)", yaxis_range=[0, 1])
        plot(fig)
        mat = pd.DataFrame([{"Rank": n + 1, "Alternative": ALTERNATIVES[a],
                             **{f"{c['name']} ({c['type']}, {c['unit']})": dss.matrix[a][c["key"]] for c in CRITERIA}}
                            for n, a in enumerate(dss.ranking)])
        st.dataframe(mat, hide_index=True, width="stretch")

        contrib = pd.DataFrame([{"Alternative": ALTERNATIVES[a], "Criterion": CRITERION_NAMES[c], "Contribution": v}
                                for a in dss.ranking for c, v in dss.wsm_contributions[a].items()])
        plot(px.bar(contrib, y="Alternative", x="Contribution", color="Criterion", orientation="h",
                    title="Why: weighted contribution of each criterion (WSM)")
             .update_layout(legend=dict(orientation="h", y=-0.25)).update_yaxes(autorange="reversed"), height=300)
        stab = pd.DataFrame([{"Alternative": ALTERNATIVES[a], "key": a, "Wins": v} for a, v in dss.stability.items()])
        plot(px.bar(stab, y="Alternative", x="Wins", color="key", color_discrete_map=ALT_COLORS, orientation="h",
                    title="Robustness: share of wins when weights are randomly perturbed (400 samples)")
             .update_layout(showlegend=False).update_xaxes(tickformat=".0%", range=[0, 1]), height=240)

        st.markdown("**What-if: sensitivity to one weight**")
        crit = st.selectbox("Criterion", list(CRITERION_NAMES), format_func=CRITERION_NAMES.get)
        sens = dss.sensitivity[crit]
        curve = pd.DataFrame(sens["curve"])
        fig = go.Figure([go.Scatter(x=curve["weight"], y=curve[a], name=ALTERNATIVES[a], mode="lines",
                                    line=dict(color=ALT_COLORS[a], width=3)) for a in ALTERNATIVES])
        fig.add_vline(x=sens["current"], line_dash="dash", annotation_text="current")
        for sw in sens["switches"]:
            fig.add_vline(x=sw["weight"], line_dash="dot", line_color="#999")
        fig.update_layout(title=f"TOPSIS score vs weight of «{CRITERION_NAMES[crit]}»",
                          xaxis_title="weight", yaxis_title="closeness", xaxis_tickformat=".0%")
        plot(fig)
        if sens["switches"]:
            st.caption("Leader changes at: " + "; ".join(
                f"{s['weight']:.0%} ({ALTERNATIVES[s['from']]} → {ALTERNATIVES[s['to']]})" for s in sens["switches"]))

        st.divider()
        st.markdown("**Your decision** — the system only advises; you decide.")
        keys = list(ALTERNATIVES)
        choice = st.radio("I will…", keys, index=keys.index(dss.recommended), format_func=ALTERNATIVES.get,
                          horizontal=True)
        comment = st.text_input("Comment (optional)")
        if st.button("✅ Confirm my decision", type="primary"):
            action.record_decision(store, cycle.perception.source_name, dec, choice, comment)
            ss.human_choice = choice
            st.success(f"Logged: {ALTERNATIVES[choice]}"
                       + ("" if choice == dss.recommended else " (you overrode the recommendation)"))

# ---------------- ⑥ Action ----------------
with tabs[5]:
    st.caption("The result leaves the system: downloadable HTML report + decision log (SQLite). "
               "The same report is sent by the Telegram bot (telegram_bot.py).")
    report = action.build_report(cycle.perception.source_name, cycle.attention.total_clauses, cycle.attention.items,
                                 A, dec, cycle.reasoning.provider_name, ss.get("human_choice"))
    st.download_button("⬇ Download report (HTML)", report, file_name=f"{Path(cycle.perception.source_name).stem}_report.html",
                       mime="text/html", type="primary")
    st.markdown("**Decision log**")
    log = store.decisions()
    if log:
        df = pd.DataFrame(log)[["created_at", "document_name", "recommended", "chosen", "comment"]]
        df["recommended"] = df["recommended"].map(ALTERNATIVES.get)
        df["chosen"] = df["chosen"].map(ALTERNATIVES.get)
        st.dataframe(df, hide_index=True, width="stretch")
    else:
        st.caption("No decisions yet — confirm one in the Decision tab.")
    with st.expander("Report preview", expanded=False):
        components.html(report, height=600, scrolling=True)

# ---------------- ⑦ Learning ----------------
with tabs[6]:
    st.caption("Correct the system's ratings → it stores the cases and updates category weights "
               "(w ← w + 0.3·(your level − system level)). Then the cycle re-runs to show the change.")
    items = [it for it in cycle.attention.items if it.id in A]
    if not items:
        st.info("Nothing to rate.")
    else:
        edit_df = pd.DataFrame([{"id": it.id, "Clause": it.clause.features.number or f"#{it.id}",
                                 "Category": category_en(it.categories[0]), "Text": it.text[:90] + "…",
                                 "System": A[it.id].risk, "Your rating": A[it.id].risk} for it in items])
        edited = st.data_editor(
            edit_df, hide_index=True, width="stretch", disabled=["id", "Clause", "Category", "Text", "System"],
            column_config={"id": None, "Your rating": st.column_config.SelectboxColumn(
                options=["none", "low", "medium", "high"], required=True)},
            key=f"feedback_editor_{ss.get('editor_version', 0)}")
        diff = edited[edited["Your rating"] != edited["System"]]
        st.write(f"Corrections: **{len(diff)}**")
        if st.button("🎓 Teach the system and re-run", type="primary", disabled=len(diff) == 0):
            by_id = {it.id: it for it in items}
            fb = [FeedbackItem(clause_text=by_id[int(r["id"])].text, categories=by_id[int(r["id"])].categories,
                               system_risk=r["System"], user_risk=r["Your rating"],
                               document_name=cycle.perception.source_name)
                  for _, r in diff.iterrows()]
            before = {"score": dec.score, "level": dec.level, "rec": dss.recommended,
                      "risks": {i: a.risk for i, a in A.items()}}
            result = learning.learn(store, fb)
            run_cycle(ss.file_path, ss.file_name)
            ss.learning_view = {"before": before, "updates": result.weight_updates}
            ss.editor_version = ss.get("editor_version", 0) + 1   # fresh editor: rows may have changed
            st.rerun()
    view = ss.get("learning_view")
    if view:
        st.markdown("### Behaviour after learning")
        b = view["before"]
        c = st.columns(3)
        c[0].metric("Risk score", dec.score, delta=dec.score - b["score"], delta_color="inverse")
        c[1].metric("Overall level", dec.level, delta=None if b["level"] == dec.level else f"was {b['level']}", delta_color="off")
        c[2].metric("DSS recommendation", ALTERNATIVES[dss.recommended],
                    delta=None if b["rec"] == dss.recommended else f"was {ALTERNATIVES[b['rec']]}", delta_color="off")
        if view["updates"]:
            up = pd.DataFrame([{"Category": category_en(u["category"]), "Before": u["old"], "After": u["new"]}
                               for u in view["updates"]])
            fig = go.Figure([go.Bar(name="Before", x=up["Category"], y=up["Before"], marker_color="#b0bec5"),
                             go.Bar(name="After", x=up["Category"], y=up["After"], marker_color="#1e88e5")])
            fig.update_layout(barmode="group", title="Category weights changed by learning")
            plot(fig)
        changed = [{"Clause": it.clause.features.number or f"#{it.id}", "Before": b["risks"].get(it.id),
                    "After": A[it.id].risk, "Why": "memory of your rating" if A[it.id].memory_applied else "new weight"}
                   for it in cycle.attention.items if it.id in A and b["risks"].get(it.id) != A[it.id].risk]
        if changed:
            st.dataframe(pd.DataFrame(changed), hide_index=True, width="stretch")
    with st.expander("Learning log"):
        st.dataframe(pd.DataFrame(store.learning_log()), hide_index=True, width="stretch")
    with st.expander("Quality evaluation on the labelled test set (57 clauses, empty memory)"):
        if st.button("Run evaluation"):
            with st.spinner("Evaluating…"):
                ss.evaluation = evaluation.evaluate()
        ev = ss.get("evaluation")
        if ev:
            c = st.columns(5)
            c[0].metric("Precision (flag)", f"{ev['precision']:.2f}")
            c[1].metric("Recall (flag)", f"{ev['recall']:.2f}")
            c[2].metric("F1", f"{ev['f1']:.2f}")
            c[3].metric("Exact level", f"{ev['exact_accuracy']:.0%}")
            c[4].metric("Within one level", f"{ev['within_one_accuracy']:.0%}")
            levels = ["none", "low", "medium", "high"]
            z = [[ev["confusion"][g][p] for p in levels] for g in levels]
            fig = px.imshow(z, x=levels, y=levels, text_auto=True, color_continuous_scale="Blues",
                            labels=dict(x="Predicted", y="Gold label", color="Clauses"), title="Confusion matrix")
            plot(fig, height=360)
            st.dataframe(pd.DataFrame(ev["documents"]), hide_index=True, width="stretch")
            st.caption("Flag = medium or high. Labels were made by the author — a small, single-annotator test set.")
