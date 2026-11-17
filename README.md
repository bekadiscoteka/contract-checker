# Contract Risk Advisor: a cognitive system with a DSS element

The system reads a contract (PDF / DOCX / TXT, Russian-language, Kazakhstan context: tenge, MRP),
finds risky clauses, explains why they are risky, and helps the user decide **what to do with the
contract**: sign as is, sign with amendments, send to a lawyer, or reject.
The final decision stays with the user. **This is not legal advice.**

- **User:** a small-business owner or individual entrepreneur (ИП/ТОО) who signs contracts without an in-house lawyer.
- **Decision:** what to do with this contract.

## Quick start

Requires **Python 3.10–3.12**. Streamlit does not support Python 3.9.7.

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. In the sidebar, pick a sample contract from `data/` (or upload
your own file) and press **Run cognitive cycle**.
On the first run the multilingual embedding model (`paraphrase-multilingual-MiniLM-L12-v2`, ~470 MB)
is downloaded from Hugging Face. Without internet access the system falls back to keywords only.

Other entry points:

```bash
python contract_checker.py data/contract_1_dangerous.txt -o report.html   # CLI, HTML report
python -m cognitive.evaluation                                             # quality metrics
python -m unittest discover -s tests                                       # 24 tests
python telegram_bot.py      # optional Telegram channel (needs TELEGRAM_BOT_TOKEN)
```

## Cognitive cycle

Each function is a separate module in `cognitive/` with a typed input and output (`cognitive/schemas.py`).
`cognitive/pipeline.py` chains them, so the output of one stage is the input of the next.

| # | Module | Input → Output | Method | In the UI |
|---|---|---|---|---|
| T1 | `perception.py` | file → cleaned clauses + features | pdfplumber / python-docx / Tesseract OCR; cleaning; regex features: %, "per day", money (₸, MRP, ₽, $, €), periods → days | ① clause table with features, cleaning stats |
| T2 | `attention.py` | clauses → top-N risky clauses | keyword match + cosine similarity of multilingual embeddings to risky examples; evidence strength; prioritisation by score | ② attention score chart |
| T3 | `memory.py` | selected clauses → context | SQLite long-term memory: learned category weights, rated cases (case-based retrieval), analysed documents, decision log; Streamlit session = short-term memory | ③ weights base vs learned, previous run of the same document, similar rated clauses |
| T4 | `reasoning.py` | clauses + memory → risk per clause with trace | forward-chaining rules R1–R10 over the features (e.g. penalty ≥ 0.5 %/day → high; liability cap < 500 000 ₸ → high; mutual wording lowers risk) + case-based override; optional LLM (Ollama / Claude) | ④ risk donut, rules fired, inference steps per clause |
| T5 | `decision.py` | assessed clauses + user weights → ranked alternatives | scenario model + TOPSIS (main) / WSM (cross-check), sensitivity analysis | ⑤ DSS |
| T6 | `action.py` | decision → outside world | HTML report (download / CLI file / Telegram), decision log in SQLite | ⑥ download button, decision log |
| T7 | `learning.py` | user feedback → updated memory | stores rated cases + weight update `w ← clip(w + 0.3·(user − system), 0.5, 3.5)` | ⑦ rate clauses → re-run → weights and results before/after |

## DSS element

**Chain:** data (assessed clauses) → model (scenarios) → alternatives → criteria → recommendation → decision by the user.

- **Alternatives (4):** sign as is · sign with amendments · send to a lawyer · reject.
- **Criteria (4, weights set with sliders):**
  - residual risk (cost);
  - direct cost, ₸ (cost);
  - delay, days (cost);
  - expected deal value, ₸ (benefit).
- **Scenario model.** S is the total severity (low 1, medium 3, high 6). F is the part that can be negotiated away (severity × negotiability of the category). p, e and w are user parameters:
  - sign: risk S, value V;
  - amend: risk S − p·F, value V·(1−w);
  - lawyer: risk S·(1−e), cost L, value V·(1−w);
  - reject: risk 0, value 0.
- **Models.**
  - TOPSIS: closeness to the ideal point is the main model.
  - WSM: weighted sum, used as a cross-check.
  - Criteria are normalised against fixed references (risk 80 pts, deal value, 30 days). Min–max normalisation over the alternatives would make a nearly safe contract look risky just because "reject" has zero risk.
- **Explanation.** The UI shows:
  - the weighted contribution of each criterion;
  - a comparison with the runner-up;
  - whether TOPSIS and WSM agree;
  - which clauses to negotiate first.
- **What-if.** Every slider recomputes the ranking immediately. The UI also shows:
  - a one-at-a-time weight sweep with the points where the leader changes;
  - the range in which the recommendation holds;
  - Monte-Carlo robustness: share of wins under 400 randomly perturbed weight vectors.
- **Human in the loop.** The user picks the final option, which may differ from the recommendation. The choice is logged.

## Test data and quality

`data/` contains three contracts (dangerous, medium, balanced) and `labels.csv`, where every clause
(57 in total) is labelled none/low/medium/high by the author.
Run `python -m cognitive.evaluation` or open **⑦ Learning → Quality evaluation**:

| Metric (flag = medium/high) | Value |
|---|---|
| Precision | 0.87 |
| Recall | 0.77 |
| F1 | 0.82 |
| Exact level accuracy | 58% |
| Within one level | 90% |
| Document order dangerous > medium > balanced | preserved |

DSS on the samples with default weights: dangerous → *send to a lawyer*, medium → *send to a lawyer*
(amendments are a close second), balanced → *sign*.

## Optional LLM

The rules always run and produce the trace. An LLM can add its own assessment of each clause:

```bash
ollama pull qwen2.5:7b
set OLLAMA_MODEL=qwen2.5:7b      # PowerShell: $env:OLLAMA_MODEL="qwen2.5:7b"
```

Then choose **Ollama local LLM + rules** in the sidebar. A local 7B model on a CPU needs about 10–20 s per clause.
For Claude, set `ANTHROPIC_API_KEY`.

## Limitations and ethics

- Rules and patterns cover typical Russian-language contract wording. Unusual wording can be missed: recall 0.77, for example deemed acceptance and a waiver clause in the test set.
- The test set is small and labelled by one person. Rules were designed on the same contracts, so the metrics are optimistic.
- Clause splitting is heuristic. Text that is not selected is not checked.
- Scenario parameters (acceptance probability, lawyer effectiveness) are user estimates. The recommendation is only as good as these inputs, and the sensitivity analysis shows how fragile it is.
- Feedback from one user can bias the weights (bias risk). Memory can be reset in the sidebar.
- The system does not replace a lawyer and never states that a contract is safe. Responsibility for the decision stays with the human.
- Files sent to the Telegram bot pass through Telegram's servers. The Streamlit app and Ollama work locally.

## Project structure

```
app.py                  Streamlit UI (7 tabs + DSS)
contract_checker.py     CLI
telegram_bot.py         optional Telegram channel
cognitive/              perception, attention, memory, reasoning, decision, action, learning,
                        pipeline, schemas, config, evaluation
risk_patterns.json      knowledge base: 12 risk categories, keywords, example clauses
data/                   sample contracts + labels.csv
tests/                  unit and integration tests
```
