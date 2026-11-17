#!/usr/bin/env python3
"""Command-line interface: run a contract through the cognitive cycle and save the HTML report.
Not legal advice.

    python contract_checker.py contract.pdf [-o report.html] [--provider rules|ollama|anthropic]

Streamlit UI:  streamlit run app.py
Telegram bot:  python telegram_bot.py
"""
import argparse
import logging
import os

from cognitive.config import DISCLAIMER, PROVIDERS
from cognitive.pipeline import run_pipeline

LOGGER = logging.getLogger("contract_checker")


def main():
    ap = argparse.ArgumentParser(description="Contract risk checker with decision support (not legal advice)")
    ap.add_argument("file")
    ap.add_argument("-o", "--output", default="report.html")
    ap.add_argument("--top", type=int, default=25, help="max clauses to analyse")
    ap.add_argument("--provider", choices=PROVIDERS, default=os.getenv("LLM_PROVIDER", "rules"),
                    help="analysis mode (default: rules)")
    ap.add_argument("--no-llm", action="store_true", help="rules only, no LLM")
    a = ap.parse_args()

    provider = "rules" if a.no_llm else a.provider
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    LOGGER.info("Contract checker started; configured provider: %s", provider)
    try:
        result = run_pipeline(a.file, provider=provider, top=a.top, output_path=a.output)
    except (OSError, RuntimeError, ValueError) as exc:
        ap.exit(2, f"Error: {exc}\n")
    d = result.decision
    print(f"Result: {d.level} risk, score {d.score}, high-risk clauses {d.highs}. Report: {a.output}")
    if d.dss:
        print(f"DSS recommendation: {d.dss.alternatives[d.dss.recommended]} "
              f"(runner-up: {d.dss.alternatives[d.dss.runner_up]})")
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
