"""Shared settings and constants."""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
PATTERNS_PATH = PROJECT_DIR / "risk_patterns.json"
MEMORY_DB = Path(os.getenv("MEMORY_DB", PROJECT_DIR / "cognitive_memory.db"))
DATA_DIR = PROJECT_DIR / "data"

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

PROVIDERS = ("rules", "ollama", "anthropic")
LEVELS = {"none": 0, "low": 1, "medium": 2, "high": 3}
LEVEL_NAMES = ["none", "low", "medium", "high"]
RU = {"none": "нет риска", "low": "низкий", "medium": "средний", "high": "высокий"}
DISCLAIMER = ("This report is generated automatically and is not legal advice. "
              "It highlights potentially risky wording for review by a lawyer. "
              "No findings does not mean the contract is safe. The final decision is yours.")
DISCLAIMER_RU = ("Отчёт создан автоматически и не является юридической консультацией. "
                 "Он помогает найти потенциально рискованные формулировки для проверки юристом. "
                 "Отсутствие находок не означает, что договор безопасен.")

# --- Money (Kazakhstan context). Values are approximate; update for the current year. ---
MRP_KZT = float(os.getenv("MRP_KZT", "4325"))  # monthly calculation index (МРП), tenge
FX_TO_KZT = {"KZT": 1.0, "MRP": MRP_KZT, "RUB": 6.0, "USD": 510.0, "EUR": 560.0}

# --- Risk categories from risk_patterns.json ---
CATEGORY_EN = {
    "Штрафы и неустойка": "Penalties and forfeits",
    "Автоматическое продление": "Automatic renewal",
    "Односторонний отказ или изменение условий": "Unilateral termination / changes",
    "Неограниченная ответственность": "Unlimited liability",
    "Невозвратная предоплата": "Non-refundable prepayment",
    "Конфиденциальность": "Confidentiality",
    "Форс-мажор": "Force majeure",
    "Подсудность и разрешение споров": "Jurisdiction and disputes",
    "Уступка прав и субподряд": "Assignment and subcontracting",
    "Интеллектуальная собственность": "Intellectual property",
    "Размытые формулировки": "Vague wording",
    "Расторжение и выход из договора": "Termination and exit",
}

# Expert prior: how likely the counterparty is to accept an amendment of this kind (0..1).
# Used by the DSS to estimate how much risk "sign with amendments" can remove.
NEGOTIABILITY = {
    "Штрафы и неустойка": 0.8,
    "Автоматическое продление": 0.9,
    "Односторонний отказ или изменение условий": 0.7,
    "Неограниченная ответственность": 0.6,
    "Невозвратная предоплата": 0.6,
    "Конфиденциальность": 0.7,
    "Форс-мажор": 0.3,
    "Подсудность и разрешение споров": 0.5,
    "Уступка прав и субподряд": 0.7,
    "Интеллектуальная собственность": 0.5,
    "Размытые формулировки": 0.9,
    "Расторжение и выход из договора": 0.6,
}

RECOMMENDATIONS_EN = {
    "Штрафы и неустойка": "Cap the total penalty (e.g. 10% of the contract value) and make it mutual.",
    "Автоматическое продление": "Require explicit renewal or a short notice period (30 days).",
    "Односторонний отказ или изменение условий": "Make the right mutual or require written consent of both parties.",
    "Неограниченная ответственность": "Set a reasonable, mutual liability cap and exclude lost profit only for both sides.",
    "Невозвратная предоплата": "Make the prepayment refundable for undelivered services.",
    "Конфиденциальность": "Limit what can be disclosed and to whom; make obligations mutual.",
    "Форс-мажор": "Check that force majeure is defined and applies to both parties.",
    "Подсудность и разрешение споров": "Agree on a neutral venue and keep a pre-trial claim procedure.",
    "Уступка прав и субподряд": "Require your consent for assignment to third parties.",
    "Интеллектуальная собственность": "Keep ownership of your materials; transfer only paid-for results.",
    "Размытые формулировки": "Replace discretionary wording with concrete scope, dates and amounts.",
    "Расторжение и выход из договора": "Add a mutual right to terminate with notice and fair settlement.",
}


def category_en(category: str) -> str:
    return CATEGORY_EN.get(category, category)
