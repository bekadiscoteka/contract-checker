"""T1 Perception: read the document, clean it, split into clauses and extract features.

Input:  path to a PDF / DOCX / TXT file.
Output: PerceptionResult — cleaned text and Clause objects with ClauseFeatures.

Methods:
- parsing: pdfplumber (Tesseract OCR for scans), python-docx, plain text;
- cleaning: join hyphenated line breaks, drop page numbers, normalize quotes/dashes/spaces;
- splitting: by clause numbering, otherwise by blank lines;
- features (regex): clause number, percents and "for every day", money with currency
  (tenge, MRP, roubles, dollars, euro), periods (hours/days/weeks/months/years → days).
  Reasoning uses these numbers in its rules.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .schemas import Clause, ClauseFeatures, PerceptionResult


def load_text(path) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        import pdfplumber
        with pdfplumber.open(p) as pdf:
            text = "\n".join((pg.extract_text() or "") for pg in pdf.pages)
        if len(text.strip()) < 100:  # probably a scan: try OCR
            try:
                from pdf2image import convert_from_path
                import pytesseract
                text = "\n".join(pytesseract.image_to_string(im, lang="rus+eng")
                                 for im in convert_from_path(str(p)))
            except Exception as e:
                raise RuntimeError(
                    f"Looks like a scanned PDF but OCR is unavailable ({e}). "
                    "Install Tesseract and Poppler, plus pytesseract and pdf2image."
                ) from e
        return text
    if ext == ".docx":
        import docx
        d = docx.Document(str(p))
        parts = [x.text for x in d.paragraphs]
        for t in d.tables:
            for row in t.rows:
                parts.append(" ".join(c.text for c in row.cells))
        return "\n".join(parts)
    return p.read_text(encoding="utf-8", errors="ignore")


# ---------- Очистка ----------
_CHAR_MAP = str.maketrans({"«": '"', "»": '"', "„": '"', "“": '"', "”": '"',
                           "—": "-", "–": "-", " ": " ", " ": " ", "­": ""})
_PAGE_NUMBER = re.compile(r"^\s*(?:стр\.?|страница|page)?\s*\d{1,3}\s*(?:из\s*\d{1,3})?\s*$",
                          re.I | re.M)
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


def clean_text(text: str) -> tuple[str, dict[str, int]]:
    """Нормализует текст и возвращает его вместе со статистикой изменений."""
    stats = {"chars_before": len(text)}
    text = text.replace("\r\n", "\n").replace("\r", "\n").translate(_CHAR_MAP)
    text, stats["hyphenations_fixed"] = _HYPHEN_BREAK.subn(r"\1\2", text)
    text, stats["page_numbers_removed"] = _PAGE_NUMBER.subn("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    stats["chars_after"] = len(text)
    return text, stats


def split_clauses(text: str) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text)
    parts = re.split(r"\n(?=\s*\d+(?:\.\d+)*[.)]?\s)", text)
    if len(parts) < 2:
        parts = re.split(r"\n\s*\n", text)
    out = [re.sub(r"\s*\n\s*", " ", s).strip() for s in parts]
    return [s for s in out if len(s) > 30 and not _is_heading(s)]


def _is_heading(text: str) -> bool:
    """Section titles such as "5. СРОК ДЕЙСТВИЯ И РАСТОРЖЕНИЕ" are not clauses."""
    letters = [ch for ch in text if ch.isalpha()]
    return (len(text.split()) <= 10 and bool(letters)
            and sum(ch.isupper() for ch in letters) / len(letters) > 0.8)


# ---------- Признаки ----------
_WORD_NUMBERS = {
    "одного": 1, "один": 1, "одна": 1, "одной": 1, "одну": 1,
    "двух": 2, "два": 2, "две": 2, "трёх": 3, "трех": 3, "три": 3,
    "пяти": 5, "пять": 5, "десяти": 10, "десять": 10,
}
_NUM = (r"\b(?P<num>\d{1,3}(?:[ ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?|"
        + "|".join(sorted(_WORD_NUMBERS, key=len, reverse=True)) + r")")
_PAREN = r"(?:\s*\([^)]{1,40}\))?"             # «3 (три) года»
_CLAUSE_NUMBER = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)]?\s")
_PERCENT = re.compile(_NUM + _PAREN + r"\s*(?:%|процент\w*)", re.I)
_PER_DAY = re.compile(r"кажд\w*\s+(?:календарн\w*\s+|рабоч\w*\s+)?(?:день|дня)|в\s+день|ежедневн", re.I)
_CURRENCIES = {  # шаблон после числа → код валюты
    "KZT": r"тенге|тг\.?(?!\w)|₸|kzt\b",
    "MRP": r"мрп\b|месячн\w+\s+расчётн\w+\s+показател\w*|месячн\w+\s+расчетн\w+\s+показател\w*",
    "RUB": r"руб\w*|₽|rub\b",
    "USD": r"долл\w*|\$|usd\b",
    "EUR": r"евро\b|€|eur\b",
}
_MONEY = re.compile(
    _NUM + _PAREN + r"\s*(?P<mult>тыс\.?|тысяч\w*|млн\.?|миллион\w*|млрд\.?|миллиард\w*)?\s*(?P<cur>"
    + "|".join(f"(?P<{code}>{pattern})" for code, pattern in _CURRENCIES.items()) + ")",
    re.I)
_DURATION = re.compile(_NUM + _PAREN + r"\s*(?:календарн\w*\s+|рабоч\w*\s+)?"
                       r"(?P<unit>час(?:а|ов)?|дн(?:я|ей|и)|день|недел(?:я|и|ь|ю)|"
                       r"месяц(?:а|ев)?|год(?:а|ов)?|лет)\b", re.I)
_UNIT_DAYS = {"час": 1 / 24, "дн": 1, "ден": 1, "нед": 7, "мес": 30, "год": 365, "лет": 365}


def _to_number(raw: str) -> float:
    raw = raw.lower()
    if raw in _WORD_NUMBERS:
        return float(_WORD_NUMBERS[raw])
    return float(raw.replace(" ", "").replace(",", "."))


def _unit_days(unit: str) -> float:
    unit = unit.lower()
    for prefix, days in _UNIT_DAYS.items():
        if unit.startswith(prefix):
            return days
    raise ValueError(unit)


def extract_features(text: str) -> ClauseFeatures:
    """Извлекает из фрагмента числовые признаки, нужные для правил рассуждения."""
    number_match = _CLAUSE_NUMBER.match(text)
    body = text[number_match.end():] if number_match else text  # номер пункта — не число условия
    percents = [_to_number(m["num"]) for m in _PERCENT.finditer(body)]
    money = []
    for m in _MONEY.finditer(body):
        mult = (m["mult"] or "").lower()
        factor = (1_000_000_000 if mult.startswith(("млрд", "миллиард")) else
                  1_000_000 if mult.startswith(("млн", "миллион")) else
                  1_000 if mult.startswith("тыс") else 1)
        currency = next(code for code in _CURRENCIES if m[code])
        money.append((_to_number(m["num"]) * factor, currency))
    durations = [round(_to_number(m["num"]) * _unit_days(m["unit"]), 3) for m in _DURATION.finditer(body)
                 if not (m["unit"].lower().startswith("год") and _to_number(m["num"]) >= 1900)]  # "2027 года" is a date
    return ClauseFeatures(
        number=number_match.group(1) if number_match else None,
        percents=percents,
        percent_per_day=bool(percents) and bool(_PER_DAY.search(body)),
        money=money,
        durations_days=durations,
        word_count=len(body.split()),
    )


def perceive(path, source_name: str | None = None) -> PerceptionResult:
    """Читает файл, очищает текст и возвращает фрагменты с признаками."""
    raw = load_text(path)
    cleaned, stats = clean_text(raw)
    clauses = [Clause(id=i + 1, text=t, features=extract_features(t))
               for i, t in enumerate(split_clauses(cleaned))]
    if not clauses:
        raise ValueError("Could not find any contract clauses in the document.")
    return PerceptionResult(source_name=source_name or Path(path).name, raw_text=raw,
                            clauses=clauses, cleaned_text=cleaned, cleaning_stats=stats,
                            doc_hash=hashlib.sha1(cleaned.encode("utf-8")).hexdigest())
