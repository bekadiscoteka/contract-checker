#!/usr/bin/env python3
"""Сканер договоров: ищет рискованные пункты и строит отчёт.
Не является юридической консультацией.

CLI:  python contract_checker.py договор.pdf [-o report.html] [--provider rules|ollama|anthropic]
Telegram-бот: python telegram_bot.py
"""
import argparse, html, json, logging, os, re, sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HERE = Path(__file__).parent
LOGGER = logging.getLogger(__name__)
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LEVELS = {"none": 0, "low": 1, "medium": 2, "high": 3}
RU = {"none": "нет риска", "low": "низкий", "medium": "средний", "high": "высокий"}
DISCLAIMER = ("Отчёт создан автоматически и не является юридической консультацией. "
              "Он помогает найти потенциально рискованные формулировки для проверки юристом. "
              "Отсутствие находок не означает, что договор безопасен.")

# ---------- Perception: чтение и разбиение ----------
def load_text(path):
    p = Path(path); ext = p.suffix.lower()
    if ext == ".pdf":
        import pdfplumber
        with pdfplumber.open(p) as pdf:
            text = "\n".join((pg.extract_text() or "") for pg in pdf.pages)
        if len(text.strip()) < 100:  # вероятно скан: пробуем OCR
            try:
                from pdf2image import convert_from_path
                import pytesseract
                text = "\n".join(pytesseract.image_to_string(im, lang="rus+eng")
                                 for im in convert_from_path(str(p)))
            except Exception as e:
                raise RuntimeError(
                    f"Похоже, это скан, а OCR недоступен ({e}). "
                    "Установите Tesseract и Poppler, а также pytesseract и pdf2image."
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

def split_clauses(text):
    text = re.sub(r"[ \t]+", " ", text)
    parts = re.split(r"\n(?=\s*\d+(?:\.\d+)*[.)]?\s)", text)
    if len(parts) < 2:
        parts = re.split(r"\n\s*\n", text)
    out = [re.sub(r"\s*\n\s*", " ", s).strip() for s in parts]
    return [s for s in out if len(s) > 30]

# ---------- Memory: база паттернов ----------
def load_patterns():
    return json.loads((HERE / "risk_patterns.json").read_text(encoding="utf-8"))

# ---------- Attention: ключевые слова + эмбеддинги ----------
def get_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception:
        print("sentence-transformers не найден: используются только ключевые слова.", file=sys.stderr)
        return None

@lru_cache(maxsize=1)
def cached_embedder():
    return get_embedder()

def select_clauses(clauses, patterns, embedder, threshold=0.55):
    sims = None
    if embedder is not None:
        import numpy as np
        ex = [(i, e) for i, p in enumerate(patterns) for e in p["examples"]]
        E = embedder.encode([e for _, e in ex], normalize_embeddings=True)
        C = embedder.encode(clauses, normalize_embeddings=True)
        S = C @ E.T
        sims = np.zeros((len(clauses), len(patterns)))
        for col, (pi, _) in enumerate(ex):
            sims[:, pi] = np.maximum(sims[:, pi], S[:, col])
    selected = []
    for ci, c in enumerate(clauses):
        low = c.lower(); cats = []; score = 0.0
        for pi, p in enumerate(patterns):
            kw = any(k in low for k in p["keywords"])
            sm = float(sims[ci, pi]) if sims is not None else 0.0
            if kw or sm >= threshold:
                cats.append(p["category"])
                score = max(score, p["weight"] * (1 + sm + (0.5 if kw else 0)))
        if cats:
            selected.append({"id": ci + 1, "text": c, "categories": cats, "score": score,
                             "weight": max(p["weight"] for p in patterns if p["category"] in cats)})
    return sorted(selected, key=lambda x: -x["score"])

# ---------- Decision: анализ по правилам, через Ollama или Claude ----------
SYSTEM = ("Ты помощник по проверке договоров. Для каждого пункта оцени риск для стороны, которая "
          "собирается подписать договор. Отвечай только JSON-массивом без пояснений и без markdown. "
          "Формат элемента: {\"id\": число, \"risk\": \"none|low|medium|high\", "
          "\"issue\": \"в чём проблема, 1-2 предложения\", \"recommendation\": \"что уточнить или изменить\"}. "
          "Пиши по-русски. Не давай юридических заключений и не пиши, что договор можно подписывать.")

def parse_llm_response(raw):
    print(f"\n--- [DEBUG] Сырой ответ от LLM ---\n{raw}\n----------------------------------\n", file=sys.stderr)
    LOGGER.info("Сырой ответ от LLM (первые 200 символов): %s", raw[:200])

    # Очищаем markdown-блоки
    raw_cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    
    # Пытаемся найти массив [...] или одиночный объект {...}
    match = re.search(r"(\[.*\]|\{.*\})", raw_cleaned, re.DOTALL)
    if match:
        raw_cleaned = match.group(0)

    try:
        parsed = json.loads(raw_cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Модель вернула ответ не в формате JSON. Сырой ответ: {raw[:300]}") from exc

    # Если модель вернула один объект вместо массива — заворачиваем его в список
    if isinstance(parsed, dict):
        parsed = [parsed]

    if not isinstance(parsed, list):
        raise ValueError("Ответ модели должен быть JSON-массивом.")

    results = {}
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("В ответе модели найден элемент неверного формата.")
        risk = item.get("risk")
        if type(item.get("id")) is not int or not isinstance(risk, str) or risk not in LEVELS:
            raise ValueError("В ответе модели отсутствует корректный id или уровень риска.")
        if item["id"] in results:
            raise ValueError("В ответе модели повторяется id пункта.")
        results[item["id"]] = {
            "id": item["id"],
            "risk": risk,
            "issue": str(item.get("issue", "")),
            "recommendation": str(item.get("recommendation", "")),
        }
    return results

def analyze_anthropic(items, batch=10):
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("Для режима Anthropic задайте переменную ANTHROPIC_API_KEY.")
    import anthropic
    client = anthropic.Anthropic()
    results = {}
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        prompt = "Проанализируй пункты договора:\n" + json.dumps(
            [{"id": x["id"], "categories": x["categories"], "text": x["text"]} for x in chunk],
            ensure_ascii=False)
        r = client.messages.create(model=CLAUDE_MODEL, max_tokens=4000, system=SYSTEM,
                                   messages=[{"role": "user", "content": prompt}])
        text = next((block.text for block in r.content if getattr(block, "type", None) == "text"), "")
        chunk_results = parse_llm_response(text)
        expected_ids = {item["id"] for item in chunk}
        if set(chunk_results) != expected_ids:
            raise ValueError("Claude вернула неполный или лишний набор пунктов.")
        results.update(chunk_results)
    return results

def analyze_ollama(items, batch=1):
    results = {}
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        prompt = "Проанализируй пункты договора:\n" + json.dumps(
            [{"id": x["id"], "categories": x["categories"], "text": x["text"]} for x in chunk],
            ensure_ascii=False)
        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "stream": False,
            "format": "json",
            "keep_alive": "5m",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0},
        }).encode("utf-8")
        request = Request(
            OLLAMA_URL.rstrip("/") + "/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        LOGGER.info(
            "Отправляю запрос в Ollama: модель %s, пакет %d из %d",
            OLLAMA_MODEL,
            i // batch + 1,
            (len(items) + batch - 1) // batch,
        )
        try:
            with urlopen(request, timeout=600) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Ошибка Ollama HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Не удалось подключиться к Ollama ({OLLAMA_URL}). "
                "Запустите Ollama и убедитесь, что модель загружена."
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Ollama вернула некорректный ответ.") from exc

        if not isinstance(data, dict):
            raise RuntimeError("Ollama вернула ответ неверного формата.")
        message = data.get("message", {})
        if data.get("error"):
            raise RuntimeError(f"Ошибка Ollama: {data['error']}")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise RuntimeError("Ollama вернула ответ без содержимого.")
        chunk_results = parse_llm_response(message["content"])
        expected_ids = {item["id"] for item in chunk}
        if set(chunk_results) != expected_ids:
            raise ValueError("Ollama вернула неполный или лишний набор пунктов.")
        results.update(chunk_results)
    return results

def analyze_rules(items):
    lv = {1: "low", 2: "medium", 3: "high"}
    return {x["id"]: {"id": x["id"], "risk": lv[x["weight"]],
                      "issue": "Найдена типовая рискованная формулировка: " + ", ".join(x["categories"]) + ".",
                      "recommendation": "Проверьте условие внимательно и обсудите его с юристом."} for x in items}

def analyze_document(file_path, provider="rules", top=25):
    if top < 1:
        raise ValueError("Параметр top должен быть не меньше 1.")
    if provider not in {"rules", "ollama", "anthropic"}:
        raise ValueError(f"Неизвестный LLM-провайдер: {provider}")
    LOGGER.info("Начинаю анализ документа; запрошенный провайдер: %s", provider)

    text = load_text(file_path)
    clauses = split_clauses(text)
    if not clauses:
        raise ValueError("Не удалось выделить пункты договора.")

    embedder = cached_embedder()
    items = select_clauses(clauses, load_patterns(), embedder)[:top]
    if not items:
        analysis = {}
        provider_name = "LLM не вызывалась: фрагменты не отобраны"
    elif provider == "ollama":
        analysis = analyze_ollama(items)
        provider_name = "Ollama"
    elif provider == "anthropic":
        analysis = analyze_anthropic(items)
        provider_name = "Anthropic"
    else:
        analysis = analyze_rules(items)
        provider_name = "Встроенные правила (без LLM)"

    LOGGER.info("Провайдер, фактически использованный для анализа: %s", provider_name)
    level, score, highs = aggregate(items, analysis)
    return {
        "total": len(clauses),
        "items": items,
        "analysis": analysis,
        "level": level,
        "score": score,
        "highs": highs,
        "mode": provider_name,
        "report_metadata": {
            "provider": provider_name,
        },
    }

# ---------- Aggregation ----------
def aggregate(items, analysis):
    pts = {"low": 1, "medium": 2, "high": 4}
    score = sum(pts.get(analysis.get(x["id"], {}).get("risk"), 0) for x in items)
    highs = sum(1 for x in items if analysis.get(x["id"], {}).get("risk") == "high")
    if highs >= 2 or score >= 10: level = "high"
    elif highs == 1 or score >= 4: level = "medium"
    elif score > 0: level = "low"
    else: level = "none"
    return level, score, highs

# ---------- Output: HTML-отчёт ----------
def build_report(name, total, items, analysis, level, score, mode, metadata=None):
    colors = {"high": "#b3261e", "medium": "#b26a00", "low": "#2e7d32", "none": "#5d6b7a"}
    metadata = metadata or {"provider": mode}
    analyzed_ids = {item["id"] for item in items if item["id"] in analysis}
    assessed_count = len(analyzed_ids)
    findings_count = sum(
        analysis[item_id].get("risk") in {"low", "medium", "high"}
        for item_id in analyzed_ids
    )
    not_assessed_count = max(0, total - assessed_count)
    rows = []
    for x in items:
        a = analysis.get(x["id"], {})
        risk = a.get("risk")
        if risk in {"low", "medium", "high"}:
            status = f"Выявлен риск: {RU[risk]}"
            color = colors[risk]
        elif risk == "none":
            status = "Риск не выявлен"
            color = colors["none"]
        else:
            status = "Не оценён"
            color = colors["none"]
        details = ""
        if a.get("issue"):
            details += f'<p><b>Комментарий:</b> {html.escape(str(a["issue"]), quote=True)}</p>'
        if a.get("recommendation"):
            details += f'<p><b>Рекомендация:</b> {html.escape(str(a["recommendation"]), quote=True)}</p>'
        rows.append(
            f'<article class="it" style="border-left-color:{color}"><h3>Фрагмент №{x["id"]}: '
            f'{html.escape(", ".join(x["categories"]), quote=True)} — {status}</h3>'
            f'<p class="q">{html.escape(x["text"], quote=True)}</p>{details}</article>'
        )
    verdict = {"high": "Высокий риск. Обязательно покажите договор юристу до подписания.",
               "medium": "Средний риск. Рекомендуется проверка юристом.",
               "low": "Низкий риск, но отмеченные пункты стоит уточнить.",
               "none": "Явных красных флагов не найдено. Это не гарантия безопасности."}[level]
    escaped_name = html.escape(name, quote=True)
    provider = html.escape(str(metadata.get("provider", mode)), quote=True)
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    findings_section = "".join(rows) or (
        '<p class="empty">Фрагменты для рассмотрения не отобраны. '
        'Это не означает, что в документе отсутствуют риски.</p>'
    )
    return f"""<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8"><title>Отчёт: {html.escape(name)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{font:16px/1.5 Georgia,serif;max-width:800px;margin:24px auto;padding:0 16px;color:#1c2530}}
section{{margin:18px 0}}.sum{{border:1px solid #d5dce3;border-radius:6px;padding:14px 18px}}
.it{{border:1px solid #d5dce3;border-left:5px solid;border-radius:6px;padding:4px 16px;margin-bottom:12px}}
.it h3{{font:700 1rem system-ui,sans-serif;margin:10px 0 4px}}.q{{color:#5d6b7a;font-style:italic}}
.w{{background:#fff4e0;border-left:4px solid #b26a00;padding:8px 12px;font-size:.9rem}}
.empty{{background:#f2f5f8;padding:12px;border-radius:5px}}dt{{font-weight:bold;margin-top:8px}}
dd{{margin-left:0}}footer{{font-size:.85rem;color:#5d6b7a;margin-top:24px}}</style></head><body>
<h1>Отчёт по договору: {escaped_name}</h1>
<aside id="disclaimer" class="w">{DISCLAIMER}</aside>
<section id="summary" aria-labelledby="summary-title"><div class="sum">
<h2 id="summary-title" style="color:{colors[level]};margin-top:0">Итог проверки: {verdict}</h2>
<dl><dt>Фрагментов выделено эвристическим разбиением</dt><dd>{total}</dd>
<dt>Фрагментов отобрано для анализа</dt><dd>{len(items)}</dd>
<dt>Фрагментов фактически оценено</dt><dd>{assessed_count}</dd>
<dt>Фрагментов не оценено</dt><dd>{not_assessed_count}</dd>
<dt>Фрагментов с риском</dt><dd>{findings_count}</dd>
<dt>Сторона, чьи риски оцениваются</dt><dd>Сторона, которая собирается подписать договор; автоматически не определяется.</dd></dl>
</div></section>
<section id="provider" aria-labelledby="provider-title"><h2 id="provider-title">Провайдер анализа</h2>
<p>{provider}</p></section>
<section id="findings" aria-labelledby="findings-title"><h2 id="findings-title">Фрагменты, попавшие на рассмотрение</h2>
{findings_section}</section>
<footer>Отчёт сформирован: {html.escape(generated_at, quote=True)}</footer>
</body></html>"""

def main():
    ap = argparse.ArgumentParser(description="Сканер договоров (не юридическая консультация)")
    ap.add_argument("file"); ap.add_argument("-o", "--output", default="report.html")
    ap.add_argument("--top", type=int, default=25, help="максимум пунктов для анализа")
    ap.add_argument("--provider", choices=("rules", "ollama", "anthropic"),
                    default=os.getenv("LLM_PROVIDER", "rules"),
                    help="режим анализа (по умолчанию: rules)")
    ap.add_argument("--no-llm", action="store_true", help="только правила, без LLM")
    a = ap.parse_args()

    provider = "rules" if a.no_llm else a.provider
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    LOGGER.info("Запуск проверки договоров; настроенный провайдер: %s", provider)
    try:
        result = analyze_document(a.file, provider=provider, top=a.top)
        Path(a.output).write_text(
            build_report(Path(a.file).name, result["total"], result["items"], result["analysis"],
                         result["level"], result["score"], result["mode"], result["report_metadata"]),
            encoding="utf-8",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        ap.exit(2, f"Ошибка: {exc}\n")
    print(f"Итог: {RU[result['level']]} риск, баллы {result['score']}, "
          f"высоких рисков {result['highs']}. Отчёт: {a.output}")
    print(DISCLAIMER)

if __name__ == "__main__":
    main()
