import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import telegram_bot
from cognitive import attention, config, decision, learning, perception, reasoning
from cognitive.action import build_report
from cognitive.memory import MemoryStore
from cognitive.pipeline import run_pipeline
from cognitive.schemas import (ActionResult, Assessment, AttentionItem, AttentionResult, Clause,
                               CycleResult, DecisionResult, DSSParams, FeedbackItem, ReasoningResult)

DATA = Path(__file__).resolve().parent.parent / "data"
PEN = "Штрафы и неустойка"


def item(i, categories, text, weight=3, evidence=1.0, keywords=("kw",)):
    return AttentionItem(clause=Clause(id=i, text=text, features=perception.extract_features(text)),
                         categories=categories, score=1.0, weight=weight, evidence=evidence,
                         keyword_hits=list(keywords), similarities={c: 0.7 for c in categories})


class TempStoreMixin:
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self._tmp.name) / "memory.db")
        self._embed = patch.object(attention, "cached_embedder", return_value=None)
        self._embed.start()

    def tearDown(self):
        self._embed.stop()
        self._tmp.cleanup()


# ---------------- T1 ----------------
class PerceptionTests(unittest.TestCase):
    def test_clean_text_fixes_hyphenation_page_numbers_and_quotes(self):
        text, stats = perception.clean_text("Сто-\nимость «услуг»\n\n  2  \nдалее — текст")
        self.assertIn("Стоимость", text)
        self.assertIn('"услуг"', text)
        self.assertIn("далее - текст", text)
        self.assertEqual(stats["hyphenations_fixed"], 1)
        self.assertEqual(stats["page_numbers_removed"], 1)

    def test_penalty_percent_per_day(self):
        f = perception.extract_features(
            "4.1. За просрочку любого платежа Заказчик уплачивает неустойку в размере 3% "
            "от общей суммы Договора за каждый день просрочки.")
        self.assertEqual(f.number, "4.1")
        self.assertEqual(f.percents, [3.0])
        self.assertTrue(f.percent_per_day)

    def test_word_numbers_money_and_durations(self):
        f = perception.extract_features(
            "Ответственность ограничена суммой 1 000 (одна тысяча) тенге, претензии принимаются "
            "в течение 1 (одного) часа, договор действует 3 (три) года, неустойка одного процента.")
        self.assertEqual(f.money, [(1000.0, "KZT")])
        self.assertEqual(f.durations_days, [round(1 / 24, 3), 1095.0])
        self.assertEqual(f.percents, [1.0])
        self.assertFalse(f.percent_per_day)

    def test_annual_interest_and_dates_are_not_durations(self):
        f = perception.extract_features("Проценты составляют 20% годовых до 31 декабря 2027 года, "
                                        "сумма 1 500 тыс. тг возвращается через 6 месяцев.")
        self.assertEqual(f.percents, [20.0])
        self.assertEqual(f.money, [(1_500_000.0, "KZT")])
        self.assertEqual(f.durations_days, [180.0])

    def test_currency_variants(self):
        f = perception.extract_features(
            "Штраф 50 МРП, аванс 2 млн ₸, доплата 300 KZT, услуги 100 рублей, лицензия 20 долларов.")
        self.assertEqual(f.money, [(50.0, "MRP"), (2_000_000.0, "KZT"), (300.0, "KZT"),
                                   (100.0, "RUB"), (20.0, "USD")])

    def test_headings_are_not_clauses(self):
        parts = perception.split_clauses("1. ПРЕДМЕТ ДОГОВОРА И ОБЩИЕ ПОЛОЖЕНИЯ\n"
                                         "1.1. Исполнитель оказывает услуги в объёме, согласованном сторонами.\n")
        self.assertEqual(len(parts), 1)
        self.assertTrue(parts[0].startswith("1.1."))


# ---------------- T4 ----------------
class ReasoningTests(unittest.TestCase):
    def test_parse_llm_response_rejects_invalid_json(self):
        with self.assertRaises(ValueError):
            reasoning.parse_llm_response("not json")

    def test_daily_penalty_rule_fires_with_trace(self):
        a = reasoning.infer(item(1, [PEN], "4.1. Неустойка 3% от суммы за каждый день просрочки.", weight=2))
        self.assertEqual(a.risk, "high")
        self.assertIn("R1", a.rules_fired)
        self.assertTrue(any("1095% per year" in t for t in a.trace))

    def test_symbolic_liability_cap_in_tenge(self):
        a = reasoning.infer(item(1, [PEN], "Ответственность Исполнителя ограничена суммой 1 000 тенге.",
                                 weight=3, evidence=0.2, keywords=()))
        self.assertEqual(a.base_risk, "none")
        self.assertEqual(a.risk, "high")
        self.assertIn("R3", a.rules_fired)

    def test_mutual_wording_lowers_risk(self):
        a = reasoning.infer(item(1, ["Конфиденциальность"], "Стороны обязуются взаимно сохранять конфиденциальность.",
                                 weight=2))
        self.assertEqual(a.base_risk, "medium")
        self.assertEqual(a.risk, "low")

    def test_weak_evidence_without_rules_is_not_flagged(self):
        a = reasoning.infer(item(1, [PEN], "Стоимость услуг составляет 250 000 тенге в месяц.",
                                 weight=3, evidence=0.2, keywords=()))
        self.assertEqual(a.risk, "none")

    def test_ollama_provider_sends_local_request_and_parses_result(self):
        items = [item(1, [PEN], "Условие договора")]
        expected = [{"id": 1, "risk": "high", "issue": "Risk", "recommendation": "Check"}]
        response = io.BytesIO(json.dumps({"message": {"content": json.dumps(expected)}}).encode())
        with patch.object(reasoning, "urlopen", return_value=response) as mocked_urlopen:
            with self.assertLogs("cognitive", level="INFO") as logs:
                result = reasoning.analyze_ollama(items)
        request = mocked_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, config.OLLAMA_URL.rstrip("/") + "/api/chat")
        self.assertEqual(json.loads(request.data)["format"], "json")
        self.assertEqual(result[1].risk, "high")
        self.assertIn("Sending request to Ollama", logs.output[0])

    def test_ollama_provider_rejects_incomplete_result(self):
        items = [item(1, [PEN], "Первое условие"), item(2, [PEN], "Второе условие")]
        partial = [{"id": 1, "risk": "high", "issue": "Risk", "recommendation": "Check"}]
        body = json.dumps({"message": {"content": json.dumps(partial)}}).encode()
        with patch.object(reasoning, "urlopen", side_effect=lambda *a, **k: io.BytesIO(body)):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                reasoning.analyze_ollama(items, batch=2)


# ---------------- T5 DSS ----------------
def _attention_and_reasoning(risks):
    items = [item(i, [PEN], f"Пункт {i} с неустойкой") for i in range(1, len(risks) + 1)]
    assessments = {i: Assessment(id=i, risk=r) for i, r in enumerate(risks, 1)}
    return (AttentionResult(items=items, total_clauses=len(items), threshold=0.55, embeddings_used=False),
            ReasoningResult(assessments=assessments, provider_name="test"))


class DSSTests(unittest.TestCase):
    def test_topsis_prefers_dominating_alternative(self):
        norm = {"a": {"risk": 1, "cost": 1, "delay": 1, "value": 1},
                "b": {"risk": 0.5, "cost": 0.5, "delay": 0.5, "value": 0.5},
                "c": {"risk": 0, "cost": 0, "delay": 0, "value": 0}}
        w = decision.normalize_weights(None)
        t = decision.topsis(norm, w)
        self.assertAlmostEqual(t["a"], 1.0)
        self.assertAlmostEqual(t["b"], 0.5)
        self.assertAlmostEqual(t["c"], 0.0)

    def test_weights_are_normalized(self):
        w = decision.normalize_weights({"risk": 50, "cost": 25, "delay": 25, "value": 0})
        self.assertAlmostEqual(sum(w.values()), 1.0)
        self.assertAlmostEqual(w["risk"], 0.5)

    def test_scenario_matrix(self):
        att, rea = _attention_and_reasoning(["high", "low"])
        profile = decision.risk_profile(att, rea)
        self.assertEqual(profile.total_severity, 7)          # 6 + 1
        m = decision.build_matrix(profile, DSSParams(deal_value=1000, p_accept=0.5, walk_away=0.1))
        self.assertEqual(m["sign"]["risk"], 7)
        self.assertAlmostEqual(m["amend"]["risk"], 7 - 0.5 * 7 * 0.8)
        self.assertAlmostEqual(m["amend"]["value"], 900)
        self.assertEqual(m["reject"]["value"], 0)

    def test_safe_contract_is_signed_and_risky_is_not(self):
        safe = decision.run_dss(*_attention_and_reasoning(["low"]))
        risky = decision.run_dss(*_attention_and_reasoning(["high"] * 12))
        self.assertEqual(safe.recommended, "sign")
        self.assertNotEqual(risky.recommended, "sign")
        self.assertEqual(len(safe.ranking), 4)
        self.assertIn(safe.alternatives[safe.recommended], safe.explanation[0])

    def test_what_if_risk_weight_changes_the_leader(self):
        risky = decision.run_dss(*_attention_and_reasoning(["high"] * 12))
        self.assertTrue(risky.sensitivity["risk"]["switches"])
        only_value = decision.run_dss(*_attention_and_reasoning(["high"] * 12),
                                      weights={"risk": 0, "cost": 0, "delay": 0, "value": 1})
        self.assertEqual(only_value.recommended, "sign")
        self.assertAlmostEqual(sum(risky.stability.values()), 1.0, places=2)


# ---------------- T3 + T7 + full cycle ----------------
class CycleTests(TempStoreMixin, unittest.TestCase):
    def test_rules_provider_runs_full_cycle(self):
        with self.assertLogs("cognitive", level="INFO") as logs:
            result = run_pipeline(DATA / "contract_1_dangerous.txt", store=self.store)
        self.assertGreater(len(result.perception.clauses), 10)
        self.assertTrue(result.attention.items)
        self.assertEqual(result.reasoning.provider_name, "Built-in rules (no LLM)")
        self.assertTrue(all(a.trace for a in result.reasoning.assessments.values()))
        self.assertEqual(result.decision.level, "high")
        self.assertIsNotNone(result.decision.dss)
        self.assertIn("Decision support", result.action.report_html)
        self.assertEqual(self.store.count("documents"), 1)
        self.assertTrue(any("запрошенный провайдер): rules" in line for line in logs.output))
        self.assertTrue(any("Provider actually used: Built-in rules" in line for line in logs.output))

    def test_llm_is_not_called_when_no_fragments_were_selected(self):
        path = Path(self._tmp.name) / "c.txt"
        path.write_text("Нейтральный текст без каких-либо условий договора вообще.", encoding="utf-8")
        with patch.object(reasoning, "analyze_ollama") as analyze_ollama:
            result = run_pipeline(path, provider="ollama", store=self.store)
        analyze_ollama.assert_not_called()
        self.assertIn("LLM not called", result.reasoning.provider_name)

    def test_learning_changes_behaviour_on_next_run(self):
        path = DATA / "contract_1_dangerous.txt"
        first = run_pipeline(path, store=self.store)
        target = next(it for it in first.attention.items if first.reasoning.assessments[it.id].risk == "high")
        fb = [FeedbackItem(clause_text=target.text, categories=target.categories,
                           system_risk="high", user_risk="low", document_name=path.name)]
        res = learning.learn(self.store, fb)
        self.assertTrue(res.weight_updates)
        self.assertLess(res.weight_updates[0]["new"], res.weight_updates[0]["old"])

        second = run_pipeline(path, store=self.store)
        a = second.reasoning.assessments[target.id]
        self.assertEqual(a.risk, "low")
        self.assertTrue(a.memory_applied)
        self.assertIsNotNone(second.memory.previous_run)
        self.assertLess(second.decision.score, first.decision.score)

    def test_three_samples_are_ranked_by_risk(self):
        levels = [run_pipeline(DATA / n, store=self.store).decision for n in
                  ("contract_1_dangerous.txt", "contract_2_medium.txt", "contract_3_balanced.txt")]
        self.assertGreater(levels[0].score, levels[1].score)
        self.assertGreater(levels[1].score, levels[2].score)


# ---------------- T6 ----------------
class ActionTests(unittest.TestCase):
    def test_report_lists_selection_coverage_and_escapes_html(self):
        items = [item(1, [PEN], "Пункт <один>"), item(2, ["Автоматическое продление"], "Пункт два"),
                 item(3, ["Расторжение и выход из договора"], "Пункт три")]
        assessments = {1: Assessment(id=1, risk="high", issue="Problem", recommendation="Check",
                                     trace=["Base: x"]),
                       2: Assessment(id=2, risk="none")}
        report = build_report("contract <test>.docx", 5, items, assessments,
                              DecisionResult(level="medium", score=4, highs=1), "Ollama")
        for anchor in ('id="summary"', 'id="provider"', 'id="findings"', 'id="disclaimer"'):
            self.assertIn(anchor, report)
        self.assertIn("<p>Ollama</p>", report)
        self.assertIn("Fragments assessed</dt><dd>2", report)
        self.assertIn("Fragments not assessed</dt><dd>3", report)
        self.assertIn("Fragments with risk</dt><dd>1", report)
        self.assertIn("No risk found", report)
        self.assertIn("Not assessed", report)
        self.assertIn("Inference steps", report)
        self.assertIn("Пункт &lt;один&gt;", report)
        self.assertIn("contract &lt;test&gt;.docx", report)


class TelegramBotTests(unittest.IsolatedAsyncioTestCase):
    async def test_private_user_without_allowlist_can_submit_document(self):
        downloaded_paths = []

        async def download_to_drive(custom_path):
            path = Path(custom_path)
            downloaded_paths.append(path)
            path.write_text("Test contract text", encoding="utf-8")

        telegram_file = SimpleNamespace(download_to_drive=AsyncMock(side_effect=download_to_drive))
        bot = SimpleNamespace(get_file=AsyncMock(return_value=telegram_file))
        status_message = SimpleNamespace(edit_text=AsyncMock())
        message = SimpleNamespace(
            document=SimpleNamespace(file_name="contract.txt", file_size=18, file_id="file-id"),
            reply_text=AsyncMock(return_value=status_message),
            reply_document=AsyncMock(),
        )
        update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=987654321))
        context = SimpleNamespace(bot=bot)
        cycle = CycleResult(
            perception=None, attention=None, memory=None,
            reasoning=ReasoningResult(assessments={}, provider_name="test"),
            decision=DecisionResult(level="none", score=0, highs=0),
            action=ActionResult(report_html="<html/>"),
        )
        with patch.object(telegram_bot, "run_pipeline", return_value=cycle) as run:
            with self.assertLogs("telegram_bot", level="INFO") as logs:
                await telegram_bot.handle_document(update, context)
        run.assert_called_once()
        message.reply_document.assert_awaited_once()
        self.assertFalse(downloaded_paths[0].exists())
        self.assertTrue(any("настроенный провайдер: rules" in line for line in logs.output))
        self.assertTrue(any("фактически использованный провайдер: test" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
