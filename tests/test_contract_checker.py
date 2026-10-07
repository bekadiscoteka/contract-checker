import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import contract_checker
import telegram_bot


class ContractCheckerTests(unittest.TestCase):
    def test_parse_llm_response_rejects_invalid_json(self):
        with self.assertRaises(ValueError):
            contract_checker.parse_llm_response("not json")

    def test_rules_provider_analyzes_text_without_model(self):
        contract = (
            "1. За просрочку оплаты начисляется неустойка в размере одного процента "
            "за каждый день просрочки исполнения обязательств.\n"
            "2. Исполнитель вправе в одностороннем порядке изменить условия договора "
            "без согласия заказчика и без предварительного уведомления.\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            contract_path = Path(temp_dir) / "contract.txt"
            contract_path.write_text(contract, encoding="utf-8")
            with patch.object(contract_checker, "cached_embedder", return_value=None):
                result = contract_checker.analyze_document(contract_path, provider="rules")

        self.assertEqual(result["total"], 2)
        self.assertEqual(len(result["items"]), 2)
        self.assertEqual(result["mode"], "правила и ключевые слова, без LLM")
        self.assertEqual(len(result["analysis"]), 2)

    def test_ollama_provider_sends_local_request_and_parses_result(self):
        items = [{"id": 1, "categories": ["Штрафы"], "text": "Условие договора"}]
        expected = [{"id": 1, "risk": "high", "issue": "Риск", "recommendation": "Проверить"}]
        response = io.BytesIO(json.dumps({"message": {"content": json.dumps(expected)}}).encode())

        with patch.object(contract_checker, "urlopen", return_value=response) as mocked_urlopen:
            result = contract_checker.analyze_ollama(items)

        request = mocked_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, contract_checker.OLLAMA_URL.rstrip("/") + "/api/chat")
        self.assertEqual(result[1]["risk"], "high")

    def test_ollama_provider_rejects_incomplete_result(self):
        items = [
            {"id": 1, "categories": ["Штрафы"], "text": "Первое условие"},
            {"id": 2, "categories": ["Неустойка"], "text": "Второе условие"},
        ]
        partial = [{"id": 1, "risk": "high", "issue": "Риск", "recommendation": "Проверить"}]
        response = io.BytesIO(json.dumps({"message": {"content": json.dumps(partial)}}).encode())

        with patch.object(contract_checker, "urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "неполный"):
                contract_checker.analyze_ollama(items)

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
        update = SimpleNamespace(
            effective_message=message,
            effective_user=SimpleNamespace(id=987654321),
        )
        context = SimpleNamespace(bot=bot)
        analysis = {
            "total": 1, "items": [], "analysis": {}, "level": "none",
            "score": 0, "highs": 0, "mode": "test",
        }

        with patch.object(telegram_bot, "analyze_document", return_value=analysis) as analyze:
            with patch.object(telegram_bot, "build_report", return_value="<html/>"):
                await telegram_bot.handle_document(update, context)

        analyze.assert_called_once()
        message.reply_document.assert_awaited_once()
        self.assertFalse(downloaded_paths[0].exists())


if __name__ == "__main__":
    unittest.main()
