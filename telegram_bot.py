#!/usr/bin/env python3
"""Telegram-бот для локальной проверки договоров."""
import asyncio
import logging
import os
import re
import tempfile
from pathlib import Path

from telegram import Update
from telegram.error import TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from contract_checker import DISCLAIMER, RU, analyze_document, build_report

LOGGER = logging.getLogger(__name__)
MAX_FILE_SIZE = 18 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
TOP_CLAUSES = 25


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "Пришлите договор в формате PDF, DOCX или TXT, и я подготовлю HTML-отчёт. "
        "Автоматический анализ не является юридической консультацией.\n\n"
        "Важно: даже если используется локальная Ollama, файл передаётся через инфраструктуру Telegram."
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    document = message.document

    original_name = document.file_name or "contract"
    extension = Path(original_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        await message.reply_text("Поддерживаются только PDF, DOCX и TXT.")
        return
    if document.file_size is not None and document.file_size > MAX_FILE_SIZE:
        await message.reply_text("Файл слишком большой. Максимальный размер — 18 МБ.")
        return

    input_path = None
    report_path = None
    status_message = None
    try:
        with tempfile.NamedTemporaryFile(prefix="contract-", suffix=extension, delete=False) as tmp:
            input_path = Path(tmp.name)
        telegram_file = await context.bot.get_file(document.file_id)
        await telegram_file.download_to_drive(custom_path=str(input_path))
        if input_path.stat().st_size > MAX_FILE_SIZE:
            await message.reply_text("Файл слишком большой. Максимальный размер — 18 МБ.")
            return
        status_message = await message.reply_text("Документ получен, выполняю анализ…")

        provider = os.getenv("LLM_PROVIDER", "rules").lower()
        LOGGER.info(
            "Обработка документа %s; настроенный провайдер: %s",
            original_name,
            provider,
        )
        result = await asyncio.to_thread(
            analyze_document, input_path, provider=provider, top=TOP_CLAUSES
        )
        LOGGER.info(
            "Документ %s обработан; фактически использованный провайдер: %s",
            original_name,
            result["mode"],
        )
        html_report = build_report(
            original_name, result["total"], result["items"], result["analysis"],
            result["level"], result["score"], result["mode"], result["report_metadata"],
        )
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".html", delete=False
        ) as tmp:
            report_path = Path(tmp.name)
            tmp.write(html_report)

        safe_stem = re.sub(r"[^A-Za-z0-9А-Яа-яЁё_.-]+", "_", Path(original_name).stem).strip("._")
        safe_stem = safe_stem or "contract"
        caption = (
            f"Итог: {RU[result['level']]} риск; баллы: {result['score']}; "
            f"высоких рисков: {result['highs']}.\n{DISCLAIMER}"
        )
        with report_path.open("rb") as report_file:
            await message.reply_document(
                document=report_file,
                filename=f"{safe_stem}_report.html",
                caption=caption[:1024],
            )
        try:
            await status_message.edit_text(
                f"Готово: {RU[result['level']]} риск, баллы {result['score']}, "
                f"высоких рисков {result['highs']}. Отчёт отправлен отдельным файлом."
            )
        except TelegramError:
            LOGGER.warning("Report was sent, but the status message could not be updated")
    except TelegramError:
        LOGGER.error("Telegram request failed while processing document")
        await message.reply_text("Ошибка Telegram при обработке файла. Попробуйте отправить его ещё раз.")
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.error("Failed to process Telegram document: %s", exc)
        await message.reply_text(f"Не удалось обработать документ: {exc}")
    finally:
        for temporary_path in (input_path, report_path):
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    LOGGER.exception("Failed to remove temporary file")


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN.")

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    application = Application.builder().token(token).concurrent_updates(False).build()
    private_chat = filters.ChatType.PRIVATE
    application.add_handler(CommandHandler("start", start, filters=private_chat))
    application.add_handler(
        MessageHandler(private_chat & filters.Document.ALL, handle_document)
    )
    provider = os.getenv("LLM_PROVIDER", "rules").lower()
    LOGGER.info(
        "Starting public contract checker bot (private chats only); configured provider: %s",
        provider,
    )
    application.run_polling()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"Ошибка конфигурации: {exc}") from exc
