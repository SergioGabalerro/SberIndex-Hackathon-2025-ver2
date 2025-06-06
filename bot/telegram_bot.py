import asyncio
import html
import logging

from aiogram import Bot, Dispatcher, types
from aiogram.client.bot import DefaultBotProperties
from aiogram.filters import Command

from .config import TELEGRAM_TOKEN, DATA_FILES, DATA_DESCRIPTIONS
from .agents.orchestrator import OrchestratorAgent


class TelegramBot:
    def __init__(self, token: str = TELEGRAM_TOKEN, data_files: dict = DATA_FILES, data_descriptions: dict = DATA_DESCRIPTIONS):
        self.token = token
        self.data_files = data_files
        self.data_descriptions = data_descriptions
        default_props = DefaultBotProperties(parse_mode='HTML')
        self.bot = Bot(self.token, default=default_props)
        self.dp = Dispatcher()
        self.orch = OrchestratorAgent(
            data_files=self.data_files,
            data_descriptions=self.data_descriptions,
        )
        self._setup_handlers()

    def _setup_handlers(self):
        self.dp.message(Command("start"))(self._cmd_start)
        self.dp.message()(self._handle_message)

    async def _cmd_start(self, message: types.Message):
        await message.answer("Привет! Напишите свой запрос.")

    async def _handle_message(self, message: types.Message):
        result = self.orch.process_message(message.from_user.id, message.text)
        if result is None:
            logging.error("process_message returned None")
            await message.answer("Произошла ошибка при обработке запроса")
            return

        answer, classification, _ = result
        sanitized_answer = html.escape(answer)
        await message.answer(f"<pre>{sanitized_answer}</pre>")

    async def run(self):
        await self.dp.start_polling(self.bot, skip_updates=True)


def run_bot():
    bot = TelegramBot()
    asyncio.run(bot.run())