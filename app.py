#from bot.__main__ import run_bot

#if __name__ == "__main__":
#    run_bot()

import os
import json
import logging
import pandas as pd
import numpy as np
import asyncio
import re
from pathlib import Path
import matplotlib.pyplot as plt

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.client.bot import DefaultBotProperties

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain.memory import ConversationBufferMemory
import html

# Импорт ChatOpenAI из актуального модуля langchain_openai
from langchain_openai import ChatOpenAI
from langchain_gigachat.chat_models import GigaChat
import openai
from types import SimpleNamespace


# === Загрузка переменных окружения из .env ===
load_dotenv()  # автоматически найдёт файл .env в корне проекта

MODEL_TYPE = os.getenv("MODEL_TYPE", "gigachat").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GIGACHAT_AUTH_KEY = os.getenv("GIGACHAT_AUTH_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")


class DeepSeekLLM:
    """Минимальная обёртка для API DeepSeek."""

    def __init__(self, api_key: str, model: str = "deepseek-chat", temperature: float = 0.0, timeout: int = 30):
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def invoke(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": m.content} for m in messages],
            temperature=self.temperature,
            timeout=self.timeout,
        )
        return SimpleNamespace(content=response.choices[0].message.content)

# Проверки корректности настроек
if MODEL_TYPE not in ("openai", "gigachat", "deepseek"):
    raise ValueError(f"Неверный MODEL_TYPE={MODEL_TYPE} в .env — должно быть 'openai', 'gigachat' или 'deepseek'.")
if MODEL_TYPE == "openai" and not OPENAI_API_KEY:
    raise ValueError("Вы выбрали MODEL_TYPE=openai, но не указали OPENAI_API_KEY в .env.")
if MODEL_TYPE == "gigachat" and not GIGACHAT_AUTH_KEY:
    raise ValueError("Вы выбрали MODEL_TYPE=gigachat, но не указали GIGACHAT_AUTH_KEY в .env.")
if MODEL_TYPE == "deepseek" and not DEEPSEEK_API_KEY:
    raise ValueError("Вы выбрали MODEL_TYPE=deepseek, но не указали DEEPSEEK_API_KEY в .env.")
if not TELEGRAM_TOKEN:
    raise ValueError("Не указан TELEGRAM_TOKEN в .env.")

# === Настройка логирования ===
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)

# === Папка с данными и пути к файлам ===
DATA_DIR = Path('data')

DATA_FILES = {
    "market_access": DATA_DIR / "1_market_access.parquet",
    "population":    DATA_DIR / "2_bdmo_population.parquet",  # Assuming this is Parquet, not CSV
    "migration":     DATA_DIR / "3_bdmo_migration.parquet",
    "salary":        DATA_DIR / "4_bdmo_salary.parquet",  # Assuming this is Parquet
    "connections":   DATA_DIR / "5_connection.parquet",
    "mo_ref":        DATA_DIR / "mo_ref.xlsx"
}


# Класс для проверки и загрузки файлов с поддержкой форматов Parquet, CSV, XLSX
class DataLoader:
    @staticmethod
    def check_data_files(data_files: dict) -> None:
        missing = [str(f) for f in data_files.values() if not f.exists()]
        if missing:
            error_msg = f"Отсутствуют файлы данных: {', '.join(missing)}"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)

    @staticmethod
    def load_data(data_files: dict) -> dict:
        data = {}
        for key, path in data_files.items():
            try:
                # Определяем формат файла по расширению и загружаем соответствующими методами
                if path.suffix == '.parquet':
                    data[key] = pd.read_parquet(path)
                elif path.suffix == '.csv':
                    data[key] = pd.read_csv(path)
                elif path.suffix in ['.xlsx', '.xls']:
                    data[key] = pd.read_excel(path)
                else:
                    raise ValueError(
                        f"Невозможно обработать файл с расширением {path.suffix}. Поддерживаются: .parquet, .csv, .xlsx.")

                logging.info(f"Успешно загружен файл: {path.name}")
            except Exception as e:
                logging.error(f"Ошибка загрузки файла {path.name}: {str(e)}")
                data[key] = pd.DataFrame()  # В случае ошибки, используем пустой DataFrame
        return data


# === Ручные описания дата-сетов для классификации ===
DATA_DESCRIPTIONS = {
    "market_access": "таблица с индексами рыночной доступности для каждого муниципалитета (territory_id, market_access, year)",
    "population": "таблица с численностью населения по муниципалитетам (territory_id, year, value)",
    "migration": "таблица с миграционными потоками по муниципалитетам (territory_id, year, in_migration, out_migration)",
    "salary": "таблица с информацией о зарплатах по оквэдам (okved_letter, territory_id, year, average_salary)",
    "connections": "таблица с показателями связности муниципалитетов (например, infrastructure_index, territory_id, year)",
    "mo_ref": "справочник муниципалитетов: territory_id, name, region, parent_district"
}


# === Инициализация LLM в зависимости от MODEL_TYPE ===
def init_llm():
    if MODEL_TYPE == "openai":
        logging.info("Инициализация ChatOpenAI (OpenAI) в качестве LLM")
        return ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-4o",
            temperature=0.0,
            request_timeout=30
        )
    if MODEL_TYPE == "deepseek":
        logging.info("Инициализация DeepSeek в качестве LLM")
        return DeepSeekLLM(
            api_key=DEEPSEEK_API_KEY,
            model="deepseek-chat",
            temperature=0.0,
            timeout=30
        )
    else:
        logging.info("Инициализация GigaChat в качестве LLM")
        return GigaChat(
            credentials=GIGACHAT_AUTH_KEY,
            verify_ssl_certs=False,
            model="GigaChat:2-Max",
            timeout=30
        )
llm = init_llm()


def call_llm(prompt: str, history: list | None = None) -> str:
    """
    Отправляем prompt в LLM и возвращаем «сырой» текстовый ответ.
    Используем метод invoke вместо устаревшего run.
    """
    logging.info(f"[LLM] Отправка запроса: {prompt[:100]}...")
    try:
        messages = []
        if history:
            messages.extend(history)
        messages.append(HumanMessage(content=prompt))
        response = llm.invoke(messages)
        raw = response.content
        logging.info(f"[LLM] Получен ответ: {raw[:100]}...")
        return raw
    except Exception as e:
        logging.error(f"Ошибка при обращении к LLM: {str(e)}")
        return ""

def parse_json_response(response_str: str) -> dict:
    """
    Парсим JSON-ответ из LLM. Обрезаем ```json``` или ``` если есть.
    """
    try:
        cleaned = re.sub(r'```json|```', '', response_str)
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {}
    except Exception as e:
        logging.error(f"Ошибка парсинга JSON: {str(e)}. Ответ: {response_str[:200]}...")
        return {}


def validate_query(query: str) -> bool:
    """
    Быстро проверяем, содержит ли запрос ключевые слова,
    связанные с муниципальными данными. Если нет —
    спрашиваем у LLM, valid / ambiguous.
    """
    keywords = [
        "зарплата", "население", "миграция", "рынок", "доступность",
        "бизнес", "регион", "город", "молодежь", "рост", "развитие",
        "территория", "муниципальный", "оквэд", "связность"
    ]
    query_lower = query.lower()
    if any(phrase in query_lower for phrase in keywords):
        return True

    prompt = (
        "Определи, относится ли запрос к анализу муниципальных данных. "
        "Критерии: население, зарплаты, миграция, бизнес, развитие регионов, связность. "
        "Ответ только 'valid' или 'ambiguous' без пояснений.\n"
        f"Запрос: '{query}'"
    )
    response = call_llm(prompt).lower().strip()
    return "valid" in response


# === ClassifierAgent: выбираем нужные таблицы по описанию ===
class ClassifierAgent:
    def __init__(self, data_descriptions: dict):
        self.role = "Классификатор"
        self.data_descriptions = data_descriptions
        logging.info(f"[{self.role}] Инициализирован.")

    def _build_prompt(self, query: str) -> str:
        datasets_info = "\n".join(
            f"- {name}: {desc}"
            for name, desc in self.data_descriptions.items()
        )
        return (
            "У тебя есть следующие таблицы с муниципальными данными. "
            "Ниже указано имя таблицы и её краткое описание (какие колонки, за какой год данные и пр.):\n\n"
            f"{datasets_info}\n\n"
            "Пользователь задаёт вопрос:\n"
            f"«{query}»\n\n"
            "Твоя задача — определить, какие таблицы (названия) нужно использовать, "
            "чтобы ответить на этот запрос. "
            #"Также учитывай какие другие таблицы из имеющихся датасетов большое влияние на таблицы которые ты выбрал для анализа и включи их (к примеру мы выбрали popultion,а на него скорее всего влияет migration)"
            "Верни ответ ровно в формате JSON: {\"datasets\": [\"имя1\", \"имя2\", …]}"
        )

    def classify(self, query: str) -> dict:
        prompt = self._build_prompt(query)
        result = parse_json_response(call_llm(prompt))
        return result if result.get("datasets") else {"datasets": []}


# === KnowledgeSearchAgent: загружаем и объединяем датафреймы ===
class KnowledgeSearchAgent:
    def __init__(self, data_files: dict):
        self.role = "Поисковик"
        self.data_files = data_files
        # загружаем все таблицы в словарь pandas.DataFrame
        self.data = DataLoader.load_data(data_files)
        logging.info(f"[{self.role}] Инициализирован.")

    def combine_dataframes(self, datasets: list) -> pd.DataFrame:
        """
        Принимаем список имён таблиц (например, ["salary", "population"]),
        добавляем всегда 'mo_ref' (для названий), если его нет,
        и объединяем все по ключу territory_id (inner join).
        Возвращаем объединённый DataFrame.
        """
        # Убедимся, что mo_ref есть в объединении
        dfs_to_merge = []
        for ds in datasets:
            if ds in self.data:
                dfs_to_merge.append(self.data[ds].copy())
        if "mo_ref" in self.data and "mo_ref" not in datasets:
            dfs_to_merge.append(self.data["mo_ref"].copy())

        if not dfs_to_merge:
            return pd.DataFrame()

        # Начинаем с первого датафрейма
        combined = dfs_to_merge[0]
        for df in dfs_to_merge[1:]:
            if "territory_id" in combined.columns and "territory_id" in df.columns:
                combined = combined.merge(df, on="territory_id", how="inner")
            else:
                # Если нет ключа, просто объединяем по индексу (маловероятно)
                combined = pd.concat([combined, df], axis=1)

        return combined


# === AnswerGenerationAgent: генерируем pandas-код у LLM и исполняем его ===
class AnswerGenerationAgent:
    def __init__(self):
        self.role = "Генератор ответов"
        logging.info(f"[{self.role}] Инициализирован.")

    def _build_prompt_for_code(self, query: str, available_columns: list, datasets: list) -> str:
        """
        Строим prompt, чтобы LLM вернул фрагмент Pandas-кода,
        который обрабатывает переменную df (объединённый DataFrame)
        и сохраняет результат в переменную 'result'.
        Также перечисляем доступные колонки для контекста.
        """
        cols_list = ", ".join(available_columns)
        datasets_str = ", ".join(datasets)
        return (
            "У тебя есть pandas DataFrame с именем df, который содержит данные из таблиц: "
            f"{datasets_str}. Данные объединены по ключу territory_id.\n"
            f"Доступные колонки: {cols_list}.\n\n"
            f"Запрос пользователя: «{query}».\n\n"
            "Нужно написать только Python-код на pandas, numpy, который даст ответ на запрос, "
            "используя df и сохраняя итог в переменную result. "
            "Добавляй комментарии при необходимости, но не включай пояснений вне кода. "
            "Перед приведением значений к целочисленному типу убирай или заменяй пропуски и бесконечные значения, "
            "чтобы не возникала ошибка 'Cannot convert non-finite values to integer'."
        )


    def generate_and_execute(self, combined_df: pd.DataFrame, query: str, datasets: list, history=None) -> tuple[str, object, str]:
        """
        Запрашиваем у LLM код, исполняем его и возвращаем текстовый результат,
        сам объект результата и сгенерированный код.
        """
        if combined_df.empty:
            return "Данные оказались пустыми после объединения. Нечего обрабатывать.", None, ""

        # Составляем список доступных колонок
        available_columns = list(combined_df.columns)
        prompt = self._build_prompt_for_code(query, available_columns, datasets)
        code_str = call_llm(prompt, history=history)
        logging.info("[GeneratedCode]\n%s", code_str)

        if not code_str.strip():
            return "Не удалось получить код от модели.", None, code_str

        # Убираем возможные ```python``` или ```, затем исполняем код в локальной среде
        code_clean = re.sub(r'```(?:python)?', '', code_str).strip()
        logging.info("[ExecutingCode]\n%s", code_clean)

        # Подготовим локальную среду для выполнения
        clean_df = combined_df.replace([np.inf, -np.inf], pd.NA).fillna(0)
        local_vars = {"df": clean_df.copy(), "pd": pd, "np": np}
        try:
            # Выполняем код. Ожидаем, что в коде в конце появится переменная result.
            exec(code_clean, {}, local_vars)
        except Exception as e:
            logging.error(f"Ошибка при выполнении сгенерированного кода: {e}")
            return f"Ошибка при выполнении кода: {e}", None, code_str

        # Проверяем, есть ли result в локальных переменных
        if "result" not in local_vars:
            return "Модель не сохранила результат в переменной result.", None, code_str

        result = local_vars["result"]

        # Форматируем результат для пользователя
        if isinstance(result, pd.DataFrame):
            # Отображаем первые 10 строк
            text_result = result.head(10).to_string(index=False)
        elif isinstance(result, pd.Series):
            text_result = result.head(10).to_string()
        else:
            try:
                text_result = str(result)
            except Exception:
                text_result = "Не удалось отобразить результат."

        return text_result, result, code_str

    def summarize(self, query: str, datasets: list, eda_summary: str, result_str: str, history=None) -> str:
        datasets_str = ", ".join(datasets)
        prompt = (
            "Ты аналитик, который отвечает на вопросы о муниципальных данных." \
            " На основе запроса пользователя и вычисленных результатов сформулируй" \
            " понятный ответ.\n" \
            f"Запрос: {query}\n" \
            f"Использованные таблицы: {datasets_str}\n" \
            f"Краткий анализ данных:\n{eda_summary}\n\n" \
            f"Результат вычислений:\n{result_str}\n\n" \
            "Сформулируй довольно краткий итог, посмотри на EDA и прочии и предложи гипотезы почему такая картина," \
            "пример вопроса: В каких МО высокая зарплата не помогает удержать население?" \
            "пример ответа: МО Y имеет среднюю зарплату 79,000₽ (TOP-10%), но потерял 2.4% населения" \
            "Причина может крыться в низкой рыночной доступности (MA = 150) и изоляции "

        )
        summary = call_llm(prompt, history=history)
        return summary.strip()



# === EDAAgent: выполняет краткий анализ данных ===
class EDAAgent:
    def __init__(self):
        self.role = "EDA"
        logging.info(f"[{self.role}] Инициализирован.")

    def analyze(self, df: pd.DataFrame) -> str:
        if df.empty:
            return "Данные отсутствуют или объединение вернуло пустой DataFrame."

        lines = [
            f"Размер данных: {len(df)} строк, {len(df.columns)} колонок.",
        ]

        missing = df.isna().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            lines.append("Пропуски (топ 5):")
            lines.append(missing.sort_values(ascending=False).head(5).to_string())

        desc = df.describe(include='all').transpose()
        lines.append("Статистика (первые 5 строк):")
        lines.append(desc.head().to_string())

        return "\n".join(lines)


# === ContextAgent без изменений ===
class ContextAgent:
    def __init__(self):
        self.user_context = {}
        self.user_memory = {}
        logging.info("[Контекст] Инициализирован.")

    def _get_memory(self, user_id: int) -> ConversationBufferMemory:
        if user_id not in self.user_memory:
            self.user_memory[user_id] = ConversationBufferMemory(return_messages=True)
        return self.user_memory[user_id]

    def add_user_message(self, user_id: int, text: str):
        self._get_memory(user_id).chat_memory.add_user_message(text)

    def add_ai_message(self, user_id: int, text: str):
        self._get_memory(user_id).chat_memory.add_ai_message(text)

    def get_history(self, user_id: int):
        return self._get_memory(user_id).chat_memory.messages

    def update_context(self, user_id: int, data: dict):
        self.user_context[user_id] = {
            **self.user_context.get(user_id, {}),
            **data
        }

    def get_context(self, user_id: int) -> dict:
        return self.user_context.get(user_id, {})

    def clear_context(self, user_id: int):
        if user_id in self.user_context:
            del self.user_context[user_id]
        if user_id in self.user_memory:
            self.user_memory[user_id].clear()


# === OrchestratorAgent: объединяем все шаги вместе ===
class OrchestratorAgent:
    def __init__(self, data_files: dict, data_descriptions: dict):
        # Проверяем наличие файлов и загружаем описания
        DataLoader.check_data_files(data_files)
        self.classifier = ClassifierAgent(data_descriptions)
        self.searcher = KnowledgeSearchAgent(data_files)
        self.generator = AnswerGenerationAgent()
        self.eda = EDAAgent()
        self.context = ContextAgent()
        logging.info("[Оркестратор] Инициализирован.")

    def process_message(self, user_id: int, text: str) -> tuple[str, dict, bool]:
        logging.info("[User %s] Запрос: %s", user_id, text)
        ctx = self.context.get_context(user_id)
        history = self.context.get_history(user_id)

        # Если это не уточнение предыдущего, сбрасываем контекст
        if not self._is_continuation(text, ctx.get("last_query", ""), history=history):
            self.context.clear_context(user_id)
            history = self.context.get_history(user_id)
            ctx = {}

        # Сохраняем сообщение пользователя в историю
        self.context.add_user_message(user_id, text)

        # 1. Проверяем, имеет ли смысл двигаться дальше
        if not validate_query(text, history=history):
            self.context.update_context(user_id, {
                "last_query": text,
                "is_ambiguous": True
            })
            return "Пожалуйста, уточните ваш запрос. Например: 'Где самые высокие зарплаты в IT?'", {}, False

        # 2. Классифицируем: получаем список имён таблиц
        classification = self.classifier.classify(text, history=history)
        tables = classification.get("datasets", [])
        if not tables:
            return "Не удалось определить подходящие данные. Попробуйте переформулировать запрос.", {}, False


        # 3. Объединяем все указанные таблицы (и mo_ref для названий) в один DataFrame
        available_tables = [t for t in tables if t in self.searcher.data and not self.searcher.data[t].empty]
        missing_tables = [t for t in tables if t not in self.searcher.data or self.searcher.data[t].empty]
        combined_df = self.searcher.combine_dataframes(available_tables)

        # 4. Проводим базовый EDA
        eda_summary = self.eda.analyze(combined_df)

        # 5. Генерируем и выполняем pandas-код у LLM для ответа
        result_text, _, code_snippet = self.generator.generate_and_execute(combined_df, text, available_tables, history=history)

        # 6. Формируем развёрнутый ответ с помощью LLM
        final_answer = self.generator.summarize(text, available_tables, eda_summary, result_text, history=history)
       #Разкомментировать если вернуть отображение кода
       # if code_snippet:
       #     final_answer += "\n\nСгенерированный код:\n" + code_snippet
        if missing_tables:
            final_answer += "\nОтсутствуют данные для: " + ", ".join(missing_tables)

        # 7. Обновляем контекст
        self.context.add_ai_message(user_id, final_answer)
        self.context.update_context(user_id, {
            "last_query": text,
            "classification": classification,
            "last_answer": final_answer,
            "has_data": not combined_df.empty
        })

        return final_answer, classification, True

    def _is_continuation(self, new_text: str, prev_text: str, history=None) -> bool:
        if not prev_text:
            return False

        prompt = (
            "Определи, является ли новый запрос уточнением предыдущего.\n"
            f"Предыдущий: '{prev_text}'\n"
            f"Новый: '{new_text}'\n"
            "Ответ в формате JSON: {\"is_continuation\": true/false}"
        )
        response = parse_json_response(call_llm(prompt, history=history))
        return response.get("is_continuation", False)


# === TelegramBot (Aiogram) ===
# === TelegramBot (Aiogram) ===
class TelegramBot:
    def __init__(self, token: str, data_files: dict, data_descriptions: dict):
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
        self.dp.message(Command("logs"))(self._cmd_logs)
        self.dp.message()(self._handle_message)

    async def _cmd_start(self, message: types.Message):
        self.orch.context.clear_context(message.from_user.id)
        await message.answer("Привет! Напишите свой запрос.")

    async def _cmd_logs(self, message: types.Message):
        try:
            with open("bot.log", "r") as f:
                logs = f.read()[-4000:]
            await message.answer(f"<pre>{html.escape(logs)}</pre>")
        except Exception as e:
            logging.error("Не удалось отправить лог: %s", e)
            await message.answer("Не удалось прочитать лог-файл.")

    async def _handle_message(self, message: types.Message):
        result = self.orch.process_message(message.from_user.id, message.text)
        if result is None:
            logging.error("process_message returned None")
            await message.answer("Произошла ошибка при обработке запроса")
            return

        answer, classification, _ = result
        sanitized_answer = html.escape(answer)
        # Разкомментировать, чтобы вернуть отображение кода
        #classification_json = json.dumps(classification, ensure_ascii=False)
        #sanitized_class = html.escape(classification_json)
        #await message.answer(f"<pre>{sanitized_answer}\n\nClassifierAgent: {sanitized_class}</pre>")

        await message.answer(f"<pre>{sanitized_answer}</pre>") # Это наоборот закомментировать, чтобы вернуть отображение кода

    async def run(self):
        await self.dp.start_polling(self.bot, skip_updates=True)

def main() -> None:
    """Запуск телеграм-бота с параметрами из конфигурации."""
    try:
        bot = TelegramBot(
            token=TELEGRAM_TOKEN,
            data_files=DATA_FILES,
            data_descriptions=DATA_DESCRIPTIONS,
        )
        asyncio.run(bot.run())
    except Exception as e:
        logging.error("Fatal error: %s", e)

# Передайте токен и другие параметры
if __name__ == '__main__':
    main()