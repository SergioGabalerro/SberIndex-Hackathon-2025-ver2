import logging
import re
import pandas as pd
import numpy as np

from ..utils import call_llm


class AnswerGenerationAgent:
    def __init__(self):
        self.role = "Генератор ответов"
        logging.info(f"[{self.role}] Инициализирован.")

    def _build_prompt_for_code(self, query: str, available_columns: list, datasets: list) -> str:
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

    def generate_and_execute(self, combined_df: pd.DataFrame, query: str, datasets: list) -> tuple[str, object, str]:
        if combined_df.empty:
            return "Данные оказались пустыми после объединения. Нечего обрабатывать.", None, ""

        available_columns = list(combined_df.columns)
        prompt = self._build_prompt_for_code(query, available_columns, datasets)
        code_str = call_llm(prompt)

        if not code_str.strip():
            return "Не удалось получить код от модели.", None, code_str

        code_clean = re.sub(r'```(?:python)?', '', code_str).strip()

        clean_df = combined_df.replace([np.inf, -np.inf], pd.NA).fillna(0)
        local_vars = {"df": clean_df.copy(), "pd": pd, "np": np}
        try:
            exec(code_clean, {}, local_vars)
        except Exception as e:
            logging.error(f"Ошибка при выполнении сгенерированного кода: {e}")
            return f"Ошибка при выполнении кода: {e}", None, code_str

        if "result" not in local_vars:
            return "Модель не сохранила результат в переменной result.", None, code_str

        result = local_vars["result"]
        if isinstance(result, pd.DataFrame):
            text_result = result.head(10).to_string(index=False)
        elif isinstance(result, pd.Series):
            text_result = result.head(10).to_string()
        else:
            try:
                text_result = str(result)
            except Exception:
                text_result = "Не удалось отобразить результат."

        return text_result, result, code_str

    def summarize(self, query: str, datasets: list, eda_summary: str, result_str: str) -> str:
        datasets_str = ", ".join(datasets)
        prompt = (
            "Ты аналитик, который отвечает на вопросы о муниципальных данных. "
            "На основе запроса пользователя и вычисленных результатов сформулируй понятный ответ.\n"
            f"Запрос: {query}\n"
            f"Использованные таблицы: {datasets_str}\n"
            f"Краткий анализ данных:\n{eda_summary}\n\n"
            f"Результат вычислений:\n{result_str}\n\n"
            "Сформулируй довольно краткий итог, посмотри на EDA и прочии и предложи гипотезы почему такая картина, "
            "пример вопроса: В каких МО высокая зарплата не помогает удержать население? "
            "пример ответа: МО Y имеет среднюю зарплату 79,000₽ (TOP-10%), но потерял 2.4% населения "
            "Причина может крыться в низкой рыночной доступности (MA = 150) и изоляции "
        )
        summary = call_llm(prompt)
        return summary.strip()