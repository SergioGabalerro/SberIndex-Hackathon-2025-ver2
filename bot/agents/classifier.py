import logging

from ..utils import call_llm, parse_json_response


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
            "Верни ответ ровно в формате JSON: {\"datasets\": [\"имя1\", \"имя2\", …]}"
        )

    def classify(self, query: str) -> dict:
        prompt = self._build_prompt(query)
        result = parse_json_response(call_llm(prompt))
        return result if result.get("datasets") else {"datasets": []}