import logging

from ..config import DATA_FILES, DATA_DESCRIPTIONS
from ..data_loader import DataLoader
from ..utils import validate_query
from .classifier import ClassifierAgent
from .knowledge_search import KnowledgeSearchAgent
from .answer_generation import AnswerGenerationAgent
from .eda import EDAAgent
from .context import ContextAgent


class OrchestratorAgent:
    def __init__(self, data_files: dict = DATA_FILES, data_descriptions: dict = DATA_DESCRIPTIONS):
        DataLoader.check_data_files(data_files)
        self.classifier = ClassifierAgent(data_descriptions)
        self.searcher = KnowledgeSearchAgent(data_files)
        self.generator = AnswerGenerationAgent()
        self.eda = EDAAgent()
        self.context = ContextAgent()
        logging.info("[Оркестратор] Инициализирован.")

    def process_message(self, user_id: int, text: str):
        ctx = self.context.get_context(user_id)

        if not self._is_continuation(text, ctx.get("last_query", "")):
            self.context.clear_context(user_id)
            ctx = {}

        if not validate_query(text):
            self.context.update_context(user_id, {"last_query": text, "is_ambiguous": True})
            return "Пожалуйста, уточните ваш запрос. Например: 'Где самые высокие зарплаты в IT?'", {}, False

        classification = self.classifier.classify(text)
        tables = classification.get("datasets", [])
        if not tables:
            return "Не удалось определить подходящие данные. Попробуйте переформулировать запрос.", {}, False

        available_tables = [t for t in tables if t in self.searcher.data and not self.searcher.data[t].empty]
        missing_tables = [t for t in tables if t not in self.searcher.data or self.searcher.data[t].empty]
        combined_df = self.searcher.combine_dataframes(available_tables)

        eda_summary = self.eda.analyze(combined_df)

        result_text, _, code_snippet = self.generator.generate_and_execute(combined_df, text, available_tables)

        final_answer = self.generator.summarize(text, available_tables, eda_summary, result_text)
        if missing_tables:
            final_answer += "\nОтсутствуют данные для: " + ", ".join(missing_tables)

        self.context.update_context(
            user_id,
            {
                "last_query": text,
                "classification": classification,
                "last_answer": final_answer,
                "has_data": not combined_df.empty,
            },
        )

        return final_answer, classification, True

    def _is_continuation(self, new_text: str, prev_text: str) -> bool:
        if not prev_text:
            return False
        prompt = (
            "Определи, является ли новый запрос уточнением предыдущего.\n"
            f"Предыдущий: '{prev_text}'\n"
            f"Новый: '{new_text}'\n"
            "Ответ в формате JSON: {\"is_continuation\": true/false}"
        )
        from ..utils import parse_json_response, call_llm

        response = parse_json_response(call_llm(prompt))
        return response.get("is_continuation", False)