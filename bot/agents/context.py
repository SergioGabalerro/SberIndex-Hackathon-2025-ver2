import logging


class ContextAgent:
    def __init__(self):
        self.user_context = {}
        logging.info("[Контекст] Инициализирован.")

    def update_context(self, user_id: int, data: dict):
        self.user_context[user_id] = {
            **self.user_context.get(user_id, {}),
            **data,
        }

    def get_context(self, user_id: int) -> dict:
        return self.user_context.get(user_id, {})

    def clear_context(self, user_id: int):
        if user_id in self.user_context:
            del self.user_context[user_id]