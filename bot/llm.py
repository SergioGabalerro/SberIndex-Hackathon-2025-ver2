import logging
from types import SimpleNamespace

from langchain_openai import ChatOpenAI
from langchain_gigachat.chat_models import GigaChat
import openai

from .config import MODEL_TYPE, OPENAI_API_KEY, GIGACHAT_AUTH_KEY, DEEPSEEK_API_KEY


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


def init_llm():
    if MODEL_TYPE == "openai":
        logging.info("Инициализация ChatOpenAI (OpenAI) в качестве LLM")
        return ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-4o",
            temperature=0.0,
            request_timeout=30,
        )
    if MODEL_TYPE == "deepseek":
        logging.info("Инициализация DeepSeek в качестве LLM")
        return DeepSeekLLM(
            api_key=DEEPSEEK_API_KEY,
            model="deepseek-chat",
            temperature=0.0,
            timeout=30,
        )
    logging.info("Инициализация GigaChat в качестве LLM")
    return GigaChat(
        credentials=GIGACHAT_AUTH_KEY,
        verify_ssl_certs=False,
        model="GigaChat:2-Max",
        timeout=30,
    )


llm = init_llm()