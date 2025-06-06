import json
import logging
import re

import pandas as pd
import numpy as np
from langchain_core.messages import HumanMessage

from .llm import llm


def call_llm(prompt: str) -> str:
    """Отправляет prompt в LLM и возвращает ответ."""
    logging.info(f"[LLM] Отправка запроса: {prompt[:100]}...")
    try:
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        raw = response.content
        logging.info(f"[LLM] Получен ответ: {raw[:100]}...")
        return raw
    except Exception as e:
        logging.error(f"Ошибка при обращении к LLM: {str(e)}")
        return ""


def parse_json_response(response_str: str) -> dict:
    """Парсим JSON из ответа LLM."""
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
    """Быстрая проверка, относится ли запрос к муниципальным данным."""
    keywords = [
        "зарплата", "население", "миграция", "рынок", "доступность",
        "бизнес", "регион", "город", "молодежь", "рост", "развитие",
        "территория", "муниципальный", "оквэд", "связность",
    ]
    q_lower = query.lower()
    if any(phrase in q_lower for phrase in keywords):
        return True
    prompt = (
        "Определи, относится ли запрос к анализу муниципальных данных. "
        "Критерии: население, зарплаты, миграция, бизнес, развитие регионов, связность. "
        "Ответ только 'valid' или 'ambiguous' без пояснений.\n"
        f"Запрос: '{query}'"
    )
    response = call_llm(prompt).lower().strip()
    return "valid" in response