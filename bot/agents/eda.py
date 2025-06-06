import logging
import pandas as pd


class EDAAgent:
    def __init__(self):
        self.role = "EDA"
        logging.info(f"[{self.role}] Инициализирован.")

    def analyze(self, df: pd.DataFrame) -> str:
        if df.empty:
            return "Данные отсутствуют или объединение вернуло пустой DataFrame."

        lines = [f"Размер данных: {len(df)} строк, {len(df.columns)} колонок."]

        missing = df.isna().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            lines.append("Пропуски (топ 5):")
            lines.append(missing.sort_values(ascending=False).head(5).to_string())

        desc = df.describe(include='all').transpose()
        lines.append("Статистика (первые 5 строк):")
        lines.append(desc.head().to_string())

        return "\n".join(lines)