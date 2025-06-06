import logging
from pathlib import Path
import pandas as pd


class DataLoader:
    """Утилита для проверки и загрузки файлов."""

    @staticmethod
    def check_data_files(data_files: dict) -> None:
        missing = [str(f) for f in data_files.values() if not Path(f).exists()]
        if missing:
            error_msg = f"Отсутствуют файлы данных: {', '.join(missing)}"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)

    @staticmethod
    def load_data(data_files: dict) -> dict:
        data = {}
        for key, path in data_files.items():
            try:
                path = Path(path)
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
                data[key] = pd.DataFrame()
        return data