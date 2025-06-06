import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MODEL_TYPE = os.getenv("MODEL_TYPE", "gigachat").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GIGACHAT_AUTH_KEY = os.getenv("GIGACHAT_AUTH_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")

DATA_DIR = Path('data')

DATA_FILES = {
    "market_access": DATA_DIR / "1_market_access.parquet",
    "population":    DATA_DIR / "2_bdmo_population.parquet",
    "migration":     DATA_DIR / "3_bdmo_migration.parquet",
    "salary":        DATA_DIR / "4_bdmo_salary.parquet",
    "connections":   DATA_DIR / "5_connection.parquet",
    "mo_ref":        DATA_DIR / "mo_ref.xlsx",
}

DATA_DESCRIPTIONS = {
    "market_access": "таблица с индексами рыночной доступности для каждого муниципалитета (territory_id, market_access, year)",
    "population":    "таблица с численностью населения по муниципалитетам (territory_id, year, value)",
    "migration":     "таблица с миграционными потоками по муниципалитетам (territory_id, year, in_migration, out_migration)",
    "salary":        "таблица с информацией о зарплатах по оквэдам (okved_letter, territory_id, year, average_salary)",
    "connections":   "таблица с показателями связности муниципалитетов (например, infrastructure_index, territory_id, year)",
    "mo_ref":        "справочник муниципалитетов: territory_id, name, region, parent_district",
}