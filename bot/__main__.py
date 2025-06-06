import logging

from .telegram_bot import run_bot

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler(),
    ],
)

if __name__ == "__main__":
    run_bot()