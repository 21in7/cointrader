import sys
from loguru import logger


def setup_logger(log_level: str = "INFO"):
    logger.remove()
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level=log_level,
        colorize=True,
    )
    logger.add(
        "logs/bot_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        level="DEBUG",
        encoding="utf-8",
    )
