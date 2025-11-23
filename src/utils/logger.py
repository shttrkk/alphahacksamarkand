"""
Централизованная система логирования для Alpha RAG Elite
Логи пишутся одновременно в файл и stdout
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Цветной форматтер для консоли"""

    COLORS = {
        'DEBUG': '\033[0;36m',    # Cyan
        'INFO': '\033[0;32m',     # Green
        'WARNING': '\033[1;33m',  # Yellow
        'ERROR': '\033[0;31m',    # Red
        'CRITICAL': '\033[1;35m', # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname:8s}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    Создает логгер с записью в файл и консоль

    Args:
        name: Имя логгера (обычно __name__)
        log_file: Путь к лог-файлу (опционально)
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
        console: Выводить ли логи в консоль

    Returns:
        Настроенный логгер

    Example:
        logger = setup_logger("preprocess", "logs/preprocess.log")
        logger.info("Processing started")
        logger.error("Something went wrong", exc_info=True)
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Удаляем существующие handlers (чтобы не было дубликатов)
    logger.handlers.clear()

    # Формат для файла (с временем)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Формат для консоли (без времени, с цветом)
    console_formatter = ColoredFormatter(
        '%(levelname)s | %(message)s'
    )

    # Handler для файла
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Handler для консоли
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


def log_section(logger: logging.Logger, title: str, char: str = "=", width: int = 80):
    """
    Логирует красивый заголовок секции

    Example:
        log_section(logger, "PREPROCESSING")
        # ================================================================================
        # PREPROCESSING
        # ================================================================================
    """
    separator = char * width
    logger.info(separator)
    logger.info(title)
    logger.info(separator)


def log_summary(logger: logging.Logger, title: str, items: dict):
    """
    Логирует summary блок

    Args:
        logger: Логгер
        title: Заголовок блока
        items: Словарь с ключ-значение парами для вывода

    Example:
        log_summary(logger, "Results", {
            "Documents processed": 1937,
            "Chunks created": 65741,
            "Time elapsed": "2m 15s"
        })
    """
    log_section(logger, title, char="-", width=60)
    for key, value in items.items():
        logger.info(f"  {key}: {value}")
    logger.info("-" * 60)


def format_time(seconds: float) -> str:
    """
    Форматирует время в читаемый вид

    Args:
        seconds: Время в секундах

    Returns:
        Строка вида "2m 15s" или "1h 23m 45s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


# Глобальная функция для быстрого получения логгера
def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Быстрое создание логгера с автоматическим именем файла

    Args:
        name: Имя модуля (обычно "preprocess", "build_indices", "search")
        log_dir: Директория для логов

    Returns:
        Настроенный логгер
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/{name}.log"
    return setup_logger(name, log_file=log_file, level=logging.INFO)
