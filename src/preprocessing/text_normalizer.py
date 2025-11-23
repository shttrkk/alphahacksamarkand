"""
Enhanced text normalization for banking domain.
Унификация банковских терминов, entity preservation, noise removal.
"""
import re
from typing import Dict, List


class BankingTextNormalizer:
    """Нормализация текста с фокусом на банковский домен."""

    # Унификация банковских терминов (все варианты → канонический)
    BANKING_NORMALIZATIONS = {
        # БИК
        r'\bбик\b': 'bik',
        r'БИК': 'bik',
        r'бик\s+банка': 'bik',
        r'банковский\s+идентификационный\s+код': 'bik',

        # Расчетный счет
        r'р\s*/\s*с(?:чет)?': 'расчетный_счет',
        r'\bрс\b': 'расчетный_счет',
        r'расч[её]тн(?:ый|ого)?\s*с(?:чет|ч[её]т)': 'расчетный_счет',
        r'расчетный\s+счет': 'расчетный_счет',
        r'расчётный\s+счёт': 'расчетный_счет',

        # Корреспондентский счет
        r'к\s*/\s*с(?:чет)?': 'корр_счет',
        r'корр\s+счет': 'корр_счет',
        r'корреспондентск(?:ий|ого)?\s*счет': 'корр_счет',

        # СМС
        r'\bсмс\b': 'sms',
        r'СМС': 'sms',
        r's\s*m\s*s': 'sms',
        r'с\s*м\s*с': 'sms',

        # Альфа-Онлайн
        r'альфа[\s\-]?онлайн': 'альфа_онлайн',
        r'alfa[\s\-]?online': 'альфа_онлайн',
        r'альфа\s+мобайл': 'альфа_онлайн',
        r'мобильн(?:ое|ый)?\s+приложение': 'альфа_онлайн',

        # Зарплата
        r'\bз\s*/\s*п\b': 'зарплата',
        r'\bзп\b': 'зарплата',
        r'зарплатн': 'зарплата',

        # Кредит
        r'кредитн': 'кредит',
        r'кредитка': 'кредит',

        # Карта
        r'карт[аоы]': 'карта',

        # Перевод
        r'перевод[аыов]': 'перевод',

        # Блокировка
        r'блокировк[аи]': 'блокировка',
        r'заблокирова': 'блокировка',
    }

    # Паттерны мусора (удаляем)
    NOISE_PATTERNS = [
        r'cookie|cookies|куки|баннер\s*cookie',
        r'политик[аи]\s+(?:конфиденц|персональн)[^\n]{0,300}',
        r'карта\s+сайта',
        r'генеральная\s+лицензия\s+банка\s+россии[^\n]{0,200}',
        r'центр\s+раскрытия\s+корпоративной\s+информации[^\n]{0,250}',
        r'является\s+оператором\s+по\s+обработке\s+персональных\s+данных[^\n]{0,300}',
        r'©\s*\d{4}.*?альфа[\s\-]?банк',
        r'огрн\s*:?\s*\d{13,15}',
        r'инн\s*:?\s*\d{10,12}',
        r'кпп\s*:?\s*\d{9}',
        r'лицензия\s+№?\s*\d+.*?от\s+\d{2}\.\d{2}\.\d{4}',
        r'все\s+права\s+защищены',
        r'нажмите\s+согласен',
        r'подтверждая\s+регистрацию',
        r'мы\s+используем\s+файлы\s+cookie',
    ]

    def __init__(self):
        # Компилируем регулярки для скорости
        self.banking_patterns = [
            (re.compile(pattern, re.IGNORECASE), replacement)
            for pattern, replacement in self.BANKING_NORMALIZATIONS.items()
        ]
        self.noise_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.NOISE_PATTERNS
        ]

    def normalize(self, text: str) -> str:
        """
        Полная нормализация текста.

        Args:
            text: Исходный текст

        Returns:
            Нормализованный текст
        """
        if not isinstance(text, str) or not text:
            return ""

        # 1. Базовая очистка
        text = text.replace('\u00A0', ' ')  # non-breaking space
        text = text.lower()
        text = text.replace('ё', 'е')

        # 2. Удаление мусора
        for pattern in self.noise_patterns:
            text = pattern.sub(' ', text)

        # 3. Унификация банковских терминов
        for pattern, replacement in self.banking_patterns:
            text = pattern.sub(replacement, text)

        # 4. Очистка символов (оставляем буквы, цифры, базовую пунктуацию)
        text = re.sub(r'[^\w\s\.\,\-\+\%\$\d\:\;\/]', ' ', text)

        # 5. Нормализация пробелов
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def normalize_query(self, query: str) -> str:
        """
        Нормализация запроса (более легкая, сохраняем больше информации).

        Args:
            query: Исходный запрос

        Returns:
            Нормализованный запрос
        """
        if not isinstance(query, str) or not query:
            return ""

        # Не удаляем noise patterns для запросов
        text = query.replace('\u00A0', ' ')
        text = text.lower()
        text = text.replace('ё', 'е')

        # Унификация банковских терминов
        for pattern, replacement in self.banking_patterns:
            text = pattern.sub(replacement, text)

        # Более мягкая очистка для запросов
        text = re.sub(r'[^\w\s\.\,\-\+\%\$\d\:\;\?\!\/]', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()


# Удобная функция для быстрого использования
_normalizer = None

def normalize_text(text: str, is_query: bool = False) -> str:
    """
    Глобальная функция для нормализации.

    Args:
        text: Текст для нормализации
        is_query: Если True, используется более легкая нормализация для запросов

    Returns:
        Нормализованный текст
    """
    global _normalizer
    if _normalizer is None:
        _normalizer = BankingTextNormalizer()

    if is_query:
        return _normalizer.normalize_query(text)
    return _normalizer.normalize(text)
