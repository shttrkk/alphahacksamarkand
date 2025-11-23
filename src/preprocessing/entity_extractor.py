"""
Entity extraction для банковского домена.
Извлекает БИК, р/с, к/с, телефоны, суммы, проценты.
"""
import re
from typing import Dict, List, Optional


class EntityExtractor:
    """Извлечение entities из текста."""

    # Паттерны для извлечения
    PATTERNS = {
        # БИК: ровно 9 цифр
        'bik': r'\b\d{9}\b',

        # Расчетный счет: ровно 20 цифр
        'rs': r'\b\d{20}\b',

        # Корреспондентский счет: начинается с 301, всего 20 цифр
        'ks': r'\b301\d{17}\b',

        # Телефоны: +7 или 8 с 10 цифрами
        'phone': r'(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}',

        # Суммы: число + руб/рублей/₽
        'sum': r'\d+(?:\s*\d{3})*(?:[\.,]\d{1,2})?\s*(?:руб(?:лей|ля|ль)?|₽|рублей)',

        # Проценты: число + %
        'percent': r'\d+(?:[\.,]\d{1,2})?\s*%',
    }

    def __init__(self):
        # Компилируем паттерны
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.PATTERNS.items()
        }

    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Извлекает все entities из текста.

        Args:
            text: Текст для обработки

        Returns:
            Dict с найденными entities: {entity_type: [values]}
        """
        if not isinstance(text, str) or not text:
            return {key: [] for key in self.PATTERNS.keys()}

        entities = {}

        for entity_type, pattern in self.compiled_patterns.items():
            matches = pattern.findall(text)
            entities[entity_type] = self._clean_matches(matches, entity_type)

        return entities

    def _clean_matches(self, matches: List[str], entity_type: str) -> List[str]:
        """Очистка и нормализация найденных значений."""
        if not matches:
            return []

        cleaned = []
        for match in matches:
            match = match.strip()

            if entity_type == 'phone':
                # Нормализация телефона: убираем все кроме цифр, добавляем +7
                digits = re.sub(r'\D', '', match)
                if digits.startswith('8') and len(digits) == 11:
                    digits = '7' + digits[1:]
                if len(digits) == 11:
                    cleaned.append('+' + digits)
            elif entity_type in ['bik', 'rs', 'ks']:
                # Только цифры
                digits = re.sub(r'\D', '', match)
                cleaned.append(digits)
            elif entity_type == 'sum':
                # Извлекаем число
                num = re.search(r'\d+(?:[\s\.,]\d+)*(?:[\.,]\d{1,2})?', match)
                if num:
                    cleaned.append(num.group().replace(' ', '').replace(',', '.'))
            elif entity_type == 'percent':
                # Извлекаем число
                num = re.search(r'\d+(?:[\.,]\d{1,2})?', match)
                if num:
                    cleaned.append(num.group().replace(',', '.'))
            else:
                cleaned.append(match)

        # Убираем дубликаты, сохраняя порядок
        seen = set()
        result = []
        for item in cleaned:
            if item not in seen:
                seen.add(item)
                result.append(item)

        return result

    def has_entity(self, text: str, entity_type: str) -> bool:
        """Проверяет наличие entity определенного типа."""
        entities = self.extract(text)
        return len(entities.get(entity_type, [])) > 0

    def match_entities(
        self,
        query_entities: Dict[str, List[str]],
        doc_entities: Dict[str, List[str]]
    ) -> Dict[str, bool]:
        """
        Проверяет совпадение entities между запросом и документом.

        Args:
            query_entities: Entities из запроса
            doc_entities: Entities из документа

        Returns:
            Dict с результатами совпадения для каждого типа entity
        """
        matches = {}

        for entity_type in self.PATTERNS.keys():
            query_vals = set(query_entities.get(entity_type, []))
            doc_vals = set(doc_entities.get(entity_type, []))

            # Есть ли пересечение?
            matches[entity_type] = bool(query_vals & doc_vals)

        return matches


# Глобальный экстрактор
_extractor = None

def extract_entities(text: str) -> Dict[str, List[str]]:
    """Глобальная функция для извлечения entities."""
    global _extractor
    if _extractor is None:
        _extractor = EntityExtractor()
    return _extractor.extract(text)
