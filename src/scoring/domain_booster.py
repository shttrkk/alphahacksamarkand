"""
Domain-specific boosts для банковской тематики.
Агрессивные boosts за entity matching и intent matching.
"""
from typing import Dict, Set, List
import re

from ..preprocessing.entity_extractor import EntityExtractor
from ..preprocessing.intent_classifier import IntentClassifier


class DomainBooster:
    """Вычисление domain-specific boosts."""

    def __init__(
        self,
        entity_boosts: Dict[str, float] = None,
        intent_boosts: Dict[str, float] = None,
        title_boost: float = 0.3,
        high_bm25_boost: float = 0.2,
    ):
        """
        Args:
            entity_boosts: Boosts за точное совпадение entities
            intent_boosts: Boosts за совпадение интентов
            title_boost: Boost если ключевые слова запроса в заголовке
            high_bm25_boost: Boost если документ в top-10 по BM25
        """
        # Default boosts (агрессивные для достижения 60%)
        self.entity_boosts = entity_boosts or {
            "bik": 3.0,  # Огромный boost за точный БИК
            "rs": 2.0,   # р/с
            "ks": 1.5,   # к/с
            "phone": 1.0,
            "sum": 0.5,
            "percent": 0.5,
        }

        self.intent_boosts = intent_boosts or {
            "БИК": 0.8,
            "РАСЧЕТНЫЙ_СЧЕТ": 0.8,
            "КОРР_СЧЕТ": 0.7,
            "СМС_ПРОБЛЕМЫ": 0.6,
            "СМС_НАСТРОЙКА": 0.6,
            "ЗАРПЛАТА": 0.6,
            "КРЕДИТ": 0.6,
            "ИПОТЕКА": 0.7,
            "БЛОКИРОВКА": 0.7,
            "АЛЬФА_ОНЛАЙН": 0.6,
            "ПЕРЕВОДЫ": 0.5,
            "КАРТА": 0.5,
            "ВКЛАД": 0.6,
        }

        self.title_boost = title_boost
        self.high_bm25_boost = high_bm25_boost

        self.entity_extractor = EntityExtractor()
        self.intent_classifier = IntentClassifier()

    def compute_boost(
        self,
        query: str,
        doc_text: str,
        doc_title: str = "",
        query_entities: Dict[str, List[str]] = None,
        query_intents: Set[str] = None,
        is_high_bm25: bool = False,
    ) -> float:
        """
        Вычисляет domain boost для пары (query, document).

        Args:
            query: Запрос
            doc_text: Текст документа
            doc_title: Заголовок документа
            query_entities: Entities из запроса (если уже извлечены)
            query_intents: Интенты из запроса (если уже извлечены)
            is_high_bm25: Документ в топ-10 по BM25

        Returns:
            Суммарный boost
        """
        total_boost = 0.0

        # 1. Entity matching
        if query_entities is None:
            query_entities = self.entity_extractor.extract(query)

        doc_entities = self.entity_extractor.extract(doc_text)
        entity_matches = self.entity_extractor.match_entities(query_entities, doc_entities)

        for entity_type, matches in entity_matches.items():
            if matches:
                total_boost += self.entity_boosts.get(entity_type, 0.0)

        # 2. Intent matching
        if query_intents is None:
            query_intents = self.intent_classifier.classify(query)

        doc_intents = self.intent_classifier.classify(doc_text)

        # Пересечение интентов
        matching_intents = query_intents & doc_intents
        for intent in matching_intents:
            total_boost += self.intent_boosts.get(intent, 0.0)

        # 3. Title keyword matching
        if doc_title and self._has_query_keywords_in_title(query, doc_title):
            total_boost += self.title_boost

        # 4. High BM25 rank
        if is_high_bm25:
            total_boost += self.high_bm25_boost

        return total_boost

    def _has_query_keywords_in_title(self, query: str, title: str) -> bool:
        """Проверяет наличие ключевых слов запроса в заголовке."""
        query_lower = query.lower()
        title_lower = title.lower()

        # Извлекаем слова длиннее 3 символов
        query_words = [w for w in re.findall(r'\w+', query_lower) if len(w) > 3]

        if not query_words:
            return False

        # Проверяем, есть ли хотя бы половина слов в заголовке
        matches = sum(1 for word in query_words if word in title_lower)
        return matches >= len(query_words) / 2

    def compute_boosts_batch(
        self,
        query: str,
        documents: List[Dict],
        query_entities: Dict[str, List[str]] = None,
        query_intents: Set[str] = None,
        high_bm25_ids: Set[int] = None,
    ) -> Dict[int, float]:
        """
        Вычисляет boosts для батча документов.

        Args:
            query: Запрос
            documents: Список документов [{"doc_id": int, "text": str, "title": str}, ...]
            query_entities: Entities из запроса
            query_intents: Интенты из запроса
            high_bm25_ids: Set doc_id которые в топ-10 BM25

        Returns:
            Dict[doc_id -> boost]
        """
        if query_entities is None:
            query_entities = self.entity_extractor.extract(query)

        if query_intents is None:
            query_intents = self.intent_classifier.classify(query)

        if high_bm25_ids is None:
            high_bm25_ids = set()

        boosts = {}

        for doc in documents:
            doc_id = doc["doc_id"]
            doc_text = doc.get("text", "")
            doc_title = doc.get("title", "")
            is_high_bm25 = doc_id in high_bm25_ids

            boost = self.compute_boost(
                query=query,
                doc_text=doc_text,
                doc_title=doc_title,
                query_entities=query_entities,
                query_intents=query_intents,
                is_high_bm25=is_high_bm25,
            )

            boosts[doc_id] = boost

        return boosts
