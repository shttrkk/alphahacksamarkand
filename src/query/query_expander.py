"""
Multi-strategy query expansion.
Rule-based + BM25-based PRF (Pseudo-Relevance Feedback).
"""
import re
from typing import List, Set, Dict
from collections import defaultdict, Counter

from ..preprocessing.text_normalizer import normalize_text
from ..preprocessing.intent_classifier import IntentClassifier
from ..preprocessing.entity_extractor import EntityExtractor
from ..preprocessing.tokenizer import tokenize


class QueryExpander:
    """Расширение запросов несколькими стратегиями."""

    # Аббревиатуры и их полные формы
    ABBREVIATIONS = {
        r'\bбик\b': 'банковский идентификационный код',
        r'\bр\s*/\s*с\b': 'расчетный счет',
        r'\bк\s*/\s*с\b': 'корреспондентский счет',
        r'\bз\s*/\s*п\b': 'зарплата',
        r'\bсмс\b': 'сообщение',
        r'\bп\s*/\s*п\b': 'платежное поручение',
    }

    def __init__(
        self,
        enable_rule_based: bool = True,
        enable_prf: bool = True,
        prf_top_docs: int = 20,
        prf_top_terms: int = 15,
        max_variants: int = 2,
    ):
        self.enable_rule_based = enable_rule_based
        self.enable_prf = enable_prf
        self.prf_top_docs = prf_top_docs
        self.prf_top_terms = prf_top_terms
        self.max_variants = max_variants

        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()

    def expand(
        self,
        query: str,
        bm25=None,
        tokenized_docs: List[List[str]] = None,
    ) -> List[str]:
        """
        Генерирует варианты запроса.

        Args:
            query: Исходный запрос
            bm25: BM25 индекс (опционально, для PRF)
            tokenized_docs: Токенизированные документы (для PRF)

        Returns:
            Список вариантов запроса (включая оригинал)
        """
        variants = []

        # 1. Оригинальный запрос (нормализованный)
        query_normalized = normalize_text(query, is_query=True)
        variants.append(query_normalized)

        # 2. Rule-based expansion
        if self.enable_rule_based:
            rule_based = self._expand_rule_based(query_normalized)
            variants.extend(rule_based)

        # 3. BM25-based PRF
        if self.enable_prf and bm25 is not None and tokenized_docs is not None:
            prf_variant = self._expand_prf(query_normalized, bm25, tokenized_docs)
            if prf_variant:
                variants.append(prf_variant)

        # Убираем дубликаты, сохраняя порядок
        unique_variants = []
        seen = set()
        for v in variants:
            if v and v not in seen:
                unique_variants.append(v)
                seen.add(v)
            # Limit to max_variants for speed
            if len(unique_variants) >= self.max_variants:
                break

        return unique_variants

    def _expand_rule_based(self, query: str) -> List[str]:
        """Rule-based expansion."""
        variants = []

        # Извлекаем интенты и entities
        intents = self.intent_classifier.classify(query)
        entities = self.entity_extractor.extract(query)

        # Вариант 1: С добавлением "Альфа-Банк"
        if 'альфа' not in query and 'alfa' not in query:
            variants.append(f"{query} Альфа-Банк")

        # Вариант 2: Без вопросительных слов
        q_no_question = re.sub(
            r'\b(как|где|что|когда|почему|зачем|куда|откуда|чем|кто|сколько)\b',
            '',
            query,
            flags=re.I
        )
        q_no_question = re.sub(r'\s+', ' ', q_no_question).strip()
        if q_no_question and q_no_question != query:
            variants.append(q_no_question)

        # Вариант 3: Замена сокращений на полные формы
        q_expanded = query
        for abbr, full in self.ABBREVIATIONS.items():
            q_expanded = re.sub(abbr, full, q_expanded, flags=re.I)
        if q_expanded != query:
            variants.append(q_expanded)

        # Вариант 4: Entity-focused (только ключевые слова)
        tokens = tokenize(query, use_stopwords=True)
        keywords = [w for w in tokens if len(w) > 3]
        if len(keywords) >= 2:
            variants.append(' '.join(keywords))

        # Вариант 5: Intent-based templates
        for intent in intents:
            template_variants = self._get_intent_templates(intent, entities)
            variants.extend(template_variants)

        return variants[:10]  # Ограничиваем количество

    def _get_intent_templates(self, intent: str, entities: Dict) -> List[str]:
        """Шаблоны для конкретных интентов."""
        templates = []

        if intent == "БИК":
            templates.append("bik Альфа-Банка реквизиты")
            templates.append("банковский идентификационный код")

        elif intent == "РАСЧЕТНЫЙ_СЧЕТ":
            templates.append("расчетный_счет реквизиты организации")
            templates.append("счет организации Альфа-Банк")

        elif intent == "СМС_ПРОБЛЕМЫ":
            templates.append("sms код подтверждения не приходит")
            templates.append("проблемы sms уведомления")

        elif intent == "ЗАРПЛАТА":
            templates.append("зарплата зарплатная карта проект")

        elif intent == "КРЕДИТ":
            templates.append("кредит кредитная карта условия")

        elif intent == "АЛЬФА_ОНЛАЙН":
            templates.append("альфа_онлайн мобильное приложение")
            templates.append("интернет банк личный кабинет")

        elif intent == "БЛОКИРОВКА":
            templates.append("блокировка карта счет")

        return templates

    def _expand_prf(
        self,
        query: str,
        bm25,
        tokenized_docs: List[List[str]],
    ) -> str:
        """
        Pseudo-Relevance Feedback expansion.

        Берет top BM25 документы, извлекает частые термины, добавляет к запросу.
        """
        try:
            query_tokens = tokenize(query)
            if not query_tokens:
                return ""

            # Получаем scores для всех документов
            scores = bm25.get_scores(query_tokens)

            # Top-k документов
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:self.prf_top_docs]

            # Собираем термины из топовых документов
            term_freq = Counter()
            for idx in top_indices:
                for term in tokenized_docs[idx]:
                    term_freq[term] += 1

            # Исключаем стоп-слова и термины из оригинального запроса
            query_tokens_set = set(query_tokens)
            stopwords = {
                'и', 'в', 'на', 'с', 'по', 'для', 'к', 'от', 'о', 'за',
                'у', 'из', 'до', 'со', 'при', 'об', 'под', 'над',
            }

            # Топ термины (не из запроса и не стоп-слова)
            expansion_terms = [
                term for term, count in term_freq.most_common(self.prf_top_terms * 2)
                if term not in query_tokens_set and term not in stopwords and len(term) > 3
            ][:self.prf_top_terms]

            if expansion_terms:
                return query + ' ' + ' '.join(expansion_terms)

        except Exception as e:
            # Если что-то пошло не так, просто возвращаем пустую строку
            return ""

        return ""
