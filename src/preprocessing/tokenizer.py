"""
Custom tokenizer для BM25 с банковской спецификой.
"""
import re
from typing import List


class BankingTokenizer:
    """Токенизатор для BM25 с учетом банковских терминов."""

    # Стоп-слова (минимальный set, не удаляем важные слова)
    STOPWORDS = {
        'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со',
        'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да',
        'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только',
        'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет',
        'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг',
        'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него',
        'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там',
        'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут',
        'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их',
        'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего',
        'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто',
        'этот', 'того', 'потому', 'этого', 'какой', 'совсем',
    }

    def __init__(self, use_stopwords: bool = True):
        self.use_stopwords = use_stopwords

    def tokenize(self, text: str) -> List[str]:
        """
        Токенизация текста.

        Args:
            text: Текст для токенизации

        Returns:
            Список токенов
        """
        if not isinstance(text, str) or not text:
            return []

        # Приводим к нижнему регистру
        text = text.lower()

        # Извлекаем слова (буквы и цифры)
        tokens = re.findall(r'\w+', text)

        # Фильтруем стоп-слова и короткие токены
        if self.use_stopwords:
            tokens = [
                tok for tok in tokens
                if tok not in self.STOPWORDS and len(tok) > 1
            ]
        else:
            tokens = [tok for tok in tokens if len(tok) > 1]

        return tokens

    def __call__(self, text: str) -> List[str]:
        """Позволяет использовать как функцию."""
        return self.tokenize(text)


# Глобальный токенизатор
_tokenizer = None

def tokenize(text: str, use_stopwords: bool = True) -> List[str]:
    """Глобальная функция для токенизации."""
    global _tokenizer
    if _tokenizer is None or _tokenizer.use_stopwords != use_stopwords:
        _tokenizer = BankingTokenizer(use_stopwords=use_stopwords)
    return _tokenizer.tokenize(text)
