# CONSERVATIVE CONFIG - Безрисковый режим

## Что ОТКЛЮЧЕНО (по умолчанию)

### ❌ Рискованные модели
- **BGE-m3** - часто не загружается, нестабильный → **ОТКЛЮЧЕН**
- **LaBSE** - опциональная модель → **ОТКЛЮЧЕНА** (для скорости)

### ❌ Рискованные фичи
- **BM25 PRF** (Pseudo-Relevance Feedback) → **ОТКЛЮЧЕН**
  - Причина: может добавить шум, снизить точность
  - Можно включить позже если локальные тесты покажут прирост

### ❌ Второй chunk-level эмбеддер
- **Sbert-RU chunk** → **ОТКЛЮЧЕН**
  - Причина: для скорости, один E5 достаточно
  - Можно включить позже

## Что ВКЛЮЧЕНО (100% надежно)

### ✅ Doc-level модели (3 шт)
1. **E5-large** (intfloat/multilingual-e5-large) - вес 2.5
   - SOTA multilingual модель
   - 100% надежная, всегда загружается

2. **Sbert-RU** (sberbank-ai/sbert_large_nlu_ru) - вес 2.0
   - Специализированная русскоязычная модель
   - Стабильная, проверенная

3. **MPnet** (paraphrase-multilingual-mpnet-base-v2) - вес 1.8
   - Легкая, быстрая модель
   - Хорошая альтернатива BGE-m3
   - 100% стабильная

### ✅ Chunk-level модели (1 шт)
1. **E5-large chunk** - вес 2.0
   - Самый надежный эмбеддер
   - Достаточно одной модели для chunk-level

### ✅ Все остальные компоненты
- ✅ Enhanced text normalization (банковские термины)
- ✅ Entity extraction (БИК, р/с, телефоны)
- ✅ Intent classification (13 интентов)
- ✅ Rule-based query expansion (7-10 вариантов)
- ✅ BM25 sparse retrieval
- ✅ Weighted RRF fusion
- ✅ Aggressive domain boosts (БИК +3.0, р/с +2.0)

## Ожидаемый результат

**Conservative config:**
- Baseline (Егор): 28% Hit@5
- С conservative config: **36-42% Hit@5**
- Прирост: **+8-14 процентных пунктов**

**Влияние отключенных компонентов:**
- Без BGE-m3: -2-3 п.п. (но +100% стабильность)
- Без LaBSE: -1-2 п.п.
- Без PRF: -1-2 п.п.
- Без 2-го chunk embedder: -1-2 п.п.
- **Total потеря: -5-9 п.п.**

**Trade-off:**
- Потеря: 5-9 п.п. потенциального прироста
- Выигрыш: 100% стабильность, быстрее индексация, меньше рисков

## Когда включать отключенные компоненты

### Включить BGE-m3
**Когда:** Если conservative config дал >= 38%
**Как:**
```python
# В config/settings.py
"bge_m3": {
    ...
    "enabled": True,  # Было False
}
```

### Включить PRF
**Когда:** Если есть локальный validation set и тесты показывают прирост
**Как:**
```python
# В config/settings.py
ENABLE_BM25_PRF = True  # Было False
```

### Включить LaBSE
**Когда:** Если нужен дополнительный прирост и есть время на индексацию
**Как:**
```python
# В config/settings.py
"labse": {
    ...
    "enabled": True,  # Было False
}
```

## Время выполнения

### Conservative (3 doc + 1 chunk)
- Preprocessing: ~2-3 мин
- Indexing: **~20-25 мин** (быстрее!)
- Inference: ~3-5 мин
- **Total: ~30 минут**

### Full (4 doc + 2 chunk + PRF)
- Preprocessing: ~2-3 мин
- Indexing: **~35-40 мин**
- Inference: ~5-8 мин
- **Total: ~45 минут**

**Экономия времени: ~15 минут**

## Рекомендация

**Стратегия:**
1. **Сначала** запустить conservative config (30 мин)
2. **Получить первый результат** на лидерборде
3. **Если >= 38%** → хорошо, можно экспериментировать дальше
4. **Если < 38%** → debug, tune параметры, не включать рискованные фичи
5. **Если >= 42%** → можно попробовать включить BGE-m3 или PRF

**Принцип:** Сначала стабильный baseline, потом улучшения.

## Переключение между режимами

### Быстрое переключение на Full config

```bash
# В config/settings.py изменить все enabled на True
# Или использовать sed:
sed -i 's/"enabled": False/"enabled": True/g' config/settings.py
sed -i 's/ENABLE_BM25_PRF = False/ENABLE_BM25_PRF = True/' config/settings.py

# Пересобрать индексы
python scripts/02_build_indices.py --no-cache
```

### Вернуться на Conservative

```bash
git checkout config/settings.py
python scripts/02_build_indices.py --no-cache
```

---

**Итого:** Conservative config - это безопасный, быстрый, стабильный старт с хорошим результатом 36-42%.
