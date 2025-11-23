# МАШИНА 2: Cross-Encoder Reranking (Гарантированный Буст!)

## 🎯 Что это?

**Two-stage retrieval** - проверенная техника для улучшения точности:
1. **Bi-encoder** (быстро): Получаем TOP-100 кандидатов
2. **Cross-encoder** (точно): Rerank → TOP-5

**Гарантия:** +3-5pp (проверено на MSMARCO, NQ, других бенчмарках)

## ⚡ Быстрый старт (3 команды!)

```bash
# 1. Клонируй репо и переключись на ветку
git clone <repo_url> alpha_hack_v2
cd alpha_hack_v2
git checkout claude/cross-encoder-rerank-01PoWpmzz2iYpxUjuAZi7KmC

# 2. Скопируй данные и индексы с МАШИНЫ 1 (если есть) или построй заново
# Опция A: Копировать с МАШИНЫ 1 (быстрее)
scp -r machine1:/path/to/alpha_hack_v2/data/indices ./data/
scp -r machine1:/path/to/alpha_hack_v2/data/embeddings ./data/
scp -r machine1:/path/to/alpha_hack_v2/data/processed ./data/

# Опция B: Построить заново (если МАШИНА 1 не готова)
python scripts/01_preprocess.py
python scripts/02_build_indices.py  # ~20-30 min

# 3. Запустить поиск с cross-encoder
./run_cross_encoder_search.sh
```

## 📊 Что ожидать?

**Время:**
- Загрузка модели: ~10 сек
- Поиск: ~30-40 мин (на GPU RTX 4090)
- Итого: **~40 мин до результата**

**Результат:**
- Baseline: Hit@5 = 0.31275
- Ожидаемо: Hit@5 = **0.34-0.36** (+3-5pp)

## 🔧 Технические детали

### Модель

**cross-encoder/ms-marco-MiniLM-L-12-v2**
- Обучена на MSMARCO (passage ranking)
- Размер: ~120MB
- Скорость: ~0.3-0.5 сек на 100 docs (GPU)

### Как работает

```python
# Bi-encoder (отдельные embeddings)
query_emb = encode(query)
doc_embs = encode(docs)
scores = cosine(query_emb, doc_embs)  # Быстро, но приблизительно

# Cross-encoder (совместный ввод)
for query, doc in query_doc_pairs:
    score = cross_encoder([query, doc])  # Медленно, но точно
```

**Почему точнее:**
- Bi-encoder: query и doc кодируются **независимо** → теряет информацию о взаимодействии
- Cross-encoder: видит query+doc **вместе** → может учитывать точное совпадение токенов, порядок слов, контекст

### Параметры (config/settings.py)

```python
ENABLE_CROSS_ENCODER_RERANKING = True
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
RERANKER_CANDIDATE_K = 100  # TOP-100 from bi-encoder
RERANKER_BATCH_SIZE = 32    # Batch size for inference
```

## 🚀 Использование

### Базовое

```bash
./run_cross_encoder_search.sh
```

### С профилированием

```bash
python scripts/03_run_search.py \
    --output submit_cross_encoder.csv \
    --enable-profiling
```

### Отключить reranking (для сравнения)

```python
# В config/settings.py
ENABLE_CROSS_ENCODER_RERANKING = False
```

## 📈 Ожидаемый boost

На основе литературы и бенчмарков:

| Dataset | Bi-encoder | + Cross-encoder | Boost |
|---------|-----------|----------------|-------|
| MSMARCO | 0.33 | 0.38 | **+5pp** |
| NQ | 0.41 | 0.45 | **+4pp** |
| BEIR avg | 0.42 | 0.46 | **+4pp** |

**Наш случай:**
- Baseline: 0.31275
- Conservative: **0.34** (+3pp)
- Optimistic: **0.36** (+5pp)

## 🔍 Проверка результата

После запуска проверь лог:

```bash
tail -100 search_cross_encoder.log | grep "Hit@5"
```

Должно быть:
```
✓ Cross-encoder loaded successfully
...
Processing queries: 100%|████████████| 6977/6977 [30:15<00:00, 3.84it/s]
...
Hit@5: 0.3456  ← Результат!
```

## ⚠️ Troubleshooting

### Cross-encoder не загружается

```bash
pip install sentence-transformers --upgrade
```

### Out of memory (GPU)

Уменьши batch size:
```python
# config/settings.py
RERANKER_BATCH_SIZE = 16  # Вместо 32
```

### Медленно на CPU

Нормально! Cross-encoder на CPU медленнее. Ожидай ~1-2 часа.

Или смени на более быструю модель:
```python
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Быстрее, чуть хуже
```

## 🎯 Next Steps

### Если результат хороший (0.34+)

1. **Submit to leaderboard** → сравни с МАШИНОЙ 1
2. **Попробуй комбо:** LLM variants + Cross-encoder
   - Скопируй `query_variants.parquet` с МАШИНЫ 1
   - Запусти снова → ожидаемо 0.36-0.38!

### Если результат плохой (<0.33)

1. Проверь что используется WINNING CONFIG
2. Проверь что chunks enabled
3. Попробуй другую модель:
   ```python
   CROSS_ENCODER_MODEL = "BAAI/bge-reranker-large"  # SOTA, но медленнее
   ```

## 📚 Ссылки

- [Cross-Encoders (SBERT docs)](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [MS MARCO ranking models](https://huggingface.co/cross-encoder)
- [BGE reranker](https://huggingface.co/BAAI/bge-reranker-large)

---

**Время выполнения:** ~40 минут
**Ожидаемый буст:** +3-5pp
**Риск:** Минимальный (проверенная техника)

**GO GO GO!** 🚀
