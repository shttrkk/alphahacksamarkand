# Alpha RAG Elite – Hackathon Crunch Edition

## Архитектура

**Hybrid Multi-Level Retrieval System:**
- 🔥 **4 dense embedders** (E5-large, BGE-m3, Sbert-RU, LaBSE)
- 🔥 **Doc + Chunk level** retrieval
- 🔥 **Weighted RRF** fusion
- 🔥 **Aggressive domain boosts** (БИК +3.0, р/с +2.0)
- 🔥 **Multi-strategy query expansion** (rule-based + PRF)
- ⚠️ **Optional lightweight reranker** (only if validated)


---

## Требования

### Железо

**Рекомендуемая конфигурация:**
- GPU: **RTX 4090 (24GB)** или **A5000 (24GB)**
- RAM: **32GB+**
- Disk: **200GB NVMe SSD**
- Python: **3.10+**

**Минимальная конфигурация:**
- GPU: RTX 3090 (24GB)
- RAM: 24GB
- Disk: 150GB

### Время выполнения

На RTX 4090:
- Preprocessing: ~2-3 мин
- Indexing (4 doc + 2 chunk embedders): ~30-35 мин
- Inference (без reranking): ~3-5 мин
- Inference (с reranking): ~8-12 мин

**Total pipeline:** ~40-50 мин

---

## Установка

### 1. Клонирование репозитория

```bash
git clone ,,,
cd ...
```

### 2. Создание окружения

**ВАРИАНТ A: Автоматическая установка (рекомендуется)**

```bash
chmod +x setup.sh
./setup.sh

# После завершения ОБЯЗАТЕЛЬНО активируйте venv:
source venv/bin/activate
```

**ВАРИАНТ Б: Ручная установка**

```bash
# Python 3.10+
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Установка зависимостей
pip install --upgrade pip
pip install -r requirements.txt

# Установка FAISS
pip install faiss-cpu
```

**Важно:**
- Проект использует `faiss-cpu` для максимальной совместимости
- faiss-gpu часто имеет проблемы с разными версиями CUDA
- PyTorch будет использовать GPU для эмбеддингов (основная нагрузка)
- **Всегда активируйте venv перед запуском скриптов!**

### 3. Подготовка данных

Положите файлы в `data/raw/`:
- `websites_updated.csv`
- `questions_clean.csv`

---

## Запуск пайплайна

**ВАЖНО:** Перед запуском активируйте виртуальное окружение!
```bash
source venv/bin/activate
```

### ВАРИАНТ A: Автоматический запуск (рекомендуется)

```bash
# Активируйте venv
source venv/bin/activate

# Запустите весь пайплайн одной командой
chmod +x run_pipeline.sh
./run_pipeline.sh

# Output: submit_alpha_rag_elite.csv
```

Скрипт автоматически выполнит все 3 этапа и проверит результат.

**Время:** ~40-50 минут (на RTX 4090)

### ВАРИАНТ Б: Пошаговый запуск

```bash
# Активируйте venv
source venv/bin/activate

# 1. Preprocessing
python scripts/01_preprocess_data.py

# 2. Build indices (займет ~30-35 мин на RTX 4090)
python scripts/02_build_indices.py

# 3. Run search and generate submission
python scripts/03_run_search.py

# Output: submit_alpha_rag_elite.csv
```

### Пошаговый запуск

#### Step 1: Preprocessing

```bash
python scripts/01_preprocess_data.py
```

**Output:**
- `data/processed/documents_normalized.parquet` – нормализованные документы
- `data/processed/chunks.parquet` – чанки (250 chars, overlap 60)
- `data/processed/questions_processed.parquet` – обработанные вопросы

**Время:** ~2-3 мин

#### Step 2: Build Indices

```bash
python scripts/02_build_indices.py [--models MODEL1,MODEL2,...] [--no-cache]
```

**Опции:**
- `--models`: Выбор моделей (по умолчанию: все включенные в config)
- `--no-cache`: Пересчитать эмбеддинги (по умолчанию: использует кеш)
- `--doc-only`: Только doc-level (без chunk-level, быстрее)
- `--chunk-only`: Только chunk-level

**Output:**
- `data/embeddings/` – эмбеддинги
- `data/indices/` – FAISS индексы
- `data/cache/` – BM25 и метаданные

**Время:** ~30-35 мин (полный), ~20 мин (doc-only)

#### Step 3: Run Search

```bash
python scripts/03_run_search.py [--enable-reranking] [--top-k 5]
```

**Опции:**
- `--enable-reranking`: Включить reranking (по умолчанию: выключен)
- `--top-k N`: Количество документов в выдаче (default: 5)
- `--batch-size N`: Batch size для embedding queries (default: 32)

**Output:**
- `submit_alpha_rag_elite.csv` – финальный сабмит

**Время:** ~3-5 мин (без reranking), ~8-12 мин (с reranking)

---

## Конфигурация

Все гиперпараметры в `config/settings.py`:

### Модели

```python
# Включение/выключение моделей
DOC_EMBEDDERS = {
    "e5_large": {"enabled": True, "weight": 2.5},
    "bge_m3": {"enabled": True, "weight": 2.5},
    "sbert_ru": {"enabled": True, "weight": 2.0},
    "labse": {"enabled": True, "weight": 1.5},
}
```

### RRF и Boosts

```python
RRF_K = 60

ENTITY_BOOSTS = {
    "bik": 3.0,  # Точное совпадение БИК
    "rs": 2.0,   # р/с
    ...
}

INTENT_BOOSTS = {
    "БИК": 0.8,
    "РАСЧЕТНЫЙ_СЧЕТ": 0.8,
    ...
}
```

### Query Expansion

```python
ENABLE_RULE_BASED_EXPANSION = True
ENABLE_BM25_PRF = True
PRF_TOP_DOCS = 20
PRF_TOP_TERMS = 15
```

### Reranking (опционально)

```python
ENABLE_RERANKING = False  # Включать только если validated!
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_K = 50
```

---

## Валидация и тюнинг

### Создание validation set

```bash
# Вручную разметить 100-200 вопросов
# data/validation/validation_set.csv
# Format: q_id, correct_web_ids (comma-separated)
```

### Локальная валидация

```bash
python scripts/validate.py --validation-set data/validation/validation_set.csv
```

**Output:**
- Hit@5, Hit@1, MRR
- Per-component metrics
- Error analysis

### Grid search гиперпараметров

```bash
python scripts/tune_hyperparameters.py \
    --param rrf_k --values 40,60,80 \
    --param entity_boost_bik --values 2.0,3.0,4.0
```

---

## Troubleshooting

### CUDA Out of Memory

**Проблема:** `RuntimeError: CUDA out of memory`

**Решения:**
1. Уменьшить `BATCH_SIZE_EMBEDDING` в `config/settings.py`
2. Выключить некоторые модели (оставить E5 + BGE)
3. Использовать CPU для некоторых моделей

### FAISS Index Error

**Проблема:** `AssertionError: index.ntotal != len(documents)`

**Решение:** Удалить кеш и пересобрать:
```bash
rm -rf data/embeddings/* data/indices/* data/cache/*
python scripts/02_build_indices.py --no-cache
```

### BGE-m3 не загружается

**Проблема:** `OSError: BAAI/bge-m3 does not appear to exist`

**Решение:** Выключить BGE-m3 в `config/settings.py`:
```python
DOC_EMBEDDERS = {
    "bge_m3": {"enabled": False, ...},
}
```

Или использовать fallback модель (настроено автоматически).

---

## Структура проекта

```
alpha_hack_v2/
├── config/
│   └── settings.py          # Все гиперпараметры
├── data/
│   ├── raw/                 # Исходные данные
│   ├── processed/           # Обработанные данные
│   ├── embeddings/          # Эмбеддинги
│   ├── indices/             # FAISS индексы
│   └── cache/               # BM25, metadata
├── src/
│   ├── preprocessing/       # Нормализация, entities, intents
│   ├── query/               # Query expansion
│   ├── indexing/            # Embedders, FAISS
│   ├── retrieval/           # Dense + sparse retrieval
│   ├── scoring/             # RRF, domain boosts
│   ├── reranking/           # Optional cross-encoder
│   └── pipeline/            # End-to-end pipeline
├── scripts/
│   ├── 01_preprocess_data.py
│   ├── 02_build_indices.py
│   ├── 03_run_search.py
│   └── validate.py
└── requirements.txt
```

---



## Авторы
- Gleb(DS/backend)
- Fedos (backend)
- Egor (backend)
- Mark(backend)
- Matvey (ML)


**Хакатон:** Альфа-Банк RAG Challenge
