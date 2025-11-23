# Quick Start Guide - Alpha RAG Elite

**Время до результата:** ~40-50 минут (на RTX 4090)

---

## Шаг 0: Подготовка сервера

### Аренда GPU на vast.ai

1. Зайти на https://vast.ai
2. Фильтры:
   - GPU: RTX 4090 или RTX 3090
   - VRAM: >= 24GB
   - RAM: >= 32GB
   - Disk: >= 200GB
   - Цена: ~$0.5-1.0/hour

3. Выбрать предложение, запустить instance
4. SSH подключение (команда в vast.ai dashboard)

### Установка окружения

```bash
# После подключения по SSH
cd /workspace  # или любая директория с достаточным местом

# Клонируем репо
git clone https://github.com/azama2t/alpha_hack_v2.git
cd alpha_hack_v2
git checkout claude/enhance-rag-retrieval-01PoWpmzz2iYpxUjuAZi7KmC

# АВТОМАТИЧЕСКАЯ УСТАНОВКА (рекомендуется)
chmod +x setup.sh
./setup.sh

# После установки ОБЯЗАТЕЛЬНО активируйте venv:
source venv/bin/activate

# Проверка GPU для PyTorch
python -c "import torch; print(torch.cuda.is_available())"
# Должно вывести: True (если есть GPU)
```

**Важно:** После завершения `setup.sh` всегда активируйте venv командой:
```bash
source venv/bin/activate
```

---

## Шаг 1: Подготовка данных

```bash
# Создать data/raw/
mkdir -p data/raw

# Загрузить файлы (scp/wget/etc)
# Положить в data/raw/:
#   - websites_updated.csv
#   - questions_clean.csv
```

**Проверка:**
```bash
ls -lh data/raw/
# Должны быть оба файла
```

---

## Шаг 2-4: Запуск пайплайна

### ВАРИАНТ A: Автоматический запуск (рекомендуется)

```bash
# ВАЖНО: сначала активируйте venv!
source venv/bin/activate

# Запустить весь пайплайн одной командой
chmod +x run_pipeline.sh
./run_pipeline.sh
```

**Что делает скрипт:**
1. Проверяет venv и активирует его
2. Проверяет наличие данных
3. Запускает все 3 этапа по очереди:
   - Preprocessing (~2-3 мин)
   - Building indices (~30-35 мин)
   - Search (~3-5 мин)
4. Проверяет результат

**Время:** ~40-50 минут (весь пайплайн)

**Результат:** `submit_alpha_rag_elite.csv`

---

### ВАРИАНТ Б: Пошаговый запуск

Если хотите контролировать каждый этап:

**ВАЖНО:** Всегда убедитесь, что venv активирован:
```bash
source venv/bin/activate
```

#### Шаг 2: Preprocessing

```bash
python scripts/01_preprocess_data.py
```

**Время:** ~2-3 минуты

**Output:**
- `data/processed/documents_normalized.parquet`
- `data/processed/chunks.parquet`
- `data/processed/questions_processed.parquet`

**Проверка:**
```bash
ls -lh data/processed/
# Должны быть все 3 файла
```

---

#### Шаг 3: Build Indices

```bash
# Убедитесь что venv активирован!
source venv/bin/activate

# Полный запуск (все модели)
python scripts/02_build_indices.py
```

**Время:** ~30-35 минут на RTX 4090

**Если нужно быстрее (только doc-level):**
```bash
python scripts/02_build_indices.py --doc-only
# Время: ~20 минут
```

**Output:**
- `data/embeddings/doc/` - эмбеддинги документов
- `data/embeddings/chunk/` - эмбеддинги чанков
- `data/indices/` - FAISS индексы
- `data/cache/` - BM25 индексы

**Проверка:**
```bash
ls -lh data/embeddings/doc/
ls -lh data/indices/
# Должны быть файлы для каждой модели
```

**Мониторинг GPU:**
```bash
# В отдельном терминале
watch -n 1 nvidia-smi
```

---

#### Шаг 4: Run Search

```bash
# Убедитесь что venv активирован!
source venv/bin/activate

python scripts/03_run_search.py
```

**Время:** ~3-5 минут

**Output:**
- `submit_alpha_rag_elite.csv`

**Проверка:**
```bash
head submit_alpha_rag_elite.csv
# Должны быть колонки: q_id, web_list
```

---

## Шаг 5: Сабмит на лидерборд

1. Скачать файл `submit_alpha_rag_elite.csv` с сервера
2. Загрузить на платформу хакатона
3. Ждать результата

**Ожидаемый Hit@5:** 58-68%

---

## Troubleshooting

### OOM (Out of Memory)

**Симптом:** `RuntimeError: CUDA out of memory`

**Решение 1:** Уменьшить batch size
```python
# В config/settings.py
BATCH_SIZE_EMBEDDING = 32  # было 64
```

**Решение 2:** Отключить некоторые модели
```python
# В config/settings.py
DOC_EMBEDDERS = {
    "e5_large": {"enabled": True, ...},
    "bge_m3": {"enabled": False, ...},  # Отключить
    "sbert_ru": {"enabled": True, ...},
    "labse": {"enabled": False, ...},   # Отключить
}
```

Затем пересобрать:
```bash
python scripts/02_build_indices.py --no-cache
```

### BGE-m3 не загружается

**Симптом:** `OSError: BAAI/bge-m3 does not appear to exist`

**Решение:** Код автоматически использует fallback на mpnet. Проверьте логи.

Или явно отключите:
```python
DOC_EMBEDDERS = {
    "bge_m3": {"enabled": False, ...},
}
```

### Медленная индексация

**Причина:** Медленный диск или CPU bottleneck

**Решение:**
1. Проверить тип диска:
```bash
df -Th
# Должен быть NVMe SSD
```

2. Уменьшить количество моделей (см. OOM решение 2)

### Empty submission

**Симптом:** В `submit_alpha_rag_elite.csv` пустые списки

**Причина:** Индексы не загрузились

**Решение:** Проверить логи скрипта 03:
```bash
python scripts/03_run_search.py 2>&1 | tee search.log
```

Должно быть:
```
Loading doc-level indices...
  ✓ e5_large: XXXX vectors
  ✓ bge_m3: XXXX vectors
  ...
```

Если нет - пересобрать индексы.

---

## Advanced: Включение Reranking

**ТОЛЬКО если есть время на валидацию!**

### 1. Создать validation set

```bash
# Вручную разметить 50-100 вопросов
# data/validation/validation_set.csv
# Format: q_id,correct_web_ids
# Example:
# 1,"42,105,234"
# 2,"67,89"
```

### 2. Запустить валидацию

```bash
python scripts/validate.py --validation-set data/validation/validation_set.csv
```

### 3. Если прирост > +1 п.п. - включить

```python
# В config/settings.py
ENABLE_RERANKING = True
```

```bash
# Перезапустить поиск
python scripts/03_run_search.py --enable-reranking
```

**Время:** +5-8 минут

---

## Время выполнения (summary)

| Шаг | Действие | Время (4090) |
|-----|----------|--------------|
| 0 | Setup окружения | 5-10 мин |
| 1 | Preprocessing | 2-3 мин |
| 2 | Build indices | 30-35 мин |
| 3 | Run search | 3-5 мин |
| 4 | Download & submit | 2 мин |
| **TOTAL** | | **~45-55 мин** |

---

## Чеклист перед сабмитом

- [ ] Все 3 скрипта выполнены без ошибок
- [ ] Файл `submit_alpha_rag_elite.csv` существует
- [ ] Файл содержит правильное количество строк (= количеству вопросов)
- [ ] Формат web_list корректный: `[1, 2, 3, 4, 5]`
- [ ] Нет пустых списков
- [ ] Все web_id валидные (существуют в websites_updated.csv)

**Проверка формата:**
```bash
python -c "
import pandas as pd
df = pd.read_csv('submit_alpha_rag_elite.csv')
print(f'Rows: {len(df)}')
print(f'Sample:')
print(df.head())
print(f'\\nAll web_lists valid: {all(df.web_list.str.startswith(\"[\"))}')
"
```

---

## Контакты и поддержка

**Issues:** https://github.com/azama2t/alpha_hack_v2/issues

**Хакатон:** Альфа-Банк RAG Challenge
**Deadline:** 16 ноября 2024
