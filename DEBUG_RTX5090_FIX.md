# RTX 5090 Debug Fix - Summary

## Проблема

1. **RTX 5090 CUDA ошибки**: PyTorch с CUDA 11.8 не поддерживает compute capability 8.9 (RTX 5090)
2. **Тихие скипы моделей**: Все dense модели скипались без ошибок, оставался только BM25
3. **Нет логирования**: Невозможно понять что пошло не так
4. **Скрипты работали слишком быстро**: 01 за 1 сек, 02 за 2 мин вместо 20-25 мин

## Что исправлено

### 1. setup.sh - RTX 5090 Support
- Автоопределение GPU архитектуры
- Установка PyTorch с CUDA 12.1 для RTX 40xx/50xx
- Проверка compute capability после установки
- Создание директории logs/

### 2. Централизованное логирование (src/utils/logger.py)
- Логи одновременно в файл и консоль
- Цветной вывод с уровнями (INFO, WARNING, ERROR)
- Утилиты для форматирования времени и секций

### 3. 01_preprocess_data.py - Детальное логирование
- **Лог-файл**: `logs/preprocess.log`
- Проверка входных файлов с понятными ошибками
- Примеры обработанных данных
- Детальная статистика и время выполнения
- Полные stacktrace при ошибках

### 4. 02_build_indices.py - CUDA Fallback + Strict Mode
- **Лог-файл**: `logs/build_indices.log`
- **CUDA Error Detection**: Автоматически детектирует CUDA ошибки
- **CPU Fallback**: Автоматический переход на CPU при CUDA проблемах
- **Device Info**: Показывает GPU name, compute capability, CUDA version
- **Strict Mode**: Фейлится если НИ ОДНА dense модель не построена
- **Tracking**: Подсчет успешных/failed моделей
- **Детальные логи**: Для каждой модели - время, устройство, размер индекса

## Что теперь делать

### Шаг 1: Переустановка окружения

```bash
# Активируйте venv если еще не активирован
source venv/bin/activate

# Переустановите PyTorch с правильной CUDA версией
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Проверьте что CUDA работает
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Compute capability: {props.major}.{props.minor}')
    print(f'CUDA version: {torch.version.cuda}')
"
```

**Ожидаемый вывод**:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 5090
Compute capability: 8.9
CUDA version: 12.1
```

### Шаг 2: Очистка старых индексов

```bash
# Удалите старые индексы (они построены неправильно)
rm -rf data/embeddings/
rm -rf data/indices/
rm -rf data/cache/

# Логи можно оставить или удалить
# rm -rf logs/
```

### Шаг 3: Запуск пайплайна ЗАНОВО

```bash
# Активируйте venv
source venv/bin/activate

# 1. Preprocessing (должен занять 2-3 минуты)
python scripts/01_preprocess_data.py

# Проверьте лог:
tail -50 logs/preprocess.log

# 2. Build indices (должен занять 20-25 минут на RTX 5090)
python scripts/02_build_indices.py

# Мониторинг в реальном времени:
tail -f logs/build_indices.log

# 3. После успешного завершения - запуск search
python scripts/03_run_search.py
```

## Что проверить в логах

### logs/preprocess.log
Должно быть:
```
✓ Documents: 1937
✓ Chunks: 65741
⏱ Time: 2m 15s (примерно)
```

### logs/build_indices.log
Должно быть:
```
System info:
  device: NVIDIA GeForce RTX 5090
  compute_capability: 8.9
  vram_gb: 24.0
  cuda_version: 12.1

Building e5_large (doc-level)...
  Device: cuda
  ✓ Model loaded on cuda
  ✓ Embeddings shape: (1937, 1024)
  ⏱ Time: 3m 45s

Doc-level models:
  ✓ Successful: 3
    - e5_large
    - sbert_ru
    - mpnet
```

## Если всё равно CUDA ошибки

Если даже после переустановки PyTorch с cu121 модели всё равно не грузятся на GPU:

1. Скрипт **автоматически** перейдет на CPU
2. В логах будет:
   ```
   ⚠️  CUDA ERROR detected!
   Пытаемся CPU fallback...
   Device: cpu
   ✓ Model loaded on cpu
   ```
3. Это будет медленнее (~2-3 часа вместо 20-25 мин), но **СРАБОТАЕТ**

## Strict Mode

Если ни одна dense модель не построилась, скрипт зафейлится с:

```
❌ FATAL: NO DENSE MODELS SUCCEEDED!

Все dense retriever'ы не смогли построиться.
Это означает, что search будет работать ТОЛЬКО на BM25,
что даст очень плохое качество (~5-10% Hit@5).

Возможные причины:
  1. CUDA несовместима с вашей GPU (RTX 5090 требует CUDA 12.1+)
  2. PyTorch установлен неправильно
  3. Недостаточно VRAM
  4. Модели не смогли загрузиться даже на CPU

Решение:
  1. Переустановите PyTorch с правильной версией CUDA:
     pip install torch --index-url https://download.pytorch.org/whl/cu121
  2. Или запустите ./setup.sh заново
  3. Проверьте логи выше для деталей по каждой модели
```

Это **защита от тихих скипов** - теперь невозможно не заметить проблему!

## Что дальше

После того как 02_build_indices.py успешно отработает:
- Должно быть минимум 2-3 успешных dense модели
- Индексы сохранены в data/embeddings/ и data/indices/
- Можно запускать 03_run_search.py

Если есть вопросы - проверьте логи в logs/ директории, там будет вся нужная информация!
