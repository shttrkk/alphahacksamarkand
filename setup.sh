#!/bin/bash
set -e  # Exit on error

# ================================================================
# Alpha RAG Elite - Quick Setup Script
# Автоматическая установка всего окружения
# ================================================================

echo "=========================================="
echo "Alpha RAG Elite - Quick Setup"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ================================================================
# 1. Проверка Python версии
# ================================================================
echo "[ 1/7 ] Проверка Python версии..."

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: python3 не найден${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}ERROR: Требуется Python >= 3.10, найден $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# ================================================================
# 2. Создание виртуального окружения
# ================================================================
echo ""
echo "[ 2/7 ] Создание виртуального окружения..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ venv создан${NC}"
else
    echo -e "${YELLOW}! venv уже существует, используем существующий${NC}"
fi

# Активация venv
source venv/bin/activate

# ================================================================
# 3. Обновление pip
# ================================================================
echo ""
echo "[ 3/7 ] Обновление pip..."
pip install --upgrade pip -q
echo -e "${GREEN}✓ pip обновлен${NC}"

# ================================================================
# 4. Установка PyTorch
# ================================================================
echo ""
echo "[ 4/7 ] Установка PyTorch..."

# Проверка наличия CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "  Найдена NVIDIA GPU, проверяем архитектуру..."

    # Проверяем версию CUDA и архитектуру GPU
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    echo "  GPU: $GPU_NAME"

    # RTX 40xx и 50xx серии требуют CUDA 12.1+ (compute capability 8.9)
    # Используем cu121 для максимальной совместимости с новыми картами
    if [[ "$GPU_NAME" == *"RTX 40"* ]] || [[ "$GPU_NAME" == *"RTX 50"* ]]; then
        echo "  Обнаружена новая архитектура (Ada Lovelace или новее)"
        echo "  Устанавливаем PyTorch с CUDA 12.1..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
    else
        echo "  Устанавливаем PyTorch с CUDA 11.8 (стандартная конфигурация)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
    fi
else
    echo "  GPU не найдена, устанавливаем CPU версию PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
fi

echo -e "${GREEN}✓ PyTorch установлен${NC}"

# ================================================================
# 5. Установка FAISS
# ================================================================
echo ""
echo "[ 5/7 ] Установка FAISS..."

# ВАЖНО: Используем faiss-cpu для стабильности
# faiss-gpu часто имеет проблемы совместимости с разными версиями CUDA
# PyTorch всё равно может использовать GPU для embeddings
echo "  Устанавливаем faiss-cpu (стабильная версия)..."
pip install faiss-cpu -q

echo -e "${GREEN}✓ FAISS установлен${NC}"

# ================================================================
# 6. Установка остальных зависимостей
# ================================================================
echo ""
echo "[ 6/7 ] Установка остальных зависимостей..."

pip install -r requirements.txt -q

echo -e "${GREEN}✓ Все зависимости установлены${NC}"

# ================================================================
# 7. Создание необходимых директорий
# ================================================================
echo ""
echo "[ 7/7 ] Создание директорий..."

mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/embeddings/doc
mkdir -p data/embeddings/chunk
mkdir -p data/indices
mkdir -p data/cache
mkdir -p data/validation
mkdir -p logs

echo -e "${GREEN}✓ Директории созданы${NC}"

# ================================================================
# Проверка установки
# ================================================================
echo ""
echo "=========================================="
echo "Проверка установки..."
echo "=========================================="

# Проверка CUDA и PyTorch
python -c "
import torch
import sys

print(f'✓ Python: {sys.version.split()[0]}')
print(f'✓ PyTorch: {torch.__version__}')

cuda_available = torch.cuda.is_available()
if cuda_available:
    print('✓ CUDA доступна')
    device_props = torch.cuda.get_device_properties(0)
    print(f'  Устройство: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {device_props.total_memory / 1024**3:.1f} GB')
    print(f'  Compute Capability: {device_props.major}.{device_props.minor}')
    print(f'  CUDA Version: {torch.version.cuda}')

    # ВАЖНО: Проверяем совместимость для RTX 40xx/50xx
    if device_props.major == 8 and device_props.minor == 9:
        print('  ⚠️  RTX 40xx/50xx обнаружена (sm_89) - требуется CUDA 12.1+')
        if torch.version.cuda and torch.version.cuda >= '12.1':
            print('  ✓ CUDA 12.1+ установлена - совместимость OK')
        else:
            print('  ❌ CUDA версия слишком старая! Переустановите PyTorch с cu121')
            sys.exit(1)
else:
    print('! CUDA недоступна, будет использоваться CPU')
    print('  ⚠️  Производительность будет значительно ниже')
"

# Проверка FAISS
python -c "
import faiss
print('✓ FAISS работает')
"

# Проверка sentence-transformers
python -c "
from sentence_transformers import SentenceTransformer
print('✓ sentence-transformers работает')
"

echo ""
echo "=========================================="
echo -e "${GREEN}Установка завершена успешно!${NC}"
echo "=========================================="
echo ""
echo -e "${YELLOW}ВАЖНО: Активируйте виртуальное окружение!${NC}"
echo ""
echo "  source venv/bin/activate"
echo ""
echo "=========================================="
echo ""
echo "Следующие шаги:"
echo ""
echo "1. Положите данные в data/raw/:"
echo "   - websites_updated.csv"
echo "   - questions_clean.csv"
echo ""
echo "2. Запустите пайплайн:"
echo "   python scripts/01_preprocess_data.py"
echo "   python scripts/02_build_indices.py"
echo "   python scripts/03_run_search.py"
echo ""
echo "3. Забрать результат: submit_alpha_rag_elite.csv"
echo ""
echo "Время выполнения:"
echo "  - Preprocessing: ~2-3 мин"
echo "  - Indexing: ~20-25 мин (3 модели)"
echo "  - Search: ~3-5 мин"
echo "  Total: ~30 минут"
echo ""
echo "Для помощи: см. README.md или QUICKSTART.md"
echo ""
