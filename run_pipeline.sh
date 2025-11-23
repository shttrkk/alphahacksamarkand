#!/bin/bash
set -e  # Exit on error

# ================================================================
# Alpha RAG Elite - Pipeline Runner
# Автоматический запуск всего пайплайна с активацией venv
# ================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Start time
PIPELINE_START=$(date +%s)

echo "=========================================="
echo "Alpha RAG Elite - Pipeline"
echo "=========================================="
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Создаем директорию для логов
mkdir -p logs

# ================================================================
# Проверка venv
# ================================================================
if [ ! -d "venv" ]; then
    echo -e "${RED}ERROR: venv не найден!${NC}"
    echo "Сначала запустите: ./setup.sh"
    exit 1
fi

# Активация venv
echo "Активация виртуального окружения..."
source venv/bin/activate
echo -e "${GREEN}✓ venv активирован${NC}"
echo ""

# Проверка данных
echo "Проверка наличия данных..."
if [ ! -f "data/raw/websites_updated.csv" ] || [ ! -f "data/raw/questions_clean.csv" ]; then
    echo -e "${RED}ERROR: Данные не найдены в data/raw/${NC}"
    echo "Положите файлы:"
    echo "  - data/raw/websites_updated.csv"
    echo "  - data/raw/questions_clean.csv"
    exit 1
fi
echo -e "${GREEN}✓ Данные найдены${NC}"
echo ""

# ================================================================
# Запуск пайплайна
# ================================================================

echo "=========================================="
echo -e "${BLUE}[ 1/3 ] Preprocessing...${NC}"
echo "=========================================="
python scripts/01_preprocess_data.py
echo ""

echo "=========================================="
echo -e "${BLUE}[ 2/3 ] Building Indices...${NC}"
echo "=========================================="
python scripts/02_build_indices.py
echo ""

echo "=========================================="
echo -e "${BLUE}[ 3/3 ] Running Search...${NC}"
echo "=========================================="
python scripts/03_run_search.py
echo ""

# ================================================================
# Готово
# ================================================================
PIPELINE_END=$(date +%s)
PIPELINE_DURATION=$((PIPELINE_END - PIPELINE_START))

# Форматируем время
HOURS=$((PIPELINE_DURATION / 3600))
MINUTES=$(((PIPELINE_DURATION % 3600) / 60))
SECONDS=$((PIPELINE_DURATION % 60))

if [ $HOURS -gt 0 ]; then
    TIME_STR="${HOURS}h ${MINUTES}m ${SECONDS}s"
elif [ $MINUTES -gt 0 ]; then
    TIME_STR="${MINUTES}m ${SECONDS}s"
else
    TIME_STR="${SECONDS}s"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}Пайплайн завершен успешно!${NC}"
echo "=========================================="
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total time: $TIME_STR"
echo ""

# Проверка результата
if [ -f "submit_alpha_rag_elite.csv" ]; then
    LINES=$(wc -l < submit_alpha_rag_elite.csv)
    echo "✓ Результат: submit_alpha_rag_elite.csv"
    echo "  Строк: $LINES"
    echo -e "  ${GREEN}Готово к отправке!${NC}"
else
    echo -e "${RED}❌ WARNING: Файл результата не найден${NC}"
fi

echo ""
echo "Логи сохранены:"
echo "  - logs/preprocess.log"
echo "  - logs/build_indices.log"
echo "  - logs/run_search.log (если search запускался)"
echo ""
echo "Для проверки логов:"
echo "  tail -50 logs/build_indices.log"
echo ""
