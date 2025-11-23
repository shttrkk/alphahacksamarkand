#!/bin/bash
#
# Quick test of LLM generation on small sample (100 queries)
# For testing prompts and validating quality before full run
#

set -e

echo "========================================="
echo "LLM Query Variants - QUICK TEST"
echo "========================================="
echo ""
echo "This will generate variants for FIRST 100 queries only."
echo "Use this to test prompt quality before full 2-3 hour run."
echo ""

# Check CUDA
if ! nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found. GPU required!"
    exit 1
fi

echo "✓ GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Activate environment if needed
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if input file exists
FULL_INPUT="data/raw/questions_clean.csv"
if [ ! -f "$FULL_INPUT" ]; then
    echo "❌ ERROR: Input file not found: $FULL_INPUT"
    echo "   Make sure you're in the project root directory."
    exit 1
fi

# Create test input (first 101 lines: 1 header + 100 queries)
TEST_INPUT="data/processed/questions_test_100.csv"
mkdir -p data/processed
head -n 101 "$FULL_INPUT" > "$TEST_INPUT"

TEST_OUTPUT="data/processed/query_variants_test.parquet"

echo "Configuration:"
echo "  Input: $TEST_INPUT (100 queries)"
echo "  Output: $TEST_OUTPUT"
echo "  Columns: q_id, query"
echo "  Expected time: ~2-3 minutes"
echo ""

# Run generation (no resume, force fresh)
python tools/generate_query_variants.py \
    --input "$TEST_INPUT" \
    --output "$TEST_OUTPUT" \
    --query_column query \
    --num_variants 2 \
    --batch_size 8 \
    --max_new_tokens 80 \
    --temperature 0.7 \
    --no_resume \
    --show_samples 20

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ TEST SUCCESS!"
    echo "========================================="
    echo ""
    echo "Review the samples above."
    echo ""
    echo "If quality looks good, run full generation:"
    echo "  ./run_llm_generation.sh"
    echo ""
    echo "If quality is poor:"
    echo "  1. Adjust prompt in tools/generate_query_variants.py"
    echo "  2. Run this test again: ./test_llm_generation.sh"
    echo ""

    # Clean up test file
    rm -f "$TEST_INPUT"
else
    echo ""
    echo "❌ Test failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi
