#!/bin/bash
#
# EGOR'S KILLER FEATURE: LLM Query Variants Generation
#
# This script generates semantic paraphrases of user queries to improve retrieval recall.
# Uses Qwen2.5-7B-Instruct with 4-bit quantization on RTX 4090.
#
# Expected time: ~2-3 hours for 7000 queries
# GPU required: RTX 4090 (or similar with 24GB+ VRAM)
#

set -e  # Exit on error

echo "========================================="
echo "LLM Query Variants Generation (FULL)"
echo "========================================="
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
INPUT_FILE="data/raw/questions_clean.csv"
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ ERROR: Input file not found: $INPUT_FILE"
    echo "   Run preprocessing first: python scripts/01_preprocess_data.py"
    exit 1
fi

OUTPUT_FILE="data/processed/query_variants.parquet"
mkdir -p data/processed

# Count queries
QUERY_COUNT=$(tail -n +2 "$INPUT_FILE" | wc -l)

echo "Configuration:"
echo "  Input: $INPUT_FILE"
echo "  Output: $OUTPUT_FILE"
echo "  Columns: q_id, query"
echo "  Total queries: $QUERY_COUNT"
echo "  Model: Qwen/Qwen2.5-7B-Instruct (4-bit quantization)"
echo "  Variants per query: 2"
echo "  Batch size: 8"
echo ""

# Check if output already exists
if [ -f "$OUTPUT_FILE" ]; then
    echo "⚠️  WARNING: Output file already exists!"
    echo "   File: $OUTPUT_FILE"
    read -p "   Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    echo ""
fi

echo "Starting LLM generation..."
echo "This will take ~2-3 hours for full dataset."
echo ""
echo "Progress monitoring:"
echo "  - Progress bar will show in this terminal"
echo "  - GPU usage: watch -n 5 nvidia-smi"
echo "  - Checkpoint file: ls -lh data/processed/query_variants_checkpoint.parquet"
echo ""
echo "Press Ctrl+C to stop (progress will be saved in checkpoint)."
echo "You can resume later by running this script again."
echo ""

# Run generation
python tools/generate_query_variants.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --query_column query \
    --num_variants 2 \
    --batch_size 8 \
    --max_new_tokens 80 \
    --temperature 0.7 \
    --checkpoint_every 500 \
    --show_samples 20

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ SUCCESS!"
    echo "========================================="
    echo ""
    echo "Query variants generated: $OUTPUT_FILE"

    # Show file size
    FILE_SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
    echo "File size: $FILE_SIZE"
    echo ""

    echo "Next steps:"
    echo "  1. Review samples above for quality"
    echo "  2. Run search with variants:"
    echo "     python scripts/03_run_search.py"
    echo ""
    echo "Configuration is already set in config/settings.py:"
    echo "  USE_QUERY_VARIANTS = True"
    echo "  ORIGINAL_QUERY_WEIGHT = 1.0"
    echo "  VARIANT_QUERY_WEIGHT = 0.7"
    echo ""
else
    echo ""
    echo "❌ Generation failed with exit code: $EXIT_CODE"
    echo ""
    echo "Checkpoint may be available for resume:"
    ls -lh data/processed/query_variants_checkpoint.parquet 2>/dev/null || true
    echo ""
    echo "To resume, just run this script again."
    exit $EXIT_CODE
fi
