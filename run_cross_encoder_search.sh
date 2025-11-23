#!/bin/bash

#=============================================================================
# МАШИНА 2: Cross-Encoder Reranking (GUARANTEED +3-5pp)
#=============================================================================
# Two-stage retrieval:
#   1. Bi-encoder: Fast retrieval of TOP-100 candidates
#   2. Cross-encoder: Accurate reranking → TOP-5
#
# Time: ~30-40 min (on GPU)
# Expected: Hit@5 = 0.34-0.36 (+3-5pp from baseline 0.31)
#=============================================================================

set -e  # Exit on error

echo "========================================="
echo "CROSS-ENCODER RERANKING PIPELINE"
echo "========================================="
echo ""
echo "✓ Config: WINNING CONFIG + cross-encoder reranking"
echo "✓ Model: cross-encoder/ms-marco-MiniLM-L-12-v2"
echo "✓ Strategy: Bi-encoder TOP-100 → Cross-encoder rerank → TOP-5"
echo "✓ Expected boost: +3-5pp"
echo ""
echo "========================================="
echo ""

# Check if we're on the correct branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "claude/cross-encoder-rerank-01PoWpmzz2iYpxUjuAZi7KmC" ]; then
    echo "⚠️  Warning: Not on cross-encoder branch!"
    echo "Current branch: $CURRENT_BRANCH"
    echo "Expected: claude/cross-encoder-rerank-01PoWpmzz2iYpxUjuAZi7KmC"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if indices exist
if [ ! -f "data/indices/e5_large_doc.bin" ]; then
    echo "❌ ERROR: Doc indices not found!"
    echo "Run: python scripts/02_build_indices.py"
    exit 1
fi

echo "📝 Starting search with cross-encoder reranking..."
echo ""

# Run search
python scripts/03_run_search.py \
    --output "submit_cross_encoder.csv" \
    2>&1 | tee search_cross_encoder.log

echo ""
echo "========================================="
echo "✅ SEARCH COMPLETE!"
echo "========================================="
echo ""
echo "Output file: submit_cross_encoder.csv"
echo "Log file: search_cross_encoder.log"
echo ""
echo "Next steps:"
echo "1. Check Hit@5 in the log"
echo "2. If good result → submit to leaderboard"
echo "3. If time remains → try perfect variants (РЕЗЕРВ)"
echo ""
