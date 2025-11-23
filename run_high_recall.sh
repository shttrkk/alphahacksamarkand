#!/bin/bash

#=============================================================================
# HIGH RECALL Configuration
#=============================================================================
# Strategy: Maximize recall through:
#   1. More embedders (e5, sbert_ru, mpnet, LaBSE)
#   2. More candidates per source (TOP_K increased)
#   3. Higher BM25 weights
#   4. Better RRF fusion (K=70)
#
# Expected: Higher recall → more relevant docs in TOP-N
# Time: ~35-40 min (slightly slower due to LaBSE)
#
# COMBO with Cross-Encoder (МАШИНА 2):
#   High Recall finds all relevant docs → Cross-Encoder ranks them perfectly
#=============================================================================

set -e

echo "========================================="
echo "HIGH RECALL CONFIGURATION"
echo "========================================="
echo ""
echo "✓ Embedders: 4 (e5, sbert_ru, mpnet, LaBSE)"
echo "✓ TOP_K increased: 200/250 per source"
echo "✓ BM25 weights increased: 2.5/2.0"
echo "✓ RRF_K: 70 (better fusion)"
echo ""
echo "Strategy: Maximum RECALL"
echo "Expected: Find ALL relevant docs in TOP-N"
echo "Time: ~35-40 min"
echo ""
echo "========================================="
echo ""

# Check if LaBSE indices exist
if [ ! -f "data/indices/labse_doc.bin" ]; then
    echo "⚠️  WARNING: LaBSE indices not found!"
    echo "Building indices first (this will take ~5-10 min extra)..."
    echo ""
    python scripts/02_build_indices.py
    echo ""
fi

# Run search
echo "🚀 Starting HIGH RECALL search..."
echo ""

python scripts/03_run_search.py \
    --output submit_HIGH_RECALL.csv \
    2>&1 | tee search_high_recall.log

echo ""
echo "========================================="
echo "✅ HIGH RECALL SEARCH COMPLETE!"
echo "========================================="
echo ""
echo "Output: submit_HIGH_RECALL.csv"
echo "Log: search_high_recall.log"
echo ""
echo "Next: COMBO with Cross-Encoder from МАШИНА 2!"
echo ""
