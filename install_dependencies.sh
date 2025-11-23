#!/bin/bash
#
# Install all dependencies for Egor's Killer Feature
# RTX 4090 optimized setup with CUDA support
#

set -e

echo "========================================="
echo "Installing Dependencies"
echo "========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"
echo ""

# Update pip
echo "Updating pip..."
pip install --upgrade pip
echo ""

echo "Installing PyTorch with CUDA support..."
echo "This may take a few minutes..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo ""

echo "Installing core dependencies..."
pip install transformers bitsandbytes accelerate
echo ""

echo "Installing NLP and vector search libraries..."
pip install sentence-transformers
echo ""

echo "Installing data processing libraries..."
pip install pandas pyarrow tqdm
echo ""

echo "Installing retrieval libraries..."
pip install rank-bm25
echo ""

echo "Installing FAISS (CPU version with CUDA support)..."
# faiss-cpu works with CUDA if PyTorch has CUDA
pip install faiss-cpu
echo ""

echo "========================================="
echo "Verifying installation..."
echo "========================================="
echo ""

# Test imports
python3 << 'EOF'
import sys
print("Testing imports...")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ PyTorch: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
except Exception as e:
    print(f"✗ Transformers: {e}")
    sys.exit(1)

try:
    import bitsandbytes
    print(f"✓ BitsAndBytes (4-bit quantization)")
except Exception as e:
    print(f"✗ BitsAndBytes: {e}")
    sys.exit(1)

try:
    import accelerate
    print(f"✓ Accelerate")
except Exception as e:
    print(f"✗ Accelerate: {e}")
    sys.exit(1)

try:
    import sentence_transformers
    print(f"✓ Sentence Transformers {sentence_transformers.__version__}")
except Exception as e:
    print(f"✗ Sentence Transformers: {e}")
    sys.exit(1)

try:
    import pandas
    print(f"✓ Pandas {pandas.__version__}")
except Exception as e:
    print(f"✗ Pandas: {e}")
    sys.exit(1)

try:
    import pyarrow
    print(f"✓ PyArrow {pyarrow.__version__}")
except Exception as e:
    print(f"✗ PyArrow: {e}")
    sys.exit(1)

try:
    import faiss
    print(f"✓ FAISS (CPU with CUDA support)")
except Exception as e:
    print(f"✗ FAISS: {e}")
    sys.exit(1)

try:
    import rank_bm25
    print(f"✓ Rank BM25")
except Exception as e:
    print(f"✗ Rank BM25: {e}")
    sys.exit(1)

print("")
print("✅ All dependencies installed successfully!")
EOF

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ INSTALLATION COMPLETE!"
    echo "========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Test LLM generation:"
    echo "     ./test_llm_generation.sh"
    echo ""
    echo "  2. Full generation:"
    echo "     ./run_llm_generation.sh"
    echo ""
else
    echo ""
    echo "❌ Installation verification failed!"
    echo "Please check errors above."
    exit 1
fi
