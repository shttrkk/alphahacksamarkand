#!/usr/bin/env python3
"""
Step 2: Build Indices
- Build embeddings for doc-level and chunk-level
- Build FAISS indices
- Build BM25 indices
- Save all to disk for fast loading

ВАЖНО: Этот скрипт корректно обрабатывает CUDA ошибки для RTX 5090
и делает CPU fallback при необходимости.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd
import pickle
import faiss
import torch
from tqdm import tqdm
import time
import traceback

from config.settings import (
    PROCESSED_DATA_DIR,
    EMBEDDINGS_DIR,
    INDICES_DIR,
    CACHE_DIR,
    DOC_EMBEDDERS,
    CHUNK_EMBEDDERS,
    BM25_K1,
    BM25_B,
    BATCH_SIZE_EMBEDDING,
    DEVICE,
    USE_CACHE,
)
from src.indexing.embedder import UnifiedEmbedder
from src.preprocessing.tokenizer import BankingTokenizer
from src.utils.logger import setup_logger, log_section, log_summary, format_time
from rank_bm25 import BM25Okapi


# ============================================================================
# Setup logging
# ============================================================================
logger = setup_logger("build_indices", log_file="logs/build_indices.log")


# ============================================================================
# Helper functions
# ============================================================================

def get_device_info():
    """Получить информацию об устройстве для логирования"""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return {
            "device": torch.cuda.get_device_name(0),
            "compute_capability": f"{props.major}.{props.minor}",
            "vram_gb": props.total_memory / 1024**3,
            "cuda_version": torch.version.cuda,
        }
    else:
        return {"device": "CPU", "cuda_available": False}


def build_embeddings_and_index(
    embedder_config: dict,
    embedder_name: str,
    texts: list,
    level: str,  # "doc" or "chunk"
    use_cache: bool = True,
    force_cpu: bool = False,
):
    """
    Build embeddings and FAISS index for a given embedder.

    ВАЖНО: Автоматически делает CPU fallback при CUDA ошибках.

    Args:
        embedder_config: Config dict from settings
        embedder_name: Name of embedder (e.g., "e5_large")
        texts: List of texts to embed
        level: "doc" or "chunk"
        use_cache: Use cached embeddings if available
        force_cpu: Force CPU mode (используется при CUDA fallback)

    Returns:
        (embeddings, faiss_index, success: bool)
        success=False если построить не удалось даже на CPU
    """
    logger.info(f"  Building {embedder_name} ({level}-level)...")
    t0 = time.time()

    # Paths
    emb_path = EMBEDDINGS_DIR / level / f"{embedder_name}.npy"
    index_path = INDICES_DIR / f"{embedder_name}_{level}.bin"

    # Check cache
    if use_cache and emb_path.exists() and index_path.exists():
        logger.info(f"    ✓ Loading from cache...")
        try:
            embeddings = np.load(emb_path)
            index = faiss.read_index(str(index_path))
            logger.info(f"    ✓ Loaded: {embeddings.shape[0]} vectors, dim={embeddings.shape[1]}")
            logger.info(f"    ⏱  Time: {format_time(time.time() - t0)}")
            return embeddings, index, True
        except Exception as e:
            logger.warning(f"    ⚠️  Failed to load cache: {e}")
            logger.warning(f"    Rebuilding from scratch...")

    # Определяем device
    device = "cpu" if force_cpu else DEVICE
    logger.info(f"    Device: {device}")
    logger.info(f"    Model: {embedder_config['model_name']}")
    logger.info(f"    Texts to embed: {len(texts)}")

    # Try to build embeddings
    try:
        # Build embedder
        logger.info(f"    Loading model...")
        embedder = UnifiedEmbedder(
            model_name=embedder_config['model_name'],
            prefix_passage=embedder_config.get('prefix_passage', ''),
            prefix_query=embedder_config.get('prefix_query', ''),
            max_seq_length=embedder_config.get('max_seq_length', 512),
            device=device,
            batch_size=BATCH_SIZE_EMBEDDING,
        )

        logger.info(f"    ✓ Model loaded on {device}")

        # Encode
        logger.info(f"    Encoding {len(texts)} texts...")
        embeddings = embedder.encode_passages(
            texts=texts,
            show_progress=True,
            normalize=True,
        )

        logger.info(f"    ✓ Embeddings shape: {embeddings.shape}")

        # Build FAISS index (Inner Product = cosine similarity for normalized vectors)
        logger.info(f"    Building FAISS index...")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        logger.info(f"    ✓ FAISS index size: {index.ntotal}")

        # Save
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(emb_path, embeddings)
        faiss.write_index(index, str(index_path))

        logger.info(f"    ✓ Saved to:")
        logger.info(f"      - {emb_path}")
        logger.info(f"      - {index_path}")
        logger.info(f"    ⏱  Time: {format_time(time.time() - t0)}")

        return embeddings, index, True

    except RuntimeError as e:
        error_msg = str(e)

        # Проверяем, является ли это CUDA ошибкой
        is_cuda_error = (
            "CUDA" in error_msg or
            "cuda" in error_msg or
            "kernel image" in error_msg or
            "compute capability" in error_msg or
            "sm_" in error_msg
        )

        if is_cuda_error and not force_cpu:
            logger.warning(f"    ⚠️  CUDA ERROR detected!")
            logger.warning(f"    Error: {error_msg}")
            logger.warning(f"    Это может быть из-за несовместимости GPU (RTX 5090 требует CUDA 12.1+)")
            logger.warning(f"    Пытаемся CPU fallback...")
            logger.info("")

            # Recursive call with force_cpu=True
            return build_embeddings_and_index(
                embedder_config=embedder_config,
                embedder_name=embedder_name,
                texts=texts,
                level=level,
                use_cache=False,  # Don't use cache for CPU fallback
                force_cpu=True,
            )
        else:
            # Другая ошибка или CPU fallback тоже не сработал
            logger.error(f"    ❌ FATAL: Failed to build {embedder_name}")
            logger.error(f"    Error: {error_msg}")
            logger.error(f"    Stacktrace:")
            logger.error(traceback.format_exc())
            return None, None, False

    except Exception as e:
        logger.error(f"    ❌ FATAL: Unexpected error building {embedder_name}")
        logger.error(f"    Error: {e}")
        logger.error(f"    Stacktrace:")
        logger.error(traceback.format_exc())
        return None, None, False


def build_bm25_index(texts: list, level: str, use_cache: bool = True):
    """
    Build BM25 index.

    Args:
        texts: List of texts
        level: "doc" or "chunk"
        use_cache: Use cached index if available

    Returns:
        BM25Okapi index
    """
    logger.info(f"  Building BM25 index ({level}-level)...")
    t0 = time.time()

    # Path
    cache_path = CACHE_DIR / f"bm25_{level}.pkl"

    # Check cache
    if use_cache and cache_path.exists():
        logger.info(f"    ✓ Loading from cache...")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)

        # Поддержка старого формата (голый BM25) и нового (словарь)
        if isinstance(data, dict):
            bm25 = data['bm25']
        else:
            # Старый формат - голый BM25Okapi
            bm25 = data
            logger.warning(f"    ⚠️  Old cache format detected, rebuild recommended")

        logger.info(f"    ✓ Loaded: {len(bm25.doc_freqs)} documents")
        logger.info(f"    ⏱  Time: {format_time(time.time() - t0)}")
        return bm25

    # Build tokenizer
    logger.info(f"    Tokenizing {len(texts)} texts...")
    tokenizer = BankingTokenizer()
    tokenized_corpus = [tokenizer.tokenize(text) for text in tqdm(texts, desc="    Tokenizing")]

    # Build BM25
    logger.info(f"    Building BM25 index...")
    bm25 = BM25Okapi(
        tokenized_corpus,
        k1=BM25_K1,
        b=BM25_B,
    )

    # Save - ВАЖНО: сохраняем словарь с bm25 и tokenized_corpus
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'bm25': bm25,
            'tokenized_docs': tokenized_corpus,
        }, f)

    logger.info(f"    ✓ Saved to {cache_path}")
    logger.info(f"    ⏱  Time: {format_time(time.time() - t0)}")

    return bm25


def main():
    script_start = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Build embeddings and indices")
    parser.add_argument("--no-cache", action="store_true", help="Don't use cached embeddings")
    parser.add_argument("--doc-only", action="store_true", help="Build only doc-level indices")
    parser.add_argument("--chunk-only", action="store_true", help="Build only chunk-level indices")
    parser.add_argument("--strict", action="store_true", default=True,
                       help="Fail if no dense models succeed (default: True)")
    args = parser.parse_args()

    use_cache = not args.no_cache

    log_section(logger, "ALPHA RAG ELITE - BUILD INDICES")

    # ========================================================================
    # Device info
    # ========================================================================
    logger.info("System info:")
    device_info = get_device_info()
    for key, value in device_info.items():
        logger.info(f"  {key}: {value}")
    logger.info("")

    logger.info("Config:")
    logger.info(f"  Use cache: {use_cache}")
    logger.info(f"  Doc-only: {args.doc_only}")
    logger.info(f"  Chunk-only: {args.chunk_only}")
    logger.info(f"  Strict mode: {args.strict}")
    logger.info(f"  Batch size: {BATCH_SIZE_EMBEDDING}")
    logger.info("")

    # ========================================================================
    # Load data
    # ========================================================================
    logger.info("[1/4] Loading processed data...")

    docs_path = PROCESSED_DATA_DIR / "documents_normalized.parquet"
    chunks_path = PROCESSED_DATA_DIR / "chunks.parquet"

    if not docs_path.exists():
        logger.error(f"❌ FATAL: {docs_path} not found!")
        logger.error(f"   Run 01_preprocess_data.py first")
        sys.exit(1)

    if not chunks_path.exists():
        logger.error(f"❌ FATAL: {chunks_path} not found!")
        logger.error(f"   Run 01_preprocess_data.py first")
        sys.exit(1)

    df_docs = pd.read_parquet(docs_path)
    df_chunks = pd.read_parquet(chunks_path)

    logger.info(f"  ✓ Documents: {len(df_docs)}")
    logger.info(f"  ✓ Chunks: {len(df_chunks)}")
    logger.info("")

    # Extract texts
    doc_texts = df_docs['combined'].tolist()
    chunk_texts = df_chunks['chunk_text'].tolist()

    # ========================================================================
    # Build doc-level indices
    # ========================================================================
    successful_doc_models = []
    failed_doc_models = []

    if not args.chunk_only:
        logger.info("[2/4] Building doc-level indices...")
        logger.info("")

        for embedder_name, config in DOC_EMBEDDERS.items():
            if not config.get("enabled", True):
                logger.info(f"  Skipping {embedder_name} (disabled in config)")
                continue

            embeddings, index, success = build_embeddings_and_index(
                embedder_config=config,
                embedder_name=embedder_name,
                texts=doc_texts,
                level="doc",
                use_cache=use_cache,
            )

            if success:
                successful_doc_models.append(embedder_name)
            else:
                failed_doc_models.append(embedder_name)
                logger.warning(f"  ⚠️  {embedder_name} FAILED - skipped")

            logger.info("")

        # BM25 doc-level
        bm25_doc = build_bm25_index(doc_texts, level="doc", use_cache=use_cache)
        logger.info("")

    # ========================================================================
    # Build chunk-level indices
    # ========================================================================
    successful_chunk_models = []
    failed_chunk_models = []

    if not args.doc_only:
        logger.info("[3/4] Building chunk-level indices...")
        logger.info("")

        for embedder_name, config in CHUNK_EMBEDDERS.items():
            if not config.get("enabled", True):
                logger.info(f"  Skipping {embedder_name} (disabled in config)")
                continue

            embeddings, index, success = build_embeddings_and_index(
                embedder_config=config,
                embedder_name=embedder_name,
                texts=chunk_texts,
                level="chunk",
                use_cache=use_cache,
            )

            if success:
                successful_chunk_models.append(embedder_name)
            else:
                failed_chunk_models.append(embedder_name)
                logger.warning(f"  ⚠️  {embedder_name} FAILED - skipped")

            logger.info("")

        # BM25 chunk-level
        bm25_chunk = build_bm25_index(chunk_texts, level="chunk", use_cache=use_cache)
        logger.info("")

    # ========================================================================
    # Summary
    # ========================================================================
    total_time = time.time() - script_start

    logger.info("[4/4] Summary:")
    logger.info("")

    if not args.chunk_only:
        logger.info(f"  Doc-level models:")
        logger.info(f"    ✓ Successful: {len(successful_doc_models)}")
        for model in successful_doc_models:
            logger.info(f"      - {model}")
        if failed_doc_models:
            logger.warning(f"    ❌ Failed: {len(failed_doc_models)}")
            for model in failed_doc_models:
                logger.warning(f"      - {model}")
        logger.info(f"    ✓ BM25: OK")

    if not args.doc_only:
        logger.info(f"\n  Chunk-level models:")
        logger.info(f"    ✓ Successful: {len(successful_chunk_models)}")
        for model in successful_chunk_models:
            logger.info(f"      - {model}")
        if failed_chunk_models:
            logger.warning(f"    ❌ Failed: {len(failed_chunk_models)}")
            for model in failed_chunk_models:
                logger.warning(f"      - {model}")
        logger.info(f"    ✓ BM25: OK")

    logger.info("")

    # Check if at least one dense model succeeded (strict mode)
    total_successful = len(successful_doc_models) + len(successful_chunk_models)

    if args.strict and total_successful == 0:
        logger.error("=" * 80)
        logger.error("❌ FATAL: NO DENSE MODELS SUCCEEDED!")
        logger.error("=" * 80)
        logger.error("")
        logger.error("Все dense retriever'ы не смогли построиться.")
        logger.error("Это означает, что search будет работать ТОЛЬКО на BM25,")
        logger.error("что даст очень плохое качество (~5-10% Hit@5).")
        logger.error("")
        logger.error("Возможные причины:")
        logger.error("  1. CUDA несовместима с вашей GPU (RTX 5090 требует CUDA 12.1+)")
        logger.error("  2. PyTorch установлен неправильно")
        logger.error("  3. Недостаточно VRAM")
        logger.error("  4. Модели не смогли загрузиться даже на CPU")
        logger.error("")
        logger.error("Решение:")
        logger.error("  1. Переустановите PyTorch с правильной версией CUDA:")
        logger.error("     pip install torch --index-url https://download.pytorch.org/whl/cu121")
        logger.error("  2. Или запустите ./setup.sh заново")
        logger.error("  3. Проверьте логи выше для деталей по каждой модели")
        logger.error("")
        sys.exit(1)

    if total_successful < 2:
        logger.warning("=" * 80)
        logger.warning("⚠️  WARNING: Менее 2 dense моделей успешно построено!")
        logger.warning("=" * 80)
        logger.warning("")
        logger.warning(f"Успешных моделей: {total_successful}")
        logger.warning("Рекомендуется минимум 2-3 модели для хорошего качества.")
        logger.warning("Текущая конфигурация может дать субоптимальное качество.")
        logger.warning("")

    log_summary(logger, "BUILD INDICES COMPLETED", {
        "Total time": format_time(total_time),
        "Doc models OK": len(successful_doc_models),
        "Chunk models OK": len(successful_chunk_models),
        "Total models OK": total_successful,
        "Failed models": len(failed_doc_models) + len(failed_chunk_models),
    })

    logger.info("")
    logger.info("✅ Indexing completed successfully!")
    logger.info(f"   Logs saved to: logs/build_indices.log")
    logger.info("")
    logger.info("Next step: python scripts/03_run_search.py")
    logger.info("")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.error("\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ FATAL ERROR: {e}", exc_info=True)
        sys.exit(1)
