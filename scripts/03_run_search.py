#!/usr/bin/env python3
"""
Step 3: Run Search
- Load all indices
- Process queries with expansion
- Multi-model retrieval (doc + chunk)
- Weighted RRF fusion
- Domain boosts
- Optional reranking
- Generate submission
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd
import pickle
import faiss
from tqdm import tqdm
import time
import traceback
from collections import defaultdict
from typing import List, Dict, Tuple

from config.settings import (
    PROCESSED_DATA_DIR,
    EMBEDDINGS_DIR,
    INDICES_DIR,
    CACHE_DIR,
    SUBMIT_CSV,
    DOC_EMBEDDERS,
    CHUNK_EMBEDDERS,
    DOC_TOP_K_PER_EMBEDDER,
    DOC_BM25_TOP_K,
    CHUNK_TOP_K_PER_EMBEDDER,
    CHUNK_BM25_TOP_K,
    RRF_K,
    ENABLE_CROSS_ENCODER_RERANKING,
    CROSS_ENCODER_MODEL,
    RERANKER_CANDIDATE_K,
    RERANKER_BATCH_SIZE,
    BM25_DOC_WEIGHT,
    BM25_CHUNK_WEIGHT,
    ENABLE_RULE_BASED_EXPANSION,
    ENABLE_BM25_PRF,
    PRF_TOP_DOCS,
    PRF_TOP_TERMS,
    MAX_QUERY_VARIANTS,
    ENABLE_DOMAIN_BOOSTING,
    DEVICE,
    BATCH_SIZE_EMBEDDING,
    TOP_K,
    # LLM Query Variants (Egor's killer feature)
    USE_QUERY_VARIANTS,
    QUERY_VARIANTS_PATH,
    ORIGINAL_QUERY_WEIGHT,
    VARIANT_QUERY_WEIGHT,
    MAX_VARIANTS_TO_USE,
)
from src.indexing.embedder import UnifiedEmbedder
from src.query.query_expander import QueryExpander
from src.scoring.rrf_scorer import weighted_rrf
from src.scoring.domain_booster import DomainBooster
from src.preprocessing.entity_extractor import EntityExtractor
from src.preprocessing.intent_classifier import IntentClassifier
from src.preprocessing.text_normalizer import normalize_text
from src.preprocessing.tokenizer import tokenize
from src.utils.logger import setup_logger, log_section, log_summary, format_time

# Cross-encoder for reranking
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.warning("CrossEncoder not available. Install: pip install sentence-transformers")


# ============================================================================
# Setup logging
# ============================================================================
logger = setup_logger("run_search", log_file="logs/run_search.log")


class AlphaRAGElite:
    """Complete retrieval pipeline."""

    def __init__(self, enable_reranking: bool = False):
        self.enable_reranking = enable_reranking

        logger.info("Initializing Alpha RAG Elite...")

        # Load documents and chunks
        logger.info("  Loading documents...")
        try:
            self.df_docs = pd.read_parquet(PROCESSED_DATA_DIR / "documents_normalized.parquet")
            self.df_chunks = pd.read_parquet(PROCESSED_DATA_DIR / "chunks.parquet")
        except Exception as e:
            logger.error(f"❌ Failed to load processed data: {e}", exc_info=True)
            raise

        logger.info(f"    Documents: {len(self.df_docs)}")
        logger.info(f"    Chunks: {len(self.df_chunks)}")

        # Load LLM query variants (Egor's killer feature)
        self.query_variants_map = {}
        if USE_QUERY_VARIANTS:
            if QUERY_VARIANTS_PATH.exists():
                logger.info(f"  Loading LLM query variants from {QUERY_VARIANTS_PATH}")
                try:
                    df_variants = pd.read_parquet(QUERY_VARIANTS_PATH)
                    # Create q_id -> variants mapping
                    for _, row in df_variants.iterrows():
                        q_id = row['q_id']
                        variants = []
                        for i in range(1, MAX_VARIANTS_TO_USE + 1):
                            variant_col = f'variant_{i}'
                            if variant_col in row and pd.notna(row[variant_col]):
                                variants.append(row[variant_col])
                        self.query_variants_map[q_id] = variants
                    logger.info(f"    ✓ Loaded variants for {len(self.query_variants_map)} queries")
                    logger.info(f"    Average variants per query: {sum(len(v) for v in self.query_variants_map.values()) / len(self.query_variants_map):.2f}")
                except Exception as e:
                    logger.warning(f"    ⚠️  Failed to load query variants: {e}")
                    logger.warning(f"    Falling back to original queries only")
                    # query_variants_map останется пустым - будет использован fallback в search()
            else:
                logger.warning(f"    ⚠️  Query variants file not found: {QUERY_VARIANTS_PATH}")
                logger.warning(f"    Run: python tools/generate_query_variants.py")
                logger.warning(f"    Falling back to original queries only")
        logger.info("")

        # Create web_id mappings
        self.doc_id_to_web_id = self.df_docs['web_id'].tolist()
        self.chunk_id_to_web_id = self.df_chunks['web_id'].tolist()

        # Create web_id -> doc_data lookup for fast access (avoid DataFrame scans)
        self.web_id_to_doc = {}
        for _, row in self.df_docs.iterrows():
            self.web_id_to_doc[row['web_id']] = {
                'combined': row['combined'],
                'title': row['title'],
            }
        logger.info(f"  Created web_id lookup for {len(self.web_id_to_doc)} documents")

        # Load indices
        self.doc_embedders = {}
        self.doc_indices = {}
        self.chunk_embedders = {}
        self.chunk_indices = {}

        self._load_doc_indices()
        self._load_chunk_indices()
        self._load_bm25_indices()

        # Initialize components
        self.query_expander = QueryExpander(
            enable_rule_based=ENABLE_RULE_BASED_EXPANSION,
            enable_prf=ENABLE_BM25_PRF,
            prf_top_docs=PRF_TOP_DOCS,
            prf_top_terms=PRF_TOP_TERMS,
            max_variants=MAX_QUERY_VARIANTS,
        )

        self.domain_booster = DomainBooster()
        self.entity_extractor = EntityExtractor()
        self.intent_classifier = IntentClassifier()

        # Load cross-encoder for reranking (if enabled)
        self.cross_encoder = None
        if ENABLE_CROSS_ENCODER_RERANKING:
            if CROSS_ENCODER_AVAILABLE:
                logger.info(f"  Loading cross-encoder: {CROSS_ENCODER_MODEL}")
                try:
                    self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)
                    logger.info(f"    ✓ Cross-encoder loaded successfully")
                except Exception as e:
                    logger.warning(f"    ⚠️  Failed to load cross-encoder: {e}")
                    logger.warning(f"    Reranking will be disabled")
            else:
                logger.warning("  ⚠️  Cross-encoder reranking enabled but library not available")
                logger.warning("  Install: pip install sentence-transformers")

        logger.info("✅ Initialization complete!")
        logger.info("")

    def _load_doc_indices(self):
        """Load doc-level embedders and FAISS indices."""
        logger.info("  Loading doc-level indices...")
        loaded_count = 0

        for name, config in DOC_EMBEDDERS.items():
            if not config.get('enabled', True):
                logger.info(f"    Skipping {name} (disabled in config)")
                continue

            try:
                # Load FAISS index
                index_path = INDICES_DIR / f"{name}_doc.bin"
                if not index_path.exists():
                    logger.warning(f"    ⚠️  Index not found for {name}: {index_path}")
                    continue

                index = faiss.read_index(str(index_path))

                # Initialize embedder (for query encoding)
                logger.info(f"    Loading model: {config['model_name']}")
                logger.info(f"      Device: {DEVICE}")
                embedder = UnifiedEmbedder(
                    model_name=config['model_name'],
                    prefix_passage=config.get('prefix_passage', ''),
                    prefix_query=config.get('prefix_query', ''),
                    max_seq_length=config.get('max_seq_length', 512),
                    device=DEVICE,
                    batch_size=BATCH_SIZE_EMBEDDING,
                )

                self.doc_embedders[name] = embedder
                self.doc_indices[name] = index
                loaded_count += 1

                logger.info(f"    ✓ {name}: {index.ntotal} vectors")

            except Exception as e:
                logger.error(f"    ❌ Failed to load {name}: {e}", exc_info=True)

        if loaded_count == 0:
            logger.error("❌ No doc-level dense indices loaded!")
        else:
            logger.info(f"  ✓ Loaded {loaded_count} doc-level dense models")
        logger.info("")

    def _load_chunk_indices(self):
        """Load chunk-level embedders and FAISS indices."""
        logger.info("  Loading chunk-level indices...")
        loaded_count = 0
        for name, config in CHUNK_EMBEDDERS.items():
            if not config.get('enabled', True):
                logger.info(f"    Skipping {name} (disabled in config)")
                continue

            try:
                # Load FAISS index
                index_path = INDICES_DIR / f"{name}_chunk.bin"
                if not index_path.exists():
                    logger.warning(f"    ⚠️  Index not found for {name}: {index_path}")
                    continue

                index = faiss.read_index(str(index_path))

                # Initialize embedder (for query encoding)
                logger.info(f"    Loading model: {config['model_name']}")
                logger.info(f"      Device: {DEVICE}")
                embedder = UnifiedEmbedder(
                    model_name=config['model_name'],
                    prefix_passage=config.get('prefix_passage', ''),
                    prefix_query=config.get('prefix_query', ''),
                    max_seq_length=config.get('max_seq_length', 512),
                    device=DEVICE,
                    batch_size=BATCH_SIZE_EMBEDDING,
                )

                self.chunk_embedders[name] = embedder
                self.chunk_indices[name] = index
                loaded_count += 1

                logger.info(f"    ✓ {name}: {index.ntotal} vectors")

            except Exception as e:
                logger.error(f"    ❌ Failed to load {name}: {e}", exc_info=True)

        if loaded_count == 0:
            logger.warning("  ⚠️  No chunk-level dense indices loaded!")
        else:
            logger.info(f"  ✓ Loaded {loaded_count} chunk-level dense models")
        logger.info("")

    def _load_bm25_indices(self):
        """Load BM25 indices."""
        logger.info("  Loading BM25 indices...")

        # Doc-level BM25
        bm25_doc_path = CACHE_DIR / "bm25_doc.pkl"
        if bm25_doc_path.exists():
            try:
                with open(bm25_doc_path, 'rb') as f:
                    data = pickle.load(f)

                # Поддержка старого формата (голый BM25) и нового (словарь)
                if isinstance(data, dict):
                    self.bm25_doc = data['bm25']
                    self.tokenized_docs_doc = data['tokenized_docs']
                else:
                    # Старый формат - голый BM25Okapi
                    logger.warning(f"    ⚠️  Old BM25 format detected (doc-level)")
                    logger.warning(f"    Using BM25 without tokenized docs (may affect PRF)")
                    self.bm25_doc = data
                    self.tokenized_docs_doc = None

                logger.info(f"    ✓ BM25 doc-level: {len(self.bm25_doc.doc_freqs)} documents")
            except Exception as e:
                logger.error(f"    ❌ Failed to load BM25 doc-level: {e}", exc_info=True)
                self.bm25_doc = None
                self.tokenized_docs_doc = None
        else:
            logger.warning(f"    ⚠️  BM25 doc-level not found at {bm25_doc_path}")
            self.bm25_doc = None
            self.tokenized_docs_doc = None

        # Chunk-level BM25
        bm25_chunk_path = CACHE_DIR / "bm25_chunk.pkl"
        if bm25_chunk_path.exists():
            try:
                with open(bm25_chunk_path, 'rb') as f:
                    data = pickle.load(f)

                # Поддержка старого формата (голый BM25) и нового (словарь)
                if isinstance(data, dict):
                    self.bm25_chunk = data['bm25']
                    self.tokenized_docs_chunk = data['tokenized_docs']
                else:
                    # Старый формат - голый BM25Okapi
                    logger.warning(f"    ⚠️  Old BM25 format detected (chunk-level)")
                    logger.warning(f"    Using BM25 without tokenized docs (may affect PRF)")
                    self.bm25_chunk = data
                    self.tokenized_docs_chunk = None

                logger.info(f"    ✓ BM25 chunk-level: {len(self.bm25_chunk.doc_freqs)} chunks")
            except Exception as e:
                logger.error(f"    ❌ Failed to load BM25 chunk-level: {e}", exc_info=True)
                self.bm25_chunk = None
                self.tokenized_docs_chunk = None
        else:
            logger.warning(f"    ⚠️  BM25 chunk-level not found at {bm25_chunk_path}")
            self.bm25_chunk = None
            self.tokenized_docs_chunk = None

        logger.info("")

    def search(self, query: str, q_id: int = None, enable_profiling: bool = False) -> List[int]:
        """
        Search for a query with optimized batching and profiling.

        Args:
            query: Search query
            q_id: Question ID (for LLM variants lookup)
            enable_profiling: Enable detailed timing profiling

        Returns:
            List of top-K web_ids
        """
        timings = {} if enable_profiling else None
        t_start = time.time() if enable_profiling else None

        # Normalize query
        t0 = time.time()
        query_normalized = normalize_text(query, is_query=True)
        if enable_profiling:
            timings['normalize'] = time.time() - t0

        # Extract entities and intents (ONLY if domain boosting enabled)
        if ENABLE_DOMAIN_BOOSTING:
            t0 = time.time()
            query_entities = self.entity_extractor.extract(query_normalized)
            query_intents = self.intent_classifier.classify(query_normalized)
            if enable_profiling:
                timings['extract_entities_intents'] = time.time() - t0
        else:
            query_entities = None
            query_intents = None

        # Generate/get query variants
        t0 = time.time()

        # EGOR'S KILLER FEATURE: Use LLM-generated variants if available
        if USE_QUERY_VARIANTS and q_id is not None and q_id in self.query_variants_map:
            # Use LLM variants
            llm_variants = self.query_variants_map[q_id]
            query_variants = [query_normalized] + llm_variants  # Original + LLM variants
            variant_weights = [ORIGINAL_QUERY_WEIGHT] + [VARIANT_QUERY_WEIGHT] * len(llm_variants)
        else:
            # Fallback to rule-based expansion (old behavior)
            query_variants = self.query_expander.expand(
                query=query_normalized,
                bm25=self.bm25_doc,
                tokenized_docs=self.tokenized_docs_doc if hasattr(self, 'tokenized_docs_doc') else None,
            )
            variant_weights = [1.0] * len(query_variants)  # Equal weights

        if enable_profiling:
            timings['query_expansion'] = time.time() - t0
            timings['num_variants'] = len(query_variants)
            timings['using_llm_variants'] = USE_QUERY_VARIANTS and q_id is not None and q_id in self.query_variants_map

        # Retrieve candidates from all sources
        all_rankings = {}

        # === DOC-LEVEL DENSE RETRIEVAL (BATCHED) ===
        t0 = time.time()
        for name, embedder in self.doc_embedders.items():
            try:
                # OPTIMIZATION: Batch encode all variants at once
                q_embs = embedder.encode_queries(query_variants, show_progress=False, normalize=True)

                # Search for each variant
                for variant_idx in range(len(query_variants)):
                    q_emb = q_embs[variant_idx:variant_idx+1]  # Keep 2D shape
                    scores, indices = self.doc_indices[name].search(q_emb, DOC_TOP_K_PER_EMBEDDER)

                    # Convert to web_ids
                    web_ids = [self.doc_id_to_web_id[idx] for idx in indices[0] if idx >= 0]

                    key = f"{name}_doc_v{variant_idx}"
                    all_rankings[key] = web_ids

            except Exception as e:
                logger.error(f"Error in {name} doc retrieval: {e}")
        if enable_profiling:
            timings['doc_dense_retrieval'] = time.time() - t0

        # === DOC-LEVEL BM25 ===
        t0 = time.time()
        if self.bm25_doc is not None:
            for variant_idx, variant in enumerate(query_variants):
                try:
                    tokens = tokenize(variant)
                    scores = self.bm25_doc.get_scores(tokens)
                    top_indices = np.argsort(-scores)[:DOC_BM25_TOP_K]
                    web_ids = [self.doc_id_to_web_id[idx] for idx in top_indices]

                    key = f"bm25_doc_v{variant_idx}"
                    all_rankings[key] = web_ids

                except Exception as e:
                    logger.error(f"Error in BM25 doc: {e}")
        if enable_profiling:
            timings['doc_bm25_retrieval'] = time.time() - t0

        # === CHUNK-LEVEL DENSE RETRIEVAL (BATCHED) ===
        t0 = time.time()
        for name, embedder in self.chunk_embedders.items():
            try:
                # OPTIMIZATION: Batch encode all variants at once
                q_embs = embedder.encode_queries(query_variants, show_progress=False, normalize=True)

                # Search for each variant
                for variant_idx in range(len(query_variants)):
                    q_emb = q_embs[variant_idx:variant_idx+1]  # Keep 2D shape
                    scores, indices = self.chunk_indices[name].search(q_emb, CHUNK_TOP_K_PER_EMBEDDER)

                    # Aggregate chunks to web_ids (max-pooling by score)
                    chunk_scores = defaultdict(float)
                    for score, idx in zip(scores[0], indices[0]):
                        if idx >= 0:
                            web_id = self.chunk_id_to_web_id[idx]
                            if score > chunk_scores[web_id]:
                                chunk_scores[web_id] = score

                    # Sort by score
                    web_ids = [wid for wid, _ in sorted(chunk_scores.items(), key=lambda x: -x[1])]

                    key = f"{name}_chunk_v{variant_idx}"
                    all_rankings[key] = web_ids

            except Exception as e:
                logger.error(f"Error in {name} chunk retrieval: {e}")
        if enable_profiling:
            timings['chunk_dense_retrieval'] = time.time() - t0

        # === CHUNK-LEVEL BM25 ===
        t0 = time.time()
        if self.bm25_chunk is not None:
            for variant_idx, variant in enumerate(query_variants):
                try:
                    tokens = tokenize(variant)
                    scores = self.bm25_chunk.get_scores(tokens)
                    top_indices = np.argsort(-scores)[:CHUNK_BM25_TOP_K]

                    # Aggregate to web_ids
                    chunk_scores = defaultdict(float)
                    for idx in top_indices:
                        web_id = self.chunk_id_to_web_id[idx]
                        score = scores[idx]
                        if score > chunk_scores[web_id]:
                            chunk_scores[web_id] = score

                    web_ids = [wid for wid, _ in sorted(chunk_scores.items(), key=lambda x: -x[1])]

                    key = f"bm25_chunk_v{variant_idx}"
                    all_rankings[key] = web_ids

                except Exception as e:
                    logger.error(f"Error in BM25 chunk: {e}")
        if enable_profiling:
            timings['chunk_bm25_retrieval'] = time.time() - t0

        # === WEIGHTED RRF FUSION (with LLM variant weighting) ===
        t0 = time.time()
        weights = {}
        for key in all_rankings.keys():
            # Extract variant index from key (e.g., "e5_large_doc_v0" -> 0)
            variant_idx = 0
            if '_v' in key:
                try:
                    variant_idx = int(key.split('_v')[-1])
                except:
                    variant_idx = 0

            # Get variant weight (from LLM variants or default)
            if variant_idx < len(variant_weights):
                variant_weight_multiplier = variant_weights[variant_idx]
            else:
                variant_weight_multiplier = 1.0

            # Get base weight for this retrieval method
            if 'bm25_doc' in key:
                base_weight = BM25_DOC_WEIGHT
            elif 'bm25_chunk' in key:
                base_weight = BM25_CHUNK_WEIGHT
            elif '_doc_' in key:
                # Extract model name
                model_name = key.split('_doc_')[0]
                base_weight = DOC_EMBEDDERS.get(model_name, {}).get('weight', 1.0)
            elif '_chunk_' in key:
                model_name = key.split('_chunk_')[0]
                base_weight = CHUNK_EMBEDDERS.get(model_name, {}).get('weight', 1.0)
            else:
                base_weight = 1.0

            # Combine base weight with variant weight
            weights[key] = base_weight * variant_weight_multiplier

        rrf_scores = weighted_rrf(all_rankings, weights, k=RRF_K)
        if enable_profiling:
            timings['rrf_fusion'] = time.time() - t0

        # === DOMAIN BOOSTS (CONDITIONAL) ===
        if ENABLE_DOMAIN_BOOSTING:
            t0 = time.time()
            # Get top-200 candidates for boosting
            top_candidates = sorted(rrf_scores.items(), key=lambda x: -x[1])[:200]
            candidate_web_ids = [wid for wid, _ in top_candidates]

            # OPTIMIZATION: Use web_id lookup dict instead of DataFrame scans
            candidate_docs = []
            for web_id in candidate_web_ids:
                doc_data = self.web_id_to_doc.get(web_id)
                if doc_data:
                    candidate_docs.append({
                        'doc_id': web_id,
                        'text': doc_data['combined'],
                        'title': doc_data['title'],
                    })

            # Compute boosts
            boosts = self.domain_booster.compute_boosts_batch(
                query=query_normalized,
                documents=candidate_docs,
                query_entities=query_entities,
                query_intents=query_intents,
            )

            # Combine RRF + boosts
            final_scores = {}
            for web_id, rrf_score in rrf_scores.items():
                boost = boosts.get(web_id, 0.0)
                final_scores[web_id] = rrf_score + boost

            if enable_profiling:
                timings['domain_boosting'] = time.time() - t0
        else:
            # No boosting - use RRF scores directly
            final_scores = rrf_scores

        # === CROSS-ENCODER RERANKING (TWO-STAGE RETRIEVAL) ===
        if self.cross_encoder is not None and ENABLE_CROSS_ENCODER_RERANKING:
            t0 = time.time()

            # Stage 1: Get TOP-100 candidates from bi-encoder
            sorted_candidates = sorted(final_scores.items(), key=lambda x: -x[1])[:RERANKER_CANDIDATE_K]
            candidate_web_ids = [wid for wid, _ in sorted_candidates]

            # Stage 2: Rerank with cross-encoder (query + doc pairs)
            query_doc_pairs = []
            valid_web_ids = []
            for web_id in candidate_web_ids:
                doc_data = self.web_id_to_doc.get(web_id)
                if doc_data:
                    # Cross-encoder input: [query, document]
                    doc_text = doc_data['combined'][:512]  # Truncate long docs
                    query_doc_pairs.append([query_normalized, doc_text])
                    valid_web_ids.append(web_id)

            if query_doc_pairs:
                # Batch prediction for speed
                rerank_scores = self.cross_encoder.predict(
                    query_doc_pairs,
                    batch_size=RERANKER_BATCH_SIZE,
                    show_progress_bar=False
                )

                # Combine web_ids with rerank scores
                reranked_results = list(zip(valid_web_ids, rerank_scores))
                reranked_results.sort(key=lambda x: -x[1])  # Sort by rerank score

                # Take TOP-K after reranking
                top_web_ids = [wid for wid, _ in reranked_results[:TOP_K]]

                if enable_profiling:
                    timings['cross_encoder_reranking'] = time.time() - t0
                    timings['reranked_candidates'] = len(query_doc_pairs)
            else:
                # Fallback if no valid docs
                top_web_ids = candidate_web_ids[:TOP_K]
        else:
            # No reranking - use bi-encoder scores
            sorted_web_ids = sorted(final_scores.items(), key=lambda x: -x[1])
            top_web_ids = [wid for wid, _ in sorted_web_ids[:TOP_K]]

        if enable_profiling:
            timings['total'] = time.time() - t_start
            logger.debug(f"Search timings: {timings}")

        return top_web_ids


def main():
    script_start = time.time()

    parser = argparse.ArgumentParser(description="Run search and generate submission")
    parser.add_argument(
        "--enable-reranking",
        action="store_true",
        help="Enable cross-encoder reranking (requires validation!)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Number of documents to return per query"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(SUBMIT_CSV),
        help="Output submission file path"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N questions (for testing)"
    )
    parser.add_argument(
        "--enable-profiling",
        action="store_true",
        help="Enable detailed timing profiling for each query"
    )

    args = parser.parse_args()

    log_section(logger, "ALPHA RAG ELITE - RUN SEARCH (OPTIMIZED)")
    logger.info("Config:")
    logger.info(f"  Enable reranking: {args.enable_reranking}")
    logger.info(f"  Top-K: {args.top_k}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Limit: {args.limit if args.limit else 'All questions'}")
    logger.info(f"  Profiling: {args.enable_profiling}")
    logger.info("")
    logger.info("Optimizations enabled:")
    logger.info(f"  Max query variants: {MAX_QUERY_VARIANTS}")
    logger.info(f"  Domain boosting: {ENABLE_DOMAIN_BOOSTING}")
    logger.info(f"  Batched encoding: Yes")
    logger.info("")

    # Initialize pipeline
    logger.info("[1/3] Initializing pipeline...")
    t0 = time.time()
    try:
        pipeline = AlphaRAGElite(enable_reranking=args.enable_reranking)
        logger.info(f"  ✓ Pipeline initialized in {format_time(time.time() - t0)}")
        logger.info("")
    except Exception as e:
        logger.error(f"❌ FATAL: Failed to initialize pipeline: {e}", exc_info=True)
        sys.exit(1)

    # Load questions
    logger.info("[2/3] Loading questions...")
    questions_path = PROCESSED_DATA_DIR / "questions_processed.parquet"

    if not questions_path.exists():
        logger.error(f"❌ FATAL: Questions not found: {questions_path}")
        logger.error(f"   Run 01_preprocess_data.py first!")
        sys.exit(1)

    try:
        df_questions = pd.read_parquet(questions_path)
        if args.limit:
            df_questions = df_questions.head(args.limit)
            logger.info(f"  ✓ Loaded {len(df_questions)} questions (limited from full dataset)")
        else:
            logger.info(f"  ✓ Loaded {len(df_questions)} questions")
        logger.info("")
    except Exception as e:
        logger.error(f"❌ FATAL: Failed to load questions: {e}", exc_info=True)
        sys.exit(1)

    # Process queries
    logger.info("[3/3] Processing queries...")
    results = []
    error_count = 0

    # For profiling - track time per component
    if args.enable_profiling:
        component_times = defaultdict(list)

    t0_total = time.time()
    for idx, row in tqdm(df_questions.iterrows(), total=len(df_questions), desc="  Searching"):
        q_id = row['q_id']
        query = row['query_normalized']

        try:
            # EGOR'S KILLER FEATURE: Pass q_id for LLM variants lookup
            top_web_ids = pipeline.search(query, q_id=q_id, enable_profiling=args.enable_profiling)
            results.append({
                'q_id': q_id,
                'web_list': str(top_web_ids)
            })
        except Exception as e:
            error_count += 1
            logger.error(f"  ❌ Error processing q_id={q_id}: {e}")
            logger.error(f"     Query: {query[:100]}...")
            # Fallback: return first 5 web_ids
            results.append({
                'q_id': q_id,
                'web_list': str(list(range(1, 6)))
            })

    search_time = time.time() - t0_total
    avg_time = search_time / len(df_questions)
    queries_per_sec = len(df_questions) / search_time

    logger.info("")
    logger.info(f"  ✓ Search completed")
    logger.info(f"    Total time: {format_time(search_time)}")
    logger.info(f"    Avg per query: {avg_time:.3f}s ({avg_time * 1000:.1f}ms)")
    logger.info(f"    Queries/sec: {queries_per_sec:.2f}")
    if error_count > 0:
        logger.warning(f"    ⚠️  Errors: {error_count}/{len(df_questions)}")

    # Estimate time for full dataset
    if args.limit:
        full_dataset_questions = 6977  # Known from previous logs
        estimated_full_time = full_dataset_questions * avg_time
        logger.info("")
        logger.info(f"  📊 Extrapolation to full dataset ({full_dataset_questions} questions):")
        logger.info(f"    Estimated time: {format_time(estimated_full_time)}")
        logger.info(f"    Speedup needed: {estimated_full_time / 3600:.1f}x faster to reach 1 hour")

    logger.info("")

    # Save submission
    logger.info("Saving submission...")
    output_path = Path(args.output)
    df_submit = pd.DataFrame(results)

    try:
        df_submit.to_csv(output_path, index=False)
        logger.info(f"  ✓ Saved to {output_path}")
        logger.info(f"    Rows: {len(df_submit)}")
        logger.info(f"    Size: {output_path.stat().st_size / 1024:.1f} KB")
    except Exception as e:
        logger.error(f"❌ FATAL: Failed to save submission: {e}", exc_info=True)
        sys.exit(1)

    logger.info("")
    logger.info("Sample results:")
    for i in range(min(5, len(df_submit))):
        row = df_submit.iloc[i]
        logger.info(f"  Q{row['q_id']}: {row['web_list']}")

    # Summary
    total_time = time.time() - script_start
    log_summary(logger, "SEARCH COMPLETED", {
        "Total time": format_time(total_time),
        "Questions processed": len(df_questions),
        "Avg time per query": f"{avg_time:.2f}s",
        "Errors": error_count,
        "Output file": str(output_path),
    })

    logger.info("")
    logger.info("✅ Search completed successfully!")
    logger.info(f"   Submission: {output_path}")
    logger.info(f"   Logs: logs/run_search.log")
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
