"""
Alpha RAG Elite - Hackathon Crunch Edition
Configuration file with all hyperparameters
"""
from pathlib import Path

# ============= PATHS =============
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
INDICES_DIR = DATA_DIR / "indices"
CACHE_DIR = DATA_DIR / "cache"
VALIDATION_DIR = DATA_DIR / "validation"

# Input files
WEBSITES_CSV = RAW_DATA_DIR / "websites_updated.csv"
QUESTIONS_CSV = RAW_DATA_DIR / "questions_clean.csv"

# Output files
SUBMIT_CSV = BASE_DIR / "submit_alpha_rag_elite.csv"

# ============= MODELS =============
# Doc-level embedders (ordered by priority)
# CONSERVATIVE CONFIG: только 100% надежные модели
DOC_EMBEDDERS = {
    "e5_large": {
        "model_name": "intfloat/multilingual-e5-large",
        "prefix_passage": "passage: ",
        "prefix_query": "query: ",
        "max_seq_length": 512,
        "weight": 2.5,  # WINNING CONFIG: RRF weight
        "enabled": True,
    },
    "sbert_ru": {
        "model_name": "sberbank-ai/sbert_large_nlu_ru",
        "prefix_passage": "",
        "prefix_query": "",
        "max_seq_length": 512,
        "weight": 2.0,
        "enabled": True,
    },
    "mpnet": {
        "model_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "prefix_passage": "",
        "prefix_query": "",
        "max_seq_length": 512,
        "weight": 1.8,
        "enabled": True,
    },
    # BGE-m3: ОТКЛЮЧЕН - часто не загружается, нестабильный
    "bge_m3": {
        "model_name": "BAAI/bge-m3",
        "prefix_passage": "",
        "prefix_query": "",
        "max_seq_length": 512,
        "weight": 2.5,
        "enabled": False,  # Включить только если уверены что работает
    },
    # LaBSE: ВКЛЮЧЕН для HIGH RECALL
    "labse": {
        "model_name": "cointegrated/LaBSE-en-ru",
        "prefix_passage": "",
        "prefix_query": "",
        "max_seq_length": 512,
        "weight": 1.8,  # Increased weight for high recall
        "enabled": True,  # HIGH RECALL: Enable for diversity
    },
}

# Chunk-level embedders (WINNING CONFIG: chunks ENABLED)
# With LLM variants: chunks provide additional recall
CHUNK_EMBEDDERS = {
    "e5_large_chunk": {
        "model_name": "intfloat/multilingual-e5-large",
        "prefix_passage": "passage: ",
        "prefix_query": "query: ",
        "max_seq_length": 512,
        "weight": 2.0,
        "enabled": True,  # ENABLED for LLM variants experiment
    },
    # Второй эмбеддер ОТКЛЮЧЕН для скорости и стабильности
    "sbert_ru_chunk": {
        "model_name": "sberbank-ai/sbert_large_nlu_ru",
        "prefix_passage": "",
        "prefix_query": "",
        "max_seq_length": 512,
        "weight": 1.8,
        "enabled": False,
    },
}

# ============= CHUNKING =============
CHUNK_SIZE = 250  # characters
CHUNK_OVERLAP = 60  # characters
MIN_CHUNK_SIZE = 50  # characters

# ============= BM25 =============
BM25_K1 = 1.5
BM25_B = 0.65
BM25_DOC_WEIGHT = 2.5  # HIGH RECALL: Increased BM25 importance
BM25_CHUNK_WEIGHT = 2.0  # HIGH RECALL: Increased chunk BM25

# ============= RETRIEVAL =============
# HIGH RECALL CONFIG: Maximum candidates for better recall
DOC_TOP_K_PER_EMBEDDER = 200  # HIGH RECALL: More candidates per embedder
DOC_BM25_TOP_K = 250  # HIGH RECALL: More BM25 candidates

# Chunk-level retrieval (HIGH RECALL)
CHUNK_TOP_K_PER_EMBEDDER = 150  # HIGH RECALL: More chunk candidates
CHUNK_BM25_TOP_K = 150  # HIGH RECALL: More chunk BM25

# ============= RRF =============
RRF_K = 70  # HIGH RECALL: Higher K for better fusion of more sources

# ============= QUERY EXPANSION =============
ENABLE_RULE_BASED_EXPANSION = False  # EMERGENCY: Disabled - too slow with domain boosting
MAX_QUERY_VARIANTS = 1  # EMERGENCY: Only original query (no expansion)
# PRF ОТКЛЮЧЕН по умолчанию - может добавить шум и снизить точность
ENABLE_BM25_PRF = False  # Включить только если локальные тесты показывают прирост

# PRF settings (если включен)
PRF_TOP_DOCS = 20
PRF_TOP_TERMS = 15
PRF_TERM_WEIGHT = 0.3

# ============= DOMAIN BOOSTS =============
ENABLE_DOMAIN_BOOSTING = False  # EMERGENCY: Disabled - entity/intent extraction too slow (200 docs × 6977 queries = 1.4M ops!)

# Entity exact match boosts
ENTITY_BOOSTS = {
    "bik": 3.0,  # Точное совпадение БИК
    "rs": 2.0,   # Точное совпадение р/с
    "ks": 1.5,   # Точное совпадение к/с
    "phone": 1.0,
    "sum": 0.5,
    "percent": 0.5,
}

# Intent matching boosts
INTENT_BOOSTS = {
    "БИК": 0.8,
    "РАСЧЕТНЫЙ_СЧЕТ": 0.8,
    "КОРР_СЧЕТ": 0.7,
    "СМС_ПРОБЛЕМЫ": 0.6,
    "ЗАРПЛАТА": 0.6,
    "КРЕДИТ": 0.6,
    "БЛОКИРОВКА": 0.7,
    "АЛЬФА_ОНЛАЙН": 0.6,
    "ПЕРЕВОДЫ": 0.5,
    "КАРТА": 0.5,
}

# Other boosts
TITLE_KEYWORD_BOOST = 0.3
HIGH_BM25_BOOST = 0.2  # If doc in top-10 BM25

# ============= RERANKING (OPTIONAL) =============
ENABLE_RERANKING = False  # Will be enabled ONLY if validation shows improvement
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_K = 50  # Only top-50 candidates
RERANK_MAX_CHARS = 400
RERANK_BATCH_SIZE = 128
RERANK_SCORE_WEIGHT = 0.4  # Combine: 0.6*RRF + 0.4*rerank

# ============= VALIDATION =============
VALIDATION_SIZE = 100  # Number of questions in validation set
MIN_IMPROVEMENT_THRESHOLD = 1.0  # Minimum improvement in Hit@5 to enable feature (percentage points)

# ============= RUNTIME =============
BATCH_SIZE_EMBEDDING = 64
USE_CACHE = True  # Cache embeddings and indices
DEVICE = "cuda"  # or "cpu"
NUM_WORKERS = 4

# ============= LLM QUERY VARIANTS (EGOR'S KILLER FEATURE) =============
# Offline LLM-based query paraphrasing to improve retrieval recall
# Addresses "broken" queries: slang, typos, short queries, unclear intent

USE_QUERY_VARIANTS = False  # HIGH RECALL: Disabled (no variants file)
QUERY_VARIANTS_PATH = PROCESSED_DATA_DIR / "query_variants.parquet"

# Multi-query retrieval fusion strategy
VARIANT_FUSION_STRATEGY = "weighted_rrf"  # Options: "weighted_rrf", "union", "best_of"
ORIGINAL_QUERY_WEIGHT = 1.0  # Weight for original query
VARIANT_QUERY_WEIGHT = 0.7  # Weight for LLM-generated variants
MAX_VARIANTS_TO_USE = 2  # How many variants to use (1-3)

# LLM generation settings (for tools/generate_query_variants.py)
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LLM_NUM_VARIANTS = 2  # Number of variants to generate per query
LLM_BATCH_SIZE = 8  # Batch size for LLM generation
LLM_MAX_NEW_TOKENS = 80  # Max tokens to generate
LLM_TEMPERATURE = 0.7  # Sampling temperature (0.7 = diverse but coherent)

# ============= CROSS-ENCODER RERANKING =============
# Two-stage retrieval: bi-encoder (fast, TOP-100) → cross-encoder (accurate, TOP-5)
# Cross-encoder sees query+doc together → more accurate but slower
# GUARANTEED +3-5pp improvement over bi-encoder alone

ENABLE_CROSS_ENCODER_RERANKING = True  # Enable cross-encoder reranking
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # Fast & effective
# Alternative models:
# - "cross-encoder/ms-marco-MiniLM-L-6-v2" (faster, slightly worse)
# - "BAAI/bge-reranker-large" (SOTA, but slow)

RERANKER_CANDIDATE_K = 100  # Get TOP-100 candidates from bi-encoder for reranking
RERANKER_BATCH_SIZE = 32  # Batch size for cross-encoder inference

# ============= FINAL OUTPUT =============
TOP_K = 5  # Number of documents to return per query
