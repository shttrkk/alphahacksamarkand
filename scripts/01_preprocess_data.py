#!/usr/bin/env python3
"""
Step 1: Preprocess data
- Load websites_updated.csv and questions_clean.csv
- Normalize text
- Extract entities and intents
- Save processed data
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from tqdm import tqdm
import time

from config.settings import (
    WEBSITES_CSV,
    QUESTIONS_CSV,
    PROCESSED_DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
)
from src.preprocessing import (
    BankingTextNormalizer,
    EntityExtractor,
    IntentClassifier,
    BankingTokenizer,
)
from src.utils.logger import setup_logger, log_section, log_summary, format_time


# ============================================================================
# Setup logging
# ============================================================================
logger = setup_logger("preprocess", log_file="logs/preprocess.log")


def create_chunks(text: str, chunk_size: int, overlap: int, min_size: int):
    """Simple sliding window chunking."""
    if not text or len(text) < min_size:
        return [text] if text else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if len(chunk) >= min_size:
            chunks.append(chunk)

        # Sliding window with overlap
        start = start + chunk_size - overlap

        if start >= len(text):
            break

    return chunks if chunks else [text]


def main():
    script_start = time.time()

    log_section(logger, "ALPHA RAG ELITE - DATA PREPROCESSING")
    logger.info("Config:")
    logger.info(f"  Chunk size: {CHUNK_SIZE}")
    logger.info(f"  Chunk overlap: {CHUNK_OVERLAP}")
    logger.info(f"  Min chunk size: {MIN_CHUNK_SIZE}")
    logger.info("")

    # ========================================================================
    # Проверка входных файлов
    # ========================================================================
    logger.info("[0/5] Checking input files...")

    if not WEBSITES_CSV.exists():
        logger.error(f"❌ FATAL: websites_updated.csv not found!")
        logger.error(f"   Expected path: {WEBSITES_CSV}")
        logger.error(f"   Please put the file in data/raw/")
        sys.exit(1)

    if not QUESTIONS_CSV.exists():
        logger.error(f"❌ FATAL: questions_clean.csv not found!")
        logger.error(f"   Expected path: {QUESTIONS_CSV}")
        logger.error(f"   Please put the file in data/raw/")
        sys.exit(1)

    logger.info(f"✓ Found: {WEBSITES_CSV}")
    logger.info(f"  Size: {WEBSITES_CSV.stat().st_size / 1024:.1f} KB")

    logger.info(f"✓ Found: {QUESTIONS_CSV}")
    logger.info(f"  Size: {QUESTIONS_CSV.stat().st_size / 1024:.1f} KB")

    # Создание директорий
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Output directory: {PROCESSED_DATA_DIR}")
    logger.info("")

    # ========================================================================
    # Инициализация компонентов
    # ========================================================================
    logger.info("[1/5] Initializing components...")
    normalizer = BankingTextNormalizer()
    entity_extractor = EntityExtractor()
    intent_classifier = IntentClassifier()
    logger.info("✓ Components initialized")
    logger.info("")

    # ========================================================================
    # ОБРАБОТКА ДОКУМЕНТОВ
    # ========================================================================
    logger.info("[2/5] Processing documents...")
    t0 = time.time()

    try:
        df_docs = pd.read_csv(WEBSITES_CSV)
    except Exception as e:
        logger.error(f"❌ FATAL: Failed to read {WEBSITES_CSV}: {e}", exc_info=True)
        sys.exit(1)

    if len(df_docs) == 0:
        logger.error(f"❌ FATAL: {WEBSITES_CSV} is empty!")
        sys.exit(1)

    logger.info(f"  Loaded {len(df_docs)} documents")
    logger.info(f"  Columns: {list(df_docs.columns)}")

    # Проверка обязательных колонок
    required_cols = ['web_id']
    missing_cols = [col for col in required_cols if col not in df_docs.columns]
    if missing_cols:
        logger.error(f"❌ FATAL: Missing required columns: {missing_cols}")
        sys.exit(1)

    # Normalize documents
    doc_data = []
    for idx, row in tqdm(df_docs.iterrows(), total=len(df_docs), desc="  Normalizing docs"):
        web_id = int(row['web_id'])
        title = normalizer.normalize(str(row.get('title', '')))
        text = normalizer.normalize(str(row.get('text', '')))

        # Combine title and text
        combined = f"{title}. {text}" if title else text

        # Extract entities and intents from document
        entities = entity_extractor.extract(combined)
        intents = intent_classifier.classify(combined)

        doc_data.append({
            'web_id': web_id,
            'title': title,
            'text': text,
            'combined': combined,
            'text_length': len(combined),
            'has_bik': len(entities['bik']) > 0,
            'has_rs': len(entities['rs']) > 0,
            'has_phone': len(entities['phone']) > 0,
            'num_intents': len(intents),
            'intents': '|'.join(sorted(intents)) if intents else '',
        })

    df_docs_processed = pd.DataFrame(doc_data)

    # Save
    docs_path = PROCESSED_DATA_DIR / "documents_normalized.parquet"
    try:
        df_docs_processed.to_parquet(docs_path, index=False)
        logger.info(f"✓ Saved to {docs_path}")
        logger.info(f"  Rows: {len(df_docs_processed)}")
        logger.info(f"  Size: {docs_path.stat().st_size / 1024:.1f} KB")
    except Exception as e:
        logger.error(f"❌ FATAL: Failed to save {docs_path}: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"  Time: {format_time(time.time() - t0)}")

    # Примеры обработанных документов
    logger.info(f"\n  Sample processed documents:")
    for i in range(min(3, len(df_docs_processed))):
        row = df_docs_processed.iloc[i]
        logger.info(f"    [{i+1}] web_id={row['web_id']}, length={row['text_length']}, "
                   f"intents={row['num_intents']}, title='{row['title'][:50]}...'")
    logger.info("")

    # ========================================================================
    # СОЗДАНИЕ ЧАНКОВ
    # ========================================================================
    logger.info("[3/5] Creating chunks...")
    t0 = time.time()

    chunks_data = []
    chunk_id = 0

    for idx, row in tqdm(df_docs_processed.iterrows(), total=len(df_docs_processed), desc="  Chunking"):
        web_id = row['web_id']
        title = row['title']
        combined = row['combined']

        # Create chunks
        text_chunks = create_chunks(
            combined,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            min_size=MIN_CHUNK_SIZE,
        )

        for chunk_num, chunk_text in enumerate(text_chunks):
            chunks_data.append({
                'chunk_id': chunk_id,
                'web_id': web_id,
                'chunk_num': chunk_num,
                'title': title,
                'chunk_text': chunk_text,
                'chunk_length': len(chunk_text),
            })
            chunk_id += 1

    df_chunks = pd.DataFrame(chunks_data)

    if len(df_chunks) == 0:
        logger.error(f"❌ FATAL: No chunks created!")
        sys.exit(1)

    chunks_path = PROCESSED_DATA_DIR / "chunks.parquet"
    try:
        df_chunks.to_parquet(chunks_path, index=False)
        logger.info(f"✓ Created {len(df_chunks)} chunks from {len(df_docs_processed)} documents")
        logger.info(f"  Avg chunks per doc: {len(df_chunks) / len(df_docs_processed):.1f}")
        logger.info(f"  Saved to {chunks_path}")
        logger.info(f"  Size: {chunks_path.stat().st_size / 1024:.1f} KB")
    except Exception as e:
        logger.error(f"❌ FATAL: Failed to save {chunks_path}: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"  Time: {format_time(time.time() - t0)}")

    # Примеры чанков
    logger.info(f"\n  Sample chunks:")
    for i in range(min(3, len(df_chunks))):
        row = df_chunks.iloc[i]
        logger.info(f"    [{i+1}] chunk_id={row['chunk_id']}, web_id={row['web_id']}, "
                   f"length={row['chunk_length']}, text='{row['chunk_text'][:60]}...'")
    logger.info("")

    # ========================================================================
    # ОБРАБОТКА ВОПРОСОВ
    # ========================================================================
    logger.info("[4/5] Processing questions...")
    t0 = time.time()

    try:
        df_questions = pd.read_csv(QUESTIONS_CSV)
    except Exception as e:
        logger.error(f"❌ FATAL: Failed to read {QUESTIONS_CSV}: {e}", exc_info=True)
        sys.exit(1)

    if len(df_questions) == 0:
        logger.error(f"❌ FATAL: {QUESTIONS_CSV} is empty!")
        sys.exit(1)

    logger.info(f"  Loaded {len(df_questions)} questions")

    question_data = []
    for idx, row in tqdm(df_questions.iterrows(), total=len(df_questions), desc="  Normalizing queries"):
        q_id = int(row['q_id'])
        query = str(row['query'])

        # Normalize query (more gentle than documents)
        query_normalized = normalizer.normalize_query(query)

        # Extract entities and intents
        entities = entity_extractor.extract(query_normalized)
        intents = intent_classifier.classify(query_normalized)

        question_data.append({
            'q_id': q_id,
            'query_original': query,
            'query_normalized': query_normalized,
            'has_bik': len(entities['bik']) > 0,
            'has_rs': len(entities['rs']) > 0,
            'has_phone': len(entities['phone']) > 0,
            'num_intents': len(intents),
            'intents': '|'.join(sorted(intents)) if intents else '',
        })

    df_questions_processed = pd.DataFrame(question_data)

    questions_path = PROCESSED_DATA_DIR / "questions_processed.parquet"
    try:
        df_questions_processed.to_parquet(questions_path, index=False)
        logger.info(f"✓ Saved to {questions_path}")
        logger.info(f"  Rows: {len(df_questions_processed)}")
        logger.info(f"  Size: {questions_path.stat().st_size / 1024:.1f} KB")
    except Exception as e:
        logger.error(f"❌ FATAL: Failed to save {questions_path}: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"  Time: {format_time(time.time() - t0)}")

    # Примеры вопросов
    logger.info(f"\n  Sample questions:")
    for i in range(min(3, len(df_questions_processed))):
        row = df_questions_processed.iloc[i]
        logger.info(f"    [{i+1}] q_id={row['q_id']}, intents={row['num_intents']}, "
                   f"query='{row['query_normalized'][:60]}...'")
    logger.info("")

    # ========================================================================
    # СТАТИСТИКА
    # ========================================================================
    logger.info("[5/5] Statistics:")
    logger.info(f"  Documents: {len(df_docs_processed)}")
    logger.info(f"    - With BIK: {df_docs_processed['has_bik'].sum()}")
    logger.info(f"    - With р/с: {df_docs_processed['has_rs'].sum()}")
    logger.info(f"    - With phone: {df_docs_processed['has_phone'].sum()}")
    logger.info(f"    - Avg text length: {df_docs_processed['text_length'].mean():.0f} chars")

    logger.info(f"\n  Questions: {len(df_questions_processed)}")
    logger.info(f"    - With BIK: {df_questions_processed['has_bik'].sum()}")
    logger.info(f"    - With р/с: {df_questions_processed['has_rs'].sum()}")
    logger.info(f"    - With phone: {df_questions_processed['has_phone'].sum()}")

    logger.info(f"\n  Chunks: {len(df_chunks)}")
    logger.info(f"    - Avg chunk length: {df_chunks['chunk_length'].mean():.0f} chars")
    logger.info(f"    - Min/Max: {df_chunks['chunk_length'].min()}/{df_chunks['chunk_length'].max()}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    total_time = time.time() - script_start
    log_summary(logger, "PREPROCESSING COMPLETED", {
        "Total time": format_time(total_time),
        "Documents processed": len(df_docs_processed),
        "Chunks created": len(df_chunks),
        "Questions processed": len(df_questions_processed),
        "Output files": 3,
    })

    logger.info(f"\nOutput files:")
    logger.info(f"  ✓ {docs_path}")
    logger.info(f"  ✓ {chunks_path}")
    logger.info(f"  ✓ {questions_path}")
    logger.info("")
    logger.info("✅ Preprocessing completed successfully!")
    logger.info(f"   Logs saved to: logs/preprocess.log")
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
