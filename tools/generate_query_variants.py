#!/usr/bin/env python3
"""
LLM-based Query Variants Generation for RAG Retrieval Enhancement

Generates semantic paraphrases of user queries to improve retrieval recall.
Uses Qwen2.5-7B-Instruct with 4-bit quantization for efficiency.

Usage:
    python tools/generate_query_variants.py \
        --input data/raw/questions_clean.csv \
        --output data/processed/query_variants.parquet \
        --num_variants 2 \
        --batch_size 8 \
        --checkpoint_every 500

Author: Egor's killer feature
"""

import os
import sys
import argparse
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Ты эксперт по банковским услугам и FAQ. Твоя задача — переформулировать вопросы пользователей в более точные запросы для поиска в базе знаний банка.

ПРАВИЛА (СТРОГО СОБЛЮДАЙ):

1. СОХРАНЯЙ СМЫСЛ
   - НЕ меняй тему вопроса
   - НЕ добавляй информацию, которой нет в оригинале
   - НЕ делай запрос слишком общим

2. УЛУЧШАЙ ФОРМУЛИРОВКУ
   ✅ Заменяй сленг: "бабки"→"деньги", "кредитка"→"кредитная карта"
   ✅ Исправляй опечатки и грамматику
   ✅ Делай неявное явным: "снять за границей"→"снятие наличных за границей в банкомате"
   ✅ Разворачивай короткие запросы: "карта банк"→"вопросы по банковским картам"

3. ФОРМАТ ОТВЕТА
   - Ровно 2 варианта переформулировки
   - Каждый на новой строке, начинается с номера
   - Длина: 5-20 слов
   - Без лишнего текста

БАНКОВСКИЙ КОНТЕКСТ:
- Темы: карты, кредиты, вклады, переводы, мобильное приложение, комиссии, валюта, банкоматы
- Формальная лексика: средства, операции, счет, баланс, транзакции

ПРИМЕРЫ:

Вопрос: бабки на карту срочно
1. Как быстро пополнить банковскую карту деньгами
2. Срочное пополнение баланса карты

Вопрос: деньги банк
1. Операции со средствами в банке
2. Вопросы по денежным операциям в банке

Вопрос: процент по вкладу больше
1. Как увеличить процентную ставку по вкладу
2. Вклады с повышенной процентной ставкой

Вопрос: апп не работает
1. Проблемы с работой мобильного приложения банка
2. Мобильное приложение не запускается

Вопрос: кэшбэк хочу больше
1. Как увеличить процент кэшбэка по карте
2. Повышенный кэшбэк по банковским картам"""


class QueryVariantGenerator:
    """LLM-based query paraphrasing for improved retrieval"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        num_variants: int = 2,
        batch_size: int = 8,
        max_new_tokens: int = 80,
        temperature: float = 0.7,
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.num_variants = num_variants
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device

        logger.info(f"Initializing LLM: {model_name}")
        logger.info(f"Using device: {device}")

        # 4-bit quantization config for RTX 4090
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        logger.info("Model loaded successfully!")
        logger.info(f"Model memory: {self.model.get_memory_footprint() / 1e9:.2f} GB")

    def create_prompt(self, query: str) -> str:
        """Create prompt for query paraphrasing"""
        user_prompt = f"Вопрос: {query}\n\nПереформулировки:"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        # Use chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt

    def parse_variants(self, text: str, num_expected: int = 2) -> List[str]:
        """Parse LLM output to extract variants"""
        variants = []

        # Split by lines
        lines = text.strip().split('\n')

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Try to extract variant (format: "1. Text" or "1) Text")
            match = re.match(r'^(\d+)[.)]?\s+(.+)$', line)
            if match:
                variant_text = match.group(2).strip()
                variants.append(variant_text)
            # Also try lines without numbers (fallback)
            elif len(variants) < num_expected and len(line) > 10:
                # Not a meta-text
                if not any(x in line.lower() for x in ['вопрос:', 'ответ:', 'вариант:', 'переформул']):
                    variants.append(line)

        return variants[:num_expected]

    def validate_variant(self, original: str, variant: str) -> bool:
        """Validate generated variant quality"""

        # Basic checks
        if not variant or len(variant.strip()) < 5:
            return False

        if len(variant) > 250:  # Too long
            return False

        # Not duplicate of original
        if variant.lower().strip() == original.lower().strip():
            return False

        # Check for drift (at least some common words)
        orig_words = set(self._tokenize_meaningful(original.lower()))
        var_words = set(self._tokenize_meaningful(variant.lower()))

        if len(orig_words) > 2:
            common_words = orig_words & var_words
            if len(common_words) == 0:
                return False  # Complete drift

        # No garbage patterns
        garbage_patterns = [
            r'http', r'www\.', r'\d{10,}',
            r'ответ:', r'вопрос:', r'вариант:',
            r'переформул',
        ]
        for pattern in garbage_patterns:
            if re.search(pattern, variant.lower()):
                return False

        # Not too verbose
        if len(variant.split()) > len(original.split()) * 4:
            return False

        return True

    def _tokenize_meaningful(self, text: str) -> List[str]:
        """Extract meaningful words (no stopwords)"""
        stopwords = {
            'в', 'на', 'по', 'с', 'для', 'как', 'что', 'и', 'а', 'но',
            'о', 'об', 'из', 'к', 'до', 'от', 'за', 'у', 'же', 'бы'
        }
        words = re.findall(r'\w+', text.lower())
        return [w for w in words if w not in stopwords and len(w) > 2]

    def generate_batch(self, queries: List[str]) -> List[List[str]]:
        """Generate variants for a batch of queries"""

        # Create prompts
        prompts = [self.create_prompt(q) for q in queries]

        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode
        batch_results = []
        for i, output in enumerate(outputs):
            # Extract only generated part
            generated = output[inputs['input_ids'][i].shape[0]:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)

            # Parse variants
            variants = self.parse_variants(text, self.num_variants)

            # Validate
            valid_variants = []
            for variant in variants:
                if self.validate_variant(queries[i], variant):
                    valid_variants.append(variant)

            batch_results.append(valid_variants)

        return batch_results

    def generate_for_dataset(
        self,
        queries_df: pd.DataFrame,
        query_column: str = 'question',
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 500,
        resume: bool = True
    ) -> pd.DataFrame:
        """Generate variants for entire dataset with checkpointing"""

        logger.info(f"Generating variants for {len(queries_df)} queries")
        logger.info(f"Batch size: {self.batch_size}, Num variants: {self.num_variants}")

        # Check for existing checkpoint
        start_idx = 0
        results = []

        if resume and checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint_df = pd.read_parquet(checkpoint_path)
            results = checkpoint_df.to_dict('records')
            start_idx = len(results)
            logger.info(f"Resuming from index {start_idx}")

        # Process in batches
        num_batches = (len(queries_df) - start_idx + self.batch_size - 1) // self.batch_size

        with tqdm(total=len(queries_df) - start_idx, desc="Generating variants") as pbar:
            for batch_start in range(start_idx, len(queries_df), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(queries_df))
                batch_df = queries_df.iloc[batch_start:batch_end]

                # Get queries
                batch_queries = batch_df[query_column].tolist()

                # Generate variants
                try:
                    batch_variants = self.generate_batch(batch_queries)
                except Exception as e:
                    logger.error(f"Error in batch {batch_start}-{batch_end}: {e}")
                    # Use empty variants on error
                    batch_variants = [[] for _ in batch_queries]

                # Store results
                for idx, (row_idx, row) in enumerate(batch_df.iterrows()):
                    variants = batch_variants[idx]

                    result = {
                        'q_id': row.get('q_id', row_idx),
                        'original_query': row[query_column],
                        'variant_1': variants[0] if len(variants) > 0 else None,
                        'variant_2': variants[1] if len(variants) > 1 else None,
                        'variant_3': variants[2] if len(variants) > 2 else None,
                        'num_valid_variants': len(variants),
                    }
                    results.append(result)

                pbar.update(batch_end - batch_start)

                # Checkpoint
                if checkpoint_path and (batch_end - start_idx) % checkpoint_every == 0:
                    temp_df = pd.DataFrame(results)
                    temp_df.to_parquet(checkpoint_path, index=False)
                    logger.info(f"Checkpoint saved: {len(results)} queries processed")

        # Final dataframe
        final_df = pd.DataFrame(results)

        # Stats
        total_variants = final_df['num_valid_variants'].sum()
        avg_variants = final_df['num_valid_variants'].mean()
        logger.info(f"Generation complete!")
        logger.info(f"Total variants generated: {total_variants}")
        logger.info(f"Average variants per query: {avg_variants:.2f}")
        logger.info(f"Queries with 0 variants: {(final_df['num_valid_variants'] == 0).sum()}")
        logger.info(f"Queries with 1 variant: {(final_df['num_valid_variants'] == 1).sum()}")
        logger.info(f"Queries with 2+ variants: {(final_df['num_valid_variants'] >= 2).sum()}")

        return final_df


def show_samples(df: pd.DataFrame, n: int = 10):
    """Show sample variants for quality check"""
    logger.info(f"\n{'='*80}")
    logger.info("SAMPLE VARIANTS (for manual quality check):")
    logger.info(f"{'='*80}\n")

    samples = df.head(n)
    for idx, row in samples.iterrows():
        print(f"[{idx}] Original: {row['original_query']}")
        if row['variant_1']:
            print(f"    → Variant 1: {row['variant_1']}")
        if row['variant_2']:
            print(f"    → Variant 2: {row['variant_2']}")
        if row['variant_3']:
            print(f"    → Variant 3: {row['variant_3']}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Generate query variants using LLM")
    parser.add_argument('--input', type=str, default='data/raw/questions_clean.csv',
                        help='Input CSV with queries')
    parser.add_argument('--output', type=str, default='data/processed/query_variants.parquet',
                        help='Output parquet file')
    parser.add_argument('--query_column', type=str, default='query',
                        help='Column name containing queries')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='HuggingFace model name')
    parser.add_argument('--num_variants', type=int, default=2,
                        help='Number of variants per query')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for generation')
    parser.add_argument('--max_new_tokens', type=int, default=80,
                        help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--checkpoint_every', type=int, default=500,
                        help='Save checkpoint every N queries')
    parser.add_argument('--no_resume', action='store_true',
                        help='Do not resume from checkpoint')
    parser.add_argument('--show_samples', type=int, default=15,
                        help='Number of samples to show at end')

    args = parser.parse_args()

    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA not available! This script requires GPU.")
        sys.exit(1)

    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load input data
    logger.info(f"Loading input data: {args.input}")
    queries_df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(queries_df)} queries")

    if args.query_column not in queries_df.columns:
        logger.error(f"Column '{args.query_column}' not found in input CSV")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint path
    checkpoint_path = str(output_path.parent / f"{output_path.stem}_checkpoint.parquet")

    # Initialize generator
    generator = QueryVariantGenerator(
        model_name=args.model_name,
        num_variants=args.num_variants,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # Generate variants
    results_df = generator.generate_for_dataset(
        queries_df,
        query_column=args.query_column,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
    )

    # Save final results
    logger.info(f"Saving results to: {args.output}")
    results_df.to_parquet(args.output, index=False)

    # Show samples
    if args.show_samples > 0:
        show_samples(results_df, n=args.show_samples)

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info("Checkpoint cleaned up")

    logger.info("✅ Done!")


if __name__ == '__main__':
    main()
