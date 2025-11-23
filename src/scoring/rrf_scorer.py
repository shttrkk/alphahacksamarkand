"""
Weighted Reciprocal Rank Fusion (RRF).
"""
from typing import Dict, List
from collections import defaultdict


def weighted_rrf(
    rankings: Dict[str, List[int]],
    weights: Dict[str, float],
    k: int = 60,
) -> Dict[int, float]:
    """
    Weighted Reciprocal Rank Fusion.

    Args:
        rankings: Dict[retriever_name -> ranked_doc_ids]
                 Для каждого retriever - список doc_id в порядке убывания релевантности
        weights: Dict[retriever_name -> weight]
                 Веса для каждого retriever
        k: RRF constant (обычно 60)

    Returns:
        Dict[doc_id -> final_score]
    """
    scores = defaultdict(float)

    for retriever_name, ranked_ids in rankings.items():
        weight = weights.get(retriever_name, 1.0)

        for rank, doc_id in enumerate(ranked_ids):
            # RRF formula: weight / (k + rank)
            # rank starts from 0, so rank 0 = position 1
            scores[doc_id] += weight / (k + rank)

    return dict(scores)


def simple_rrf(
    rankings: List[List[int]],
    k: int = 60,
) -> Dict[int, float]:
    """
    Простой RRF без весов (все retriever'ы с весом 1.0).

    Args:
        rankings: List of ranked doc_id lists
        k: RRF constant

    Returns:
        Dict[doc_id -> score]
    """
    scores = defaultdict(float)

    for ranked_ids in rankings:
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] += 1.0 / (k + rank)

    return dict(scores)


def combine_scores(
    score_dicts: List[Dict[int, float]],
    weights: List[float] = None,
) -> Dict[int, float]:
    """
    Комбинирует несколько словарей скоров с весами.

    Args:
        score_dicts: Список Dict[doc_id -> score]
        weights: Веса для каждого словаря (если None, все веса = 1.0)

    Returns:
        Dict[doc_id -> combined_score]
    """
    if weights is None:
        weights = [1.0] * len(score_dicts)

    combined = defaultdict(float)

    for score_dict, weight in zip(score_dicts, weights):
        for doc_id, score in score_dict.items():
            combined[doc_id] += weight * score

    return dict(combined)
