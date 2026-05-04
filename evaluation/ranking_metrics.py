#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ranking_metrics.py
===================

This module defines a set of common ranking evaluation metrics. These
functions are intended for use with recommendation systems or any
application where a model returns a ranked list of items and a set of
relevant (ground truth) items is known. Each function is documented
with its intended use and return value.

Functions
---------

precision_at_k(recs, gt, k)
    Computes the fraction of the top-k recommendations that are
    relevant.

recall_at_k(recs, gt, k)
    Computes the fraction of the relevant items that appear in the
    top-k recommendations.

hit_at_k(recs, gt, k)
    Returns 1 if at least one relevant item appears in the top-k
    recommendations, else 0.

ndcg_at_k(recs, gt, k)
    Computes the Normalised Discounted Cumulative Gain at k assuming
    binary relevance.

mrr_at_k(recs, gt, k)
    Computes the Mean Reciprocal Rank of the first relevant item
    appearing in the top-k recommendations.
"""

from math import log2
from typing import Iterable, List


def precision_at_k(recs: Iterable[str], gt: Iterable[str], k: int) -> float:
    """Compute Precision@K.

    Precision@K is the proportion of the first ``k`` recommended items
    that are present in the set of ground truth items. If ``k`` is
    non-positive this function returns 0.0. The order of the ground
    truth items does not matter.

    Parameters
    ----------
    recs : iterable of str
        The ranked list of recommendations.
    gt : iterable of str
        The set of relevant items (ground truth).
    k : int
        The cutoff rank. Only the first ``k`` recommendations are
        considered.

    Returns
    -------
    float
        The precision value between 0 and 1.
    """
    if k <= 0:
        return 0.0
    recs_list: List[str] = list(recs)
    gt_set = set(gt)
    hits = len(set(recs_list[:k]).intersection(gt_set))
    return hits / float(k)


def recall_at_k(recs: Iterable[str], gt: Iterable[str], k: int) -> float:
    """Compute Recall@K.

    Recall@K is the proportion of relevant items that appear in the
    first ``k`` recommendations. If the ground truth set is empty this
    function returns 0.0. If ``k`` is larger than the number of
    recommendations the function will consider all available
    recommendations.

    Parameters
    ----------
    recs : iterable of str
        The ranked list of recommendations.
    gt : iterable of str
        The set of relevant items (ground truth).
    k : int
        The cutoff rank.

    Returns
    -------
    float
        The recall value between 0 and 1.
    """
    gt_set = set(gt)
    if not gt_set:
        return 0.0
    recs_list: List[str] = list(recs)
    hits = len(set(recs_list[:k]).intersection(gt_set))
    return hits / float(len(gt_set))


def hit_at_k(recs: Iterable[str], gt: Iterable[str], k: int) -> int:
    """Return 1 if at least one relevant item is in the top-k recommendations.

    This is a binary metric indicating whether the recommendation list
    successfully retrieved at least one item from the ground truth set
    within the first ``k`` positions. It returns ``1`` if there is a
    hit and ``0`` otherwise.

    Parameters
    ----------
    recs : iterable of str
        The ranked list of recommendations.
    gt : iterable of str
        The set of relevant items (ground truth).
    k : int
        The cutoff rank.

    Returns
    -------
    int
        ``1`` if there is a hit, otherwise ``0``.
    """
    recs_list: List[str] = list(recs)
    gt_set = set(gt)
    return 1 if len(set(recs_list[:k]).intersection(gt_set)) > 0 else 0


def dcg_at_k(recs: Iterable[str], gt: Iterable[str], k: int) -> float:
    """Compute the Discounted Cumulative Gain (DCG) at K for binary relevance.

    DCG rewards relevant items that appear higher in the ranking more
    than items that appear later. It does not normalise the value.

    Parameters
    ----------
    recs : iterable of str
        The ranked list of recommendations.
    gt : iterable of str
        The set of relevant items (ground truth).
    k : int
        The cutoff rank.

    Returns
    -------
    float
        The DCG value.
    """
    recs_list: List[str] = list(recs)
    gt_set = set(gt)
    score = 0.0
    for idx, item in enumerate(recs_list[:k]):
        if item in gt_set:
            # The first position (idx = 0) uses log2(2) = 1 for division
            score += 1.0 / log2(idx + 2)
    return score


def ndcg_at_k(recs: Iterable[str], gt: Iterable[str], k: int) -> float:
    """Compute the Normalised Discounted Cumulative Gain (NDCG) at K.

    NDCG normalises DCG by the ideal DCG (IDCG), which is the DCG
    obtained when all relevant items are ranked at the top. This
    implementation assumes binary relevance (each relevant item
    contributes equally to the gain).

    Parameters
    ----------
    recs : iterable of str
        The ranked list of recommendations.
    gt : iterable of str
        The set of relevant items (ground truth).
    k : int
        The cutoff rank.

    Returns
    -------
    float
        The NDCG value between 0 and 1. If there are no relevant items
        ``0.0`` is returned.
    """
    gt_set = set(gt)
    ideal_hits = min(len(gt_set), k)
    if ideal_hits == 0:
        return 0.0
    dcg = dcg_at_k(recs, gt, k)
    # Compute the ideal DCG (IDCG) where all relevant items are ranked at the top
    idcg = sum(1.0 / log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(recs: Iterable[str], gt: Iterable[str], k: int) -> float:
    """Compute the Mean Reciprocal Rank (MRR) at K.

    The MRR measures the inverse rank of the first relevant item in the
    recommendation list. If no relevant item appears in the top ``k``
    positions then the metric returns ``0.0``.

    Parameters
    ----------
    recs : iterable of str
        The ranked list of recommendations.
    gt : iterable of str
        The set of relevant items (ground truth).
    k : int
        The cutoff rank.

    Returns
    -------
    float
        The reciprocal rank of the first relevant item or 0.0 if none
        are found in the top ``k``.
    """
    gt_set = set(gt)
    for idx, item in enumerate(list(recs)[:k]):
        if item in gt_set:
            return 1.0 / float(idx + 1)
    return 0.0
