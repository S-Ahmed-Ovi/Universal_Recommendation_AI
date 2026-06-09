"""
utils/metrics.py – Recommendation evaluation metrics.

Implements:
  - RMSE / MAE          for explicit-feedback (rating prediction)
  - NDCG@K              normalised discounted cumulative gain   (ranking)
  - Precision@K         fraction of top-K that are relevant    (ranking)
  - Recall@K            fraction of relevant items retrieved   (ranking)

All ranking metrics treat every item the user interacted with in the test set
as "relevant". ranking_preds is expected to be a DataFrame with columns
[userID, itemID, score], already limited to the top-K items per user and
sorted descending by score (as produced by InteractionMatrixBuilder).
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Explicit-feedback metrics (rating prediction)
# ---------------------------------------------------------------------------

def evaluate_explicit(
    test: Any,
    rating_preds: Any,
) -> dict[str, float]:
    """
    Compute RMSE and MAE from rating predictions.

    Args:
        test:         DataFrame with columns [userID, itemID, rating]
        rating_preds: DataFrame with columns [userID, itemID, prediction]

    Returns zero for both metrics when predictions are unavailable.
    """
    if (
        not isinstance(rating_preds, pd.DataFrame)
        or rating_preds.empty
        or not isinstance(test, pd.DataFrame)
        or test.empty
        or "prediction" not in rating_preds.columns
        or "rating" not in test.columns
    ):
        return {"RMSE": 0.0, "MAE": 0.0}

    merged = test.merge(
        rating_preds[["userID", "itemID", "prediction"]],
        on=["userID", "itemID"],
        how="inner",
    )
    if merged.empty:
        return {"RMSE": 0.0, "MAE": 0.0}

    diff = merged["rating"].astype(float) - merged["prediction"].astype(float)
    rmse = float(math.sqrt((diff ** 2).mean()))
    mae  = float(diff.abs().mean())
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4)}


# ---------------------------------------------------------------------------
# Ranking metrics (implicit / NDCG-based)
# ---------------------------------------------------------------------------

def evaluate_ranking(
    test: Any,
    ranking_preds: Any,
    top_k: int,
) -> dict[str, float]:
    """
    Compute NDCG@K, Precision@K, and Recall@K averaged over all test users.

    Args:
        test:          DataFrame with columns [userID, itemID, ...]
                       Every (userID, itemID) pair is treated as relevant.
        ranking_preds: DataFrame with columns [userID, itemID, score],
                       top-K items per user sorted descending by score.
        top_k:         Cutoff K.

    Returns zeros when inputs are invalid or empty.
    """
    if (
        not isinstance(ranking_preds, pd.DataFrame)
        or ranking_preds.empty
        or not isinstance(test, pd.DataFrame)
        or test.empty
    ):
        return {"NDCG@K": 0.0, "Precision@K": 0.0, "Recall@K": 0.0}

    # Build ground-truth: userID -> set of relevant itemIDs
    relevant: dict[str, set] = (
        test.groupby("userID")["itemID"]
        .apply(set)
        .to_dict()
    )

    # Build predictions: userID -> ordered list of predicted itemIDs
    preds_by_user: dict[str, list] = (
        ranking_preds.sort_values("score", ascending=False)
        .groupby("userID")["itemID"]
        .apply(list)
        .to_dict()
    )

    # Ideal DCG for a user who has `n_rel` relevant items, capped at top_k
    def _idcg(n_rel: int) -> float:
        return sum(
            1.0 / math.log2(i + 2)
            for i in range(min(n_rel, top_k))
        )

    ndcg_scores:      list[float] = []
    precision_scores: list[float] = []
    recall_scores:    list[float] = []

    # Only evaluate users that appear in both test and predictions
    eval_users = set(relevant.keys()) & set(preds_by_user.keys())
    if not eval_users:
        return {"NDCG@K": 0.0, "Precision@K": 0.0, "Recall@K": 0.0}

    for user in eval_users:
        rel   = relevant[user]
        recs  = preds_by_user[user][:top_k]
        n_rel = len(rel)

        if n_rel == 0:
            continue

        hits = [1 if item in rel else 0 for item in recs]
        n_hits = sum(hits)

        # NDCG@K
        dcg  = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
        idcg = _idcg(n_rel)
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

        # Precision@K
        precision_scores.append(n_hits / top_k)

        # Recall@K
        recall_scores.append(n_hits / n_rel)

    if not ndcg_scores:
        return {"NDCG@K": 0.0, "Precision@K": 0.0, "Recall@K": 0.0}

    return {
        "NDCG@K":      round(sum(ndcg_scores)      / len(ndcg_scores),      4),
        "Precision@K": round(sum(precision_scores)  / len(precision_scores), 4),
        "Recall@K":    round(sum(recall_scores)     / len(recall_scores),    4),
    }


# ---------------------------------------------------------------------------
# Combined entry-point used by BenchmarkEngine
# ---------------------------------------------------------------------------

def evaluate_all(
    test: Any,
    rating_preds: Any,
    ranking_preds: Any,
    name: str,
    top_k: int,
) -> dict[str, float | str]:
    """
    Compute all metrics and return a single flat dict, prefixed with Algorithm.
    """
    return {
        "Algorithm": name,
        **evaluate_explicit(test, rating_preds),
        **evaluate_ranking(test, ranking_preds, top_k),
    }