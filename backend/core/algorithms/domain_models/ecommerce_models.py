"""
algorithms/domain_models/ecommerce_models.py – E-commerce specific recommendation logic

Domain-specific enhancements for e-commerce recommendation:
  - Purchase-weighted ALS: boosts confidence on purchase events vs. views
  - Popularity + recency hybrid: blends item popularity with recency decay
  - Category-aware item-KNN: uses category co-occurrence to enrich similarity
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.core.data_processing.interaction_matrix import InteractionMatrixBuilder
from backend.config.settings import RANDOM_SEED, TOP_K
from backend.utils.logger import get_logger

log = get_logger(__name__)

_builder = InteractionMatrixBuilder()


def run_ecommerce_popularity(
    train: pd.DataFrame,
    test:  pd.DataFrame,
    top_k: int = TOP_K,
    recency_decay: float = 0.95,
    min_interactions: int = 5,
    return_model: bool = False,
) -> tuple:
    """
    E-commerce Popularity Recommender with recency decay.

    Scores items by interaction frequency, optionally discounted by age.
    Items with fewer than min_interactions are excluded.
    Seen items are removed per user.
    """
    im = _builder.build(train)

    if "timestamp" in train.columns:
        ts = pd.to_numeric(train["timestamp"], errors="coerce")
        ts_min, ts_max = ts.min(), ts.max()
        ts_range = ts_max - ts_min if ts_max > ts_min else 1.0
        # Normalise timestamps to [0, 1]
        train = train.copy()
        train["_ts_norm"] = (ts - ts_min) / ts_range
        item_scores = (
            train.groupby("itemID")
            .apply(lambda g: (recency_decay ** (1 - g["_ts_norm"])).sum())
            .to_dict()
        )
    else:
        item_scores = train["itemID"].value_counts().to_dict()

    # Filter min interactions
    item_counts = train["itemID"].value_counts()
    eligible    = set(item_counts[item_counts >= min_interactions].index)

    seen_per_user = train.groupby("userID")["itemID"].apply(set).to_dict()

    records = []
    for user_id in test["userID"].unique():
        seen = seen_per_user.get(str(user_id), set())
        candidates = [
            (iid, score) for iid, score in item_scores.items()
            if iid in eligible and iid not in seen
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        for rank, (item_id, score) in enumerate(candidates[:top_k]):
            records.append({"userID": user_id, "itemID": item_id, "score": score})

    log.info("E-commerce Popularity trained | eligible_items=%d", len(eligible))
    ranking_preds = pd.DataFrame(records)
    if return_model:
        return None, ranking_preds, {
            "type": "ecommerce_popularity",
            "item_scores": item_scores,
            "eligible": eligible,
            "seen_per_user": seen_per_user,
            "top_k": top_k,
            "interaction_matrix": im,
        }
    return None, ranking_preds


def run_ecommerce_purchase_als(
    train: pd.DataFrame,
    test:  pd.DataFrame,
    top_k: int = TOP_K,
    n_factors: int = 64,
    n_iterations: int = 20,
    regularization: float = 0.01,
    purchase_weight: float = 5.0,
    view_weight: float = 1.0,
    event_col: str = "event_type",
    return_model: bool = False,
) -> tuple:
    """
    Purchase-weighted ALS for e-commerce.

    If the dataset has an event_type column distinguishing 'purchase'
    from 'view', applies higher confidence weights to purchase events.
    Falls back to standard ALS if event_type column is absent.
    """
    from backend.core.algorithms.implicit_models.als import run_als

    if event_col in train.columns:
        # Re-weight: purchases → purchase_weight, others → view_weight
        train = train.copy()
        is_purchase = train[event_col].str.lower().str.contains("purchase|buy|order", na=False)
        train["rating"] = np.where(is_purchase, purchase_weight, view_weight)
        log.info(
            "E-commerce ALS | purchases=%d views=%d",
            is_purchase.sum(), (~is_purchase).sum(),
        )
    else:
        log.info("No '%s' column — running standard ALS", event_col)

    return run_als(train, test, top_k, n_factors=n_factors, n_iterations=n_iterations,
                   regularization=regularization, return_model=return_model)
