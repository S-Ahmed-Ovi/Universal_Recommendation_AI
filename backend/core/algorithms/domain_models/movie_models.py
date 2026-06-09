"""
algorithms/domain_models/movie_models.py – Movie-domain recommendation logic

Domain-specific enhancements for movie recommendation:
  - Genre-aware Item-KNN: enriches item similarity with genre overlap
  - Temporal SVD: incorporates time-drifting user preferences
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from backend.core.data_processing.interaction_matrix import InteractionMatrixBuilder
from backend.config.settings import RANDOM_SEED, TOP_K
from backend.utils.logger import get_logger

log = get_logger(__name__)

_builder = InteractionMatrixBuilder()


def _genre_similarity_matrix(item_ids: list, genre_map: dict) -> np.ndarray:
    """
    Vectorised Jaccard genre-overlap matrix.

    Builds a (n_items, n_genres) binary indicator matrix, then computes
    overlap entirely with numpy — no Python loops over item pairs.
    """
    # Collect all genre tokens
    all_genres = sorted({
        g
        for iid in item_ids
        for g in str(genre_map.get(iid, "")).split("|")
        if g
    })
    if not all_genres:
        return np.zeros((len(item_ids), len(item_ids)), dtype=np.float32)

    genre_idx = {g: j for j, g in enumerate(all_genres)}
    n, m = len(item_ids), len(all_genres)

    # Binary indicator  (n_items × n_genres)
    indicator = np.zeros((n, m), dtype=np.float32)
    for i, iid in enumerate(item_ids):
        for g in str(genre_map.get(iid, "")).split("|"):
            if g in genre_idx:
                indicator[i, genre_idx[g]] = 1.0

    # Jaccard = intersection / union  (fully vectorised)
    inter = indicator @ indicator.T                          # n×n
    row_sum = indicator.sum(axis=1, keepdims=True)          # n×1
    union = row_sum + row_sum.T - inter                     # n×n
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union > 0, inter / union, 0.0)
    np.fill_diagonal(jaccard, 0.0)
    return jaccard.astype(np.float32)


def run_movie_item_knn(
    train: pd.DataFrame,
    test:  pd.DataFrame,
    top_k: int = TOP_K,
    k_neighbours: int = 30,
    genre_boost: float = 0.2,
    item_metadata: pd.DataFrame | None = None,
    return_model: bool = False,
) -> tuple:
    """
    Genre-aware Item-KNN for movie recommendations.

    Computes cosine similarity on the interaction matrix then optionally
    adds a vectorised genre-overlap bonus if item_metadata with a 'genres'
    column is provided.
    """
    im    = _builder.build(train)
    dense = im.matrix.toarray()
    item_sim = cosine_similarity(dense.T)           # (n_items × n_items)

    # ── Vectorised genre enrichment (no Python loops over item pairs) ──
    if item_metadata is not None and "genres" in item_metadata.columns:
        genre_map = item_metadata.set_index("itemID")["genres"].to_dict()
        jaccard   = _genre_similarity_matrix(im.item_ids, genre_map)
        item_sim += genre_boost * jaccard
        log.info("Applied vectorised genre boost | boost=%.2f", genre_boost)

    # Keep only top-k_neighbours per item (vectorised partition)
    if k_neighbours < item_sim.shape[0]:
        # Zero out self-similarity first
        np.fill_diagonal(item_sim, 0.0)
        # For each row, find the k-th largest value and threshold below it
        thresh = np.partition(item_sim, -k_neighbours, axis=1)[:, -k_neighbours]
        item_sim = np.where(item_sim >= thresh[:, None], item_sim, 0.0)
    else:
        np.fill_diagonal(item_sim, 0.0)

    sc = (dense @ item_sim) / (np.abs(item_sim).sum(axis=1, keepdims=True).T + 1e-9)

    # Rating preds
    records = []
    for _, row in test.iterrows():
        uid = str(row["userID"]); iid = str(row["itemID"])
        u = im.user_index.get(uid); i = im.item_index.get(iid)
        if u is not None and i is not None:
            records.append({"userID": uid, "itemID": iid, "prediction": float(sc[u, i])})

    log.info("Movie Item-KNN trained | k=%d genre_boost=%s", k_neighbours, genre_boost)
    rating_preds  = pd.DataFrame(records)
    target_users = test["userID"].astype(str).unique().tolist()
    ranking_preds = _builder.score_matrix_to_ranking_df(
        im, sc, train, top_k, target_user_ids=target_users
    )
    if return_model:
        return rating_preds, ranking_preds, {
            "type": "movie_item_knn",
            "item_sim": item_sim,
            "interaction_matrix": im,
        }
    return rating_preds, ranking_preds


def run_temporal_svd(
    train: pd.DataFrame,
    test:  pd.DataFrame,
    top_k: int = TOP_K,
    n_components: int = 50,
    n_time_bins: int = 5,
    return_model: bool = False,
) -> tuple:
    """
    Temporal SVD – models time-drifting user preferences.

    Splits training history into NON-OVERLAPPING time bins, fits one SVD per
    bin, then computes a recency-weighted average via vectorised index mapping.

    Previously used cumulative (growing) slices, which caused each early-bin
    SVD to re-process all prior data, and mapped indices with a Python double
    for-loop — both are now fixed.

    Falls back to standard SVD if no timestamp column is present.
    """
    from backend.core.algorithms.explicit_models.svd import run_svd

    if "timestamp" not in train.columns:
        log.info("No timestamp column — falling back to standard SVD")
        return run_svd(train, test, top_k, n_components=n_components, return_model=return_model)

    ts = pd.to_numeric(train["timestamp"], errors="coerce")
    train = train.copy()
    train["_bin"] = pd.qcut(ts, q=n_time_bins, labels=False, duplicates="drop")
    n_bins = int(train["_bin"].max() + 1)

    ref_im = _builder.build(train.drop(columns=["_bin"]))
    sc     = np.zeros((ref_im.n_users, ref_im.n_items), dtype=np.float32)
    total_weight = 0.0

    # Build lookup arrays once — avoids dict lookups inside the loop
    ref_user_arr = np.array(ref_im.user_ids)
    ref_item_arr = np.array(ref_im.item_ids)

    for bin_idx in range(n_bins):
        # NON-OVERLAPPING bin (was cumulative — re-processed all prior data)
        chunk = train[train["_bin"] == bin_idx].drop(columns=["_bin"])
        if chunk.empty:
            continue
        weight = (bin_idx + 1) / n_bins    # more recent → higher weight

        try:
            from sklearn.decomposition import TruncatedSVD as TSVD
            im_chunk = _builder.build(chunk)
            k = min(n_components, im_chunk.n_users - 1, im_chunk.n_items - 1)
            if k < 1:
                continue
            tsvd     = TSVD(n_components=k, random_state=RANDOM_SEED)
            U        = tsvd.fit_transform(im_chunk.matrix)
            sc_chunk = (U @ tsvd.components_).astype(np.float32)  # (chunk_u × chunk_i)

            # ── Vectorised index mapping (was a Python double for-loop) ──────
            # Map chunk user indices → reference user indices
            chunk_user_to_ref = np.array([
                ref_im.user_index.get(uid, -1) for uid in im_chunk.user_ids
            ])
            chunk_item_to_ref = np.array([
                ref_im.item_index.get(iid, -1) for iid in im_chunk.item_ids
            ])

            valid_u = chunk_user_to_ref >= 0   # boolean mask
            valid_i = chunk_item_to_ref >= 0

            # Sub-select valid rows/cols and scatter into reference matrix
            src = sc_chunk[np.ix_(valid_u, valid_i)]
            dst_u = chunk_user_to_ref[valid_u]
            dst_i = chunk_item_to_ref[valid_i]
            sc[np.ix_(dst_u, dst_i)] += weight * src

            total_weight += weight
        except Exception as e:
            log.warning("Temporal SVD bin %d failed: %s", bin_idx, e)

    if total_weight > 0:
        sc /= total_weight

    log.info("Temporal SVD trained | bins=%d components=%d", n_bins, n_components)
    target_users = test["userID"].astype(str).unique().tolist()
    ranking_preds = _builder.score_matrix_to_ranking_df(
        ref_im, sc, train, top_k, target_user_ids=target_users
    )
    if return_model:
        return None, ranking_preds, {
            "type": "temporal_svd",
            "score_matrix": sc,
            "interaction_matrix": ref_im,
        }
    return None, ranking_preds
