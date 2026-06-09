"""
General-purpose neighborhood and linear recommendation models.

Algorithms
----------
UserKNN    – User-based k-nearest neighbors collaborative filtering.

ItemKNN    – Item-based k-nearest neighbors collaborative filtering.

EASE       – Embarrassingly Shallow Autoencoder: a linear recommender
             solved in closed form.

SLIM       – Sparse Linear Methods (SLIM): item-item sparse regression model.
             Reference: Ning & Karypis (2011) — "SLIM: Sparse Linear Methods
             for Top-N Recommender Systems"

P3Alpha    – Random-walk graph-based CF with teleportation exponent alpha.
             Reference: Cooper et al. (2014) — "Random walks in recommender
             systems: exact computation and simulations"

RP3Beta    – P3Alpha with popularity-normalised transition matrix (beta).
             Reference: Paudel et al. (2017) — "Updatable, Accurate,
             Diverse, and Scalable Recommendations for Interactive Applications"

WARP       – Weighted Approximate-Rank Pairwise loss trained MF.
             Reference: Weston et al. (2011) — "WSABIE: Scaling Up To Large
             Vocabulary Image Annotation"
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import ElasticNet
from sklearn.metrics.pairwise import cosine_similarity

from backend.config.settings import RANDOM_SEED, TOP_K
from backend.core.data_processing.interaction_matrix import InteractionMatrix, InteractionMatrixBuilder
from backend.utils.logger import get_logger

log = get_logger(__name__)

_builder = InteractionMatrixBuilder()


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _rating_preds(score_matrix: np.ndarray, im: InteractionMatrix, test: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in test.iterrows():
        uid = str(row["userID"])
        iid = str(row["itemID"])
        u_idx = im.user_index.get(uid)
        i_idx = im.item_index.get(iid)
        if u_idx is not None and i_idx is not None:
            records.append({"userID": uid, "itemID": iid, "prediction": float(score_matrix[u_idx, i_idx])})
    return pd.DataFrame(records)


def _ranking_preds(
    score_matrix: np.ndarray,
    im: InteractionMatrix,
    train: pd.DataFrame,
    test: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    target_users = test["userID"].astype(str).unique().tolist()
    return _builder.score_matrix_to_ranking_df(
        im,
        score_matrix,
        train,
        top_k,
        target_user_ids=target_users,
    )


def _prune_top_k(similarity: np.ndarray, k_neighbors: int) -> np.ndarray:
    if similarity.shape[0] <= 1:
        return similarity

    pruned = similarity.copy()
    np.fill_diagonal(pruned, 0.0)
    k_neighbors = max(1, min(int(k_neighbors), pruned.shape[0] - 1))
    if k_neighbors >= pruned.shape[0] - 1:
        return pruned

    thresholds = np.partition(pruned, -k_neighbors, axis=1)[:, -k_neighbors]
    return np.where(pruned >= thresholds[:, None], pruned, 0.0)


# ── User-KNN ───────────────────────────────────────────────────────────────────

def run_user_knn(
    train: pd.DataFrame,
    test: pd.DataFrame,
    top_k: int = TOP_K,
    k_neighbors: int = 40,
    shrinkage: float = 10.0,
    return_model: bool = False,
) -> tuple:
    """
    User-based k-nearest neighbors collaborative filtering.
    """
    im = _builder.build(train)
    dense = im.matrix.toarray().astype(np.float32)

    user_sim = cosine_similarity(dense)
    if shrinkage > 0:
        counts = (dense > 0).astype(np.float32) @ (dense > 0).astype(np.float32).T
        user_sim *= counts / (counts + float(shrinkage))
    user_sim = _prune_top_k(user_sim, k_neighbors)

    denom = np.abs(user_sim).sum(axis=1, keepdims=True) + 1e-9
    score_matrix = (user_sim @ dense) / denom

    log.info("User-KNN trained | neighbors=%d shrinkage=%.2f", k_neighbors, shrinkage)

    rating_preds = _rating_preds(score_matrix, im, test)
    ranking_preds = _ranking_preds(score_matrix, im, train, test, top_k)

    if return_model:
        return rating_preds, ranking_preds, {
            "type": "user_knn",
            "score_matrix": score_matrix.astype(np.float32),
            "interaction_matrix": im,
        }
    return rating_preds, ranking_preds


# ── Item-KNN ───────────────────────────────────────────────────────────────────

def run_item_knn(
    train: pd.DataFrame,
    test: pd.DataFrame,
    top_k: int = TOP_K,
    k_neighbors: int = 40,
    shrinkage: float = 10.0,
    return_model: bool = False,
) -> tuple:
    """
    Item-based k-nearest neighbors collaborative filtering.
    """
    im = _builder.build(train)
    dense = im.matrix.toarray().astype(np.float32)

    item_sim = cosine_similarity(dense.T)
    if shrinkage > 0:
        counts = (dense > 0).astype(np.float32).T @ (dense > 0).astype(np.float32)
        item_sim *= counts / (counts + float(shrinkage))
    item_sim = _prune_top_k(item_sim, k_neighbors)

    denom = np.abs(item_sim).sum(axis=1, keepdims=True).T + 1e-9
    score_matrix = (dense @ item_sim) / denom

    log.info("Item-KNN trained | neighbors=%d shrinkage=%.2f", k_neighbors, shrinkage)

    rating_preds = _rating_preds(score_matrix, im, test)
    ranking_preds = _ranking_preds(score_matrix, im, train, test, top_k)

    if return_model:
        return rating_preds, ranking_preds, {
            "type": "item_knn",
            "score_matrix": score_matrix.astype(np.float32),
            "item_sim": item_sim.astype(np.float32),
            "interaction_matrix": im,
        }
    return rating_preds, ranking_preds


# ── EASE ───────────────────────────────────────────────────────────────────────

def run_ease(
    train: pd.DataFrame,
    test: pd.DataFrame,
    top_k: int = TOP_K,
    l2_penalty: float = 300.0,
    return_model: bool = False,
) -> tuple:
    """
    EASE: a linear recommender solved in closed form.
    """
    im = _builder.build(train)
    x = im.matrix.toarray().astype(np.float64)

    gram = x.T @ x
    diag = np.arange(gram.shape[0])
    gram[diag, diag] += float(l2_penalty)

    precision = np.linalg.inv(gram)
    coefficients = precision / (-np.diag(precision)[None, :])
    np.fill_diagonal(coefficients, 0.0)

    score_matrix = x @ coefficients

    log.info("EASE trained | items=%d lambda=%.2f", im.n_items, l2_penalty)

    rating_preds = _rating_preds(score_matrix, im, test)
    ranking_preds = _ranking_preds(score_matrix, im, train, test, top_k)

    if return_model:
        return rating_preds, ranking_preds, {
            "type": "ease",
            "coefficients": coefficients.astype(np.float32),
            "interaction_matrix": im,
        }
    return rating_preds, ranking_preds


# ── SLIM ───────────────────────────────────────────────────────────────────────

def run_slim(
    train: pd.DataFrame,
    test: pd.DataFrame,
    top_k: int = TOP_K,
    l1_ratio: float = 0.1,
    alpha: float = 0.0005,
    max_iter: int = 100,
    return_model: bool = False,
) -> tuple:
    """
    SLIM: Sparse Linear Methods for top-N recommendations.

    Learns a sparse item-item weight matrix W by fitting an ElasticNet
    regression for each item column of the interaction matrix.  Predictions
    are computed as R * W where W_ii = 0 (no self-loops).

    Parameters
    ----------
    l1_ratio  : ElasticNet mixing parameter (0 = Ridge, 1 = Lasso).
    alpha     : Regularisation strength.
    max_iter  : Maximum ElasticNet solver iterations.
    """
    im = _builder.build(train)
    R = im.matrix.toarray().astype(np.float32)
    n_items = im.n_items

    W = np.zeros((n_items, n_items), dtype=np.float32)

    model = ElasticNet(
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        fit_intercept=False,
        max_iter=int(max_iter),
        positive=True,       # enforce non-negative weights
        selection="random",
        random_state=RANDOM_SEED,
    )

    for j in range(n_items):
        # target: column j of R
        y = R[:, j].copy()
        # zero out item j in the feature matrix to prevent self-loops
        X_j = R.copy()
        X_j[:, j] = 0.0
        model.fit(X_j, y)
        W[:, j] = model.coef_

    score_matrix = (R @ W).astype(np.float32)

    log.info(
        "SLIM trained | items=%d alpha=%.5f l1_ratio=%.2f",
        n_items, alpha, l1_ratio,
    )

    rating_preds = _rating_preds(score_matrix, im, test)
    ranking_preds = _ranking_preds(score_matrix, im, train, test, top_k)

    if return_model:
        return rating_preds, ranking_preds, {
            "type": "slim",
            "W": W,
            "interaction_matrix": im,
        }
    return rating_preds, ranking_preds


# ── P3Alpha ────────────────────────────────────────────────────────────────────

def run_p3alpha(
    train: pd.DataFrame,
    test: pd.DataFrame,
    top_k: int = TOP_K,
    alpha: float = 1.0,
    normalize_similarity: bool = True,
    return_model: bool = False,
) -> tuple:
    """
    P3Alpha: three-hop random-walk graph-based collaborative filtering.

    Constructs a user-item bipartite graph, performs a three-step random walk
    (user → item → user → item) with a teleportation exponent *alpha* to
    reduce popularity bias, and uses the resulting item-item similarity matrix
    as a score kernel.

    Parameters
    ----------
    alpha               : Teleportation exponent applied to transition rows.
                          alpha=1 gives vanilla P3; higher values suppress
                          popular items more aggressively.
    normalize_similarity: Row-normalise the final item-item similarity.
    """
    im = _builder.build(train)
    R = im.matrix.astype(np.float64)  # (n_users, n_items)

    # Row-normalised user→item transition matrix P_ui
    row_sums = np.asarray(R.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    # apply alpha exponent to degree before normalising
    row_sums_alpha = np.power(row_sums, alpha)
    D_u_inv = sp.diags(1.0 / row_sums_alpha)
    P_ui = D_u_inv @ R  # (n_users, n_items)

    # Row-normalised item→user transition matrix P_iu
    col_sums = np.asarray(R.sum(axis=0)).ravel()
    col_sums[col_sums == 0] = 1.0
    D_i_inv = sp.diags(1.0 / col_sums)
    P_iu = D_i_inv @ R.T  # (n_items, n_users)

    # Three-hop walk: item-item similarity = P_iu @ P_ui
    # Then score_matrix = R @ (P_iu @ P_ui)
    item_sim = (P_iu @ P_ui).toarray().astype(np.float32)  # (n_items, n_items)
    np.fill_diagonal(item_sim, 0.0)

    if normalize_similarity:
        row_max = item_sim.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        item_sim /= row_max

    score_matrix = (P_ui @ item_sim).astype(np.float32)

    log.info("P3Alpha trained | items=%d alpha=%.2f", im.n_items, alpha)

    rating_preds = _rating_preds(score_matrix, im, test)
    ranking_preds = _ranking_preds(score_matrix, im, train, test, top_k)

    if return_model:
        return rating_preds, ranking_preds, {
            "type": "p3alpha",
            "item_sim": item_sim,
            "interaction_matrix": im,
        }
    return rating_preds, ranking_preds


# ── RP3Beta ────────────────────────────────────────────────────────────────────

def run_rp3beta(
    train: pd.DataFrame,
    test: pd.DataFrame,
    top_k: int = TOP_K,
    alpha: float = 1.0,
    beta: float = 0.6,
    normalize_similarity: bool = True,
    return_model: bool = False,
) -> tuple:
    """
    RP3Beta: popularity-normalised P3Alpha.

    Extends P3Alpha by dividing the item-item similarity scores by each
    item's popularity raised to the power *beta*, further penalising
    blockbuster items that would otherwise dominate recommendations.

    Parameters
    ----------
    alpha               : Random-walk exponent (same as P3Alpha).
    beta                : Popularity normalisation exponent.  beta=0 disables
                          popularity normalisation (equivalent to P3Alpha).
    normalize_similarity: Row-normalise after popularity correction.
    """
    im = _builder.build(train)
    R = im.matrix.astype(np.float64)

    # Row-normalised user→item transition P_ui
    row_sums = np.asarray(R.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    row_sums_alpha = np.power(row_sums, alpha)
    D_u_inv = sp.diags(1.0 / row_sums_alpha)
    P_ui = D_u_inv @ R

    # Row-normalised item→user transition P_iu
    col_sums = np.asarray(R.sum(axis=0)).ravel()
    col_sums[col_sums == 0] = 1.0
    D_i_inv = sp.diags(1.0 / col_sums)
    P_iu = D_i_inv @ R.T

    # Three-hop similarity
    item_sim = (P_iu @ P_ui).toarray().astype(np.float32)
    np.fill_diagonal(item_sim, 0.0)

    # Popularity normalisation: divide column j by pop(j)^beta
    popularity = col_sums.astype(np.float32)
    pop_beta = np.power(popularity, float(beta))
    pop_beta[pop_beta == 0] = 1.0
    item_sim /= pop_beta[None, :]  # broadcast over rows

    if normalize_similarity:
        row_max = item_sim.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        item_sim /= row_max

    score_matrix = (P_ui @ item_sim).astype(np.float32)

    log.info(
        "RP3Beta trained | items=%d alpha=%.2f beta=%.2f",
        im.n_items, alpha, beta,
    )

    rating_preds = _rating_preds(score_matrix, im, test)
    ranking_preds = _ranking_preds(score_matrix, im, train, test, top_k)

    if return_model:
        return rating_preds, ranking_preds, {
            "type": "rp3beta",
            "item_sim": item_sim,
            "interaction_matrix": im,
        }
    return rating_preds, ranking_preds


# ── WARP ───────────────────────────────────────────────────────────────────────

def run_warp(
    train: pd.DataFrame,
    test: pd.DataFrame,
    top_k: int = TOP_K,
    n_factors: int = 32,
    n_epochs: int = 10,
    learning_rate: float = 0.01,
    regularization: float = 0.01,
    max_sampled: int = 10,
    return_model: bool = False,
) -> tuple:
    """
    WARP: Weighted Approximate-Rank Pairwise loss matrix factorisation.

    For each positive (user, item) pair, samples negative items until a
    rank violation is found, then updates via a gradient step weighted by
    the approximate rank of the violating item.  This directly optimises
    for ranking quality (precision@k / AUC).

    Parameters
    ----------
    n_factors     : Latent dimension for user and item embeddings.
    n_epochs      : Number of passes over positive interactions.
    learning_rate : SGD step size.
    regularization: L2 regularisation applied to factors.
    max_sampled   : Maximum negative samples per positive before giving up.
    """
    im = _builder.build(train)
    mat = im.matrix  # CSR, (n_users, n_items)
    n_users, n_items = im.n_users, im.n_items

    rng = np.random.default_rng(RANDOM_SEED)
    U = rng.normal(0, 0.1, (n_users, n_factors)).astype(np.float32)
    V = rng.normal(0, 0.1, (n_items, n_factors)).astype(np.float32)

    rows, cols = mat.nonzero()
    n_pos = len(rows)

    # Precompute per-user positive item sets for fast negative sampling
    user_positives: list[set] = [set() for _ in range(n_users)]
    for u, i in zip(rows, cols):
        user_positives[u].add(int(i))

    # WARP rank weight: w(rank) = sum_{k=1}^{rank} 1/k  (truncated harmonic)
    _harmonic = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, n_items + 1))]).astype(np.float32)

    for epoch in range(int(n_epochs)):
        order = rng.permutation(n_pos)
        total_loss = 0.0

        for idx in order:
            u = int(rows[idx])
            i = int(cols[idx])  # positive item

            score_pos = float(U[u] @ V[i])

            # Sample negatives until violation or max_sampled
            sampled = 0
            violating_j = -1
            for _ in range(max_sampled):
                j = int(rng.integers(0, n_items))
                if j in user_positives[u]:
                    continue
                sampled += 1
                score_neg = float(U[u] @ V[j])
                if score_neg > score_pos - 1.0:  # margin violation
                    violating_j = j
                    break

            if violating_j < 0:
                continue  # no violation found; skip

            # Approximate rank of violating item
            approx_rank = max(1, int((n_items - len(user_positives[u])) / sampled))
            weight = _harmonic[min(approx_rank, n_items - 1)]

            # Gradient step
            u_vec = U[u].copy()
            v_pos = V[i].copy()
            v_neg = V[violating_j].copy()

            grad_u = weight * (v_neg - v_pos)
            grad_vi = -weight * u_vec
            grad_vj = weight * u_vec

            U[u] -= learning_rate * (grad_u + regularization * u_vec)
            V[i] -= learning_rate * (grad_vi + regularization * v_pos)
            V[violating_j] -= learning_rate * (grad_vj + regularization * v_neg)

            total_loss += weight

        log.debug("WARP epoch %d/%d | approx_loss=%.4f", epoch + 1, n_epochs, total_loss / max(n_pos, 1))

    score_matrix = (U @ V.T).astype(np.float32)
    log.info("WARP trained | factors=%d epochs=%d", n_factors, n_epochs)

    rating_preds = _rating_preds(score_matrix, im, test)
    ranking_preds = _ranking_preds(score_matrix, im, train, test, top_k)

    if return_model:
        return rating_preds, ranking_preds, {
            "type": "warp",
            "user_factors": U,
            "item_factors": V,
            "interaction_matrix": im,
        }
    return rating_preds, ranking_preds
