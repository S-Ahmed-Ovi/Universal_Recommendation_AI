"""
algorithms/explicit_models/svd.py – SVD and SVD++ matrix factorisation

SVD:   Truncated SVD via sklearn (fast, pure numpy, no extra dependencies)
SVD++: SVD with implicit feedback signal incorporated into user factors

Return convention
-----------------
All runner functions accept an optional return_model=False flag.

  return_model=False (default, used by BenchmarkEngine):
      returns (rating_preds_df, ranking_preds_df)

  return_model=True (used by TrainingPipeline._save_model):
      returns (rating_preds_df, ranking_preds_df, trained_model_object)

The trained_model_object is the fitted sklearn estimator so the serving
layer can call model.transform(matrix) or read model.components_ at
inference time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from backend.core.data_processing.interaction_matrix import InteractionMatrixBuilder, InteractionMatrix
from backend.config.settings import RANDOM_SEED, TOP_K
from backend.utils.logger import get_logger

log = get_logger(__name__)

_builder = InteractionMatrixBuilder()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _rating_preds(score_matrix, im: InteractionMatrix, test: pd.DataFrame) -> pd.DataFrame:
    """Produce (userID, itemID, prediction) for all user-item pairs in test."""
    records = []
    for _, row in test.iterrows():
        uid = str(row["userID"])
        iid = str(row["itemID"])
        u   = im.user_index.get(uid)
        i   = im.item_index.get(iid)
        if u is not None and i is not None:
            records.append({"userID": uid, "itemID": iid, "prediction": float(score_matrix[u, i])})
    return pd.DataFrame(records)


def _ranking_preds(score_matrix, im, train, test, k) -> pd.DataFrame:
    target_users = test["userID"].astype(str).unique().tolist()
    return _builder.score_matrix_to_ranking_df(
        im, score_matrix, train, k, target_user_ids=target_users
    )


# ══════════════════════════════════════════════════════════════════════════════
# SVD
# ══════════════════════════════════════════════════════════════════════════════

def run_svd(
    train: pd.DataFrame,
    test:  pd.DataFrame,
    top_k: int = TOP_K,
    n_components: int = 50,
    return_model: bool = False,
) -> tuple:
    """
    Truncated SVD Matrix Factorisation (sklearn).

    Decomposes the user-item interaction matrix into U x S x Vt and
    reconstructs predicted scores as U x (S x Vt).

    When return_model=True the fitted TruncatedSVD estimator and the
    InteractionMatrix used during training are returned as a third element
    so the serving layer can reconstruct scores at inference time.
    """
    im  = _builder.build(train)
    k   = min(n_components, im.n_users - 1, im.n_items - 1)

    svd = TruncatedSVD(n_components=k, random_state=RANDOM_SEED)
    U   = svd.fit_transform(im.matrix)
    sc  = U @ svd.components_

    log.info("SVD trained | components=%d users=%d items=%d", k, im.n_users, im.n_items)

    rating_preds  = _rating_preds(sc, im, test)
    ranking_preds = _ranking_preds(sc, im, train, test, top_k)

    if return_model:
        # Wrap in a container that exposes the interface expected by
        # RecommenderEngine._get_score_matrix (components_ + transform).
        return rating_preds, ranking_preds, {"model": svd, "interaction_matrix": im}
    return rating_preds, ranking_preds


# ══════════════════════════════════════════════════════════════════════════════
# SVD++
# ══════════════════════════════════════════════════════════════════════════════

def run_svdpp(
    train: pd.DataFrame,
    test:  pd.DataFrame,
    top_k: int = TOP_K,
    n_components: int = 50,
    n_epochs: int = 20,
    lr: float = 0.005,
    reg: float = 0.02,
    return_model: bool = False,
) -> tuple:
    """
    SVD++ – Extends SVD by incorporating implicit feedback (which items a user
    has interacted with, regardless of the rating value) into the user latent
    factors.  Implemented as SGD-based matrix factorisation.

    User factor:  p_u + (|N(u)|^{-0.5}) * Σ_{j ∈ N(u)} y_j
    Item factor:  q_i
    Prediction:   μ + b_u + b_i + (p_u + implicit_factor) · q_i
    """
    im = _builder.build(train)
    mat = im.matrix.toarray().astype(np.float64)

    n_users, n_items = mat.shape
    rng = np.random.default_rng(RANDOM_SEED)

    # Initialise parameters
    global_mean = mat[mat > 0].mean() if mat.any() else 0.0
    b_u = np.zeros(n_users)
    b_i = np.zeros(n_items)
    P   = rng.normal(0, 0.01, (n_users, n_components))    # user factors
    Q   = rng.normal(0, 0.01, (n_items, n_components))    # item factors
    Y   = rng.normal(0, 0.01, (n_items, n_components))    # implicit item factors

    # Precompute implicit sets per user
    implicit_sets = {
        u: im.matrix[u].nonzero()[1].tolist()
        for u in range(n_users)
    }

    # SGD training loop
    rows, cols = mat.nonzero()
    indices = list(zip(rows, cols))
    rng.shuffle(indices)   # type: ignore

    for epoch in range(n_epochs):
        total_loss = 0.0
        for u, i in indices:
            Nu    = implicit_sets[u]
            Nu_sq = len(Nu) ** -0.5 if Nu else 0.0
            imp   = Nu_sq * Y[Nu].sum(axis=0) if Nu else np.zeros(n_components)
            pu_   = P[u] + imp
            pred  = global_mean + b_u[u] + b_i[i] + pu_ @ Q[i]
            err   = mat[u, i] - pred

            total_loss += err ** 2
            # Update biases
            b_u[u] += lr * (err - reg * b_u[u])
            b_i[i] += lr * (err - reg * b_i[i])
            # Update factors
            P[u]   += lr * (err * Q[i] - reg * P[u])
            Q[i]   += lr * (err * pu_  - reg * Q[i])
            for j in Nu:
                Y[j] += lr * (err * Nu_sq * Q[i] - reg * Y[j])

        if (epoch + 1) % 5 == 0:
            log.debug("SVD++ epoch %d/%d | loss=%.4f", epoch + 1, n_epochs, total_loss)

    # Reconstruct full score matrix
    imp_factors = np.array([
        (len(implicit_sets[u]) ** -0.5 * Y[implicit_sets[u]].sum(axis=0))
        if implicit_sets[u] else np.zeros(n_components)
        for u in range(n_users)
    ])
    sc = global_mean + b_u[:, None] + b_i[None, :] + (P + imp_factors) @ Q.T

    log.info("SVD++ trained | components=%d epochs=%d", n_components, n_epochs)

    rating_preds  = _rating_preds(sc, im, test)
    ranking_preds = _ranking_preds(sc, im, train, test, top_k)

    if return_model:
        model_obj = {
            "type":          "svdpp",
            "global_mean":   global_mean,
            "b_u":           b_u,
            "b_i":           b_i,
            "P":             P,
            "Q":             Q,
            "Y":             Y,
            "implicit_sets": implicit_sets,
            "n_components":  n_components,
            "interaction_matrix": im,
        }
        return rating_preds, ranking_preds, model_obj
    return rating_preds, ranking_preds
