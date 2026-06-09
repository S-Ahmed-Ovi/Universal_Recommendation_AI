"""
algorithms/implicit_models/als.py – Alternating Least Squares (ALS)

ALS is the standard algorithm for implicit feedback collaborative filtering.
Treats the interaction matrix as confidence-weighted preferences and learns
latent user/item factors by alternately solving closed-form least squares.

Reference: Hu, Koren & Volinsky (2008) — "Collaborative Filtering for
           Implicit Feedback Datasets"
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from backend.core.data_processing.interaction_matrix import InteractionMatrixBuilder
from backend.config.settings import RANDOM_SEED, TOP_K
from backend.utils.logger import get_logger

log = get_logger(__name__)

_builder = InteractionMatrixBuilder()


def run_als(
    train: pd.DataFrame,
    test:  pd.DataFrame,
    top_k: int = TOP_K,
    n_factors: int = 50,
    n_iterations: int = 15,
    regularization: float = 0.01,
    alpha: float = 40.0,
    return_model: bool = False,
) -> tuple:
    """
    Implicit ALS matrix factorisation.

    Parameters
    ----------
    n_factors      : number of latent factors
    n_iterations   : number of ALS iterations
    regularization : L2 regularisation strength
    alpha          : confidence scaling factor (c_ui = 1 + alpha * r_ui)

    Strategy
    --------
    1. Try the implicit library (fast C++ backend) if available.
    2. Fall back to a pure-numpy ALS implementation.
    """
    im  = _builder.build(train)
    mat = im.matrix   # CSR

    try:
        return _run_als_implicit_lib(im, mat, train, test, top_k, n_factors, n_iterations, regularization, alpha, return_model)
    except ImportError:
        log.warning("'implicit' library not installed — using pure-numpy ALS fallback")
        return _run_als_numpy(im, mat, train, test, top_k, n_factors, n_iterations, regularization, alpha, return_model)


# ── implicit library backend ───────────────────────────────────────────────────

def _run_als_implicit_lib(im, mat, train, test, top_k, n_factors, n_iters, reg, alpha, return_model=False):
    from implicit.als import AlternatingLeastSquares
    from implicit.nearest_neighbours import bm25_weight

    # Apply BM25 confidence weighting
    mat_weighted = bm25_weight(mat, K1=100, B=0.8).T.tocsr()

    model = AlternatingLeastSquares(
        factors        = n_factors,
        iterations     = n_iters,
        regularization = reg,
        random_state   = RANDOM_SEED,
    )
    model.fit(mat_weighted)

    # Build score matrix: user_factors @ item_factors.T
    sc = model.user_factors @ model.item_factors.T
    log.info("ALS (implicit lib) trained | factors=%d iters=%d", n_factors, n_iters)
    target_users = test["userID"].astype(str).unique().tolist()
    ranking_preds = _builder.score_matrix_to_ranking_df(
        im, sc, train, top_k, target_user_ids=target_users
    )
    if return_model:
        return None, ranking_preds, {"model": model, "interaction_matrix": im}
    return None, ranking_preds


# ── pure-numpy ALS fallback ────────────────────────────────────────────────────

def _run_als_numpy(im, mat, train, test, top_k, n_factors, n_iters, reg, alpha, return_model=False):
    n_users, n_items = mat.shape
    rng = np.random.default_rng(RANDOM_SEED)

    # Confidence matrix: C_ui = 1 + alpha * R_ui
    C = mat.copy().astype(np.float64)
    C.data = 1.0 + alpha * C.data
    C_csc  = C.tocsc()

    # Preference matrix: P_ui = 1 if R_ui > 0
    P = mat.copy()
    P.data = np.ones_like(P.data)

    # Initialise factors
    X = rng.normal(0, 0.01, (n_users, n_factors))  # user factors
    Y = rng.normal(0, 0.01, (n_items, n_factors))  # item factors
    I = np.eye(n_factors) * reg

    for iteration in range(n_iters):
        # Fix Y, solve for X
        YtY = Y.T @ Y
        for u in range(n_users):
            c_u = C[u].toarray().flatten()
            p_u = P[u].toarray().flatten()
            C_u = sp.diags(c_u)
            A   = YtY + Y.T @ C_u @ Y + I
            b   = Y.T @ (C_u @ p_u)
            X[u] = np.linalg.solve(A, b)

        # Fix X, solve for Y
        XtX = X.T @ X
        for i in range(n_items):
            c_i = C_csc[:, i].toarray().flatten()
            p_i = P.tocsc()[:, i].toarray().flatten()
            C_i = sp.diags(c_i)
            A   = XtX + X.T @ C_i @ X + I
            b   = X.T @ (C_i @ p_i)
            Y[i] = np.linalg.solve(A, b)

        log.debug("ALS numpy iteration %d/%d", iteration + 1, n_iters)

    sc = X @ Y.T
    log.info("ALS (numpy) trained | factors=%d iters=%d", n_factors, n_iters)
    target_users = test["userID"].astype(str).unique().tolist()
    ranking_preds = _builder.score_matrix_to_ranking_df(
        im, sc, train, top_k, target_user_ids=target_users
    )
    if return_model:
        return None, ranking_preds, {"type": "als_numpy", "user_factors": X, "item_factors": Y, "interaction_matrix": im}
    return None, ranking_preds
