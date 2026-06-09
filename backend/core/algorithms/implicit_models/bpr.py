"""
algorithms/implicit_models/bpr.py – Bayesian Personalised Ranking (BPR)

BPR optimises for the ranking of observed items above unobserved items
using pairwise training. It maximises the posterior probability of
user preferences: p(u prefers i over j).

Reference: Rendle et al. (2009) — "BPR: Bayesian Personalized Ranking
           from Implicit Feedback"
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.core.data_processing.interaction_matrix import InteractionMatrixBuilder
from backend.config.settings import RANDOM_SEED, TOP_K
from backend.utils.logger import get_logger

log = get_logger(__name__)

_builder = InteractionMatrixBuilder()


def run_bpr(
    train: pd.DataFrame,
    test:  pd.DataFrame,
    top_k: int = TOP_K,
    n_factors: int = 50,
    n_epochs: int = 40,
    lr: float = 0.01,
    reg: float = 0.001,
    neg_sample_ratio: int = 3,
    return_model: bool = False,
) -> tuple:
    """
    Bayesian Personalised Ranking via SGD on sampled triplets (u, i, j).

    For each observed (u, i) pair, sample neg_sample_ratio unobserved
    items j and update factors to rank i above j.

    Strategy:
      1. Try the implicit library (fast C++ backend) if available.
      2. Fall back to pure-numpy SGD BPR implementation.
    """
    im = _builder.build(train)

    try:
        return _run_bpr_implicit_lib(
            im, train, test, top_k, n_factors, n_epochs, lr, reg, return_model
        )
    except ImportError:
        log.warning("'implicit' library not installed — using numpy BPR fallback")
        return _run_bpr_numpy(
            im, train, test, top_k, n_factors, n_epochs, lr, reg, neg_sample_ratio, return_model
        )


def _run_bpr_implicit_lib(im, train, test, top_k, n_factors, n_epochs, lr, reg, return_model=False):
    from implicit.bpr import BayesianPersonalizedRanking

    model = BayesianPersonalizedRanking(
        factors     = n_factors,
        iterations  = n_epochs,
        learning_rate = lr,
        regularization= reg,
        random_state  = RANDOM_SEED,
    )
    model.fit(im.matrix.T.tocsr())
    sc = model.user_factors @ model.item_factors.T

    log.info("BPR (implicit lib) trained | factors=%d epochs=%d", n_factors, n_epochs)
    target_users = test["userID"].astype(str).unique().tolist()
    ranking_preds = _builder.score_matrix_to_ranking_df(
        im, sc, train, top_k, target_user_ids=target_users
    )
    if return_model:
        return None, ranking_preds, {"model": model, "interaction_matrix": im}
    return None, ranking_preds


def _run_bpr_numpy(im, train, test, top_k, n_factors, n_epochs, lr, reg, neg_ratio, return_model=False):
    n_users, n_items = im.n_users, im.n_items
    rng = np.random.default_rng(RANDOM_SEED)

    # Initialise factors
    P = rng.normal(0, 0.01, (n_users, n_factors))
    Q = rng.normal(0, 0.01, (n_items, n_factors))

    # Precompute positive item sets per user
    pos_items = {
        u: set(im.matrix[u].nonzero()[1].tolist())
        for u in range(n_users)
    }

    # Collect (u, i) positive pairs for sampling
    rows, cols = im.matrix.nonzero()
    pairs = list(zip(rows.tolist(), cols.tolist()))

    for epoch in range(n_epochs):
        rng.shuffle(pairs)   # type: ignore
        total_loss = 0.0

        for u, i in pairs:
            if len(pos_items[u]) >= n_items:
                # This user interacted with every item; no negative sample exists.
                continue
            for _ in range(neg_ratio):
                # Sample a negative item j not interacted with
                j = rng.integers(0, n_items)
                attempts = 0
                while j in pos_items[u] and attempts < 50:
                    j = rng.integers(0, n_items)
                    attempts += 1
                if j in pos_items[u]:
                    continue

                x_uij  = float(P[u] @ (Q[i] - Q[j]))
                sigmoid = 1.0 / (1.0 + np.exp(-x_uij))
                err     = 1.0 - sigmoid
                total_loss += -np.log(sigmoid + 1e-10)

                # Gradient update
                diff    = Q[i] - Q[j]
                P[u]   += lr * (err * diff   - reg * P[u])
                Q[i]   += lr * (err * P[u]   - reg * Q[i])
                Q[j]   += lr * (-err * P[u]  - reg * Q[j])

        if (epoch + 1) % 20 == 0:
            log.debug("BPR epoch %d/%d | loss=%.4f", epoch + 1, n_epochs, total_loss)

    sc = P @ Q.T
    log.info("BPR (numpy) trained | factors=%d epochs=%d", n_factors, n_epochs)
    target_users = test["userID"].astype(str).unique().tolist()
    ranking_preds = _builder.score_matrix_to_ranking_df(
        im, sc, train, top_k, target_user_ids=target_users
    )
    if return_model:
        return None, ranking_preds, {"type": "bpr_numpy", "user_factors": P, "item_factors": Q, "interaction_matrix": im}
    return None, ranking_preds
