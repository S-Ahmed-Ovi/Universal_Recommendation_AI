"""
algorithms/explicit_models/nmf.py – Non-Negative Matrix Factorisation

NMF constrains both user and item factors to be non-negative, which
produces additive, parts-based representations and often leads to more
interpretable latent factors than SVD.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

from backend.core.data_processing.interaction_matrix import InteractionMatrixBuilder, InteractionMatrix
from backend.config.settings import RANDOM_SEED, TOP_K
from backend.utils.logger import get_logger

log = get_logger(__name__)

_builder = InteractionMatrixBuilder()


def run_nmf(
    train: pd.DataFrame,
    test:  pd.DataFrame,
    top_k: int = TOP_K,
    n_components: int = 20,
    max_iter: int = 500,
    l1_ratio: float = 0.0,
    alpha_W: float = 0.0,
    return_model: bool = False,
) -> tuple:
    """
    Non-Negative Matrix Factorisation (sklearn).

    Factorises the interaction matrix R ≈ W × H, where W is the user
    factor matrix and H is the item factor matrix, both non-negative.
    Predicted score for user u and item i: W[u] · H[:, i]
    """
    im  = _builder.build(train)
    k   = min(n_components, im.n_users - 1, im.n_items - 1)

    nmf = NMF(
        n_components = k,
        random_state = RANDOM_SEED,
        max_iter     = max_iter,
        l1_ratio     = l1_ratio,
        alpha_W      = alpha_W,
    )
    W  = nmf.fit_transform(im.matrix)
    sc = W @ nmf.components_

    log.info("NMF trained | components=%d users=%d items=%d", k, im.n_users, im.n_items)

    # Rating predictions
    records = []
    for _, row in test.iterrows():
        uid = str(row["userID"]); iid = str(row["itemID"])
        u   = im.user_index.get(uid); i = im.item_index.get(iid)
        if u is not None and i is not None:
            records.append({"userID": uid, "itemID": iid, "prediction": float(sc[u, i])})
    rating_preds = pd.DataFrame(records)

    target_users = test["userID"].astype(str).unique().tolist()
    ranking_preds = _builder.score_matrix_to_ranking_df(
        im, sc, train, top_k, target_user_ids=target_users
    )

    if return_model:
        return rating_preds, ranking_preds, {"model": nmf, "interaction_matrix": im}
    return rating_preds, ranking_preds
