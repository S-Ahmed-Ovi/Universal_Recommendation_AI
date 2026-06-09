"""
Hybrid, neural, and metadata-aware recommendation models.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor

from backend.config.settings import RANDOM_SEED, TOP_K
from backend.core.data_processing.interaction_matrix import InteractionMatrix, InteractionMatrixBuilder
from backend.utils.logger import get_logger

log = get_logger(__name__)

_builder = InteractionMatrixBuilder()
_CORE_COLS = {"userID", "itemID", "rating", "timestamp"}
_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def _normalize_token(text: object) -> str:
    value = _TOKEN_RE.sub(" ", str(text or "").lower()).strip()
    return value or "unknown"


def _dominant_value(series: pd.Series) -> object:
    non_null = series.dropna()
    if non_null.empty:
        return None
    mode = non_null.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else non_null.iloc[0]


def _build_entity_text_map(df: pd.DataFrame, entity_col: str) -> dict[str, str]:
    work = df.copy()
    work[entity_col] = work[entity_col].astype(str)
    extra_cols = [c for c in work.columns if c not in _CORE_COLS and c != entity_col]
    grouped = work.groupby(entity_col, dropna=False)
    entity_map: dict[str, str] = {}

    for entity_id, group in grouped:
        parts = [f"{entity_col.lower()} {_normalize_token(entity_id)}"]
        for col in extra_cols:
            value = _dominant_value(group[col])
            if value is None:
                continue
            if pd.api.types.is_number(value):
                parts.append(f"{col.lower()}_{float(value):.3f}")
            else:
                parts.append(f"{col.lower()}_{_normalize_token(value)}")
        entity_map[str(entity_id)] = " ".join(parts)
    return entity_map


def _aligned_text_matrix(
    df: pd.DataFrame,
    entity_col: str,
    entity_ids: list[str],
    max_features: int,
) -> tuple[sp.csr_matrix, TfidfVectorizer]:
    text_map = _build_entity_text_map(df, entity_col)
    corpus = [text_map.get(str(entity_id), f"{entity_col.lower()} {_normalize_token(entity_id)}") for entity_id in entity_ids]
    vectorizer = TfidfVectorizer(max_features=max_features, token_pattern=r"(?u)\b\w+\b")
    matrix = vectorizer.fit_transform(corpus)
    if matrix.shape[1] == 0:
        matrix = sp.csr_matrix(np.ones((len(entity_ids), 1), dtype=np.float32))
    return matrix.tocsr(), vectorizer


def _dense_embeddings(matrix: sp.csr_matrix, n_components: int) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.zeros((0, 1), dtype=np.float32)
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 1), dtype=np.float32)
    max_k = min(matrix.shape[0] - 1, matrix.shape[1] - 1, n_components)
    if max_k >= 2:
        svd = TruncatedSVD(n_components=max_k, random_state=RANDOM_SEED)
        return svd.fit_transform(matrix).astype(np.float32)
    return matrix.toarray().astype(np.float32)


def _row_normalize(matrix: sp.csr_matrix) -> sp.csr_matrix:
    row_sums = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
    row_sums[row_sums == 0] = 1.0
    inv = sp.diags(1.0 / row_sums)
    return inv @ matrix


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
    return _builder.score_matrix_to_ranking_df(im, score_matrix, train, top_k, target_user_ids=target_users)


def _safe_latent_dim(im: InteractionMatrix, requested: int) -> int:
    return max(2, min(int(requested), max(2, im.n_users - 1), max(2, im.n_items - 1)))


def run_lightfm_hybrid(
    train: pd.DataFrame,
    test: pd.DataFrame,
    top_k: int = TOP_K,
    n_components: int = 32,
    content_weight: float = 0.35,
    user_feature_weight: float = 0.20,
    max_features: int = 2000,
    return_model: bool = False,
) -> tuple:
    """
    LightFM-style hybrid recommender using collaborative factors plus side-feature factors.
    """
    im = _builder.build(train)
    k = _safe_latent_dim(im, n_components)

    cf_svd = TruncatedSVD(n_components=min(k, im.n_users - 1, im.n_items - 1), random_state=RANDOM_SEED)
    user_cf = cf_svd.fit_transform(im.matrix).astype(np.float32)
    item_cf = cf_svd.components_.T.astype(np.float32)

    item_text, _ = _aligned_text_matrix(train, "itemID", im.item_ids, max_features=max_features)
    user_text, _ = _aligned_text_matrix(train, "userID", im.user_ids, max_features=max_features)
    item_meta = _dense_embeddings(item_text, k)
    user_meta = _dense_embeddings(user_text, k)

    user_content = (_row_normalize(im.matrix) @ item_meta).astype(np.float32)
    user_side = (content_weight * user_content) + (user_feature_weight * user_meta)
    user_factors = np.concatenate([user_cf, user_side], axis=1)
    item_factors = np.concatenate([item_cf, item_meta], axis=1)

    global_bias = float(train["rating"].mean()) if "rating" in train.columns and len(train) else 0.0
    user_bias = np.asarray(im.matrix.mean(axis=1)).ravel().astype(np.float32)
    item_bias = np.asarray(im.matrix.mean(axis=0)).ravel().astype(np.float32)
    score_matrix = global_bias + user_bias[:, None] + item_bias[None, :] + (user_factors @ item_factors.T)
    score_matrix = score_matrix.astype(np.float32)

    log.info(
        "LightFM-style hybrid trained | k=%d item_features=%d user_features=%d",
        k,
        item_text.shape[1],
        user_text.shape[1],
    )

    rating_preds = _rating_preds(score_matrix, im, test)
    ranking_preds = _ranking_preds(score_matrix, im, train, test, top_k)

    if return_model:
        return rating_preds, ranking_preds, {
            "type": "lightfm_hybrid",
            "global_bias": global_bias,
            "user_bias": user_bias,
            "item_bias": item_bias,
            "user_factors": user_factors,
            "item_factors": item_factors,
            "interaction_matrix": im,
        }
    return rating_preds, ranking_preds


def run_autoencoder_cf(
    train: pd.DataFrame,
    test: pd.DataFrame,
    top_k: int = TOP_K,
    hidden_dim: int = 64,
    alpha: float = 1e-4,
    max_iter: int = 150,
    return_model: bool = False,
) -> tuple:
    """
    Shallow autoencoder collaborative filtering using sklearn MLPRegressor.
    """
    im = _builder.build(train)
    x = im.matrix.toarray().astype(np.float32)
    hidden_dim = max(4, min(int(hidden_dim), max(4, im.n_items // 2 or 4)))

    model = MLPRegressor(
        hidden_layer_sizes=(hidden_dim,),
        activation="relu",
        solver="adam",
        alpha=float(alpha),
        learning_rate_init=1e-3,
        max_iter=int(max_iter),
        early_stopping=True,
        n_iter_no_change=10,
        random_state=RANDOM_SEED,
    )
    model.fit(x, x)
    score_matrix = model.predict(x).astype(np.float32)

    log.info("Autoencoder-CF trained | hidden=%d iter=%d", hidden_dim, getattr(model, "n_iter_", -1))

    rating_preds = _rating_preds(score_matrix, im, test)
    ranking_preds = _ranking_preds(score_matrix, im, train, test, top_k)

    if return_model:
        return rating_preds, ranking_preds, {
            "type": "autoencoder_cf",
            "model": model,
            "item_factors": np.asarray(model.coefs_[0], dtype=np.float32),
            "interaction_matrix": im,
        }
    return rating_preds, ranking_preds


def run_content_tfidf(
    train: pd.DataFrame,
    test: pd.DataFrame,
    top_k: int = TOP_K,
    max_features: int = 4000,
    embedding_dim: int = 64,
    return_model: bool = False,
) -> tuple:
    """
    Content-based recommender using TF-IDF item metadata and user content profiles.
    """
    im = _builder.build(train)
    item_tfidf, _ = _aligned_text_matrix(train, "itemID", im.item_ids, max_features=max_features)
    user_profiles = (_row_normalize(im.matrix) @ item_tfidf).tocsr()
    score_matrix = (user_profiles @ item_tfidf.T).toarray().astype(np.float32)
    item_factors = _dense_embeddings(item_tfidf, embedding_dim)

    log.info("Content TF-IDF trained | vocab=%d embed_dim=%d", item_tfidf.shape[1], item_factors.shape[1])

    rating_preds = _rating_preds(score_matrix, im, test)
    ranking_preds = _ranking_preds(score_matrix, im, train, test, top_k)

    if return_model:
        return rating_preds, ranking_preds, {
            "type": "content_tfidf",
            "score_matrix": score_matrix,
            "item_factors": item_factors,
            "interaction_matrix": im,
        }
    return rating_preds, ranking_preds


def run_factorization_machines(
    train: pd.DataFrame,
    test: pd.DataFrame,
    top_k: int = TOP_K,
    n_factors: int = 24,
    n_epochs: int = 20,
    learning_rate: float = 0.01,
    regularization: float = 0.01,
    max_features: int = 1500,
    return_model: bool = False,
) -> tuple:
    """
    Feature-aware factorization machine style recommender.
    """
    im = _builder.build(train)
    item_text, _ = _aligned_text_matrix(train, "itemID", im.item_ids, max_features=max_features)
    item_side = _dense_embeddings(item_text, min(32, n_factors)).astype(np.float32)

    rng = np.random.default_rng(RANDOM_SEED)
    rows, cols = im.matrix.nonzero()
    values = im.matrix.data.astype(np.float32)
    order = np.arange(len(values))

    global_bias = float(values.mean()) if len(values) else 0.0
    user_bias = np.zeros(im.n_users, dtype=np.float32)
    item_bias = np.zeros(im.n_items, dtype=np.float32)
    linear_side = np.zeros(item_side.shape[1], dtype=np.float32)
    user_factors = rng.normal(0, 0.05, (im.n_users, n_factors)).astype(np.float32)
    item_factors = rng.normal(0, 0.05, (im.n_items, n_factors)).astype(np.float32)
    side_factors = rng.normal(0, 0.05, (item_side.shape[1], n_factors)).astype(np.float32)

    for _ in range(max(1, int(n_epochs))):
        rng.shuffle(order)
        for idx in order:
            u = rows[idx]
            i = cols[idx]
            rating = values[idx]
            x = item_side[i]
            side_vec = x @ side_factors

            p_u = user_factors[u].copy()
            q_i = item_factors[i].copy()
            pred = global_bias + user_bias[u] + item_bias[i] + (x @ linear_side) + (p_u @ (q_i + side_vec))
            err = rating - pred

            user_bias[u] += learning_rate * (err - regularization * user_bias[u])
            item_bias[i] += learning_rate * (err - regularization * item_bias[i])
            linear_side += learning_rate * (err * x - regularization * linear_side)
            user_factors[u] += learning_rate * (err * (q_i + side_vec) - regularization * p_u)
            item_factors[i] += learning_rate * (err * p_u - regularization * q_i)
            side_factors += learning_rate * (err * np.outer(x, p_u) - regularization * side_factors)

    item_latent = item_factors + (item_side @ side_factors)
    item_linear = item_side @ linear_side
    score_matrix = (
        global_bias
        + user_bias[:, None]
        + item_bias[None, :]
        + item_linear[None, :]
        + (user_factors @ item_latent.T)
    ).astype(np.float32)

    log.info("Factorization Machines trained | factors=%d epochs=%d", n_factors, n_epochs)

    rating_preds = _rating_preds(score_matrix, im, test)
    ranking_preds = _ranking_preds(score_matrix, im, train, test, top_k)

    if return_model:
        return rating_preds, ranking_preds, {
            "type": "factorization_machine",
            "global_bias": global_bias,
            "user_bias": user_bias,
            "item_bias": item_bias,
            "item_linear": item_linear.astype(np.float32),
            "user_factors": user_factors,
            "item_factors": item_latent.astype(np.float32),
            "interaction_matrix": im,
        }
    return rating_preds, ranking_preds
