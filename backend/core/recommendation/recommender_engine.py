"""
core/recommendation/recommender_engine.py – Tenant-aware recommendation engine.

v3 Changes vs v2:
  - `RecommenderEngine.__init__` now requires `tenant_id`.
  - All `loader.load_promoted()` calls pass `tenant_id` so models are always
    fetched from tenants/{tenant_id}/models/ — cross-tenant access is impossible.
  - `RecommendationResult` carries `tenant_id` for downstream auditability.
  - All log calls include `tenant=` for clean per-tenant traces.
  - Imports updated from flat v2 layout → core.* v3 hierarchy.
  - Recommendation, scoring, and fallback logic is completely unchanged.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from backend.core.data_processing.interaction_matrix import InteractionMatrix
from backend.core.models.model_loader import LoadedModel, ModelLoader
from backend.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RecommendationRequest:
    user_id: str
    top_n: int = 10
    exclude_seen: bool = True
    algorithm: Optional[str] = None


@dataclass
class RecommendationResult:
    user_id: str
    algorithm: str
    model_id: str
    recommendations: list[dict]   # [{item_id, score, rank}]
    top_n: int
    generated_at: float
    tenant_id: str = ""           # v3: stamped at result creation time


class RecommenderEngine:
    def __init__(
        self,
        loader: ModelLoader,
        interaction_matrix: Optional[InteractionMatrix],
        default_algorithm: str = "SVD",
        tenant_id: str = "",
    ):
        """Initialise a tenant-scoped recommender over a prepared interaction matrix."""
        if not tenant_id or not isinstance(tenant_id, str):
            raise ValueError(
                "tenant_id must be a non-empty string. "
                "Ensure TenantMiddleware has set request.state.tenant_id."
            )
        self.tenant_id        = tenant_id
        self.loader           = loader
        self.im               = interaction_matrix
        self.default_algorithm = default_algorithm

    # ── Public API ─────────────────────────────────────────────────────────────

    def recommend(
        self,
        user_id: str,
        top_n: int = 10,
        algorithm: Optional[str] = None,
        exclude_seen: bool = True,
    ) -> RecommendationResult:
        """Load the active model for an algorithm and return recommendations."""
        algo   = algorithm or self.default_algorithm
        loaded = self.loader.load_promoted(tenant_id=self.tenant_id, algorithm=algo)
        return self.recommend_with_loaded_model(
            user_id=user_id,
            loaded=loaded,
            top_n=top_n,
            algorithm_label=algo,
            exclude_seen=exclude_seen,
        )

    def recommend_with_loaded_model(
        self,
        user_id: str,
        loaded: LoadedModel,
        top_n: int = 10,
        algorithm_label: Optional[str] = None,
        exclude_seen: bool = True,
    ) -> RecommendationResult:
        """Score one user against an already-loaded model artifact."""
        algo  = algorithm_label or loaded.algorithm
        im    = self._resolve_interaction_matrix(loaded)
        top_n = max(1, min(int(top_n), im.n_items))

        u_idx = im.user_index.get(str(user_id))
        if u_idx is None:
            log.warning(
                "User not found — popularity fallback | tenant=%s user=%s",
                self.tenant_id, user_id,
            )
            return self._popularity_fallback(
                user_id, algo, loaded.model_id, top_n, im, self.tenant_id
            )

        scores = self._score_user(loaded, im, u_idx)
        if exclude_seen:
            seen = im.matrix[u_idx].nonzero()[1]
            scores[seen] = -np.inf

        if top_n >= len(scores):
            top_idx = np.argsort(scores)[::-1]
        else:
            top_idx = np.argpartition(scores, -top_n)[-top_n:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        recs = [
            {"item_id": im.item_ids[i], "score": float(scores[i]), "rank": rank + 1}
            for rank, i in enumerate(top_idx)
            if np.isfinite(scores[i])
        ]
        log.debug(
            "recommend | tenant=%s user=%s algo=%s recs=%d",
            self.tenant_id, user_id, algo, len(recs),
        )
        return RecommendationResult(
            user_id=str(user_id),
            algorithm=algo,
            model_id=loaded.model_id,
            recommendations=recs,
            top_n=top_n,
            generated_at=time.time(),
            tenant_id=self.tenant_id,
        )

    def recommend_batch(
        self,
        user_ids: list[str],
        top_n: int = 10,
        algorithm: Optional[str] = None,
    ) -> list[RecommendationResult]:
        """Generate recommendations for a batch of users using one algorithm choice."""
        log.debug(
            "recommend_batch | tenant=%s users=%d top_n=%d",
            self.tenant_id, len(user_ids), top_n,
        )
        return [
            self.recommend(uid, top_n=top_n, algorithm=algorithm)
            for uid in user_ids
        ]

    def similar_items(
        self,
        item_id: str,
        top_n: int = 10,
        algorithm: Optional[str] = None,
    ) -> list[dict]:
        """Return top-N similar items using model factors or a safe fallback."""
        algo   = algorithm or self.default_algorithm
        loaded = self.loader.load_promoted(tenant_id=self.tenant_id, algorithm=algo)
        im     = self._resolve_interaction_matrix(loaded)

        item_factors = self._get_item_factors(loaded)
        if item_factors is None:
            # Fallback: co-occurrence similarity from the interaction matrix.
            # Works for ALS, BPR, and any model that lacks explicit item factors.
            log.info(
                "similar_items: no item factors for algo '%s' — "
                "using co-occurrence fallback | tenant=%s",
                algo, self.tenant_id,
            )
            return self._cooccurrence_similar_items(item_id, top_n, im)

        i_idx = im.item_index.get(str(item_id))
        if i_idx is None:
            raise KeyError(
                f"Item '{item_id}' not found (tenant='{self.tenant_id}')."
            )

        top_n = max(1, min(int(top_n), len(im.item_ids) - 1))
        query  = item_factors[i_idx]
        norms  = np.linalg.norm(item_factors, axis=1, keepdims=True) + 1e-9
        q_norm = np.linalg.norm(query) + 1e-9
        sims   = (item_factors @ query) / (norms.flatten() * q_norm)
        sims[i_idx] = -1.0

        top_idx = np.argpartition(sims, -top_n)[-top_n:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        return [
            {"item_id": im.item_ids[j], "similarity": float(sims[j]), "rank": rank + 1}
            for rank, j in enumerate(top_idx)
        ]

    # ── Interaction matrix resolution ──────────────────────────────────────────

    def _resolve_interaction_matrix(self, loaded: LoadedModel) -> InteractionMatrix:
        model = loaded.model
        if isinstance(model, dict):
            im = model.get("interaction_matrix")
            if isinstance(im, InteractionMatrix):
                return im
        if self.im is None:
            raise RuntimeError(
                f"Model '{loaded.algorithm}' does not include an interaction matrix "
                f"and no serving matrix is configured (tenant='{self.tenant_id}')."
            )
        return self.im

    # ── Scoring ────────────────────────────────────────────────────────────────

    def _score_user(
        self,
        loaded: LoadedModel,
        im: InteractionMatrix,
        u_idx: int,
    ) -> np.ndarray:
        model = loaded.model
        if isinstance(model, dict):
            model_type = model.get("type", "")

            if model_type == "svdpp":
                global_mean  = float(model.get("global_mean", 0.0))
                b_u          = np.asarray(model.get("b_u", []), dtype=np.float64)
                b_i          = np.asarray(model.get("b_i", []), dtype=np.float64)
                P            = np.asarray(model.get("P", []),   dtype=np.float64)
                Q            = np.asarray(model.get("Q", []),   dtype=np.float64)
                Y            = np.asarray(model.get("Y", []),   dtype=np.float64)
                implicit_sets = model.get("implicit_sets", {}) or {}
                nu = implicit_sets.get(u_idx, implicit_sets.get(str(u_idx), [])) or []
                imp = (len(nu) ** -0.5) * Y[nu].sum(axis=0) if len(nu) > 0 else np.zeros(P.shape[1], dtype=np.float64)
                pu = P[u_idx] + imp
                return np.asarray(global_mean + b_u[u_idx] + b_i + (pu @ Q.T), dtype=np.float32)

            if model_type == "movie_item_knn" and "item_sim" in model:
                item_sim = np.asarray(model["item_sim"], dtype=np.float32)
                user_vec = im.matrix[u_idx].toarray().flatten().astype(np.float32)
                denom    = np.abs(item_sim).sum(axis=1) + 1e-9
                return (user_vec @ item_sim) / denom

            if model_type == "ecommerce_popularity":
                item_scores = model.get("item_scores", {})
                eligible    = set(model.get("eligible", im.item_ids))
                return np.asarray(
                    [float(item_scores.get(iid, 0.0)) if iid in eligible else 0.0 for iid in im.item_ids],
                    dtype=np.float32,
                )

            if model_type == "autoencoder_cf":
                inner = model.get("model")
                if inner is not None:
                    user_vec = im.matrix[u_idx].toarray().astype(np.float32)
                    return np.asarray(inner.predict(user_vec)).flatten().astype(np.float32)

            if model_type == "lightfm_hybrid":
                global_bias  = float(model.get("global_bias", 0.0))
                user_bias    = np.asarray(model.get("user_bias", []),    dtype=np.float32)
                item_bias    = np.asarray(model.get("item_bias", []),    dtype=np.float32)
                user_factors = np.asarray(model.get("user_factors", []), dtype=np.float32)
                item_factors = np.asarray(model.get("item_factors", []), dtype=np.float32)
                return (
                    global_bias + user_bias[u_idx] + item_bias + (user_factors[u_idx] @ item_factors.T)
                ).astype(np.float32)

            if model_type == "factorization_machine":
                global_bias  = float(model.get("global_bias", 0.0))
                user_bias    = np.asarray(model.get("user_bias", []),    dtype=np.float32)
                item_bias    = np.asarray(model.get("item_bias", []),    dtype=np.float32)
                item_linear  = np.asarray(model.get("item_linear", []),  dtype=np.float32)
                user_factors = np.asarray(model.get("user_factors", []), dtype=np.float32)
                item_factors = np.asarray(model.get("item_factors", []), dtype=np.float32)
                return (
                    global_bias + user_bias[u_idx] + item_bias + item_linear + (user_factors[u_idx] @ item_factors.T)
                ).astype(np.float32)

            if "score_matrix" in model:
                return np.asarray(model["score_matrix"])[u_idx].astype(np.float32)

            inner = model.get("model")
            if inner is not None:
                if hasattr(inner, "components_") and hasattr(inner, "transform"):
                    u = inner.transform(im.matrix[u_idx])
                    return np.asarray(u @ inner.components_).flatten().astype(np.float32)
                if hasattr(inner, "user_factors") and hasattr(inner, "item_factors"):
                    return (np.asarray(inner.user_factors[u_idx]) @ np.asarray(inner.item_factors).T).astype(np.float32)

            if "user_factors" in model and "item_factors" in model:
                return (np.asarray(model["user_factors"][u_idx]) @ np.asarray(model["item_factors"]).T).astype(np.float32)

        if hasattr(model, "user_factors") and hasattr(model, "item_factors"):
            return (np.asarray(model.user_factors[u_idx]) @ np.asarray(model.item_factors).T).astype(np.float32)
        if hasattr(model, "components_") and hasattr(model, "transform"):
            u = model.transform(im.matrix[u_idx])
            return np.asarray(u @ model.components_).flatten().astype(np.float32)

        return self._get_score_matrix(loaded, im)[u_idx].astype(np.float32)

    def _get_score_matrix(self, loaded: LoadedModel, im: InteractionMatrix) -> np.ndarray:
        model = loaded.model
        if isinstance(model, dict):
            if "model" in model:
                inner = model["model"]
                if hasattr(inner, "components_") and hasattr(inner, "transform"):
                    return inner.transform(im.matrix) @ inner.components_
                if hasattr(inner, "user_factors") and hasattr(inner, "item_factors"):
                    return np.asarray(inner.user_factors) @ np.asarray(inner.item_factors).T
            if "user_factors" in model and "item_factors" in model:
                return np.asarray(model["user_factors"]) @ np.asarray(model["item_factors"]).T
            if "score_matrix" in model:
                return np.asarray(model["score_matrix"])
            if model.get("type") == "ecommerce_popularity":
                scores = np.asarray(
                    [model["item_scores"].get(iid, 0.0) for iid in im.item_ids], dtype=np.float32
                )
                return np.tile(scores, (im.n_users, 1))
        if hasattr(model, "user_factors") and hasattr(model, "item_factors"):
            return np.asarray(model.user_factors) @ np.asarray(model.item_factors).T
        if hasattr(model, "components_") and hasattr(model, "transform"):
            return model.transform(im.matrix) @ model.components_
        if isinstance(model, np.ndarray) and model.ndim == 2:
            return model
        raise TypeError(
            f"Cannot extract score matrix from model type: {type(model).__name__} "
            f"(tenant='{self.tenant_id}')."
        )

    def _cooccurrence_similar_items(
        self,
        item_id: str,
        top_n: int,
        im: InteractionMatrix,
    ) -> list[dict]:
        """
        Cosine-similarity fallback for similar_items when the model does not
        expose item-factor matrices (e.g. ALS via implicit, BPR).

        Uses the item columns of the interaction matrix directly: each item is
        represented by the vector of user interactions with that item.
        """
        i_idx = im.item_index.get(str(item_id))
        if i_idx is None:
            raise KeyError(
                f"Item '{item_id}' not found (tenant='{self.tenant_id}')."
            )

        # item_vecs: shape (n_items, n_users)
        import scipy.sparse as sp
        mat = im.matrix  # (n_users, n_items)
        if sp.issparse(mat):
            item_vecs = np.asarray(mat.T.toarray(), dtype=np.float64)
        else:
            item_vecs = np.asarray(mat, dtype=np.float64).T

        query  = item_vecs[i_idx]
        norms  = np.linalg.norm(item_vecs, axis=1) + 1e-9
        q_norm = np.linalg.norm(query) + 1e-9
        sims   = (item_vecs @ query) / (norms * q_norm)
        sims[i_idx] = -1.0  # exclude the query item

        top_n   = max(1, min(int(top_n), len(im.item_ids) - 1))
        top_idx = np.argpartition(sims, -top_n)[-top_n:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        return [
            {"item_id": im.item_ids[j], "similarity": float(sims[j]), "rank": rank + 1}
            for rank, j in enumerate(top_idx)
        ]

    def _get_item_factors(self, loaded: LoadedModel) -> Optional[np.ndarray]:
        model = loaded.model
        if isinstance(model, dict):
            inner = model.get("model")
            if inner is not None:
                if hasattr(inner, "item_factors"):
                    return np.asarray(inner.item_factors)
                if hasattr(inner, "components_"):
                    return np.asarray(inner.components_).T
            if "item_factors" in model:
                return np.asarray(model["item_factors"])
            if "Q" in model:
                return np.asarray(model["Q"])
            return None
        if hasattr(model, "item_factors"):
            return np.asarray(model.item_factors)
        if hasattr(model, "components_"):
            return np.asarray(model.components_).T
        return None

    # ── Fallback ───────────────────────────────────────────────────────────────

    @staticmethod
    def _popularity_fallback(
        user_id: str,
        algo: str,
        model_id: str,
        top_n: int,
        im: InteractionMatrix,
        tenant_id: str,
    ) -> RecommendationResult:
        item_counts = np.asarray(im.matrix.sum(axis=0)).flatten()
        top_n       = max(1, min(int(top_n), len(item_counts)))
        top_idx     = np.argpartition(item_counts, -top_n)[-top_n:]
        top_idx     = top_idx[np.argsort(item_counts[top_idx])[::-1]]
        recs = [
            {"item_id": im.item_ids[i], "score": float(item_counts[i]), "rank": rank + 1}
            for rank, i in enumerate(top_idx)
        ]
        return RecommendationResult(
            user_id=str(user_id),
            algorithm=f"{algo}+popularity_fallback",
            model_id=model_id,
            recommendations=recs,
            top_n=top_n,
            generated_at=time.time(),
            tenant_id=tenant_id,
        )
