"""
core/pipeline/serving_pipeline.py – Tenant-aware serving pipeline for real-time recommendations.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from backend.core.data_processing.embedding_enricher import EmbeddingStore
from backend.core.data_processing.interaction_matrix import InteractionMatrix, InteractionMatrixBuilder
from backend.core.insights.explainability import ExplainabilityEngine, Explanation
from backend.core.models.model_loader import ModelLoader
from backend.core.models.model_registry import ModelRegistry
from backend.core.recommendation.recommender_engine import RecommendationResult, RecommenderEngine
from backend.utils.logger import get_logger

log = get_logger(__name__)


class ServingPipeline:
    """
    Tenant-scoped serving pipeline for recommendations, similar-item lookup,
    and explainability.
    """

    def __init__(
        self,
        tenant_id: str,
        engine: RecommenderEngine,
        explainer: ExplainabilityEngine,
        model_id: Optional[str] = None,
        embedding_store: Optional[EmbeddingStore] = None,  # content-aware similarity
    ):
        _validate_tenant(tenant_id)
        self.tenant_id       = tenant_id
        self.model_id        = model_id
        self._engine         = engine
        self._explainer      = explainer
        self.embedding_store = embedding_store

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_registry(
        cls,
        tenant_id: str,
        algorithm: str,
        train_df: Optional[pd.DataFrame] = None,
        item_metadata: Optional[pd.DataFrame] = None,
        embedding_store: Optional[EmbeddingStore] = None,  # pass TrainingResult.embedding_store here
    ) -> "ServingPipeline":
        """
        Build a ServingPipeline by loading the promoted model for *algorithm*
        from the tenant-scoped registry.

        Args:
            tenant_id:       The tenant whose promoted model should be loaded.
            algorithm:       Algorithm name (e.g. "svd", "als").
            train_df:        Optional training DataFrame used to rebuild the
                             interaction matrix when it isn't persisted inside
                             the model artifact.
            item_metadata:   Optional item-level metadata forwarded to the
                             ExplainabilityEngine.
            embedding_store: Optional EmbeddingStore from the training pipeline.
                             When provided, similar_items() and recommend()
                             blend collaborative scores with embedding similarity.
        """
        _validate_tenant(tenant_id)

        loader = ModelLoader()
        loaded = loader.load_promoted(tenant_id=tenant_id, algorithm=algorithm)

        model = loaded.model
        im: Optional[InteractionMatrix] = None

        if isinstance(model, dict) and "interaction_matrix" in model:
            im = model["interaction_matrix"]
        elif train_df is not None:
            im = InteractionMatrixBuilder().build(train_df)
        else:
            raise RuntimeError(
                f"No interaction matrix available for tenant='{tenant_id}' "
                f"algorithm='{algorithm}'. "
                "Provide train_df or retrain with the interaction_matrix persisted inside the model artifact."
            )

        engine = RecommenderEngine(
            loader=loader,
            interaction_matrix=im,
            default_algorithm=algorithm,
            tenant_id=tenant_id,
        )

        user_factors, item_factors = cls._extract_latent_factors(model)
        explainer = ExplainabilityEngine(
            im=im,
            user_factors=user_factors,
            item_factors=item_factors,
            item_metadata=item_metadata,
        )

        if embedding_store is not None:
            log.info(
                "ServingPipeline embedding attached | tenant=%s items=%d dim=%d model=%s",
                tenant_id, embedding_store.n_items, embedding_store.dim, embedding_store.model_name,
            )

        log.info(
            "ServingPipeline initialised | tenant=%s algo=%s model_id=%s embedding=%s",
            tenant_id, algorithm, loaded.model_id,
            embedding_store.model_name if embedding_store else "none",
        )
        return cls(
            tenant_id=tenant_id,
            engine=engine,
            explainer=explainer,
            model_id=loaded.model_id,
            embedding_store=embedding_store,
        )

    # ── Recommendation ─────────────────────────────────────────────────────────

    def recommend(
        self,
        user_id: str,
        top_n: int = 10,
        algorithm: Optional[str] = None,
        exclude_seen: bool = True,
        embedding_alpha: float = 0.7,    # CF weight; (1-alpha) = embedding weight
        seed_item_id: Optional[str] = None,  # reference item for content blending
    ) -> RecommendationResult:
        """
        Return top-N recommendations for *user_id*.

        When an EmbeddingStore is attached and *seed_item_id* is provided,
        collaborative scores are blended with embedding similarity so that
        content relevance boosts accuracy for cold/sparse items.
        """
        log.debug(
            "recommend | tenant=%s user=%s top_n=%d algo=%s seed_item=%s",
            self.tenant_id, user_id, top_n, algorithm, seed_item_id,
        )
        result = self._engine.recommend(
            user_id=user_id,
            top_n=top_n,
            algorithm=algorithm,
            exclude_seen=exclude_seen,
        )

        # Blend with embeddings when a store is attached.
        # If no seed_item_id was given, fall back to the user's most recently
        # interacted item so embeddings are never silently bypassed.
        if self.embedding_store is not None:
            effective_seed = seed_item_id
            if effective_seed is None:
                effective_seed = self._last_interacted_item(user_id)
            if effective_seed is not None:
                result = self._blend_result(result, effective_seed, embedding_alpha, top_n)

        return result

    def recommend_batch(
        self,
        user_ids: list[str],
        top_n: int = 10,
        algorithm: Optional[str] = None,
    ) -> list[RecommendationResult]:
        """
        Return top-N recommendations for each user in *user_ids*.
        Results are in the same order as the input list.
        """
        log.debug(
            "recommend_batch | tenant=%s users=%d top_n=%d",
            self.tenant_id, len(user_ids), top_n,
        )
        return self._engine.recommend_batch(
            user_ids=user_ids,
            top_n=top_n,
            algorithm=algorithm,
        )

    def similar_items(
        self,
        item_id: str,
        top_n: int = 10,
        algorithm: Optional[str] = None,
        use_embeddings: bool = True,     # prefer embedding similarity when store is attached
        embedding_alpha: float = 0.5,    # equal blend for item-item (0.5 CF / 0.5 content)
    ) -> list[dict]:
        """
        Return the top-N items most similar to *item_id*.

        When an EmbeddingStore is attached and *use_embeddings=True*, results
        are ranked by a blend of latent-factor similarity and semantic embedding
        similarity — substantially more accurate for cold items.
        """
        log.debug(
            "similar_items | tenant=%s item=%s top_n=%d use_embeddings=%s",
            self.tenant_id, item_id, top_n, use_embeddings,
        )

        if self.embedding_store is not None and use_embeddings:
            # Fast path: use embedding similarity as primary signal
            try:
                emb_similar = self.embedding_store.most_similar(item_id, k=top_n * 2)

                # Get CF scores for the same candidates and blend.
                # similar_items returns list[dict] with key 'similarity', not 'score'.
                cf_scores: dict[str, float] = {}
                try:
                    cf_raw = self._engine.similar_items(
                        item_id=item_id, top_n=top_n * 2, algorithm=algorithm
                    )
                    cf_scores = {
                        rec["item_id"]: rec.get("similarity", rec.get("score", 0.0))
                        for rec in cf_raw
                        if isinstance(rec.get("similarity", rec.get("score")), (int, float))
                    }
                except (KeyError, Exception) as cf_err:
                    log.debug(
                        "similar_items | CF lookup failed for item '%s': %s — using embeddings only.",
                        item_id, cf_err,
                    )

                if cf_scores:
                    blended = self.embedding_store.blend_scores(
                        collaborative_scores=cf_scores,
                        query_item_id=item_id,
                        alpha=embedding_alpha,
                    )
                    ranked = sorted(blended.items(), key=lambda x: x[1], reverse=True)
                else:
                    ranked = emb_similar  # no CF available; pure embedding

                return [
                    {"item_id": iid, "score": float(score)}
                    for iid, score in ranked[:top_n]
                ]
            except KeyError:
                # item_id not in embedding store — fall through to CF-only path
                log.debug(
                    "similar_items | item '%s' not in embedding store — using CF only.", item_id
                )

        # Default: pure collaborative filtering
        return self._engine.similar_items(
            item_id=item_id,
            top_n=top_n,
            algorithm=algorithm,
        )

    # ── Explainability ─────────────────────────────────────────────────────────

    def explain(
        self,
        user_id: str,
        item_id: str,
        score: float = 0.0,
    ) -> Explanation:
        log.debug("explain | tenant=%s user=%s item=%s", self.tenant_id, user_id, item_id)
        return self._explainer.explain(user_id=user_id, item_id=item_id, score=score)

    def explain_recommendations(
        self,
        user_id: str,
        result: RecommendationResult,
    ) -> list[Explanation]:
        log.debug(
            "explain_recommendations | tenant=%s user=%s items=%d",
            self.tenant_id, user_id, len(result.recommendations),
        )
        return self._explainer.explain_batch(user_id, result.recommendations)

    # ── Health ─────────────────────────────────────────────────────────────────

    def health(self) -> dict:
        loaded_models = self._engine.loader.list_loaded(tenant_id=self.tenant_id)
        return {
            "status": "ok",
            "tenant_id": self.tenant_id,
            "loaded_models": loaded_models,
            "embedding_model": self.embedding_store.model_name if self.embedding_store else None,
            "embedding_items": self.embedding_store.n_items if self.embedding_store else 0,
        }

    # ── Internals ──────────────────────────────────────────────────────────────

    def _last_interacted_item(self, user_id: str) -> Optional[str]:
        """
        Return the last item this user interacted with from the interaction
        matrix, to use as an embedding seed when none is explicitly provided.
        Returns None if the user is unknown or the matrix is unavailable.
        """
        try:
            im = self._engine.im
            if im is None:
                return None
            u_idx = im.user_index.get(str(user_id))
            if u_idx is None:
                return None
            row = im.matrix[u_idx]
            nonzero_cols = row.nonzero()[1]
            if len(nonzero_cols) == 0:
                return None
            # Pick the item with the highest interaction value as the proxy
            # for "most recent / strongest signal"
            best_col = nonzero_cols[int(np.argmax(row[0, nonzero_cols].toarray()))]
            return im.item_ids[best_col]
        except Exception:
            return None

    def _blend_result(
        self,
        result: RecommendationResult,
        seed_item_id: str,
        alpha: float,
        top_n: int,
    ) -> RecommendationResult:
        """
        Re-rank the recommendations in *result* by blending CF scores with
        embedding similarity relative to *seed_item_id*.

        Returns a new RecommendationResult with blended scores, top_n items.
        """
        try:
            cf_scores = {r["item_id"]: r["score"] for r in result.recommendations}
            blended = self.embedding_store.blend_scores(
                collaborative_scores=cf_scores,
                query_item_id=seed_item_id,
                alpha=alpha,
            )
            reranked = sorted(blended.items(), key=lambda x: x[1], reverse=True)[:top_n]
            # Rebuild recommendations list preserving any extra fields
            lookup = {r["item_id"]: r for r in result.recommendations}
            new_recs = []
            for iid, score in reranked:
                rec = dict(lookup.get(iid, {"item_id": iid}))
                rec["score"] = score
                new_recs.append(rec)
            return RecommendationResult(
                user_id=result.user_id,
                recommendations=new_recs,
                algorithm=result.algorithm,
                model_id=getattr(result, "model_id", None),
            )
        except Exception as exc:
            log.warning("Embedding blend failed (%s) — returning original CF result.", exc)
            return result

    @staticmethod
    def _extract_latent_factors(
        model,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if isinstance(model, dict):
            inner = model.get("model")
            if inner is not None and hasattr(inner, "user_factors") and hasattr(inner, "item_factors"):
                return np.asarray(inner.user_factors), np.asarray(inner.item_factors)
            if "user_factors" in model and "item_factors" in model:
                return np.asarray(model["user_factors"]), np.asarray(model["item_factors"])
            if "P" in model and "Q" in model:
                return np.asarray(model["P"]), np.asarray(model["Q"])
        elif hasattr(model, "user_factors") and hasattr(model, "item_factors"):
            return np.asarray(model.user_factors), np.asarray(model.item_factors)
        return None, None


# ── Module-level helper ────────────────────────────────────────────────────────

def _validate_tenant(tenant_id: str) -> None:
    if not tenant_id or not isinstance(tenant_id, str):
        raise ValueError(
            "tenant_id must be a non-empty string. "
            "Ensure TenantMiddleware has set request.state.tenant_id before "
            "constructing ServingPipeline."
        )