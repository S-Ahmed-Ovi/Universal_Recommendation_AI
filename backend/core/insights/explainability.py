"""
insights/explainability.py – Recommendation explainability engine

Explains WHY an item was recommended to a user.
Techniques:
  - Item similarity analysis (items the user liked → similar items)
  - User similarity analysis (similar users liked this item)
  - Feature importance (which latent dimensions drove the score)
  - Popularity-based explanations
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from backend.core.data_processing.interaction_matrix import InteractionMatrix
from backend.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Explanation:
    """Structured explanation for a single recommendation."""
    user_id:       str
    item_id:       str
    score:         float
    primary_reason: str
    supporting_items: list[dict]   # items user liked that led to this rec
    similar_users:    list[dict]   # similar users who interacted with this item
    feature_importance: list[dict] # top latent dimensions
    confidence:    str             # HIGH | MEDIUM | LOW


class ExplainabilityEngine:
    """
    Generates human-readable explanations for recommendations.

    Usage
    -----
    engine = ExplainabilityEngine(interaction_matrix, item_factors, user_factors)
    exp    = engine.explain("user_123", "item_456", score=0.87)
    print(exp.primary_reason)
    """

    def __init__(
        self,
        im:           InteractionMatrix,
        item_factors: Optional[np.ndarray] = None,   # (n_items, n_factors)
        user_factors: Optional[np.ndarray] = None,   # (n_users, n_factors)
        item_metadata: Optional[pd.DataFrame] = None, # optional item titles/categories
    ):
        """Cache the matrices and optional metadata needed for repeated explanations."""
        self.im           = im
        self.item_factors = item_factors
        self.user_factors = user_factors
        self.item_metadata = item_metadata
        self._item_titles = self._build_item_title_lookup(item_metadata)

    def explain(
        self,
        user_id: str,
        item_id: str,
        score: float = 0.0,
        n_similar_items: int = 3,
        n_similar_users: int = 3,
    ) -> Explanation:
        """
        Generate a full explanation for why item_id was recommended to user_id.
        """
        u_idx = self.im.user_index.get(str(user_id))
        i_idx = self.im.item_index.get(str(item_id))

        supporting_items   = []
        similar_users      = []
        feature_importance = []

        if u_idx is not None and i_idx is not None:
            supporting_items = self._similar_items_in_history(
                u_idx, i_idx, n_similar_items
            )
            similar_users = self._similar_users_for_item(
                u_idx, i_idx, n_similar_users
            )
            if self.item_factors is not None and self.user_factors is not None:
                feature_importance = self._feature_importance(u_idx, i_idx)

        primary_reason = self._build_primary_reason(
            user_id, item_id, supporting_items, similar_users, score
        )
        confidence = self._confidence(score, len(supporting_items), len(similar_users))

        log.debug("Explained | user=%s item=%s reason=%s", user_id, item_id, primary_reason)
        return Explanation(
            user_id           = user_id,
            item_id           = item_id,
            score             = score,
            primary_reason    = primary_reason,
            supporting_items  = supporting_items,
            similar_users     = similar_users,
            feature_importance = feature_importance,
            confidence        = confidence,
        )

    def explain_batch(
        self,
        user_id: str,
        recommendations: list[dict],
    ) -> list[Explanation]:
        """Explain a list of recommendations for one user."""
        return [
            self.explain(user_id, rec["item_id"], score=rec.get("score", 0.0))
            for rec in recommendations
        ]

    # ── Item similarity analysis ───────────────────────────────────────────────

    def _similar_items_in_history(
        self, u_idx: int, i_idx: int, n: int
    ) -> list[dict]:
        """
        Find items in the user's history that are most similar to item_id.
        Uses cosine similarity of item latent factors if available,
        otherwise falls back to co-occurrence.
        """
        if self.item_factors is None:
            return self._cooccurrence_items(u_idx, i_idx, n)

        seen_indices = self.im.matrix.getrow(u_idx).indices
        candidate_indices = seen_indices[seen_indices != i_idx]
        if candidate_indices.size == 0:
            return []

        target_vec = self.item_factors[i_idx]
        candidate_vectors = self.item_factors[candidate_indices]
        target_norm = float(np.linalg.norm(target_vec)) + 1e-9
        candidate_norms = np.linalg.norm(candidate_vectors, axis=1) + 1e-9
        sims = (candidate_vectors @ target_vec) / (candidate_norms * target_norm + 1e-9)
        top_positions = self._top_k_desc(sims, n)

        return [
            {
                "item_id": self.im.item_ids[candidate_indices[pos]],
                "similarity": round(float(sims[pos]), 4),
                "title": self._item_title(self.im.item_ids[candidate_indices[pos]]),
            }
            for pos in top_positions
        ]

    def _cooccurrence_items(self, u_idx, i_idx, n) -> list[dict]:
        """Fallback: items seen by users who also interacted with item_id."""
        seen_items = self.im.matrix.getrow(u_idx).indices[: max(int(n), 0)]
        return [
            {"item_id": self.im.item_ids[j], "similarity": None, "title": self._item_title(self.im.item_ids[j])}
            for j in seen_items
        ]

    # ── User similarity analysis ───────────────────────────────────────────────

    def _similar_users_for_item(
        self, u_idx: int, i_idx: int, n: int
    ) -> list[dict]:
        """
        Find users similar to the target user who also interacted with item_id.
        """
        if self.user_factors is None:
            # Fallback: just return users who interacted with item_id
            users_who_interacted = self.im.matrix[:, i_idx].nonzero()[0]
            return [
                {"user_id": self.im.user_ids[u], "similarity": None}
                for u in users_who_interacted[:n]
                if u != u_idx
            ]

        users_with_item = self.im.matrix[:, i_idx].nonzero()[0]
        candidate_indices = users_with_item[users_with_item != u_idx]
        if candidate_indices.size == 0:
            return []

        target_user = self.user_factors[u_idx]
        candidate_vectors = self.user_factors[candidate_indices]
        target_norm = float(np.linalg.norm(target_user)) + 1e-9
        candidate_norms = np.linalg.norm(candidate_vectors, axis=1) + 1e-9
        sims = (candidate_vectors @ target_user) / (candidate_norms * target_norm + 1e-9)
        top_positions = self._top_k_desc(sims, n)

        return [
            {
                "user_id": self.im.user_ids[candidate_indices[pos]],
                "similarity": round(float(sims[pos]), 4),
            }
            for pos in top_positions
        ]

    # ── Feature importance ─────────────────────────────────────────────────────

    def _feature_importance(self, u_idx: int, i_idx: int, top_k: int = 5) -> list[dict]:
        """
        Compute element-wise product of user and item latent vectors.
        The top dimensions indicate which latent features drove the score.
        """
        u_vec   = self.user_factors[u_idx]
        i_vec   = self.item_factors[i_idx]
        contrib = u_vec * i_vec
        top_idx = self._top_k_desc(np.abs(contrib), top_k)
        return [
            {"dimension": int(d), "contribution": round(float(contrib[d]), 4)}
            for d in top_idx
        ]

    # ── Primary reason builder ─────────────────────────────────────────────────

    @staticmethod
    def _build_primary_reason(user_id, item_id, supporting_items, similar_users, score) -> str:
        """Choose the clearest human-facing summary from the strongest explanation signals."""
        if supporting_items:
            titles = [s.get("title") or s.get("item_id") for s in supporting_items[:2]]
            return f"Recommended because you interacted with similar items: {', '.join(titles)}"
        if similar_users:
            return (
                f"Users with similar taste to you also interacted with this item "
                f"({len(similar_users)} similar users found)"
            )
        if score > 0.7:
            return "Highly rated item that matches your interaction patterns"
        return "Recommended based on your overall interaction history"

    @staticmethod
    def _confidence(score: float, n_items: int, n_users: int) -> str:
        """Map explanation evidence strength into a coarse confidence label."""
        if n_items >= 2 or n_users >= 2:  return "HIGH"
        if n_items >= 1 or n_users >= 1:  return "MEDIUM"
        return "LOW"

    @staticmethod
    def _top_k_desc(values: np.ndarray, k: int) -> np.ndarray:
        """Return indices for the top-k values in descending order without a full sort."""
        if values.size == 0 or k <= 0:
            return np.asarray([], dtype=np.int64)
        limit = min(int(k), values.size)
        if limit == values.size:
            return np.argsort(values)[::-1]
        top_idx = np.argpartition(values, -limit)[-limit:]
        return top_idx[np.argsort(values[top_idx])[::-1]]

    @staticmethod
    def _build_item_title_lookup(item_metadata: Optional[pd.DataFrame]) -> dict[str, str]:
        """Build a one-pass item-id to title lookup for repeated explanation reads."""
        if (
            item_metadata is None
            or item_metadata.empty
            or "itemID" not in item_metadata.columns
        ):
            return {}

        title_col = next(
            (col for col in ("title", "name", "description") if col in item_metadata.columns),
            None,
        )
        if title_col is None:
            return {}

        frame = item_metadata[["itemID", title_col]].dropna(subset=["itemID"]).copy()
        frame["itemID"] = frame["itemID"].astype(str)
        frame = frame.drop_duplicates(subset=["itemID"], keep="first")
        return {
            str(row["itemID"]): str(row[title_col])
            for row in frame.to_dict(orient="records")
            if row.get(title_col) is not None
        }

    def _item_title(self, item_id: str) -> Optional[str]:
        """Resolve an item title from the cached metadata lookup when available."""
        return self._item_titles.get(str(item_id))
