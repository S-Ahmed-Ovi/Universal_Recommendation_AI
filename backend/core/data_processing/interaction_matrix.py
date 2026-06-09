"""
data_processing/interaction_matrix.py – Sparse interaction matrix construction

Builds a CSR sparse matrix from a cleaned interaction DataFrame.
Used by all matrix-factorisation and neighbourhood-based algorithms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp

from backend.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class InteractionMatrix:
    """
    Container for the sparse user-item interaction matrix plus index mappings.

    Attributes
    ----------
    matrix      : scipy CSR sparse matrix (n_users × n_items)
    user_index  : dict mapping userID → row index
    item_index  : dict mapping itemID → column index
    user_ids    : list of all userIDs (ordered)
    item_ids    : list of all itemIDs (ordered)
    is_implicit : whether ratings are implicit (0/1) or explicit (numeric)
    """
    matrix:     sp.csr_matrix
    user_index: dict
    item_index: dict
    user_ids:   list
    item_ids:   list
    is_implicit: bool

    @property
    def n_users(self) -> int:
        """Return the number of users represented in the matrix."""
        return self.matrix.shape[0]

    @property
    def n_items(self) -> int:
        """Return the number of items represented in the matrix."""
        return self.matrix.shape[1]

    @property
    def density(self) -> float:
        """Return the non-zero density of the sparse interaction matrix."""
        return self.matrix.nnz / (self.n_users * self.n_items)

    def to_dense(self) -> np.ndarray:
        """Convert to a dense numpy array. Use only for small matrices."""
        if self.n_users * self.n_items > 50_000_000:
            raise MemoryError(
                f"Dense conversion would require {self.n_users * self.n_items * 4 / 1e9:.1f} GB. "
                f"Use the sparse matrix directly."
            )
        return self.matrix.toarray()

    def get_user_vector(self, user_id: str) -> np.ndarray:
        """Return the dense interaction vector for a single user."""
        idx = self.user_index.get(str(user_id))
        if idx is None:
            raise KeyError(f"User '{user_id}' not found in interaction matrix.")
        return self.matrix[idx].toarray().flatten()

    def get_item_vector(self, item_id: str) -> np.ndarray:
        """Return the dense interaction vector for a single item."""
        idx = self.item_index.get(str(item_id))
        if idx is None:
            raise KeyError(f"Item '{item_id}' not found in interaction matrix.")
        return self.matrix[:, idx].toarray().flatten()


class InteractionMatrixBuilder:
    """
    Builds an InteractionMatrix from a standardised interaction DataFrame.
    """

    def build(self, df: pd.DataFrame) -> InteractionMatrix:
        """
        Construct the sparse user-item interaction matrix.

        Parameters
        ----------
        df : cleaned DataFrame with columns [userID, itemID, rating]

        Returns
        -------
        InteractionMatrix
        """
        user_ids = sorted(df["userID"].unique().tolist())
        item_ids = sorted(df["itemID"].unique().tolist())

        user_index = {uid: i for i, uid in enumerate(user_ids)}
        item_index = {iid: i for i, iid in enumerate(item_ids)}

        rows   = df["userID"].map(user_index).values
        cols   = df["itemID"].map(item_index).values
        data   = df["rating"].values.astype(np.float32)

        matrix = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(len(user_ids), len(item_ids)),
            dtype=np.float32,
        )

        is_implicit = df.attrs.get("is_implicit", False)

        log.info(
            "Interaction matrix | %d users × %d items | nnz=%d | density=%.4f%% | implicit=%s",
            len(user_ids), len(item_ids), matrix.nnz,
            matrix.nnz / (len(user_ids) * len(item_ids)) * 100,
            is_implicit,
        )

        return InteractionMatrix(
            matrix     = matrix,
            user_index = user_index,
            item_index = item_index,
            user_ids   = user_ids,
            item_ids   = item_ids,
            is_implicit= is_implicit,
        )

    def top_k_for_user(
        self,
        im: InteractionMatrix,
        score_matrix: np.ndarray,
        user_id: str,
        k: int,
        exclude_seen: bool = True,
    ) -> list[Tuple[str, float]]:
        """
        Return top-K (item_id, score) pairs for a user from a dense score matrix.

        Parameters
        ----------
        im           : InteractionMatrix for index lookups
        score_matrix : dense ndarray (n_users × n_items) of predicted scores
        user_id      : user to generate recommendations for
        k            : number of recommendations
        exclude_seen : if True, mask items the user has already interacted with
        """
        u_idx  = im.user_index.get(str(user_id))
        if u_idx is None:
            return []

        scores = score_matrix[u_idx].copy()

        if exclude_seen:
            seen_items = im.matrix[u_idx].nonzero()[1]
            scores[seen_items] = -np.inf

        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(im.item_ids[i], float(scores[i])) for i in top_indices if scores[i] > -np.inf]

    def score_matrix_to_ranking_df(
        self,
        im: InteractionMatrix,
        score_matrix: np.ndarray,
        train_df: pd.DataFrame,
        k: int,
        target_user_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Convert a full score matrix into a ranking DataFrame.

        Returns DataFrame with columns: [userID, itemID, score]
        """
        records = []
        k = max(1, min(int(k), im.n_items))
        seen_per_user = train_df.groupby("userID")["itemID"].apply(set).to_dict()
        users = target_user_ids or im.user_ids
        for user_id in users:
            if user_id not in im.user_index:
                continue
            u_idx  = im.user_index[user_id]
            scores = score_matrix[u_idx].copy()
            seen   = seen_per_user.get(user_id, set())

            for item_id in seen:
                i_idx = im.item_index.get(item_id)
                if i_idx is not None:
                    scores[i_idx] = -np.inf

            top_idx = np.argpartition(scores, -k)[-k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

            for i in top_idx:
                if scores[i] > -np.inf:
                    records.append({"userID": user_id, "itemID": im.item_ids[i], "score": float(scores[i])})

        return pd.DataFrame(records)
