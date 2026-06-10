"""
core/data_processing/embedding_enricher.py

Generates dense item embeddings and stores them in OpenSearch.
Each tenant gets an isolated index: embeddings_{tenant_id}

Replaces the previous in-memory NumPy EmbeddingStore with a
persistent, multi-tenant-safe vector store backed by OpenSearch.
"""
from __future__ import annotations

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional

from opensearchpy import OpenSearch, helpers

log = logging.getLogger(__name__)

DEFAULT_MODEL  = "all-MiniLM-L6-v2"
EMBEDDING_DIM  = 384
INDEX_PREFIX   = "proactive-ai"
BULK_CHUNK     = 500   # documents per bulk upsert batch


# ── OpenSearch connection ──────────────────────────────────────────────────────

def _get_client() -> OpenSearch:
    """
    Build an OpenSearch client from environment variables.

    Required env vars:
        OPENSEARCH_HOST      e.g. localhost or your-cluster.us-east-1.es.amazonaws.com
        OPENSEARCH_PORT      e.g. 9200
        OPENSEARCH_USER      e.g. admin
        OPENSEARCH_PASSWORD  e.g. your-password
        OPENSEARCH_USE_SSL   true / false  (default false for local dev)
    """
    host     = os.environ.get("OPENSEARCH_HOST", "localhost")
    port     = int(os.environ.get("OPENSEARCH_PORT", 9200))
    user     = os.environ.get("OPENSEARCH_USER", "admin")
    password = os.environ.get("OPENSEARCH_PASSWORD", "@Aa1510778084")
    use_ssl  = os.environ.get("OPENSEARCH_USE_SSL", "false").lower() == "true"

    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=(user, password),
        use_ssl=use_ssl,
        verify_certs=use_ssl,
        ssl_show_warn=False,
        timeout=30,
        retry_on_timeout=True,
        max_retries=3,
    )


# ── Index name helper ──────────────────────────────────────────────────────────

def _index_name(tenant_id: str) -> str:
    # Lowercase + replace special chars — OpenSearch index names are strict
    safe = tenant_id.lower().replace(" ", "-").replace("_", "-")
    return f"{INDEX_PREFIX}-{safe}"


# ── Index mapping ──────────────────────────────────────────────────────────────

def _index_mapping(dim: int) -> dict:
    return {
        "settings": {
            "index": {
                "knn": True,                      # enable k-NN plugin
                "knn.algo_param.ef_search": 512,  # recall vs speed tradeoff
            }
        },
        "mappings": {
            "properties": {
                "item_id":    {"type": "keyword"},
                "embedding":  {
                    "type":       "knn_vector",
                    "dimension":  dim,
                    "method": {
                        "name":       "hnsw",        # Hierarchical NSW — best for ANN
                        "space_type": "cosinesimil", # cosine similarity
                        "engine":     "nmslib",
                        "parameters": {
                            "ef_construction": 512,
                            "m":               16,
                        }
                    }
                },
                "text":       {"type": "text"},
                "model":      {"type": "keyword"},
                "updated_at": {"type": "date"},
            }
        }
    }


# ── EmbeddingStore (OpenSearch-backed) ────────────────────────────────────────

@dataclass
class EmbeddingStore:
    """
    Thin handle returned after building embeddings.
    All actual data lives in OpenSearch — this object holds
    connection info and exposes the same query interface as before
    so the serving pipeline needs zero changes.
    """
    tenant_id:  str
    model_name: str
    n_items:    int
    dim:        int
    _client:    OpenSearch = None

    def __post_init__(self):
        if self._client is None:
            self._client = _get_client()
        self._index = _index_name(self.tenant_id)

    # ── Similarity queries ─────────────────────────────────────────────────────

    def get_vector(self, item_id: str) -> np.ndarray:
        """Fetch a single item's vector from OpenSearch."""
        resp = self._client.get(
            index=self._index,
            id=str(item_id),
            _source=["embedding"],
        )
        return np.array(resp["_source"]["embedding"], dtype=np.float32)

    def similarity(self, item_a: str, item_b: str) -> float:
        """Cosine similarity between two items."""
        va = self.get_vector(item_a)
        vb = self.get_vector(item_b)
        return float(np.dot(va, vb))  # vectors are L2-normalised → dot = cosine

    def most_similar(
        self,
        item_id: str,
        k: int = 10,
        exclude_self: bool = True,
        filters: Optional[dict] = None,
    ) -> list[tuple[str, float]]:
        """
        Return the top-k most similar items using OpenSearch k-NN.

        Parameters
        ----------
        item_id      : query item
        k            : number of results
        exclude_self : remove the query item from results
        filters      : optional OpenSearch filter clause, e.g.
                       {"term": {"genre": "Action"}}
                       Allows filtered ANN — not possible with in-memory store.
        """
        vector = self.get_vector(item_id)
        fetch_k = k + 1 if exclude_self else k

        query: dict = {
            "size": fetch_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": vector.tolist(),
                        "k":      fetch_k,
                    }
                }
            },
            "_source": ["item_id"],
        }

        # Optional filter — narrows the ANN search space
        if filters:
            query["query"] = {
                "bool": {
                    "must":   [query["query"]],
                    "filter": [filters],
                }
            }

        resp    = self._client.search(index=self._index, body=query)
        results = []
        for hit in resp["hits"]["hits"]:
            iid   = hit["_source"]["item_id"]
            score = float(hit["_score"])
            if exclude_self and iid == str(item_id):
                continue
            results.append((iid, score))

        return results[:k]

    def score_candidates(
        self,
        query_item_id: str,
        candidate_ids: list[str],
    ) -> np.ndarray:
        """
        Score a specific list of candidates against a query item.
        Used for blending CF shortlists with embedding similarity.
        Fetches candidate vectors in one bulk mget call.
        """
        query_vec = self.get_vector(query_item_id)

        # Bulk fetch candidate vectors
        resp = self._client.mget(
            index=self._index,
            body={"ids": [str(c) for c in candidate_ids]},
            _source=["item_id", "embedding"],
        )
        id_to_vec = {}
        for doc in resp["docs"]:
            if doc.get("found"):
                id_to_vec[doc["_source"]["item_id"]] = np.array(
                    doc["_source"]["embedding"], dtype=np.float32
                )

        scores = np.zeros(len(candidate_ids), dtype=np.float32)
        for i, cid in enumerate(candidate_ids):
            vec = id_to_vec.get(str(cid))
            if vec is not None:
                scores[i] = float(np.dot(query_vec, vec))

        return scores

    def blend_scores(
        self,
        collaborative_scores: dict[str, float],
        query_item_id: Optional[str] = None,
        alpha: float = 0.7,
    ) -> dict[str, float]:
        """
        Blend CF scores with embedding similarity.
        alpha = CF weight, (1-alpha) = embedding weight.
        """
        if not collaborative_scores:
            return collaborative_scores

        cf_ids  = list(collaborative_scores.keys())
        cf_vals = np.array(
            [collaborative_scores[i] for i in cf_ids], dtype=np.float64
        )

        # Normalise CF to [0, 1]
        cf_range = cf_vals.max() - cf_vals.min()
        cf_norm  = (cf_vals - cf_vals.min()) / cf_range if cf_range > 0 else np.ones_like(cf_vals)

        if query_item_id and self._item_exists(str(query_item_id)):
            emb = self.score_candidates(query_item_id, cf_ids).astype(np.float64)
            emb = (emb + 1) / 2   # cosine [-1,1] → [0,1]
        else:
            emb = np.zeros(len(cf_ids), dtype=np.float64)

        blended = alpha * cf_norm + (1 - alpha) * emb
        return {iid: float(s) for iid, s in zip(cf_ids, blended)}

    # ── Internals ──────────────────────────────────────────────────────────────

    def _item_exists(self, item_id: str) -> bool:
        return self._client.exists(index=self._index, id=item_id)


# ── EmbeddingEnricher ─────────────────────────────────────────────────────────

class EmbeddingEnricher:
    """
    Generates embeddings for all items and upserts them into
    the tenant's OpenSearch index.

    One index per tenant → full isolation, no cross-tenant leakage.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model     = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                log.info("Loading embedding model: %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install: pip install sentence-transformers"
                )
        return self._model

    def build(
        self,
        tenant_id: str,                          # required now — determines index
        item_ids: list[str],
        metadata_df: Optional[pd.DataFrame] = None,
        text_col: Optional[str] = None,
        batch_size: int = 512,
        show_progress: bool = True,
    ) -> EmbeddingStore:
        """
        Encode all items and upsert into OpenSearch.

        Parameters
        ----------
        tenant_id    : determines which OpenSearch index to write to
        item_ids     : deduplicated item IDs from the interaction matrix
        metadata_df  : optional DataFrame with 'itemID' + text_col
        text_col     : column to embed; falls back to item ID strings if None
        batch_size   : encoding batch size
        show_progress: tqdm progress bar during encoding
        """
        client = _get_client()
        index  = _index_name(tenant_id)
        model  = self._load_model()

        # Ensure index exists with correct k-NN mapping
        self._ensure_index(client, index)

        # Build text representations
        texts = self._build_texts(item_ids, metadata_df, text_col)

        log.info(
            "Encoding %d items | tenant=%s model=%s",
            len(texts), tenant_id, self.model_name,
        )

        vectors = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalise → cosine = dot product
        ).astype(np.float32)

        # Bulk upsert into OpenSearch
        now = datetime.now(timezone.utc).isoformat()
        self._bulk_upsert(client, index, item_ids, vectors, texts, now)

        log.info(
            "Embedding upsert complete | tenant=%s index=%s items=%d dim=%d",
            tenant_id, index, len(item_ids), vectors.shape[1],
        )

        return EmbeddingStore(
            tenant_id=tenant_id,
            model_name=self.model_name,
            n_items=len(item_ids),
            dim=int(vectors.shape[1]),
            _client=client,
        )

    # ── Index management ───────────────────────────────────────────────────────

    @staticmethod
    def _ensure_index(client: OpenSearch, index: str) -> None:
        """Create the index if it doesn't exist yet."""
        if not client.indices.exists(index=index):
            client.indices.create(
                index=index,
                body=_index_mapping(EMBEDDING_DIM),
            )
            log.info("Created OpenSearch index: %s", index)

    def delete_tenant_index(self, tenant_id: str) -> None:
        """
        Hard-delete all embeddings for a tenant.
        Call this when a tenant is offboarded.
        """
        client = _get_client()
        index  = _index_name(tenant_id)
        if client.indices.exists(index=index):
            client.indices.delete(index=index)
            log.info("Deleted OpenSearch index: %s", index)

    # ── Bulk upsert ────────────────────────────────────────────────────────────

    def _bulk_upsert(
        self,
        client: OpenSearch,
        index: str,
        item_ids: list[str],
        vectors: np.ndarray,
        texts: list[str],
        timestamp: str,
    ) -> None:
        """
        Upsert all item vectors in chunks.
        Uses index (not create) so re-training updates existing vectors.
        """
        def _actions():
            for iid, vec, txt in zip(item_ids, vectors, texts):
                yield {
                    "_op_type": "index",   # upsert — overwrites if exists
                    "_index":   index,
                    "_id":      str(iid),
                    "_source": {
                        "item_id":    str(iid),
                        "embedding":  vec.tolist(),
                        "text":       txt,
                        "model":      self.model_name,
                        "updated_at": timestamp,
                    }
                }

        success, failed = helpers.bulk(
            client,
            _actions(),
            chunk_size=BULK_CHUNK,
            raise_on_error=False,
        )
        if failed:
            log.warning(
                "Bulk upsert: %d succeeded, %d failed | index=%s",
                success, len(failed), index,
            )
        else:
            log.info(
                "Bulk upsert: %d documents | index=%s", success, index
            )

    # ── Text building (unchanged from before) ─────────────────────────────────

    @staticmethod
    def _build_texts(
        item_ids: list[str],
        metadata_df: Optional[pd.DataFrame],
        text_col: Optional[str],
    ) -> list[str]:
        if metadata_df is None or text_col is None:
            log.warning(
                "No metadata — embedding item ID strings. "
                "Pass metadata_df + text_col for better accuracy."
            )
            return [str(iid) for iid in item_ids]

        if text_col not in metadata_df.columns:
            raise ValueError(
                f"text_col '{text_col}' not in metadata_df. "
                f"Available: {list(metadata_df.columns)}"
            )

        meta_series = (
            metadata_df.set_index("itemID")[text_col].astype(str)
            if "itemID" in metadata_df.columns
            else metadata_df[text_col].astype(str)
        )
        # Normalise the index to str — raw DataFrames may have int/float item IDs
        # (e.g. MovieLens movieId=1,2,3) while InteractionMatrix.item_ids are always
        # str after DataCleaner.clean(). Without this cast, every lookup misses and
        # the enricher silently falls back to embedding bare item ID strings.
        meta_series.index = meta_series.index.astype(str)
        # Convert to plain dict for O(1) scalar lookups — avoids the
        # ambiguous pd.Series.get() behaviour when the index has duplicates
        # (it returns a Series instead of a scalar, causing NaN fall-through).
        meta: dict[str, str] = meta_series.to_dict()

        texts, missing = [], 0
        for iid in item_ids:
            text = meta.get(str(iid))
            if text is None or text == "nan":
                texts.append(str(iid))
                missing += 1
            else:
                texts.append(str(text))

        if missing:
            log.warning(
                "%d / %d items had no metadata — fell back to item ID string.",
                missing, len(item_ids),
            )
        return texts
