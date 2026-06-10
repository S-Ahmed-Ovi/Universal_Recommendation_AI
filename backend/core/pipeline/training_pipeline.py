"""
core/pipeline/training_pipeline.py – Tenant-aware end-to-end training pipeline.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional


import pandas as pd
from colorama import Fore, Style

from backend.core.algorithms import ALGORITHM_REGISTRY
from backend.core.benchmark.benchmark_engine import BenchmarkEngine, BenchmarkReport
from backend.config.database import NoSQLConfig, SQLConfig
from backend.core.data_processing.embedding_enricher import EmbeddingEnricher, EmbeddingStore
from backend.config.settings import (
    ALGORITHM_MODES,
    BASE_TENANTS_DIR,
    DEFAULT_ALGORITHM_MODE,
    OPTUNA_TRIALS_AUTO,
    TOP_K,
    TOP_K_ALLOWED,
    TOP_MODEL_ALLOWED,
    TOP_N_MODELS,
)
from backend.core.data_processing.data_cleaning import DataCleaner
from backend.core.data_processing.dataset_analyzer import DatasetAnalyzer
from backend.core.data_processing.feedback_detector import FeedbackDetector
from backend.core.data_processing.interaction_matrix import InteractionMatrixBuilder
from backend.core.deployment.control_plane import DeploymentControlPlane
from backend.core.models.model_registry import ModelRegistry
from backend.core.optimization.optuna_tuner import OptunaTuner, TuningResult, get_optuna_trial_budgets
from backend.smart_db_csv_builder.connectors.csv import TenantFileLoader
from backend.smart_db_csv_builder.connectors.nosql import NoSQLConnector
from backend.smart_db_csv_builder.services.executor import SQLExecutor
from backend.utils.logger import get_logger

from pathlib import Path

log = get_logger(__name__)


@dataclass
class TrainingConfig:
    top_k: int = TOP_K
    n_tuning_trials: int = OPTUNA_TRIALS_AUTO
    top_model_count: int = TOP_N_MODELS
    algorithm_mode: str = DEFAULT_ALGORITHM_MODE
    force_all_algos: bool = False
    interactive: bool = False
    save_model: bool = True
    auto_promote: bool = True
    # ── Embedding config ───────────────────────────────────────────────────────
    embedding_enabled: bool = True           # set False to skip embedding step entirely
    embedding_model: str = "all-MiniLM-L6-v2"  # swap to "all-mpnet-base-v2" for higher accuracy
    embedding_text_col: Optional[str] = None    # metadata column to embed (e.g. "title", "description")
    embedding_alpha: float = 0.7             # CF weight in blend_scores (1-alpha = embedding weight)

    def validate(self) -> None:
        """Validate top-k, shortlist size, and algorithm mode configuration bounds."""
        if self.top_k not in TOP_K_ALLOWED:
            raise ValueError(f"top_k must be one of {TOP_K_ALLOWED}, got {self.top_k}")
        if self.top_model_count not in TOP_MODEL_ALLOWED:
            raise ValueError(
                f"top_model_count must be one of {TOP_MODEL_ALLOWED}, got {self.top_model_count}"
            )
        if self.algorithm_mode not in ALGORITHM_MODES:
            raise ValueError(
                f"algorithm_mode must be one of {ALGORITHM_MODES}, got {self.algorithm_mode}"
            )
        if not (0.0 <= self.embedding_alpha <= 1.0):
            raise ValueError(
                f"embedding_alpha must be between 0.0 and 1.0, got {self.embedding_alpha}"
            )


@dataclass
class TrainingResult:
    tenant_id: str
    report: BenchmarkReport
    tuning_results: list[TuningResult]
    best_model_id: Optional[str]
    best_algorithm: Optional[str]
    best_params: dict
    elapsed_s: float
    all_model_ids: list[str] = field(default_factory=list)
    resolved_mode: str = "implicit"
    feedback_profile: dict = field(default_factory=dict)
    top_model_recommendations: list[dict] = field(default_factory=list)
    ranking_logic: dict = field(default_factory=dict)
    optuna_note: str = ""
    optuna_policy: dict = field(default_factory=dict)
    model_selection_policy: dict = field(default_factory=dict)
    embedding_store: Optional[EmbeddingStore] = None   # populated when embedding_enabled=True

    @property
    def all_tuning_results(self) -> list[TuningResult]:
        return self.tuning_results


class TrainingPipeline:
    """
    End-to-end training pipeline, scoped to a single tenant.
    """

    def __init__(
        self,
        tenant_id: str,
        config: Optional[TrainingConfig] = None,
    ):
        self._validate_tenant(tenant_id)
        self.tenant_id = tenant_id
        self.config = config or TrainingConfig()
        self.config.validate()

        self._tenant_data_dir: Path = (
            Path(BASE_TENANTS_DIR) / tenant_id / "data"
        ).resolve()
        self._tenant_data_dir.mkdir(parents=True, exist_ok=True)

        self.analyzer  = DatasetAnalyzer()
        self.detector  = FeedbackDetector()
        self.cleaner   = DataCleaner()
        self.builder   = InteractionMatrixBuilder()
        self.benchmark = BenchmarkEngine()
        self.tuner     = OptunaTuner()
        self.registry  = ModelRegistry(tenant_id=tenant_id)

        # Embedding enricher — instantiated only when enabled
        self.embedding_enricher = (
            EmbeddingEnricher(model_name=self.config.embedding_model)
            if self.config.embedding_enabled
            else None
        )

        log.info(
            "TrainingPipeline ready | tenant=%s algo_mode=%s top_k=%d embedding=%s",
            self.tenant_id, self.config.algorithm_mode, self.config.top_k,
            self.config.embedding_model if self.config.embedding_enabled else "disabled",
        )

    # ── Public entry-points ────────────────────────────────────────────────────

    def run_from_file(self, path: str, file_format: str = "auto") -> TrainingResult:
        self._validate_tenant(self.tenant_id)
        resolved = self._resolve_data_path(path)
        print(f"\n{Fore.CYAN}[tenant={self.tenant_id}] Loading from file: {resolved}{Style.RESET_ALL}")
        log.info("run_from_file | tenant=%s path=%s", self.tenant_id, resolved)
        loader = TenantFileLoader(tenant_id=self.tenant_id)
        df = loader.load(str(resolved), file_format=file_format)
        return self._run(df)

    def run_from_dataframe(self, df: pd.DataFrame) -> TrainingResult:
        self._validate_tenant(self.tenant_id)
        print(f"\n{Fore.CYAN}[tenant={self.tenant_id}] Running from in-memory DataFrame{Style.RESET_ALL}")
        log.info("run_from_dataframe | tenant=%s rows=%d", self.tenant_id, len(df))
        return self._run(df)

    def run_from_sql(
        self,
        sql_config: SQLConfig,
        sql: str,
        params: Optional[dict] = None,
    ) -> TrainingResult:
        self._validate_tenant(self.tenant_id)
        print(
            f"\n{Fore.CYAN}[tenant={self.tenant_id}] "
            f"Connecting to {sql_config.dialect} database...{Style.RESET_ALL}"
        )
        log.info("run_from_sql | tenant=%s dialect=%s", self.tenant_id, sql_config.dialect)
        executor = SQLExecutor(sql_config, tenant_id=self.tenant_id)
        query_result = executor.execute_custom(sql, params)
        return self._run(query_result.df)

    def run_from_nosql(
        self,
        nosql_config: NoSQLConfig,
        collection: str,
        query: Optional[dict] = None,
    ) -> TrainingResult:
        self._validate_tenant(self.tenant_id)
        print(
            f"\n{Fore.CYAN}[tenant={self.tenant_id}] "
            f"Connecting to {nosql_config.engine}...{Style.RESET_ALL}"
        )
        log.info("run_from_nosql | tenant=%s engine=%s", self.tenant_id, nosql_config.engine)
        with NoSQLConnector(nosql_config) as conn:
            if nosql_config.engine == "mongodb":
                df = conn.fetch_collection(collection, query)
            elif nosql_config.engine == "dynamodb":
                df = conn.fetch_dynamodb_table(collection)
            else:
                raise ValueError(f"Unsupported NoSQL engine: {nosql_config.engine}")
        return self._run(df)

    # ── Core pipeline ──────────────────────────────────────────────────────────

    def _run(self, raw_df: pd.DataFrame) -> TrainingResult:
        t0 = time.time()
        self._print_header(self.tenant_id)
        print(f"Rows: {raw_df.shape[0]:,} | Columns: {raw_df.shape[1]}")

        print(f"\n{Fore.YELLOW}[1/7] Column Detection{Style.RESET_ALL}")
        mapping = self.analyzer.detect_columns(raw_df)
        warnings = self.analyzer.validate_mapping(raw_df, mapping)
        for warning in warnings:
            print(f"{Fore.YELLOW}warning: {warning}{Style.RESET_ALL}")
        mapping = self.analyzer.confirm_or_override(
            mapping, list(raw_df.columns), interactive=self.config.interactive
        )
        if not mapping.userID or not mapping.itemID:
            raise ValueError(
                "Dataset validation failed: could not confidently detect both userID and itemID columns. "
                f"Detected mapping was userID={mapping.userID!r}, itemID={mapping.itemID!r}, "
                f"rating={mapping.rating!r}, timestamp={mapping.timestamp!r}. "
                "Inspect the built CSV/JSON and ensure it contains real user-item interaction identifiers."
            )

        print(f"\n{Fore.YELLOW}[2/7] Feedback Profiling{Style.RESET_ALL}")
        probe_df = raw_df.rename(columns={mapping.rating: "rating"}) if mapping.rating else raw_df
        feedback = self.detector.detect_from_df(probe_df)
        resolved_mode = self.detector.resolve_mode(feedback, self.config.algorithm_mode)
        print(f"Resolved mode: {resolved_mode.upper()} (requested={self.config.algorithm_mode})")

        print(f"\n{Fore.YELLOW}[3/7] Data Cleaning{Style.RESET_ALL}")
        if not mapping.rating and mapping.timestamp:
            log.warning(
                "rating column not detected; promoting timestamp mapping '%s' → rating",
                mapping.timestamp,
            )
            mapping.rating = mapping.timestamp
            mapping.timestamp = None
        clean_df, cleaning_report = self.cleaner.clean(raw_df, mapping)
        if clean_df.empty:
            raise ValueError(
                "Dataset validation failed after cleaning: 0 interactions remained. "
                f"Detected mapping was userID={mapping.userID!r}, itemID={mapping.itemID!r}, "
                f"rating={mapping.rating!r}, timestamp={mapping.timestamp!r}. "
                f"Available columns: {list(raw_df.columns)}"
            )
        clean_df.attrs["is_implicit"] = resolved_mode in {"implicit", "hybrid"}
        clean_df.attrs["resolved_mode"] = resolved_mode
        train, test = self.cleaner.split(clean_df)

        print(f"\n{Fore.YELLOW}[4/7] Interaction Matrix{Style.RESET_ALL}")
        im = self.builder.build(train)
        print(f"Matrix: {im.n_users:,} x {im.n_items:,} nnz={im.matrix.nnz:,} density={im.density:.4%}")

        # ── Step 5: Embedding Enrichment ───────────────────────────────────────
        print(f"\n{Fore.YELLOW}[5/7] Embedding Enrichment{Style.RESET_ALL}")
        embedding_store: Optional[EmbeddingStore] = None
        if self.embedding_enricher is not None:
            # Attempt to extract a metadata DataFrame from raw_df.
            # If the dataset has a text column (title, description, name, etc.),
            # it will be embedded for content-aware similarity.
            # Falls back to item ID strings if no metadata column is detected.
            metadata_df, text_col = self._extract_item_metadata(
                raw_df=raw_df,
                item_ids=im.item_ids,
                item_col=mapping.itemID,
                text_col_hint=self.config.embedding_text_col,
            )
            try:
                embedding_store = self.embedding_enricher.build(
                    tenant_id=self.tenant_id,
                    item_ids=im.item_ids,
                    metadata_df=metadata_df,
                    text_col=text_col,
                )
                print(
                    f"  Embeddings: {embedding_store.n_items:,} items | "
                    f"dim={embedding_store.dim} | model={embedding_store.model_name}"
                )
                if text_col:
                    print(f"  Text source: column '{text_col}'")
                else:
                    print(f"  {Fore.YELLOW}No metadata column found — embedded item ID strings.{Style.RESET_ALL}")
                log.info(
                    "Embedding enrichment complete | tenant=%s items=%d dim=%d text_col=%s",
                    self.tenant_id, embedding_store.n_items, embedding_store.dim, text_col,
                )
            except Exception as exc:
                # Non-fatal — log a warning and continue without embeddings
                log.warning(
                    "Embedding enrichment failed | tenant=%s error=%s — continuing without embeddings.",
                    self.tenant_id, exc,
                )
                print(
                    f"  {Fore.YELLOW}Embedding skipped due to error: {exc}{Style.RESET_ALL}"
                )
        else:
            print(f"  {Fore.YELLOW}Embedding disabled (embedding_enabled=False).{Style.RESET_ALL}")

        print(f"\n{Fore.YELLOW}[6/7] Benchmark{Style.RESET_ALL}")
        report = self.benchmark.run(
            train=train,
            test=test,
            top_k=self.config.top_k,
            force_all=self.config.force_all_algos,
            algorithm_mode=resolved_mode,
            feedback_profile=feedback,
        )
        self._print_leaderboard(report)

        print(f"\n{Fore.YELLOW}[7/7] Hyperparameter Tuning{Style.RESET_ALL}")
        top_n = self.config.top_model_count
        top_algorithms = [r.algorithm for r in report.top_n(top_n)]
        print(f"Selected algorithms ({len(top_algorithms)}): {top_algorithms}")

        bench_by_algo = {r.algorithm: r for r in report.results}
        tuning_results: list[TuningResult] = []
        optuna_policy = self._build_optuna_policy(
            selected_algorithms=top_algorithms,
            primary_metric=report.primary_metric,
            primary_metric_direction=report.primary_metric_direction,
        )
        optuna_note = optuna_policy.get("summary", "")

        if self.config.n_tuning_trials == 0:
            print("Optuna skipped (n_tuning_trials=0).")
            for algo in top_algorithms:
                tuning_results.append(
                    TuningResult(
                        algorithm=algo,
                        best_params={},
                        best_value=self._benchmark_primary_value(report, bench_by_algo.get(algo)),
                        metric_name=report.primary_metric,
                        n_trials=0,
                        elapsed_s=0.0,
                        status="skipped",
                        fallback_reason="tuning skipped by configuration",
                        trial_budget="fixed(0)",
                    )
                )
        else:
            tuning_results = self.tuner.tune_top_n(
                algorithms=top_algorithms,
                train=report.train,
                test=report.test,
                n_trials=self.config.n_tuning_trials,
                top_k=self.config.top_k,
            )

        ranked_tuning = self._rank_tuning_results(tuning_results, report, bench_by_algo)
        best_model_id = None
        best_algorithm = None
        best_params: dict = {}
        all_model_ids: list[str] = []

        if self.config.save_model:
            for rank, tune_result in enumerate(ranked_tuning, 1):
                promote = rank == 1 and self.config.auto_promote
                try:
                    model_id = self._save_model(
                        tune_result=tune_result,
                        report=report,
                        rank=rank,
                        total=len(ranked_tuning),
                        promote=promote,
                    )
                    all_model_ids.append(model_id)
                    if rank == 1:
                        best_model_id = model_id
                        best_algorithm = tune_result.algorithm
                        best_params = tune_result.best_params
                except Exception as e:
                    log.error(
                        "Model save failed | tenant=%s algo=%s error=%s",
                        self.tenant_id, tune_result.algorithm, e,
                        exc_info=True,
                    )
        elif ranked_tuning:
            best_algorithm = ranked_tuning[0].algorithm
            best_params = ranked_tuning[0].best_params

        recommendations = self._build_model_recommendations(
            ranked_results=ranked_tuning,
            report=report,
            bench_by_algo=bench_by_algo,
            model_ids=all_model_ids,
            feedback_profile=feedback.to_dict(),
            cleaning_report=cleaning_report.__dict__,
        )
        selection_policy = self._build_model_selection_policy(
            recommendations=recommendations,
            primary_metric=report.primary_metric,
            primary_metric_direction=report.primary_metric_direction,
        )

        elapsed = time.time() - t0
        log.info(
            "Training complete | tenant=%s best_algo=%s models_saved=%d elapsed=%.1fs",
            self.tenant_id, best_algorithm, len(all_model_ids), elapsed,
        )
        self._print_footer(self.tenant_id, best_algorithm, best_params, all_model_ids, elapsed)

        return TrainingResult(
            tenant_id=self.tenant_id,
            report=report,
            tuning_results=ranked_tuning,
            best_model_id=best_model_id,
            best_algorithm=best_algorithm,
            best_params=best_params,
            elapsed_s=elapsed,
            all_model_ids=all_model_ids,
            resolved_mode=resolved_mode,
            feedback_profile=feedback.to_dict(),
            top_model_recommendations=recommendations,
            ranking_logic=report.ranking_logic,
            optuna_note=optuna_note,
            optuna_policy=optuna_policy,
            model_selection_policy=selection_policy,
            embedding_store=embedding_store,
        )

    # ── Embedding helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_item_metadata(
        raw_df: pd.DataFrame,
        item_ids: list[str],
        item_col: Optional[str],
        text_col_hint: Optional[str],
    ) -> tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Try to build a per-item metadata DataFrame from raw_df.

        Resolution order:
          1. text_col_hint explicitly set in TrainingConfig → use it directly
          2. Auto-detect: score every string column by (hint_match, item_cardinality, avg_len)
             and pick the best non-ID candidate.
          3. Append a secondary keywords/tags column to the chosen text if one exists.
          4. No usable column found → return (None, None) → enricher falls back to item IDs.

        Returns
        -------
        (metadata_df, text_col)
          metadata_df : DataFrame with 'itemID' + text_col, deduplicated per item
          text_col    : resolved column name, or None
        """
        if item_col not in raw_df.columns:
            return None, None

        # ── ID/signal column guard ─────────────────────────────────────────────
        # Two-tier approach:
        #   1. Exact blocklist — common ID/signal column names across all domains.
        #   2. Token subset   — block if every underscore-split token is an ID word.
        #      This catches multi-part IDs (e.g. "user_event_id") without
        #      accidentally blocking descriptive names like "event_name" or
        #      "score_description" (those contain non-ID tokens like "name"/"description").
        _EXACT_ID_COLS: frozenset = frozenset({
            # ── user identifiers ──
            "userid", "user_id", "uid", "user", "customerid", "customer_id",
            "member_id", "memberid", "account_id", "accountid",
            # ── item identifiers ──
            "itemid", "item_id", "iid", "productid", "product_id",
            "asin", "sku", "upc", "isbn",
            "movie_id", "movieid", "song_id", "track_id", "book_id",
            "article_id", "content_id", "listing_id",
            # ── interaction signals ──
            "rating", "score", "stars",
            "timestamp", "ts", "time", "date", "created_at", "updated_at",
            "price", "cost", "amount", "qty", "quantity",
            "interaction", "event", "event_type", "action",
            "click", "purchase", "order", "order_id", "session_id",
            "lat", "lon", "latitude", "longitude",
            "id",
        })
        _ID_TOKENS: frozenset = frozenset({
            "userid", "user", "uid", "customerid", "customer",
            "itemid", "item", "iid", "productid", "product",
            "rating", "score", "timestamp", "time", "date",
            "price", "cost", "amount", "qty", "quantity",
            "interaction", "event", "click", "purchase", "order",
            "lat", "lon", "latitude", "longitude",
            "id", "type", "code", "num", "no", "idx", "index",
            "ts", "at", "on",
        })

        def _is_id_col(col: str) -> bool:
            col_norm = col.lower().replace(" ", "_")
            if col_norm in _EXACT_ID_COLS:
                return True
            tokens = {t for t in col_norm.split("_") if t}
            return bool(tokens) and tokens.issubset(_ID_TOKENS)

        # ── Text-hint vocabulary ───────────────────────────────────────────────
        # Used to boost columns whose names signal item identity/description.
        _TEXT_HINTS: tuple[str, ...] = (
            "title", "name", "description", "desc", "label", "text",
            "content", "genre", "category", "brand", "summary",
            "subject", "topic", "synopsis", "overview", "detail",
            "product", "item", "movie", "book", "song", "track",
            "artist", "author", "director",
        )

        # ── Explicit hint from TrainingConfig ──────────────────────────────────
        if text_col_hint:
            if text_col_hint in raw_df.columns:
                meta = (
                    raw_df[[item_col, text_col_hint]]
                    .rename(columns={item_col: "itemID"})
                    .drop_duplicates(subset=["itemID"])
                )
                return meta, text_col_hint
            log.warning(
                "embedding_text_col='%s' not found in dataset columns %s — auto-detecting.",
                text_col_hint, list(raw_df.columns),
            )

        # ── Candidate columns ──────────────────────────────────────────────────
        string_cols = [
            c for c in raw_df.columns
            if c != item_col
            and raw_df[c].dtype == object
            and not _is_id_col(c)
        ]

        if not string_cols:
            return None, None

        # ── Scoring ────────────────────────────────────────────────────────────
        # Priority tuple (descending importance):
        #   1. hint_match   — does the column name suggest item metadata?
        #   2. cardinality  — item-level unique values (metadata is item-stable,
        #                     reviews/logs vary per row → penalise high-cardinality)
        #   3. avg_len      — longer text is generally richer for embeddings
        #
        # Cardinality is measured as unique(col) / unique(item_col).
        # Ideal metadata: ratio ≈ 1.0.  Reviews/logs: ratio >> 1.0.
        n_items = raw_df[item_col].nunique()

        def _score(col: str) -> tuple[int, float, float]:
            hint_match  = int(any(h in col.lower() for h in _TEXT_HINTS))
            sample      = raw_df[col].dropna().astype(str).head(500)
            avg_len     = float(sample.str.len().mean()) if len(sample) else 0.0
            # cardinality ratio: lower is better (item-stable columns score higher)
            n_unique    = raw_df[col].nunique()
            card_ratio  = n_unique / max(n_items, 1)
            # invert so that item-stable (ratio≈1) ranks above per-row (ratio>>1)
            card_score  = 1.0 / (1.0 + abs(card_ratio - 1.0))
            return (hint_match, card_score, avg_len)

        best_col        = max(string_cols, key=_score)
        _, _, avg_len   = _score(best_col)

        # Reject if the winner has no meaningful text (e.g. single-char codes)
        if avg_len <= 3:
            return None, None

        meta = (
            raw_df[[item_col, best_col]]
            .rename(columns={item_col: "itemID"})
            .drop_duplicates(subset=["itemID"])
        )

        # ── Optional: append a keywords/tags column ────────────────────────────
        # Concatenating comma-separated tags to the item text gives the model
        # richer signal without the noise of free-text reviews.
        _KW_HINTS = ("keyword", "keywords", "tags", "tag")
        kw_cols = [
            c for c in raw_df.columns
            if c != item_col
            and c != best_col
            and raw_df[c].dtype == object
            and any(h in c.lower() for h in _KW_HINTS)
        ]
        if kw_cols:
            kw_col = kw_cols[0]
            kw_map: dict[str, str] = (
                raw_df[[item_col, kw_col]]
                .drop_duplicates(subset=[item_col])
                .set_index(item_col)[kw_col]
                .astype(str)
                .to_dict()
            )
            meta = meta.copy()
            meta[best_col] = meta.apply(
                lambda row: (
                    row[best_col] + " " + kw_map.get(row["itemID"], "")
                ).strip(),
                axis=1,
            )
            log.info("Enriched embedding text with keywords column '%s'", kw_col)

        return meta, best_col

    # ── Model persistence ──────────────────────────────────────────────────────

    def _save_model(
        self,
        tune_result: TuningResult,
        report: BenchmarkReport,
        rank: int,
        total: int,
        promote: bool,
    ) -> str:
        algo   = tune_result.algorithm
        meta   = ALGORITHM_REGISTRY[algo]
        params = tune_result.best_params or {}
        print(f"  [{rank}/{total}] retraining {algo} | tenant={self.tenant_id} params={params}")

        run_out = meta["fn"](
            report.train,
            report.test,
            self.config.top_k,
            **params,
            return_model=True,
        )
        if len(run_out) != 3:
            raise RuntimeError(
                f"Algorithm '{algo}' did not return a trained model object with return_model=True"
            )

        _, _, model_obj = run_out
        metrics = dict(next((r.metrics for r in report.results if r.algorithm == algo), {}))
        if tune_result.best_value is not None:
            metrics[tune_result.metric_name] = tune_result.best_value
        metrics["Composite Score"] = next(
            (
                rec.get("Composite Score")
                for rec in report.leaderboard().to_dict(orient="records")
                if rec.get("Algorithm") == algo
            ),
            None,
        )

        model_id = self.registry.save(
            model=model_obj,
            algorithm=algo,
            metrics=metrics,
            params=params,
            is_implicit=report.resolved_mode in {"implicit", "hybrid"},
            notes=(
                f"tenant={self.tenant_id} rank={rank}/{total} "
                f"status={tune_result.status} trials={tune_result.n_trials} "
                f"metric={tune_result.metric_name}"
            ),
        )
        if promote:
            DeploymentControlPlane(
                tenant_id=self.tenant_id,
                registry=self.registry,
            ).promote(
                model_id,
                actor=self.tenant_id,
                reason="Auto-promoted top-ranked training result",
                source="training_auto_promote",
            )
            log.info(
                "Model promoted | tenant=%s algo=%s id=%s",
                self.tenant_id, algo, model_id,
            )
        return model_id

    # ── Ranking & recommendations ──────────────────────────────────────────────

    def _rank_tuning_results(
        self,
        tuning_results: list[TuningResult],
        report: BenchmarkReport,
        bench_by_algo: dict[str, object],
    ) -> list[TuningResult]:
        direction = report.primary_metric_direction

        def score(result: TuningResult) -> float:
            if result.best_value is not None and math.isfinite(float(result.best_value)):
                return float(result.best_value)
            benchmark_result = bench_by_algo.get(result.algorithm)
            if benchmark_result:
                return self._benchmark_primary_value(report, benchmark_result)
            return float("-inf") if direction == "maximize" else float("inf")

        def key(result: TuningResult) -> tuple[int, float]:
            status_rank = {"ok": 0, "fallback": 1, "skipped": 2, "failed": 3}.get(result.status, 4)
            raw = score(result)
            metric_key = -raw if direction == "maximize" else raw
            return status_rank, metric_key

        out = list(tuning_results)
        out.sort(key=key)
        return out

    @staticmethod
    def _benchmark_primary_value(report: BenchmarkReport, bench_result) -> float:
        if not bench_result:
            return 0.0
        val = bench_result.metrics.get(report.primary_metric)
        if val is None:
            return 0.0
        try:
            return float(val)
        except Exception:
            return 0.0

    def _build_model_recommendations(
        self,
        ranked_results: list[TuningResult],
        report: BenchmarkReport,
        bench_by_algo: dict[str, object],
        model_ids: list[str],
        feedback_profile: dict,
        cleaning_report: dict,
    ) -> list[dict]:
        lb = report.leaderboard().set_index("Algorithm") if not report.leaderboard().empty else pd.DataFrame()
        sparsity = cleaning_report.get("sparsity", 1.0)
        sparse_label = "sparse" if sparsity >= 0.98 else "dense"
        out: list[dict] = []
        metric_values = []
        composite_values = []

        for tune in ranked_results:
            algo = tune.algorithm
            row = lb.loc[algo] if not lb.empty and algo in lb.index else None
            perf_value = (
                tune.best_value
                if tune.best_value is not None
                else (float(row.get(report.primary_metric)) if row is not None else None)
            )
            composite_value = float(row.get("Composite Score")) if row is not None else None
            if isinstance(perf_value, (int, float)) and math.isfinite(float(perf_value)):
                metric_values.append(float(perf_value))
            if isinstance(composite_value, (int, float)) and math.isfinite(float(composite_value)):
                composite_values.append(float(composite_value))

        metric_min    = min(metric_values)    if metric_values    else None
        metric_max    = max(metric_values)    if metric_values    else None
        composite_min = min(composite_values) if composite_values else None
        composite_max = max(composite_values) if composite_values else None

        for idx, tune in enumerate(ranked_results, 1):
            algo = tune.algorithm
            meta = ALGORITHM_REGISTRY.get(algo, {})
            row  = lb.loc[algo] if not lb.empty and algo in lb.index else None
            perf_value = (
                tune.best_value
                if tune.best_value is not None
                else (float(row.get(report.primary_metric)) if row is not None else None)
            )
            elapsed_s       = float(row.get("Time (s)")) if row is not None and pd.notna(row.get("Time (s)")) else None
            composite_score = float(row.get("Composite Score")) if row is not None else None
            selection_score_pct = self._estimate_selection_score_pct(
                metric_name=report.primary_metric,
                metric_value=perf_value,
                metric_direction=report.primary_metric_direction,
                composite_score=composite_score,
                metric_min=metric_min,
                metric_max=metric_max,
                composite_min=composite_min,
                composite_max=composite_max,
            )
            performance_summary = (
                f"{report.primary_metric} reached {perf_value:.4f}."
                if isinstance(perf_value, (int, float))
                else f"{report.primary_metric} was not available after tuning."
            )
            fit_summary = (
                f"{algo} matches the {report.resolved_mode} dataset mode and is rated "
                f"{meta.get('sparsity_fit', 'medium')} for {sparse_label} data."
            )
            reliability_summary = (
                f"Tuning finished with status '{tune.status}' and trial budget {tune.trial_budget or 'n/a'}."
            )
            reasons = [
                f"Feedback fit: algorithm supports {meta.get('feedback', 'unknown')} mode; dataset resolved as {report.resolved_mode}.",
                (
                    f"Expected performance: {report.primary_metric}={perf_value:.4f}."
                    if isinstance(perf_value, (int, float))
                    else f"Expected performance: benchmark primary metric is {report.primary_metric}."
                ),
                f"Scalability: {meta.get('scalability', 'medium')} with limits users={meta.get('max_users') or 'unbounded'}, items={meta.get('max_items') or 'unbounded'}.",
                f"Training speed: {meta.get('training_speed', 'medium')}"
                + (f" (observed {elapsed_s:.2f}s)." if elapsed_s is not None else "."),
                f"Robustness: {meta.get('robustness', 'medium')} (tuning status={tune.status}).",
                f"Data suitability: rated for {meta.get('sparsity_fit', 'medium')} sparsity; current dataset is {sparse_label} ({sparsity:.2%} sparsity).",
                f"Interpretability: {meta.get('interpretability', 'medium')}.",
                f"Production readiness: {meta.get('production_readiness', 'medium')}.",
            ]
            out.append(
                {
                    "rank": idx,
                    "algorithm": algo,
                    "model_id": model_ids[idx - 1] if idx - 1 < len(model_ids) else None,
                    "metric_name": report.primary_metric,
                    "metric_value": perf_value,
                    "composite_score": composite_score,
                    "status": tune.status,
                    "trial_budget": tune.trial_budget,
                    "reasons": reasons,
                    "summary": " ".join([performance_summary, fit_summary, reliability_summary]),
                    "performance_summary": performance_summary,
                    "fit_summary": fit_summary,
                    "reliability_summary": reliability_summary,
                    "selection_score_pct": selection_score_pct,
                    "feedback_profile": feedback_profile,
                }
            )
        self._annotate_ranked_recommendations(
            ranked=out,
            primary_metric=report.primary_metric,
            primary_metric_direction=report.primary_metric_direction,
        )
        return out

    def _build_optuna_policy(
        self,
        selected_algorithms: list[str],
        primary_metric: str,
        primary_metric_direction: str,
    ) -> dict:
        requested_trials = int(self.config.n_tuning_trials)
        adaptive_budgets = get_optuna_trial_budgets()
        selected_budgets = {
            algorithm: adaptive_budgets.get(algorithm)
            for algorithm in selected_algorithms
            if adaptive_budgets.get(algorithm) is not None
        }
        if requested_trials == OPTUNA_TRIALS_AUTO:
            mode    = "adaptive"
            summary = (
                "Optuna uses adaptive per-algorithm trial budgets because n_trials=-1. "
                f"Each selected algorithm receives its own preset budget before ranking on {primary_metric}."
            )
        elif requested_trials == 0:
            mode    = "disabled"
            summary = (
                "Optuna is disabled because n_trials=0. "
                "The system keeps benchmark scores and skips hyperparameter search."
            )
        else:
            mode    = "fixed"
            summary = (
                f"Optuna uses a fixed budget of {requested_trials} trials for every selected algorithm."
            )

        return {
            "requested_trials": requested_trials,
            "mode": mode,
            "top_k": int(self.config.top_k),
            "top_k_definition": (
                f"Top-K={self.config.top_k} means ranking metrics evaluate whether relevant items appear "
                f"in each user's top {self.config.top_k} recommendations."
            ),
            "objective_metric": primary_metric,
            "objective_direction": primary_metric_direction,
            "selected_algorithms": list(selected_algorithms),
            "adaptive_trial_budgets": selected_budgets if requested_trials == OPTUNA_TRIALS_AUTO else {},
            "rules": [
                "n_trials=-1 uses adaptive trial budgets per algorithm.",
                "n_trials=0 skips Optuna and keeps benchmark scores.",
                "n_trials>0 applies the same fixed trial count to every selected algorithm.",
                f"Implicit and hybrid datasets optimize {primary_metric} by ranking quality; explicit datasets minimize RMSE.",
            ],
            "summary": summary,
        }

    def _build_model_selection_policy(
        self,
        recommendations: list[dict],
        primary_metric: str,
        primary_metric_direction: str,
    ) -> dict:
        ranked = [dict(row) for row in recommendations if row.get("algorithm")]
        if not ranked:
            return {
                "selection_type": "none",
                "selected_count": 0,
                "selected_models": [],
                "reason": "No ranked models were available after training.",
                "display_title": "No recommended models",
                "recommended_strategy": None,
            }

        self._annotate_ranked_recommendations(
            ranked=ranked,
            primary_metric=primary_metric,
            primary_metric_direction=primary_metric_direction,
        )

        top           = ranked[0]
        runner_up     = ranked[1] if len(ranked) > 1 else None
        top_score     = float(top.get("selection_score_pct") or 0.0)
        runner_up_score = float(runner_up.get("selection_score_pct") or 0.0) if runner_up else 0.0
        score_gap     = round(top_score - runner_up_score, 2) if runner_up else 100.0
        shortlist_count = max(1, min(len(ranked), int(self.config.top_model_count)))
        display_selected = ranked[:shortlist_count]

        if len(ranked) == 1 or top_score >= 90.0 or (top_score >= 75.0 and score_gap >= 10.0):
            serving_selected = ranked[:1]
            serving_reason = (
                f"{top['algorithm']} is the clear winner with a selection score of {top_score:.2f}%."
                if runner_up is None
                else (
                    f"{top['algorithm']} is clearly ahead with a selection score of {top_score:.2f}% "
                    f"and a {score_gap:.2f}-point lead over {runner_up['algorithm']}. "
                    "The API will return one recommended model by default."
                )
            )
        else:
            serving_selected = ranked[:shortlist_count]
            serving_reason = (
                f"No single model separated enough from the pack. {top['algorithm']} leads with "
                f"{top_score:.2f}% selection score, but the gap to {runner_up['algorithm']} is only "
                f"{score_gap:.2f} points. The API will return the best {len(serving_selected)} models for comparison."
                if runner_up is not None
                else f"{top['algorithm']} is currently the only available model."
            )

        selection_type  = "single_model" if len(display_selected) == 1 else "top_ranked_models"
        display_title   = (
            "Recommended single model"
            if len(display_selected) == 1
            else f"Top {len(display_selected)} model recommendations and reasons"
        )
        recommended_strategy    = "single_model" if len(serving_selected) == 1 else "ensemble_weighted"
        reason = (
            f"{serving_reason} Showing the top {len(display_selected)} ranked models so you can compare why each one scored well."
            if len(display_selected) > 1
            else serving_reason
        )
        best_model_explanation  = top.get("reason") or serving_reason
        comparison_explanations = [
            row["comparison_to_next"]
            for row in display_selected
            if row.get("comparison_to_next")
        ]

        selected_algorithms = {row.get("algorithm") for row in display_selected}
        for row in ranked:
            row["selected_by_policy"] = row.get("algorithm") in selected_algorithms
            if row["selected_by_policy"]:
                row["decision_note"] = reason

        return {
            "selection_type": selection_type,
            "selected_count": len(display_selected),
            "selected_models": display_selected,
            "reason": reason,
            "display_title": display_title,
            "recommended_strategy": recommended_strategy,
            "serving_selected_count": len(serving_selected),
            "serving_selected_models": serving_selected,
            "serving_reason": serving_reason,
            "best_model_explanation": best_model_explanation,
            "comparison_explanations": comparison_explanations,
            "primary_metric": primary_metric,
            "primary_metric_direction": primary_metric_direction,
            "top_model": {
                "algorithm": top.get("algorithm"),
                "model_id": top.get("model_id"),
                "selection_score_pct": top.get("selection_score_pct"),
                "metric_name": top.get("metric_name"),
                "metric_value": top.get("metric_value"),
            },
            "runner_up": (
                {
                    "algorithm": runner_up.get("algorithm"),
                    "model_id": runner_up.get("model_id"),
                    "selection_score_pct": runner_up.get("selection_score_pct"),
                    "metric_name": runner_up.get("metric_name"),
                    "metric_value": runner_up.get("metric_value"),
                }
                if runner_up
                else None
            ),
            "score_gap_pct": score_gap,
            "decision_rules": [
                "Always show the full requested top-model count in the training results so the user can compare ranked candidates.",
                "Use one serving model when the winner is dominant enough to stand on its own.",
                "Use every shortlisted benchmark-selected model when several candidates remain competitive so the API can compare the full candidate set.",
                "Dominance is estimated from the primary metric, composite score, and relative separation between models.",
            ],
        }

    def _annotate_ranked_recommendations(
        self,
        ranked: list[dict],
        primary_metric: str,
        primary_metric_direction: str,
    ) -> None:
        ordered = sorted(
            ranked,
            key=lambda row: (
                int(row.get("rank") or 10**9),
                -float(row.get("selection_score_pct") or 0.0),
                str(row.get("algorithm") or ""),
            ),
        )
        for index, row in enumerate(ordered):
            previous_row = ordered[index - 1] if index > 0 else None
            next_row     = ordered[index + 1] if index + 1 < len(ordered) else None
            row["comparison_to_previous"] = (
                self._build_rank_comparison(
                    higher=previous_row,
                    lower=row,
                    primary_metric=primary_metric,
                    primary_metric_direction=primary_metric_direction,
                )
                if previous_row
                else None
            )
            row["comparison_to_next"] = (
                self._build_rank_comparison(
                    higher=row,
                    lower=next_row,
                    primary_metric=primary_metric,
                    primary_metric_direction=primary_metric_direction,
                )
                if next_row
                else None
            )
            row["reason"] = self._build_rank_reason(
                row=row,
                previous_row=previous_row,
                next_row=next_row,
                primary_metric=primary_metric,
                primary_metric_direction=primary_metric_direction,
            )

    def _build_rank_reason(
        self,
        row: dict,
        previous_row: Optional[dict],
        next_row: Optional[dict],
        primary_metric: str,
        primary_metric_direction: str,
    ) -> str:
        rank      = int(row.get("rank") or 0)
        algorithm = row.get("algorithm") or "Unknown model"
        score     = self._safe_float(row.get("selection_score_pct"))
        parts: list[str] = []

        if rank == 1:
            parts.append(
                f"Ranked #{rank} because it has the highest selection score at {score:.2f}%."
                if score is not None
                else f"Ranked #{rank} because it is the highest-scoring supported model."
            )
        else:
            parts.append(
                f"Ranked #{rank} with a selection score of {score:.2f}%."
                if score is not None
                else f"Ranked #{rank} in the supported model list."
            )

        if next_row is not None:
            parts.append(
                self._build_rank_comparison(
                    higher=row,
                    lower=next_row,
                    primary_metric=primary_metric,
                    primary_metric_direction=primary_metric_direction,
                )
            )
        elif previous_row is not None:
            parts.append(
                f"It remains behind #{int(previous_row.get('rank') or rank - 1)} "
                f"{previous_row.get('algorithm')} and has no lower-ranked supported competitor below it."
            )

        for key in ("performance_summary", "fit_summary", "reliability_summary"):
            if row.get(key):
                parts.append(str(row[key]))

        return " ".join(part for part in parts if part)

    def _build_rank_comparison(
        self,
        higher: Optional[dict],
        lower: Optional[dict],
        primary_metric: str,
        primary_metric_direction: str,
    ) -> str:
        if not higher or not lower:
            return ""
        higher_name = higher.get("algorithm") or "Higher-ranked model"
        lower_name  = lower.get("algorithm")  or "Lower-ranked model"
        lower_rank  = int(lower.get("rank") or 0)
        higher_score = self._safe_float(higher.get("selection_score_pct"))
        lower_score  = self._safe_float(lower.get("selection_score_pct"))

        if higher_score is not None and lower_score is not None:
            gap        = higher_score - lower_score
            score_text = (
                f"{higher_name} stays ahead of #{lower_rank} {lower_name} because its selection score is "
                f"{higher_score:.2f}% versus {lower_score:.2f}% ({gap:+.2f} points)."
            )
        else:
            score_text = f"{higher_name} stays ahead of #{lower_rank} {lower_name} on the ranking composite."

        detail_parts: list[str] = []
        metric_sentence = self._metric_comparison_sentence(
            higher=higher,
            lower=lower,
            primary_metric=primary_metric,
            primary_metric_direction=primary_metric_direction,
        )
        if metric_sentence:
            detail_parts.append(metric_sentence)

        higher_composite = self._safe_float(higher.get("composite_score"))
        lower_composite  = self._safe_float(lower.get("composite_score"))
        if higher_composite is not None and lower_composite is not None:
            detail_parts.append(
                f"Composite Score is also stronger ({higher_composite:.4f} vs {lower_composite:.4f})."
            )
        return " ".join([score_text, *detail_parts]).strip()

    def _metric_comparison_sentence(
        self,
        higher: dict,
        lower: dict,
        primary_metric: str,
        primary_metric_direction: str,
    ) -> str:
        higher_metric = self._safe_float(higher.get("metric_value"))
        lower_metric  = self._safe_float(lower.get("metric_value"))
        if higher_metric is None or lower_metric is None:
            return ""
        direction = (primary_metric_direction or "").strip().lower()
        if direction.startswith("min"):
            better_text      = f"lower {primary_metric}"
            higher_is_better = higher_metric <= lower_metric
        else:
            better_text      = f"higher {primary_metric}"
            higher_is_better = higher_metric >= lower_metric
        if not higher_is_better:
            return ""
        return f"It also posts {better_text} ({higher_metric:.4f} vs {lower_metric:.4f})."

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _resolve_data_path(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return (self._tenant_data_dir / p).resolve()

    @staticmethod
    def _validate_tenant(tenant_id: str) -> None:
        if not tenant_id or not isinstance(tenant_id, str):
            raise ValueError(
                "tenant_id must be a non-empty string. "
                "Ensure TenantMiddleware has set request.state.tenant_id before "
                "constructing TrainingPipeline."
            )

    @staticmethod
    def _safe_float(value: object) -> Optional[float]:
        try:
            if value in (None, ""):
                return None
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(out):
            return None
        return out

    @staticmethod
    def _estimate_selection_score_pct(
        metric_name: str,
        metric_value: Optional[float],
        metric_direction: str,
        composite_score: Optional[float],
        metric_min: Optional[float],
        metric_max: Optional[float],
        composite_min: Optional[float],
        composite_max: Optional[float],
    ) -> float:
        signals: list[tuple[float, float]] = []

        bounded_metric = None
        if isinstance(metric_value, (int, float)) and math.isfinite(float(metric_value)):
            metric_value = float(metric_value)
            if metric_direction == "maximize" and 0.0 <= metric_value <= 1.0:
                bounded_metric = metric_value * 100.0
            elif str(metric_name).upper() == "RMSE":
                bounded_metric = 100.0 / (1.0 + max(metric_value, 0.0))
        if bounded_metric is not None:
            signals.append((bounded_metric, 0.5))

        if isinstance(composite_score, (int, float)) and math.isfinite(float(composite_score)):
            composite_score = float(composite_score)
            if 0.0 <= composite_score <= 1.0:
                signals.append((composite_score * 100.0, 0.25))

        normalized_metric = None
        if (
            isinstance(metric_value, (int, float))
            and metric_min is not None
            and metric_max is not None
            and math.isfinite(float(metric_min))
            and math.isfinite(float(metric_max))
        ):
            span = float(metric_max) - float(metric_min)
            if span <= 1e-12:
                normalized_metric = 100.0
            elif metric_direction == "minimize":
                normalized_metric = ((float(metric_max) - float(metric_value)) / span) * 100.0
            else:
                normalized_metric = ((float(metric_value) - float(metric_min)) / span) * 100.0
        if normalized_metric is not None:
            signals.append((max(0.0, min(100.0, normalized_metric)), 0.25))

        if (
            isinstance(composite_score, (int, float))
            and composite_min is not None
            and composite_max is not None
            and math.isfinite(float(composite_min))
            and math.isfinite(float(composite_max))
        ):
            span = float(composite_max) - float(composite_min)
            normalized_composite = (
                100.0 if span <= 1e-12
                else ((float(composite_score) - float(composite_min)) / span) * 100.0
            )
            signals.append((max(0.0, min(100.0, normalized_composite)), 0.25))

        if not signals:
            return 0.0
        weighted_total = sum(value * weight for value, weight in signals)
        total_weight   = sum(weight for _, weight in signals)
        return round(weighted_total / total_weight, 2)

    # ── Console output ─────────────────────────────────────────────────────────

    @staticmethod
    def _print_header(tenant_id: str) -> None:
        print(f"\n{Fore.CYAN}{'=' * 72}")
        print("  PROACTIVE AI - TRAINING PIPELINE")
        print(f"  tenant: {tenant_id}")
        print(f"{'=' * 72}{Style.RESET_ALL}")

    @staticmethod
    def _print_footer(tenant_id: str, algo, params, all_model_ids, elapsed) -> None:
        print(f"\n{Fore.CYAN}{'=' * 72}")
        print(f"TRAINING COMPLETE in {elapsed:.1f}s  [tenant={tenant_id}]")
        if algo:
            print(f"Best algorithm: {algo}")
            print(f"Best params   : {params}")
        print(f"Models saved  : {len(all_model_ids)}")
        print(f"{'=' * 72}{Style.RESET_ALL}\n")

    @staticmethod
    def _print_leaderboard(report: BenchmarkReport) -> None:
        lb = report.leaderboard()
        if lb.empty:
            print(f"{Fore.RED}No benchmark results available.{Style.RESET_ALL}")
            return
        print("\nLeaderboard (composite ranking):")
        cols = [
            "Rank", "Algorithm", report.primary_metric,
            "Time (s)", "Scalability Score", "Performance Score", "Composite Score",
        ]
        view = lb[[c for c in cols if c in lb.columns]].head(15)
        print(view.to_string(index=False))