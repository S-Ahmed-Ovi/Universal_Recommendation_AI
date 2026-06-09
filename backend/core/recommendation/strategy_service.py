"""
Strategy-aware recommendation orchestration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from backend.config.settings import ALLOW_INACTIVE_MODEL_SERVING
from backend.core.algorithms import ALGORITHM_REGISTRY
from backend.core.deployment.control_plane import DeploymentControlPlane
from backend.core.models.model_loader import LoadedModel, ModelLoader
from backend.core.models.model_registry import ModelRecord, ModelRegistry
from backend.core.recommendation.recommender_engine import RecommenderEngine
from backend.utils.logger import get_logger

log = get_logger(__name__)

RECOMMEND_STRATEGIES = {"best_promoted_model", "single_model", "ensemble_weighted"}
_METRIC_PRIORITY = (
    "NDCG@K",
    "RMSE",
    "MAP@K",
    "Precision@K",
    "Recall@K",
    "Composite Score",
)
_MODE_TO_ALLOWED_FEEDBACK = {
    "explicit": {"explicit", "both"},
    "implicit": {"implicit", "both"},
    "hybrid": {"explicit", "implicit", "both"},
    "auto": {"explicit", "implicit", "both"},
}


@dataclass
class WeightedModelInput:
    """Resolved weighted model selector used by single-model and ensemble flows."""
    weight: float
    model_id: Optional[str] = None
    algorithm: Optional[str] = None
    label: Optional[str] = None
    allow_algorithm_fallback: bool = False


@dataclass
class RecommendationPayload:
    """Normalized input payload consumed by the strategy service."""
    user_id: str
    top_n: int
    strategy: Optional[str] = None
    model_id: Optional[str] = None
    algorithm: Optional[str] = None
    label: Optional[str] = None
    models: list[WeightedModelInput] | None = None
    auto_normalize_weights: bool = True
    allow_algorithm_fallback: bool = False


class RecommendationStrategyError(ValueError):
    """Structured validation/runtime error raised during recommendation orchestration."""
    def __init__(self, code: str, message: str, **details):
        """Capture a stable machine-readable code alongside the user-facing message."""
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def to_dict(self) -> dict:
        """Serialise the error into the API-friendly diagnostic payload."""
        return {"code": self.code, "message": self.message, **self.details}


class RecommendationStrategyService:
    """Coordinate model selection, inference, and ranking metadata for serving routes."""
    def __init__(
        self,
        registry: ModelRegistry,
        loader: ModelLoader,
        recommender: RecommenderEngine,
        tenant_id: str,
    ):
        """Bind the tenant-scoped dependencies used across all recommendation strategies."""
        self.registry = registry
        self.loader = loader
        self.recommender = recommender
        self.tenant_id = tenant_id
        self.deployments = DeploymentControlPlane(
            tenant_id=tenant_id,
            registry=registry,
            loader=loader,
        )

    def recommend(
        self,
        payload: RecommendationPayload,
        last_result=None,
    ) -> dict:
        """Execute the requested strategy and return the stable serving response payload."""
        strategy = self._resolve_strategy(payload)
        mode = getattr(last_result, "resolved_mode", None) if last_result else None
        prepared = self._prepare_models(payload=payload, strategy=strategy, last_result=last_result, mode=mode)
        selected_models = prepared["models"]
        warnings = list(prepared["warnings"])

        candidate_top_n = max(int(payload.top_n) * 5, 50)
        model_outputs, runtime_issues = self._run_models(
            selected_models=selected_models,
            user_id=payload.user_id,
            top_n=candidate_top_n,
        )
        warnings.extend(runtime_issues)

        if strategy == "single_model" and not model_outputs:
            issue = runtime_issues[0] if runtime_issues else {}
            raise RecommendationStrategyError(
                issue.get("code", "NO_PREDICTIONS_FOR_USER"),
                issue.get("message", "The selected model did not produce recommendations for this user."),
                strategy=strategy,
                user_id=payload.user_id,
                requested_model_id=payload.model_id,
                requested_algorithm=payload.algorithm,
            )

        if strategy == "ensemble_weighted" and len(model_outputs) < 2:
            raise RecommendationStrategyError(
                "INSUFFICIENT_PREDICTIVE_MODELS",
                "Fewer than two selected models produced predictions for ensemble_weighted.",
                strategy=strategy,
                requested_model_count=len(selected_models),
                predictive_model_count=len(model_outputs),
                warnings=runtime_issues,
            )

        if not model_outputs:
            raise RecommendationStrategyError(
                "NO_MODELS_AVAILABLE",
                "No valid models were available to serve recommendations.",
                strategy=strategy,
            )

        weight_rebalanced = False
        weight_rebalanced_note = ""
        if strategy == "ensemble_weighted":
            model_outputs, weight_rebalanced, weight_rebalanced_note = self._rebalance_runtime_weights(
                model_outputs,
                had_drops=len(model_outputs) != len(selected_models),
            )
            if weight_rebalanced and weight_rebalanced_note:
                warnings.append(self._warning("RUNTIME_WEIGHT_REBALANCED", weight_rebalanced_note))

        recommendations = (
            self._aggregate_weighted(model_outputs, top_n=payload.top_n)
            if strategy == "ensemble_weighted"
            else self._single_model_result(model_outputs[0], top_n=payload.top_n)
        )
        contribution_breakdown = self._build_contribution_breakdown(recommendations)

        response = {
            "strategy": strategy,
            "user_id": payload.user_id,
            "top_n": payload.top_n,
            "weight_normalization_applied": prepared["weight_normalization_applied"],
            "weight_normalization_note": prepared["weight_normalization_note"],
            "normalization_strategy": "min_max_per_model",
            "weight_rebalanced_after_runtime_failures": weight_rebalanced,
            "weight_rebalanced_note": weight_rebalanced_note,
            "models_used": [
                {
                    "model_id": model["model_id"],
                    "algorithm": model["algorithm"],
                    "label": model["label"],
                    "promoted": model["promoted"],
                    "metric_name": model["metric_name"],
                    "metric_value": model["metric_value"],
                    "feedback_mode": model["feedback_mode"],
                    "status": model["status"],
                    "input_weight": model["input_weight"],
                    "normalized_weight": model["normalized_weight"],
                    "served_algorithm": model["served_algorithm"],
                }
                for model in model_outputs
            ],
            "contribution_breakdown": contribution_breakdown,
            "recommendations": recommendations,
            "warnings": warnings,
        }
        if len(model_outputs) == 1:
            response["model_id"] = model_outputs[0]["model_id"]
            response["algorithm"] = model_outputs[0]["algorithm"]
        else:
            response["model_id"] = None
            response["algorithm"] = "ensemble"
        return response

    def recommendation_options(self, last_result=None, top_n_models: Optional[int] = None) -> dict:
        """Return the ranked serving options and strategy metadata exposed to clients."""
        option_rows = self._build_recommendation_option_rows(last_result=last_result)
        best_promoted_model = self._best_promoted_option(last_result=last_result, option_rows=option_rows)
        single_model_options = [dict(row) for row in option_rows if row.get("recommendation_eligible")]
        selection_policy = self._selection_policy(last_result=last_result, option_rows=single_model_options)
        single_model_options = self._decorate_ranked_models(single_model_options, last_result=last_result)
        selection_policy = self._apply_rank_details_to_policy(
            selection_policy=selection_policy,
            ranked_rows=single_model_options,
        )
        selected_models = list((selection_policy or {}).get("selected_models", []))
        serving_selected_models = list((selection_policy or {}).get("serving_selected_models", []))
        default_rank_limit = len(single_model_options) if single_model_options else 1
        requested_rank_limit = self._coerce_int(top_n_models)
        if requested_rank_limit is None or requested_rank_limit <= 0:
            requested_rank_limit = default_rank_limit
        ranked_models = [dict(row) for row in single_model_options[:requested_rank_limit]]
        recommended_models = list(selected_models or serving_selected_models)
        ensemble_top_models = [dict(row) for row in (ranked_models or single_model_options)]
        best_model_explanation = None
        if ranked_models:
            best_model_explanation = ranked_models[0].get("reason")
        if not best_model_explanation and selection_policy:
            best_model_explanation = selection_policy.get("best_model_explanation") or selection_policy.get("reason")

        return {
            "resolved_mode": getattr(last_result, "resolved_mode", None) if last_result else None,
            "best_model_id": getattr(last_result, "best_model_id", None) if last_result else None,
            "best_algorithm": getattr(last_result, "best_algorithm", None) if last_result else None,
            "best_promoted_model": best_promoted_model,
            "single_model_options": single_model_options,
            "supported_models": single_model_options,
            "supported_model_count": len(single_model_options),
            "selected_models": selected_models,
            "selected_model_count": len(selected_models),
            "serving_selected_models": serving_selected_models,
            "serving_selected_model_count": len(serving_selected_models),
            "ensemble_top_models": ensemble_top_models,
            "recommended_models": recommended_models,
            "recommended_model_count": len(recommended_models),
            "ranked_models": ranked_models,
            "ranked_model_count": len(ranked_models),
            "ranked_model_limit": requested_rank_limit,
            "best_model_explanation": best_model_explanation,
            "selection_policy": selection_policy,
            "strategies": sorted(RECOMMEND_STRATEGIES),
            "has_recommendation_models": bool(single_model_options),
            "top_models": single_model_options,
            "ranked_algorithms": list(
                dict.fromkeys(
                    row["algorithm"]
                    for row in (ranked_models or single_model_options)
                    if row.get("algorithm")
                )
            ),
            "model_id_map": {
                row["algorithm"]: row["model_id"]
                for row in single_model_options
                if row.get("model_id") and row.get("algorithm")
            },
        }

    def _resolve_strategy(self, payload: RecommendationPayload) -> str:
        """Normalize the requested strategy and reject unsupported values early."""
        strategy = (payload.strategy or "best_promoted_model").strip().lower()
        if strategy not in RECOMMEND_STRATEGIES:
            raise RecommendationStrategyError(
                "INVALID_STRATEGY",
                f"strategy must be one of {sorted(RECOMMEND_STRATEGIES)}",
                strategy=strategy,
            )
        return strategy

    def _prepare_models(self, payload: RecommendationPayload, strategy: str, last_result, mode: Optional[str]) -> dict:
        """Resolve the model set, weights, and warnings required for the chosen strategy."""
        if strategy == "best_promoted_model":
            return self._prepare_best_promoted(last_result=last_result)

        if strategy == "single_model":
            request_model = WeightedModelInput(
                model_id=(payload.model_id or "").strip() or None,
                algorithm=(payload.algorithm or "").strip() or None,
                label=(payload.label or "").strip() or None,
                weight=1.0,
                allow_algorithm_fallback=payload.allow_algorithm_fallback or bool(payload.algorithm),
            )
            resolved, warnings = self._resolve_requested_models(
                selected_models=[request_model],
                strategy=strategy,
                mode=mode,
            )
            for model in resolved:
                model["input_weight"] = 1.0
                model["normalized_weight"] = 1.0
            return {
                "models": resolved,
                "weight_normalization_applied": False,
                "weight_normalization_note": "",
                "warnings": warnings,
            }

        selected_models = [
            WeightedModelInput(
                model_id=(selected.model_id or "").strip() or None,
                algorithm=(selected.algorithm or "").strip() or None,
                label=(selected.label or "").strip() or None,
                weight=float(selected.weight),
                allow_algorithm_fallback=selected.allow_algorithm_fallback or bool(selected.algorithm),
            )
            for selected in (payload.models or [])
        ]
        self._validate_ensemble_selection(selected_models)
        resolved, warnings = self._resolve_requested_models(
            selected_models=selected_models,
            strategy=strategy,
            mode=mode,
        )
        resolved, applied, note = self._normalize_weights(resolved, payload.auto_normalize_weights)
        return {
            "models": resolved,
            "weight_normalization_applied": applied,
            "weight_normalization_note": note,
            "warnings": warnings,
        }

    def _prepare_best_promoted(self, last_result=None) -> dict:
        """Build the single promoted-model selection used by the default strategy."""
        record, warnings = self._resolve_best_promoted_record(last_result=last_result)
        if record is None:
            raise RecommendationStrategyError(
                "NO_PROMOTED_MODEL",
                "No promoted model is available. Train models and promote one first.",
            )

        prepared = self._selected_model_from_record(record=record, input_weight=1.0)
        prepared["normalized_weight"] = 1.0
        return {
            "models": [prepared],
            "weight_normalization_applied": False,
            "weight_normalization_note": "",
            "warnings": warnings,
        }

    def _resolve_best_promoted_record(self, last_result=None) -> tuple[Optional[ModelRecord], list[dict]]:
        """Return the active deployment record plus any drift warnings from last training."""
        warnings: list[dict] = []
        record = self.deployments.get_active_record()
        best_model_id = getattr(last_result, "best_model_id", None) if last_result else None
        best_algorithm = getattr(last_result, "best_algorithm", None) if last_result else None

        if record is None:
            return None, warnings

        if best_model_id and best_model_id != record.model_id:
            current = self.registry.get(best_model_id)
            if current is None:
                warnings.append(
                    self._warning(
                        "MODEL_ID_NOT_FOUND_FALLBACK_USED",
                        "The last best model_id was not found in the registry; the active deployment was used.",
                        requested_model_id=best_model_id,
                        active_model_id=record.model_id,
                    )
                )
            else:
                warnings.append(
                    self._warning(
                        "ACTIVE_DEPLOYMENT_DIFFERS_FROM_LAST_TRAINING_RESULT",
                        "The active deployment differs from the last training best model; the active deployment was used.",
                        requested_model_id=best_model_id,
                        active_model_id=record.model_id,
                    )
                )

        if best_algorithm and best_algorithm != record.algorithm:
            warnings.append(
                self._warning(
                    "ACTIVE_DEPLOYMENT_ALGORITHM_DIFFERS",
                    "The active deployment algorithm differs from the last training best algorithm.",
                    best_algorithm=best_algorithm,
                    active_algorithm=record.algorithm,
                    active_model_id=record.model_id,
                )
            )

        return record, warnings

    def _validate_ensemble_selection(self, selected_models: list[WeightedModelInput]) -> None:
        """Validate ensemble request shape, dedupe model picks, and enforce weight rules."""
        if len(selected_models) < 2:
            raise RecommendationStrategyError(
                "ENSEMBLE_MIN_MODELS",
                "At least 2 models are required for ensemble_weighted.",
            )

        seen: set[str] = set()
        for model in selected_models:
            model_id = (model.model_id or "").strip()
            algorithm = (model.algorithm or "").strip()
            if not model_id and not algorithm:
                raise RecommendationStrategyError(
                    "MODEL_SELECTION_REQUIRED",
                    "Each ensemble model must include either model_id or algorithm.",
                )
            dedupe_key = f"model:{model_id}" if model_id else f"algorithm:{algorithm.lower()}"
            if dedupe_key in seen:
                raise RecommendationStrategyError(
                    "DUPLICATE_MODEL_SELECTION",
                    f"Duplicate ensemble selection '{model_id or algorithm}' is not allowed.",
                    duplicate_model_id=model_id,
                    duplicate_algorithm=algorithm or None,
                )
            seen.add(dedupe_key)
            if float(model.weight) <= 0:
                raise RecommendationStrategyError(
                    "INVALID_WEIGHT",
                    f"Model '{model_id or algorithm}' has invalid weight {model.weight}. Weights must be positive.",
                    model_id=model_id or None,
                    algorithm=algorithm or None,
                    weight=model.weight,
                )

    def _resolve_requested_models(
        self,
        selected_models: list[WeightedModelInput],
        strategy: str,
        mode: Optional[str],
    ) -> tuple[list[dict], list[dict]]:
        """Resolve client selections into concrete registry records and user-facing warnings."""
        resolved = []
        warnings = []
        for selected in selected_models:
            record, warning = self._resolve_model_record(selected=selected, strategy=strategy, mode=mode)
            if warning:
                warnings.append(warning)
            resolved.append(
                self._selected_model_from_record(
                    record=record,
                    input_weight=float(selected.weight),
                    label=(selected.label or "").strip() or None,
                )
            )
        return resolved, warnings

    def _resolve_model_record(
        self,
        selected: WeightedModelInput,
        strategy: str,
        mode: Optional[str],
    ) -> tuple[ModelRecord, Optional[dict]]:
        """Resolve one requested selector to a serveable model record or raise precisely."""
        requested_model_id = (selected.model_id or "").strip() or None
        requested_algorithm = (selected.algorithm or "").strip() or None
        allow_fallback = bool(selected.allow_algorithm_fallback)
        warning = None
        fallback_label = self._fallback_resolution_label(strategy)

        if requested_model_id:
            record = self.registry.get(requested_model_id)
            if record is None:
                if allow_fallback and requested_algorithm:
                    record = self._fallback_record_for_algorithm(requested_algorithm, strategy=strategy)
                    if record is None:
                        raise RecommendationStrategyError(
                            "MODEL_ID_NOT_FOUND",
                            f"Requested model_id '{requested_model_id}' was not found and no {fallback_label} exists for algorithm '{requested_algorithm}'.",
                            requested_model_id=requested_model_id,
                            requested_algorithm=requested_algorithm,
                        )
                    warning = self._warning(
                        "MODEL_ID_NOT_FOUND_FALLBACK_USED",
                        f"The requested model_id was not found; the {fallback_label} for the requested algorithm was used instead.",
                        requested_model_id=requested_model_id,
                        fallback_model_id=record.model_id,
                        requested_algorithm=requested_algorithm,
                    )
                else:
                    raise RecommendationStrategyError(
                        "MODEL_ID_NOT_FOUND",
                        f"Requested model_id '{requested_model_id}' was not found.",
                        requested_model_id=requested_model_id,
                    )
            if requested_algorithm and record.algorithm != requested_algorithm:
                raise RecommendationStrategyError(
                    "ALGORITHM_VERIFICATION_FAILED",
                    f"Requested model_id '{record.model_id}' belongs to '{record.algorithm}', not '{requested_algorithm}'.",
                    requested_model_id=record.model_id,
                    requested_algorithm=requested_algorithm,
                    resolved_algorithm=record.algorithm,
                )
        else:
            if not allow_fallback or not requested_algorithm:
                raise RecommendationStrategyError(
                    "MODEL_ID_REQUIRED",
                    f"model_id is required when strategy={strategy}.",
                    strategy=strategy,
                )
            record = self._fallback_record_for_algorithm(requested_algorithm, strategy=strategy)
            if record is None:
                raise RecommendationStrategyError(
                    "NO_MODEL_FOR_ALGORITHM",
                    f"No {fallback_label} was found for algorithm '{requested_algorithm}'.",
                    requested_algorithm=requested_algorithm,
                )
            warning = self._warning(
                "MODEL_ID_MISSING_FALLBACK_USED",
                f"No model_id was provided; the {fallback_label} for the requested algorithm was used instead.",
                fallback_model_id=record.model_id,
                requested_algorithm=requested_algorithm,
            )

        if record.algorithm not in ALGORITHM_REGISTRY:
            raise RecommendationStrategyError(
                "UNKNOWN_ALGORITHM",
                f"Unknown algorithm '{record.algorithm}' in model registry.",
                model_id=record.model_id,
                algorithm=record.algorithm,
            )
        if mode and not self._is_mode_compatible(mode, record.algorithm):
            raise RecommendationStrategyError(
                "MODEL_INCOMPATIBLE_WITH_DATASET_MODE",
                f"Algorithm '{record.algorithm}' is incompatible with latest dataset mode '{mode}'.",
                model_id=record.model_id,
                algorithm=record.algorithm,
                resolved_mode=mode,
            )
        if not self._allows_unpromoted_serving(strategy=strategy) and not record.promoted:
            raise RecommendationStrategyError(
                "INACTIVE_MODEL_NOT_ALLOWED",
                (
                    f"Model '{record.model_id}' is not the active deployment. "
                    "Production inference is restricted to promoted models."
                ),
                model_id=record.model_id,
                algorithm=record.algorithm,
            )
        return record, warning

    def _run_models(self, selected_models: list[dict], user_id: str, top_n: int) -> tuple[list[dict], list[dict]]:
        """Run inference across the prepared models and collect non-fatal runtime issues."""
        outputs = []
        issues = []
        for model in selected_models:
            try:
                loaded = self._load_model(model)
                rec = self.recommender.recommend_with_loaded_model(
                    user_id=user_id,
                    loaded=loaded,
                    top_n=top_n,
                    algorithm_label=model["algorithm"],
                )
            except Exception as e:
                log.warning("Model recommendation failed | algorithm=%s error=%s", model["algorithm"], e)
                issues.append(
                    self._warning(
                        "MODEL_INFERENCE_FAILED",
                        f"Model '{model['algorithm']}' failed during recommendation: {e}",
                        model_id=model["model_id"],
                        algorithm=model["algorithm"],
                    )
                )
                continue

            if not rec.recommendations:
                issues.append(
                    self._warning(
                        "NO_PREDICTIONS_FOR_USER",
                        f"Model '{model['algorithm']}' returned no recommendations for user '{user_id}'.",
                        model_id=model["model_id"],
                        algorithm=model["algorithm"],
                        user_id=user_id,
                    )
                )
                continue

            if rec.algorithm != model["algorithm"]:
                issues.append(
                    self._warning(
                        "USER_NOT_FOUND_POPULARITY_FALLBACK_USED",
                        f"User '{user_id}' was not found for model '{model['algorithm']}'; popularity fallback scores were used.",
                        model_id=model["model_id"],
                        algorithm=model["algorithm"],
                        served_algorithm=rec.algorithm,
                        user_id=user_id,
                    )
                )

            outputs.append(
                {
                    **model,
                    "served_algorithm": rec.algorithm,
                    "recommendations": rec.recommendations,
                }
            )
        return outputs, issues

    def _build_recommendation_option_rows(self, last_result=None) -> list[dict]:
        """Assemble deduplicated model option rows from training output and registry state."""
        leaderboard_by_algo = self._leaderboard_rows(last_result)
        training_rows = self._training_option_sources(last_result)
        all_records = self.registry.list_models()
        latest_by_algorithm: dict[str, ModelRecord] = {}
        for record in all_records:
            latest_by_algorithm.setdefault(record.algorithm, record)
        rows: list[dict] = []
        seen: set[str] = set()

        for source in training_rows:
            algorithm = source.get("algorithm")
            model_id = source.get("model_id")
            record = self.registry.get(model_id) if model_id else None
            if record is None and algorithm:
                record = latest_by_algorithm.get(str(algorithm))
            row = self._normalize_option_row(
                row=source,
                record=record,
                leaderboard_row=leaderboard_by_algo.get(algorithm),
                default_rank=source.get("rank"),
                last_result=last_result,
            )
            key = row.get("model_id") or f"algorithm:{row.get('algorithm')}"
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

        if not rows:
            latest_records = list(latest_by_algorithm.values())
            for index, record in enumerate(latest_records, start=1):
                row = self._normalize_option_row(
                    row={},
                    record=record,
                    leaderboard_row=leaderboard_by_algo.get(record.algorithm),
                    default_rank=index,
                    last_result=last_result,
                )
                key = row.get("model_id") or f"algorithm:{row.get('algorithm')}"
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)

        rows.sort(key=self._option_sort_key)
        for index, row in enumerate(rows, start=1):
            if row.get("rank") in (None, "", 0):
                row["rank"] = index
        return rows

    def _best_promoted_option(self, last_result=None, option_rows: Optional[list[dict]] = None) -> Optional[dict]:
        """Return the currently promoted option enriched with training drift warnings."""
        option_rows = option_rows or []
        record, warnings = self._resolve_best_promoted_record(last_result=last_result)
        if record is None:
            return {
                "rank": None,
                "model_id": None,
                "algorithm": None,
                "label": "No promoted model available",
                "promoted": False,
                "metric_name": None,
                "metric_value": None,
                "recommendation_eligible": False,
                "feedback_mode": None,
                "status": "unavailable",
                "selection_warnings": warnings,
            }

        for row in option_rows:
            if row.get("model_id") == record.model_id:
                out = dict(row)
                out["selection_warnings"] = warnings
                return out

        leaderboard_by_algo = self._leaderboard_rows(last_result)
        row = self._normalize_option_row(
            row={},
            record=record,
            leaderboard_row=leaderboard_by_algo.get(record.algorithm),
            default_rank=None,
            last_result=last_result,
        )
        row["selection_warnings"] = warnings
        return row

    def _training_option_sources(self, last_result=None) -> list[dict]:
        """Extract the best available model-ranking hints from the last training result."""
        rows = list(getattr(last_result, "top_model_recommendations", []) or [])
        if rows:
            return [dict(row) for row in rows]

        best_algorithm = getattr(last_result, "best_algorithm", None) if last_result else None
        best_model_id = getattr(last_result, "best_model_id", None) if last_result else None
        if best_algorithm or best_model_id:
            return [
                {
                    "rank": 1,
                    "algorithm": best_algorithm,
                    "model_id": best_model_id,
                    "status": "ok",
                }
            ]
        return []

    def _leaderboard_rows(self, last_result=None) -> dict[str, dict]:
        """Index leaderboard rows by algorithm for fast option enrichment."""
        report = getattr(last_result, "report", None)
        if report is None or not hasattr(report, "leaderboard"):
            return {}
        leaderboard = report.leaderboard()
        if getattr(leaderboard, "empty", True):
            return {}
        return {
            str(row.get("Algorithm")): dict(row)
            for row in leaderboard.to_dict(orient="records")
            if row.get("Algorithm")
        }

    def _normalize_option_row(
        self,
        row: dict,
        record: Optional[ModelRecord],
        leaderboard_row: Optional[dict],
        default_rank: Optional[int],
        last_result=None,
    ) -> dict:
        """Merge registry, leaderboard, and training metadata into one option row."""
        raw = dict(row or {})
        algorithm = raw.get("algorithm") or getattr(record, "algorithm", None)
        model_id = raw.get("model_id") or getattr(record, "model_id", None)

        metric_name = raw.get("metric_name") or self._metric_name_from_row(leaderboard_row, last_result=last_result)
        metric_value = raw.get("metric_value")
        if metric_value is None and metric_name and leaderboard_row:
            metric_value = leaderboard_row.get(metric_name)
        if metric_value is None and record is not None:
            record_metric_name, record_metric_value = self._metric_from_record(record)
            metric_name = metric_name or record_metric_name
            metric_value = record_metric_value if metric_value is None else metric_value

        promoted = bool(getattr(record, "promoted", False))
        feedback_mode = self._feedback_mode_for_algorithm(algorithm, record=record)
        resolved_mode = getattr(last_result, "resolved_mode", None) if last_result else None
        compatible = bool(model_id and algorithm and algorithm in ALGORITHM_REGISTRY)
        if compatible and resolved_mode:
            allowed = _MODE_TO_ALLOWED_FEEDBACK.get(
                str(resolved_mode).strip().lower(),
                {"explicit", "implicit", "both"},
            )
            compatible = feedback_mode in allowed

        status = raw.get("status") or ("promoted" if promoted else ("saved" if model_id else "unavailable"))
        if not compatible and status not in {"unavailable", "error"}:
            status = "incompatible"

        rank = raw.get("rank")
        if rank is None and leaderboard_row:
            rank = leaderboard_row.get("Rank")
        if rank is None:
            rank = default_rank

        return {
            "rank": self._coerce_int(rank),
            "model_id": model_id,
            "algorithm": algorithm,
            "label": raw.get("label") or self._build_option_label(
                rank=rank,
                algorithm=algorithm,
                metric_name=metric_name,
                metric_value=metric_value,
            ),
            "promoted": promoted,
            "metric_name": metric_name,
            "metric_value": self._coerce_float(metric_value),
            "recommendation_eligible": compatible,
            "feedback_mode": feedback_mode,
            "status": status,
            "summary": raw.get("summary"),
            "performance_summary": raw.get("performance_summary"),
            "fit_summary": raw.get("fit_summary"),
            "reliability_summary": raw.get("reliability_summary"),
            "selection_score_pct": self._coerce_float(raw.get("selection_score_pct")),
            "reasons": list(raw.get("reasons") or []),
            "decision_note": raw.get("decision_note"),
        }

    _normalise_option_row = _normalize_option_row

    def _metric_name_from_row(self, leaderboard_row: Optional[dict], last_result=None) -> Optional[str]:
        """Choose the most relevant metric label for a model option row."""
        report = getattr(last_result, "report", None)
        primary_metric = getattr(report, "primary_metric", None)
        if primary_metric:
            return primary_metric
        if leaderboard_row:
            for key in _METRIC_PRIORITY:
                if key in leaderboard_row:
                    return key
        return None

    def _build_option_label(
        self,
        rank: Optional[int],
        algorithm: Optional[str],
        metric_name: Optional[str],
        metric_value: Optional[float],
    ) -> str:
        """Build the human-readable label shown in recommendation option payloads."""
        prefix = f"#{int(rank)} " if self._coerce_int(rank) else ""
        algo = algorithm or "Unknown model"
        if metric_name and metric_value is not None:
            return f"{prefix}{algo} [{metric_name} {self._format_metric(metric_value)}]"
        return f"{prefix}{algo}"

    def _selected_model_from_record(
        self,
        record: ModelRecord,
        input_weight: float,
        label: Optional[str] = None,
    ) -> dict:
        """Convert a registry record into the canonical serving-model payload shape."""
        metric_name, metric_value = self._metric_from_record(record)
        return {
            "model_id": record.model_id,
            "algorithm": record.algorithm,
            "label": label or self._build_option_label(None, record.algorithm, metric_name, metric_value),
            "promoted": bool(record.promoted),
            "metric_name": metric_name,
            "metric_value": metric_value,
            "feedback_mode": self._feedback_mode_for_algorithm(record.algorithm, record=record),
            "status": "promoted" if record.promoted else "saved",
            "input_weight": float(input_weight),
            "normalized_weight": float(input_weight),
        }

    def _metric_from_record(self, record: ModelRecord) -> tuple[Optional[str], Optional[float]]:
        """Pick the highest-priority metric available on a stored model record."""
        metrics = dict(getattr(record, "metrics", {}) or {})
        for key in _METRIC_PRIORITY:
            if key in metrics and metrics.get(key) is not None:
                return key, self._coerce_float(metrics.get(key))
        return None, None

    def _load_model(self, model: dict) -> LoadedModel:
        """Load the concrete model artifact backing a prepared serving selection."""
        return self.loader.load_by_id(tenant_id=self.tenant_id, model_id=model["model_id"])

    def _warning(self, _code: str, message: str, **_details) -> str:
        """Return the stable warning message exposed to API clients."""
        return message

    def _feedback_mode_for_algorithm(
        self,
        algorithm: Optional[str],
        record: Optional[ModelRecord] = None,
    ) -> Optional[str]:
        """Resolve whether an algorithm expects explicit, implicit, or dual feedback."""
        if not algorithm:
            return None
        meta = ALGORITHM_REGISTRY.get(str(algorithm))
        feedback = (meta or {}).get("feedback")
        if feedback in {"explicit", "implicit", "both"}:
            return feedback

        if record is not None:
            return "implicit" if bool(getattr(record, "is_implicit", False)) else "explicit"

        latest = self.registry.latest_for_algorithm(str(algorithm)) if hasattr(self.registry, "latest_for_algorithm") else None
        if latest is not None:
            return "implicit" if bool(getattr(latest, "is_implicit", False)) else "explicit"
        return None

    def _is_mode_compatible(self, mode: Optional[str], algorithm: Optional[str]) -> bool:
        """Check whether the latest dataset mode can legally serve the given algorithm."""
        if not algorithm:
            return False
        requested_mode = (mode or "auto").strip().lower()
        feedback_mode = self._feedback_mode_for_algorithm(algorithm)
        allowed = _MODE_TO_ALLOWED_FEEDBACK.get(requested_mode, {"explicit", "implicit", "both"})
        return feedback_mode in allowed

    def _normalize_weights(self, resolved: list[dict], auto_normalize: bool) -> tuple[list[dict], bool, str]:
        """Normalize client-supplied weights into serving fractions and explain what changed."""
        if not resolved:
            return [], False, ""

        total = sum(max(float(model.get("input_weight", 0.0)), 0.0) for model in resolved)
        if total <= 0:
            raise RecommendationStrategyError(
                "INVALID_WEIGHT_SUM",
                "Model weights must sum to a positive value.",
            )

        near_one = abs(total - 1.0) < 1e-8
        near_hundred = abs(total - 100.0) < 1e-8
        if not (near_one or near_hundred) and not auto_normalize:
            raise RecommendationStrategyError(
                "WEIGHT_SUM_INVALID",
                "Weights must sum to 1.0 or 100.0 when auto_normalize_weights is disabled.",
                weight_sum=round(total, 6),
            )

        divisor = 1.0 if near_one else (100.0 if near_hundred else total)
        applied = not near_one
        note = ""
        if near_hundred:
            note = "Weights were interpreted as percentages and converted to fractions."
        elif applied:
            note = f"Weights were auto-normalized from total {total:.4f} to sum to 1.0."

        normalized: list[dict] = []
        for model in resolved:
            out = dict(model)
            out["normalized_weight"] = float(model.get("input_weight", 0.0)) / divisor
            normalized.append(out)
        return normalized, applied, note

    def _rebalance_runtime_weights(
        self,
        model_outputs: list[dict],
        had_drops: bool,
    ) -> tuple[list[dict], bool, str]:
        """Renormalize weights after runtime model failures so ensemble sums stay valid."""
        if not model_outputs:
            return [], False, ""

        total = sum(max(float(model.get("normalized_weight", 0.0)), 0.0) for model in model_outputs)
        if not had_drops and abs(total - 1.0) < 1e-8:
            return model_outputs, False, ""

        if total <= 0:
            replacement = 1.0 / len(model_outputs)
            rebalanced = [{**model, "normalized_weight": replacement} for model in model_outputs]
        else:
            rebalanced = [
                {**model, "normalized_weight": float(model.get("normalized_weight", 0.0)) / total}
                for model in model_outputs
            ]
        note = "Weights were rebalanced across the surviving models after runtime drops."
        return rebalanced, True, note

    def _aggregate_weighted(self, model_outputs: list[dict], top_n: int) -> list[dict]:
        """Combine model outputs into a weighted ensemble ranking with contribution traces."""
        item_map: dict[str, dict] = {}
        for model in model_outputs:
            recommendations = list(model.get("recommendations", []) or [])
            numeric_scores = [
                float(row.get("score"))
                for row in recommendations
                if row.get("score") is not None and self._coerce_float(row.get("score")) is not None
            ]
            if not numeric_scores:
                continue
            min_score = min(numeric_scores)
            max_score = max(numeric_scores)
            span = max_score - min_score

            for row in recommendations:
                item_id = row.get("item_id")
                raw_score = self._coerce_float(row.get("score"))
                if item_id is None or raw_score is None:
                    continue
                normalized_score = 1.0 if span == 0 else (raw_score - min_score) / span
                weighted_score = normalized_score * float(model.get("normalized_weight", 0.0))
                bucket = item_map.setdefault(
                    str(item_id),
                    {"item_id": str(item_id), "final_score": 0.0, "contributions": []},
                )
                bucket["final_score"] += weighted_score
                bucket["contributions"].append(
                    {
                        "model_id": model.get("model_id"),
                        "algorithm": model.get("algorithm"),
                        "raw_score": raw_score,
                        "normalized_score": normalized_score,
                        "input_weight": self._coerce_float(model.get("input_weight")),
                        "normalized_weight": self._coerce_float(model.get("normalized_weight")),
                        "weighted_score": weighted_score,
                    }
                )

        ordered = sorted(
            item_map.values(),
            key=lambda row: (-float(row.get("final_score", 0.0)), str(row.get("item_id", ""))),
        )
        recommendations: list[dict] = []
        for index, row in enumerate(ordered[: max(int(top_n), 1)], start=1):
            final_score = float(row.get("final_score", 0.0))
            contributions = sorted(
                row.get("contributions", []),
                key=lambda item: (-float(item.get("weighted_score", 0.0)), str(item.get("algorithm", ""))),
            )
            for contribution in contributions:
                share = 0.0 if final_score <= 0 else (float(contribution["weighted_score"]) / final_score) * 100.0
                contribution["share_pct"] = round(share, 4)
            recommendations.append(
                {
                    "rank": index,
                    "item_id": row["item_id"],
                    "final_score": round(final_score, 6),
                    "contributions": contributions,
                    "explanation": "Weighted ensemble score derived from normalized model contributions.",
                }
            )
        return recommendations

    def _single_model_result(self, model_output: dict, top_n: int) -> list[dict]:
        """Reuse the weighted aggregation path for a single-model response."""
        return self._aggregate_weighted([model_output], top_n=top_n)

    def _build_contribution_breakdown(self, recommendations: list[dict]) -> list[dict]:
        """Summarize how much each model contributed across the final ranked results."""
        totals: dict[tuple[str, Optional[str]], dict] = {}
        for row in recommendations:
            for contribution in row.get("contributions", []) or []:
                key = (str(contribution.get("algorithm") or ""), contribution.get("model_id"))
                bucket = totals.setdefault(
                    key,
                    {
                        "algorithm": contribution.get("algorithm"),
                        "model_id": contribution.get("model_id"),
                        "weighted_score": 0.0,
                        "item_count": 0,
                    },
                )
                bucket["weighted_score"] += float(contribution.get("weighted_score", 0.0))
                bucket["item_count"] += 1

        total_weighted = sum(row["weighted_score"] for row in totals.values())
        ordered = sorted(
            totals.values(),
            key=lambda row: (-float(row["weighted_score"]), str(row.get("algorithm", ""))),
        )
        for row in ordered:
            share = 0.0 if total_weighted <= 0 else (row["weighted_score"] / total_weighted) * 100.0
            row["share_pct"] = round(share, 4)
            row["weighted_score"] = round(float(row["weighted_score"]), 6)
        return ordered

    def _latest_promoted_record(self) -> Optional[ModelRecord]:
        """Return the active promoted record for the tenant, if one exists."""
        return self.deployments.get_active_record()

    def _allows_unpromoted_serving(self, strategy: Optional[str]) -> bool:
        """Return whether the selected strategy may serve saved, non-promoted models."""
        return bool(ALLOW_INACTIVE_MODEL_SERVING or strategy == "ensemble_weighted")

    def _fallback_resolution_label(self, strategy: Optional[str]) -> str:
        """Describe the fallback model source used in user-facing diagnostics."""
        if self._allows_unpromoted_serving(strategy):
            return "latest available model"
        return "active deployment-compatible model"

    def _fallback_record_for_algorithm(self, algorithm: str, strategy: Optional[str] = None) -> Optional[ModelRecord]:
        """Resolve the fallback model for an algorithm based on strategy-aware serving policy."""
        if self._allows_unpromoted_serving(strategy):
            return self.registry.latest_for_algorithm(algorithm)
        return self.registry.get_promoted(algorithm)

    def _selection_policy(self, last_result=None, option_rows: Optional[list[dict]] = None) -> Optional[dict]:
        """Return the persisted selection policy or synthesize a minimal default policy."""
        if last_result is not None:
            existing = getattr(last_result, "model_selection_policy", None)
            if existing:
                return dict(existing)

        rows = [dict(row) for row in (option_rows or []) if row.get("algorithm")]
        if not rows:
            return None

        selected = rows[:1]
        return {
            "selection_type": "single_model",
            "selected_count": 1,
            "selected_models": selected,
            "reason": f"{selected[0].get('algorithm')} is the highest-ranked available model.",
            "display_title": "Recommended single model",
            "recommended_strategy": "single_model",
        }

    def _apply_rank_details_to_policy(self, selection_policy: Optional[dict], ranked_rows: list[dict]) -> Optional[dict]:
        """Enrich policy-selected models with the latest ranking explanations and comparisons."""
        if not selection_policy:
            return selection_policy
        row_map = {
            (row.get("model_id") or f"algorithm:{row.get('algorithm')}"): dict(row)
            for row in ranked_rows
        }
        enriched = dict(selection_policy)
        for key in ("selected_models", "serving_selected_models"):
            current = list(enriched.get(key) or [])
            updated = []
            for row in current:
                row_key = row.get("model_id") or f"algorithm:{row.get('algorithm')}"
                merged = dict(row)
                merged.update(row_map.get(row_key) or {})
                updated.append(merged)
            enriched[key] = updated
        selected_models = list(enriched.get("selected_models") or [])
        serving_selected_models = list(enriched.get("serving_selected_models") or [])
        if (
            enriched.get("recommended_strategy") == "ensemble_weighted"
            and selected_models
            and len(serving_selected_models) < len(selected_models)
        ):
            enriched["serving_selected_models"] = [dict(row) for row in selected_models]
            enriched["serving_selected_count"] = len(selected_models)
            for text_key in ("reason", "serving_reason"):
                text_value = enriched.get(text_key)
                if isinstance(text_value, str):
                    text_value = text_value.replace("best 5 models", f"best {len(selected_models)} models")
                    text_value = text_value.replace("at most 5 models", "every shortlisted model")
                    enriched[text_key] = text_value
            rules = []
            for rule in list(enriched.get("decision_rules") or []):
                if not isinstance(rule, str):
                    continue
                rules.append(
                    rule
                    .replace("at most 5 models", "every shortlisted model")
                    .replace("at most 5", "the full shortlisted set")
                )
            if rules:
                enriched["decision_rules"] = rules
        if enriched.get("selected_models"):
            enriched["best_model_explanation"] = (
                enriched.get("best_model_explanation")
                or enriched["selected_models"][0].get("reason")
                or enriched.get("reason")
            )
            enriched["comparison_explanations"] = [
                row["comparison_to_next"]
                for row in enriched["selected_models"]
                if row.get("comparison_to_next")
            ]
        return enriched

    def _option_sort_key(self, row: dict) -> tuple:
        """Produce the stable sort key used to rank serving options for clients."""
        rank = self._coerce_int(row.get("rank"))
        metric_name = row.get("metric_name")
        metric_value = self._coerce_float(row.get("metric_value"))
        if metric_name == "RMSE":
            metric_key = metric_value if metric_value is not None else float("inf")
        else:
            metric_key = -(metric_value if metric_value is not None else float("-inf"))
        return (
            0 if row.get("recommendation_eligible") else 1,
            0 if row.get("promoted") else 1,
            rank if rank is not None else 10**9,
            metric_key,
            str(row.get("algorithm") or ""),
        )

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        """Safely coerce optional numeric-like values into ints."""
        try:
            if value in (None, ""):
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        """Safely coerce optional numeric-like values into finite floats."""
        try:
            if value in (None, ""):
                return None
            out = float(value)
        except (TypeError, ValueError):
            return None
        if out != out or out in {float("inf"), float("-inf")}:
            return None
        return out

    @staticmethod
    def _format_metric(value: float) -> str:
        """Format metric values consistently for labels and explanations."""
        return f"{float(value):.4f}"

    def _decorate_ranked_models(self, rows: list[dict], last_result=None) -> list[dict]:
        """Attach pairwise comparisons and reason strings to ranked option rows."""
        ordered = [dict(row) for row in rows]
        primary_metric, metric_direction = self._resolve_primary_metric(last_result=last_result, rows=ordered)
        for index, row in enumerate(ordered):
            previous_row = ordered[index - 1] if index > 0 else None
            next_row = ordered[index + 1] if index + 1 < len(ordered) else None
            row["comparison_to_previous"] = (
                self._build_rank_comparison(previous_row, row, primary_metric, metric_direction)
                if previous_row
                else None
            )
            row["comparison_to_next"] = (
                self._build_rank_comparison(row, next_row, primary_metric, metric_direction)
                if next_row
                else None
            )
            row["reason"] = self._build_rank_reason(
                row=row,
                previous_row=previous_row,
                next_row=next_row,
                primary_metric=primary_metric,
                metric_direction=metric_direction,
            )
        return ordered

    def _resolve_primary_metric(self, last_result=None, rows: Optional[list[dict]] = None) -> tuple[str, str]:
        """Determine the primary comparison metric and its optimization direction."""
        policy = getattr(last_result, "model_selection_policy", None) if last_result is not None else None
        primary_metric = (policy or {}).get("primary_metric")
        direction = (policy or {}).get("primary_metric_direction")
        report = getattr(last_result, "report", None) if last_result is not None else None
        if not primary_metric and report is not None:
            primary_metric = getattr(report, "primary_metric", None)
        if not direction and report is not None:
            direction = getattr(report, "primary_metric_direction", None)
        if not primary_metric and rows:
            for row in rows:
                if row.get("metric_name"):
                    primary_metric = row.get("metric_name")
                    break
        if not primary_metric:
            primary_metric = "RMSE" if any((row.get("metric_name") == "RMSE") for row in (rows or [])) else "NDCG@K"
        if not direction:
            direction = "minimize" if primary_metric == "RMSE" else "maximize"
        return str(primary_metric), str(direction)

    def _build_rank_reason(
        self,
        row: dict,
        previous_row: Optional[dict],
        next_row: Optional[dict],
        primary_metric: str,
        metric_direction: str,
    ) -> str:
        """Generate the user-facing narrative for why a model holds its current rank."""
        rank = self._coerce_int(row.get("rank")) or 0
        score = self._coerce_float(row.get("selection_score_pct"))
        algorithm = row.get("algorithm") or "Unknown model"
        parts = []
        if rank == 1:
            if score is not None:
                parts.append(f"Top 1: {algorithm} leads with a selection score of {score:.2f}%.")
            else:
                parts.append(f"Top 1: {algorithm} is the highest-ranked supported model.")
        else:
            if score is not None:
                parts.append(f"Top {rank}: {algorithm} follows with a selection score of {score:.2f}%.")
            else:
                parts.append(f"Top {rank}: {algorithm} remains in the supported ranking.")

        if next_row:
            parts.append(self._build_rank_comparison(row, next_row, primary_metric, metric_direction))
        elif previous_row:
            previous_rank = self._coerce_int(previous_row.get("rank")) or max(rank - 1, 1)
            previous_algorithm = previous_row.get("algorithm") or "the higher-ranked model"
            parts.append(
                f"It trails #{previous_rank} {previous_algorithm} and is the lowest-ranked supported model in this list."
            )

        performance_summary = row.get("performance_summary")
        fit_summary = row.get("fit_summary")
        reliability_summary = row.get("reliability_summary")
        if performance_summary:
            parts.append(str(performance_summary))
        if fit_summary:
            parts.append(str(fit_summary))
        if reliability_summary:
            parts.append(str(reliability_summary))
        elif row.get("summary"):
            parts.append(str(row.get("summary")))
        return " ".join(part for part in parts if part)

    def _build_rank_comparison(
        self,
        higher: Optional[dict],
        lower: Optional[dict],
        primary_metric: str,
        metric_direction: str,
    ) -> str:
        """Describe why one ranked model stays ahead of the next model in the list."""
        if not higher or not lower:
            return ""
        higher_name = higher.get("algorithm") or "Higher-ranked model"
        lower_name = lower.get("algorithm") or "Lower-ranked model"
        lower_rank = self._coerce_int(lower.get("rank")) or 0
        higher_score = self._coerce_float(higher.get("selection_score_pct"))
        lower_score = self._coerce_float(lower.get("selection_score_pct"))
        if higher_score is not None and lower_score is not None:
            score_text = (
                f"It stays above #{lower_rank} {lower_name} because its selection score is "
                f"{higher_score:.2f}% versus {lower_score:.2f}% ({higher_score - lower_score:+.2f} points)."
            )
        else:
            score_text = f"It stays above #{lower_rank} {lower_name} on the current ranking inputs."

        details = []
        metric_detail = self._metric_comparison_detail(
            higher=higher,
            lower=lower,
            primary_metric=primary_metric,
            metric_direction=metric_direction,
        )
        if metric_detail:
            details.append(metric_detail)
        higher_composite = self._coerce_float(higher.get("composite_score"))
        lower_composite = self._coerce_float(lower.get("composite_score"))
        if higher_composite is not None and lower_composite is not None:
            details.append(f"Composite Score is also stronger ({higher_composite:.4f} vs {lower_composite:.4f}).")
        return " ".join([score_text, *details]).strip()

    def _metric_comparison_detail(
        self,
        higher: dict,
        lower: dict,
        primary_metric: str,
        metric_direction: str,
    ) -> str:
        """Return the metric-specific comparison sentence that supports rank ordering."""
        higher_metric = self._coerce_float(higher.get("metric_value"))
        lower_metric = self._coerce_float(lower.get("metric_value"))
        if higher_metric is None or lower_metric is None:
            return ""

        direction = (metric_direction or "").strip().lower()
        minimizing = direction.startswith("min") or primary_metric == "RMSE"
        if minimizing:
            if higher_metric > lower_metric:
                return ""
            return f"It also has lower {primary_metric} ({higher_metric:.4f} vs {lower_metric:.4f})."
        if higher_metric < lower_metric:
            return ""
        return f"It also has higher {primary_metric} ({higher_metric:.4f} vs {lower_metric:.4f})."
