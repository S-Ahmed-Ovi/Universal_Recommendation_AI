"""
Benchmarks route

GET /benchmarks — get the latest model leaderboard and metrics for the tenant
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request

from backend.api.gateway.auth import require_auth
from .shared import _get_last_result, _get_registry, _sanitise

router = APIRouter(tags=["Benchmarks"])

_EXPECTED_SHAPE_FIELDS = [
    ("user_id", "Required key"),
    ("item_id", "Required key"),
    ("rating", "Required signal"),
    ("timestamp", "Recommended"),
    ("context", "Optional metadata"),
]


def _leaderboard_rows(last_result) -> list[dict]:
    """Build leaderboard rows while backfilling model ids from training metadata."""
    if not last_result:
        return []

    rows = []

    report = getattr(last_result, "report", None)
    if report is not None and hasattr(report, "leaderboard"):
        try:
            df = report.leaderboard()
            rows = df.to_dict(orient="records") if not df.empty else []
        except Exception:
            rows = []

    if not rows:
        raw = getattr(last_result, "leaderboard", [])
        if callable(raw):
            try:
                df = raw()
                rows = df.to_dict(orient="records") if not df.empty else []
            except Exception:
                rows = []
        elif isinstance(raw, list):
            rows = list(raw)

    all_model_ids = list(getattr(last_result, "all_model_ids", []) or [])
    top_model_recommendations = list(getattr(last_result, "top_model_recommendations", []) or [])
    recommendation_map = {
        str(item.get("algorithm")): item.get("model_id")
        for item in top_model_recommendations
        if item.get("algorithm") and item.get("model_id")
    }

    for idx, row in enumerate(rows):
        if row.get("model_id"):
            continue
        rank = int(row.get("Rank", idx + 1)) if str(row.get("Rank", "")).isdigit() else idx + 1
        if 0 < rank <= len(all_model_ids):
            row["model_id"] = all_model_ids[rank - 1]
            continue
        algo = row.get("Algorithm")
        if algo and recommendation_map.get(str(algo)):
            row["model_id"] = recommendation_map[str(algo)]

    return rows


def _tuning_rows(last_result) -> list[dict]:
    """Normalise tuning results into the stable API shape expected by clients."""
    if not last_result:
        return []

    raw = getattr(last_result, "tuning", None)
    if isinstance(raw, list) and raw:
        return raw

    tuning_results = getattr(last_result, "tuning_results", None)
    if not tuning_results:
        return []

    out = []
    for t in tuning_results:
        out.append(
            {
                "algorithm": getattr(t, "algorithm", None),
                "best_value": getattr(t, "best_value", None),
                "metric": getattr(t, "metric_name", None),
                "best_params": getattr(t, "best_params", {}),
                "n_trials": getattr(t, "n_trials", None),
                "status": getattr(t, "status", None),
                "fallback_reason": getattr(t, "fallback_reason", None),
                "trial_budget": getattr(t, "trial_budget", None),
            }
        )
    return out


def _dataset_contract(last_result) -> tuple[list[dict], list[str], dict]:
    """Summarise dataset readiness, available columns, and high-level training stats."""
    if not last_result:
        return [], [], {}

    report = getattr(last_result, "report", None)
    train_df = getattr(report, "train", None) if report is not None else None
    columns = []
    stats = {}

    if train_df is not None and hasattr(train_df, "columns"):
        try:
            columns = [str(c) for c in list(train_df.columns)]
            stats = {
                "rows": int(len(train_df)),
                "columns": int(len(columns)),
                "users": int(train_df["user_id"].nunique()) if "user_id" in train_df.columns else None,
                "items": int(train_df["item_id"].nunique()) if "item_id" in train_df.columns else None,
            }
        except Exception:
            columns = []
            stats = {}

    if not columns:
        columns = list(getattr(last_result, "dataset_columns", []) or [])
        stats = dict(getattr(last_result, "dataset_stats", {}) or {})

    lower_columns = {c.lower() for c in columns}
    shape = []
    for field, requirement in _EXPECTED_SHAPE_FIELDS:
        present = field.lower() in lower_columns
        shape.append(
            {
                "field": field,
                "requirement": requirement,
                "present": present,
                "readiness": 100 if present else 0,
            }
        )
    return shape, columns, stats


def _build_benchmarks_payload(tenant_id: str, last_result) -> dict:
    """Assemble the full benchmarks response for the tenant's latest training state."""
    registry = _get_registry(tenant_id)
    promoted_records = registry.list_models(promoted_only=True)
    promoted = [
        {
            "model_id": r.model_id,
            "algorithm": r.algorithm,
            "metrics": r.metrics,
            "params": r.params,
            "created_at": r.created_at,
            "notes": r.notes,
        }
        for r in promoted_records
    ]

    leaderboard = _leaderboard_rows(last_result)
    tuning = _tuning_rows(last_result)
    best_algo = getattr(last_result, "best_algorithm", None) if last_result else None
    best_model_id = getattr(last_result, "best_model_id", None) if last_result else None
    elapsed = getattr(last_result, "elapsed_s", None) if last_result else None
    mode = getattr(last_result, "resolved_mode", None) if last_result else None
    top_model_recommendations = list(getattr(last_result, "top_model_recommendations", []) or []) if last_result else []
    model_selection_policy = dict(getattr(last_result, "model_selection_policy", {}) or {}) if last_result else {}
    dataset_shape, dataset_columns, dataset_stats = _dataset_contract(last_result)

    return {
        "tenant_id": tenant_id,
        "best_algorithm": best_algo,
        "best_model_id": best_model_id,
        "leaderboard": leaderboard,
        "tuning": tuning,
        "elapsed_s": elapsed,
        "resolved_mode": mode,
        "top_model_recommendations": top_model_recommendations,
        "model_selection_policy": model_selection_policy,
        "dataset_shape": dataset_shape,
        "dataset_columns": dataset_columns,
        "dataset_stats": dataset_stats,
        "promoted_models": promoted,
        "message": "No training results found. Run a training job first." if not last_result else None,
    }


@router.get("/benchmarks")
@require_auth(scope="models:read")
async def get_benchmarks(request: Request):
    """
    Return the leaderboard from the most recent training run AND the
    currently promoted models across all runs.
    """
    tenant_id: str = request.state.tenant_id
    last_result = _get_last_result(tenant_id)
    payload = await asyncio.to_thread(_build_benchmarks_payload, tenant_id, last_result)
    return _sanitise(payload)
