"""
FastAPI route definitions.

Public:
- /health
- /session/status
- /algorithms
- /auth/login
- /train/file
- /train/sql
- /jobs/{job_id}

Protected (Bearer token):
- /auth/me
- /recommend
- /recommend/batch
- /similar/{item_id}
- /explain/{user_id}/{item_id}
- /models
- /models/{model_id}/promote
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from backend.core.algorithms import ALGORITHM_REGISTRY
from backend.api.gateway.auth import create_access_token, require_auth, validate_credentials
from backend.config.settings import (
    ALGORITHM_MODES,
    DEFAULT_ALGORITHM_MODE,
    OPTUNA_TRIALS_AUTO,
    TOP_K,
    TOP_K_ALLOWED,
    TOP_MODEL_ALLOWED,
    TOP_N_MODELS,
)
from backend.core.models.model_registry import ModelRegistry
from backend.core.models.model_loader import ModelLoader
from backend.core.pipeline.serving_pipeline import ServingPipeline
from backend.core.pipeline.training_pipeline import TrainingConfig, TrainingPipeline
from backend.core.recommendation.recommender_engine import RecommenderEngine
from backend.core.recommendation.strategy_service import (
    RecommendationPayload,
    RecommendationStrategyError,
    RecommendationStrategyService,
    WeightedModelInput,
    RECOMMEND_STRATEGIES,
)
from backend.core.data_processing.interaction_matrix import InteractionMatrixBuilder
from backend.core.users.manager import (
    ADMIN_ROLE,
    USER_ROLE,
    UserAlreadyExistsError,
    UserManager,
    UserNotFoundError,
    UserTenantLimitError,
    UserValidationError,
)
from backend.api.routes.shared import (
    _runtime_state as tenant_runtime_state,
    _serving_cache as tenant_serving_cache,
    _strategy_service as tenant_strategy_service,
)
from backend.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter()
_executor = ThreadPoolExecutor(max_workers=4)

# In-memory runtime state
_jobs: dict[str, dict] = {}
_runtime_state: dict[str, object] = {
    "train_df": None,
    "last_result": None,
    "serving_im": None,
}
_serving_cache: dict[str, ServingPipeline] = {}
TENANT_MANAGE_SCOPE = "tenant:manage"
ADMIN_WRITE_SCOPE = "admin:write"


def _build_services(tenant_id: str):
    """Build per-request, tenant-scoped service objects."""
    registry = ModelRegistry(tenant_id=tenant_id)
    loader = ModelLoader()
    recommender = RecommenderEngine(
        loader,
        interaction_matrix=_runtime_state.get("serving_im"),
        tenant_id=tenant_id,
    )
    strategy = RecommendationStrategyService(registry, loader, recommender)
    return registry, loader, recommender, strategy


def _get_services(auth: dict = Depends(require_auth(scopes=[]))):
    tenant_id = auth.get("tenant_id") or "admin"
    return _build_services(tenant_id)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3)
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=8)
    confirm_password: str = Field(..., min_length=1)

    @field_validator("username")
    @classmethod
    def _validate_username(cls, value: str) -> str:
        return UserManager.normalize_username(value)

    @field_validator("email")
    @classmethod
    def _validate_email(cls, value: str) -> str:
        return UserManager.normalize_email(value)

    @model_validator(mode="after")
    def _validate_password_match(self):
        if self.password != self.confirm_password:
            raise ValueError("password and confirm_password must match")
        return self


def _payload_scopes(payload: dict) -> set[str]:
    return set(payload.get("scopes", []) or [])


def _is_platform_admin(payload: dict) -> bool:
    return ADMIN_WRITE_SCOPE in _payload_scopes(payload)


def _role_label(role: str) -> str:
    try:
        return UserManager.role_label(role)
    except UserValidationError:
        return role.title()


def _tenant_session_scopes(tenant_id: str = "") -> list[str]:
    scopes = {TENANT_MANAGE_SCOPE}
    if tenant_id:
        try:
            cfg = TenantManager.get(tenant_id)
        except (TenantNotFoundError, TenantValidationError, KeyError):
            cfg = None
        if cfg is not None:
            scopes.update(scope for scope in cfg.scopes if not scope.startswith("admin:"))
    return sorted(scopes)


def _require_bearer_user(payload: dict = Depends(require_auth(scopes=[]))) -> dict:
    if payload.get("auth_type") == "api_key":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin tenant management requires a bearer user session",
        )
    if not payload.get("sub"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user session")
    return payload


def _user_token_response(user, tenant_id: Optional[str] = None) -> dict:
    resolved_tenant_id = tenant_id if tenant_id is not None else user.tenant_id
    scopes = _tenant_session_scopes(resolved_tenant_id)
    token = create_access_token(
        subject=user.username,
        scopes=scopes,
        tenant_id=resolved_tenant_id or None,
        tier="user",
        role=user.role,
        email=user.email or None,
    )
    
    api_key = None
    if resolved_tenant_id:
        try:
            cfg = TenantManager.get(resolved_tenant_id)
            api_key = cfg.api_key
            if not api_key:
                log.warning("User login success but tenant '%s' has no raw API key stored.", resolved_tenant_id)
        except Exception as exc:
            log.error("Failed to fetch tenant '%s' for token response: %s", resolved_tenant_id, exc)

    log.info("Token response generated | user=%s tenant=%s has_api_key=%s", user.username, resolved_tenant_id, bool(api_key))

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user.username,
        "email": user.email or None,
        "role": user.role,
        "role_label": _role_label(user.role),
        "tenant_id": resolved_tenant_id or None,
        "api_key": api_key,
        "scopes": scopes,
        "requires_tenant": not bool(resolved_tenant_id),
    }


def _admin_token_response(username: str, tenant_id: Optional[str] = None) -> dict:
    scopes = [ADMIN_WRITE_SCOPE]
    token = create_access_token(
        subject=username,
        scopes=scopes,
        tenant_id=tenant_id or None,
        tier="admin",
        role=ADMIN_ROLE,
    )

    api_key = None
    if tenant_id:
        try:
            cfg = TenantManager.get(tenant_id)
            api_key = cfg.api_key
        except Exception:
            pass

    log.info("Admin token response generated | user=%s tenant=%s has_api_key=%s", username, tenant_id, bool(api_key))

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": username,
        "email": None,
        "role": ADMIN_ROLE,
        "role_label": _role_label(ADMIN_ROLE),
        "tenant_id": tenant_id or None,
        "api_key": api_key,
        "scopes": scopes,
        "requires_tenant": False,
    }


def _tenant_summary(cfg) -> dict:
    return {
        "tenant_id": cfg.tenant_id,
        "display_name": cfg.display_name,
        "tier": cfg.tier,
        "scopes": cfg.scopes,
        "rate_limit_rpm": cfg.rate_limit_rpm,
        "active": cfg.active,
        "algorithm_mode": cfg.algorithm_mode,
        "created_at": cfg.created_at,
        "updated_at": cfg.updated_at,
        "notes": cfg.notes,
        "owner_subject": cfg.owner_subject,
    }


def _current_user_tenant(payload: dict):
    if _is_platform_admin(payload):
        return None
    subject = str(payload.get("sub") or "").strip()
    if not subject:
        return None
    tenant = TenantManager.get_by_owner(subject)
    if tenant is not None:
        return tenant
    try:
        user = UserManager.get(subject)
    except (UserNotFoundError, UserValidationError):
        return None
    if not user.tenant_id:
        return None
    try:
        return TenantManager.get(user.tenant_id)
    except (TenantNotFoundError, TenantValidationError, KeyError):
        return None


def _require_registered_user(payload: dict) -> str:
    subject = str(payload.get("sub") or "").strip()
    if _is_platform_admin(payload):
        return subject
    try:
        UserManager.get(subject)
    except (UserNotFoundError, UserValidationError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Register before creating a tenant",
        )
    return subject


def _get_authorized_tenant(tenant_id: str, payload: dict):
    try:
        cfg = TenantManager.get(tenant_id)
    except (TenantNotFoundError, KeyError):
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found.")

    if _is_platform_admin(payload):
        return cfg

    subject = str(payload.get("sub") or "").strip()
    if cfg.owner_subject == subject:
        return cfg

    try:
        user = UserManager.get(subject)
    except (UserNotFoundError, UserValidationError):
        user = None
    if user and user.tenant_id == cfg.tenant_id:
        return cfg

    raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found.")


class WeightedModelRequest(BaseModel):
    algorithm: str
    weight: float
    model_id: Optional[str] = None

    @field_validator("algorithm")
    @classmethod
    def _validate_algorithm(cls, value: str) -> str:
        v = (value or "").strip()
        if not v:
            raise ValueError("algorithm is required")
        return v

    @field_validator("weight")
    @classmethod
    def _validate_weight(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("weight must be positive")
        return value


class RecommendRequest(BaseModel):
    user_id: str
    top_n: int = 10
    strategy: Optional[str] = None
    model_id: Optional[str] = None
    algorithm: Optional[str] = None
    models: list[WeightedModelRequest] = Field(default_factory=list)
    auto_normalize_weights: bool = True

    @field_validator("top_n")
    @classmethod
    def _validate_top_n(cls, value: int) -> int:
        if value not in TOP_K_ALLOWED:
            raise ValueError(f"top_n must be one of {TOP_K_ALLOWED}")
        return value

    @field_validator("user_id")
    @classmethod
    def _validate_user_id(cls, value: str) -> str:
        v = (value or "").strip()
        if not v:
            raise ValueError("user_id is required")
        return v

    @field_validator("strategy")
    @classmethod
    def _validate_strategy(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        v = value.strip().lower()
        if v not in RECOMMEND_STRATEGIES:
            raise ValueError(f"strategy must be one of {sorted(RECOMMEND_STRATEGIES)}")
        return v


class BatchRecommendRequest(BaseModel):
    user_ids: list[str] = Field(default_factory=list)
    top_n: int = 10
    strategy: Optional[str] = None
    model_id: Optional[str] = None
    algorithm: Optional[str] = None
    models: list[WeightedModelRequest] = Field(default_factory=list)
    auto_normalize_weights: bool = True

    @field_validator("top_n")
    @classmethod
    def _validate_top_n(cls, value: int) -> int:
        if value not in TOP_K_ALLOWED:
            raise ValueError(f"top_n must be one of {TOP_K_ALLOWED}")
        return value

    @field_validator("user_ids")
    @classmethod
    def _validate_user_ids(cls, value: list[str]) -> list[str]:
        if not value:
            return value
        cleaned = [v.strip() for v in value if v and v.strip()]
        if len(cleaned) != len(value):
            raise ValueError("user_ids must not contain empty values")
        return cleaned

    @field_validator("strategy")
    @classmethod
    def _validate_strategy(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        v = value.strip().lower()
        if v not in RECOMMEND_STRATEGIES:
            raise ValueError(f"strategy must be one of {sorted(RECOMMEND_STRATEGIES)}")
        return v


class SQLTrainRequest(BaseModel):
    dialect: str
    host: str
    port: int
    database: str
    username: str
    password: str
    sql: str
    top_k: int = TOP_K
    n_trials: int = OPTUNA_TRIALS_AUTO
    top_models: int = TOP_N_MODELS
    algorithm_mode: str = DEFAULT_ALGORITHM_MODE

    @field_validator("top_k")
    @classmethod
    def _validate_top_k(cls, value: int) -> int:
        if value not in TOP_K_ALLOWED:
            raise ValueError(f"top_k must be one of {TOP_K_ALLOWED}")
        return value

    @field_validator("top_models")
    @classmethod
    def _validate_top_models(cls, value: int) -> int:
        if value not in TOP_MODEL_ALLOWED:
            raise ValueError(f"top_models must be one of {TOP_MODEL_ALLOWED}")
        return value

    @field_validator("algorithm_mode")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        v = (value or "").lower().strip()
        if v not in ALGORITHM_MODES:
            raise ValueError(f"algorithm_mode must be one of {ALGORITHM_MODES}")
        return v


def _submit_job(fn, *args, **kwargs) -> str:
    job_id = str(uuid.uuid4())[:12]
    _jobs[job_id] = {
        "status": "pending",
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "result": None,
        "error": None,
    }

    def _run():
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started_at"] = time.time()
        try:
            result = fn(*args, **kwargs)
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = result
            _jobs[job_id]["finished_at"] = time.time()
        except Exception as e:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)
            _jobs[job_id]["finished_at"] = time.time()
            log.error("Job %s failed: %s", job_id, e, exc_info=True)

    _executor.submit(_run)
    return job_id


@router.get("/health", tags=["System"])
def health():
    """Expose a lightweight admin health check with a current timestamp."""
    return {"status": "ok", "timestamp": time.time()}


@router.get("/session/status", tags=["System"])
def session_status(
    tenant_id: Optional[str] = Query(
        None,
        description="Optional tenant to inspect. When omitted, defaults to 'admin'.",
    ),
):
    """Return the current tenant session, dataset, cache, and recommendation status."""
    resolved_tenant = (tenant_id or "admin").strip() or "admin"
    runtime = tenant_runtime_state.get(
        resolved_tenant,
        {"train_df": None, "last_result": None, "serving_im": None},
    )
    train_df = runtime.get("train_df")
    last_result = runtime.get("last_result")
    strategy_service = tenant_strategy_service(resolved_tenant)
    options = strategy_service.recommendation_options(last_result=last_result)

    promoted = ModelRegistry(tenant_id=resolved_tenant).list_models(promoted_only=True)
    latest_promoted = promoted[0] if promoted else None

    return {
        "tenant_id": resolved_tenant,
        "dataset_loaded": train_df is not None or last_result is not None,
        "last_best_algorithm": (
            getattr(last_result, "best_algorithm", None)
            if last_result
            else (latest_promoted.algorithm if latest_promoted else None)
        ),
        "last_best_model_id": (
            getattr(last_result, "best_model_id", None)
            if last_result
            else (latest_promoted.model_id if latest_promoted else None)
        ),
        "last_mode": getattr(last_result, "resolved_mode", None) if last_result else None,
        "cached_serving_algorithms": list(tenant_serving_cache.get(resolved_tenant, {}).keys()),
        "recommendation_options": options,
    }


@router.get("/algorithms", tags=["System"])
def list_algorithms():
    """List registered recommendation algorithms and their product metadata."""
    return {
        "algorithms": [
            {
                "name": name,
                "feedback": meta.get("feedback"),
                "domain": meta.get("domain"),
                "description": meta.get("description"),
                "training_speed": meta.get("training_speed"),
                "scalability": meta.get("scalability"),
                "robustness": meta.get("robustness"),
                "interpretability": meta.get("interpretability"),
                "production_readiness": meta.get("production_readiness"),
            }
            for name, meta in ALGORITHM_REGISTRY.items()
        ]
    }


@router.post("/auth/login", tags=["Auth"])
def login(req: LoginRequest):
    """Authenticate a registered user or legacy admin and mint a bearer token."""
    user = UserManager.validate_credentials(req.username, req.password)
    if user is not None:
        if user.tenant_id:
            try:
                TenantManager.activate_tenant(user.tenant_id, user.username)
            except Exception as exc:
                log.warning(
                    "Automatic tenant activation failed during login | user=%s tenant=%s error=%s",
                    user.username,
                    user.tenant_id,
                    exc,
                )
        return _user_token_response(user)

    if validate_credentials(req.username, req.password):
        return _admin_token_response(req.username.strip())

    raise HTTPException(status_code=401, detail="Invalid credentials")


@router.get("/auth/register-meta", tags=["Auth"])
def register_meta():
    """Return role metadata for the public registration form."""
    return {
        "roles": UserManager.list_roles(),
        "assignable_on_registration": [USER_ROLE],
        "default_role": USER_ROLE,
    }


@router.post("/auth/register", tags=["Auth"])
def register(req: RegisterRequest):
    """Create a user account before tenant provisioning."""
    try:
        user = UserManager.register(req.username, req.email, req.password)
    except UserAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except UserValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _user_token_response(user)


@router.get("/auth/me", tags=["Auth"])
def whoami(payload: dict = Depends(_require_bearer_user)):
    """Return the authenticated subject and token expiry details."""
    subject = str(payload.get("sub") or "").strip()
    tenant = _current_user_tenant(payload)
    user = None
    if not _is_platform_admin(payload):
        try:
            user = UserManager.get(subject)
        except (UserNotFoundError, UserValidationError):
            user = None
    role = ADMIN_ROLE if _is_platform_admin(payload) else (user.role if user else payload.get("role") or USER_ROLE)
    return {
        "user": subject,
        "email": user.email if user else payload.get("email"),
        "exp": payload.get("exp"),
        "scopes": payload.get("scopes", []),
        "role": role,
        "role_label": _role_label(role),
        "tenant_id": tenant.tenant_id if tenant else None,
        "requires_tenant": not _is_platform_admin(payload) and tenant is None,
        "is_platform_admin": _is_platform_admin(payload),
    }


@router.get("/auth/roles", tags=["Auth"])
def list_roles(payload: dict = Depends(_require_bearer_user)):
    """Return the backend role catalog used for dynamic user-role display."""
    return {
        "roles": UserManager.list_roles(),
        "assignable_on_registration": [USER_ROLE],
        "current_role": ADMIN_ROLE if _is_platform_admin(payload) else payload.get("role", USER_ROLE),
    }
# ---------------------------------------------------------------------------
# Webhook registration endpoint
# ---------------------------------------------------------------------------

from backend.core.tenants.manager import (
    TenantManager,
    TenantNotFoundError,
    TenantAlreadyExistsError,
    TenantValidationError,
    TIERS,
    ALL_SCOPES,
)
from backend.sdk.webhooks.push import verify_signature as _verify_sig  # noqa: F401 — re-exported for docs


class _WebhookConfig(BaseModel):
    webhook_url: str = Field(..., description="HTTPS URL to deliver webhook events to.")
    webhook_secret: str = Field(..., min_length=16, description="Shared secret (min 16 chars) used to sign payloads.")


@router.patch("/tenants/{tenant_id}/webhook", tags=["Admin"])
async def set_tenant_webhook(
    tenant_id: str,
    body: _WebhookConfig,
    payload: dict = Depends(_require_bearer_user),
):
    """
    Register or update the webhook URL and secret for a tenant.

    After this is set, the platform will fire signed POST requests to
    ``webhook_url`` for the following events:

    - ``job.completed`` — training job finished successfully
    - ``job.failed``    — training job terminated with an error
    - ``model.promoted`` — a model was promoted to production

    Payloads are signed with HMAC-SHA256. Verify using::

        X-ProactiveAI-Signature: sha256=<hex_digest>

    Set ``webhook_url`` and ``webhook_secret`` to empty strings to clear
    the webhook registration.
    """
    cfg = _get_authorized_tenant(tenant_id, payload)

    cfg.webhook_url = body.webhook_url.strip() or None
    cfg.webhook_secret = body.webhook_secret.strip() or None
    TenantManager._write_config(tenant_id, cfg)
    TenantManager._store_in_cache(cfg)

    return {
        "status": "updated",
        "tenant_id": tenant_id,
        "webhook_url": cfg.webhook_url,
        "webhook_secret_set": cfg.webhook_secret is not None,
    }

# ---------------------------------------------------------------------------
# Tenant management endpoints
# ---------------------------------------------------------------------------

class _TenantCreateRequest(BaseModel):
    tenant_id: str = Field(..., description="Unique tenant identifier. Lowercase alphanumeric and hyphens, 3-63 chars.")
    display_name: str = Field("", description="Human-readable label for this tenant.")
    tier: str = Field("starter", description="Pricing tier: free | starter | pro | enterprise")
    scopes: Optional[list[str]] = Field(None, description="Permission scopes. Defaults to all standard scopes.")
    algorithm_mode: str = Field("auto", description="Default algorithm mode: auto | explicit | implicit | hybrid")
    rate_limit_rpm: int = Field(60, ge=1, description="Max API requests per minute for this tenant.")
    notes: str = Field("", description="Optional internal notes about this tenant.")


class _TenantUpdateRequest(BaseModel):
    display_name: Optional[str] = Field(None, description="Update the display name.")
    tier: Optional[str] = Field(None, description="Change the tier: free | starter | pro | enterprise")
    scopes: Optional[list[str]] = Field(None, description="Replace the full scopes list.")
    algorithm_mode: Optional[str] = Field(None, description="Change the default algorithm mode.")
    rate_limit_rpm: Optional[int] = Field(None, ge=1, description="Change the requests-per-minute limit.")
    active: Optional[bool] = Field(None, description="Set to false to deactivate without deleting.")
    notes: Optional[str] = Field(None, description="Update internal notes.")


@router.post("/tenants", tags=["Tenant Management"])
def create_tenant(
    body: _TenantCreateRequest,
    payload: dict = Depends(_require_bearer_user),
):
    """
    Create a new tenant and return its API key.

    **The raw API key is only returned once here — store it immediately.**
    It is never shown again; the system only keeps its SHA-256 hash.

    Valid `tier` values: `free`, `starter`, `pro`, `enterprise`.

    Valid `scopes`: `training:write`, `recommend:read`, `models:read`,
    `models:write`, `explain:read`, `batch:read`, `similar:read`, `jobs:read`.
    """
    subject = _require_registered_user(payload)
    owner_subject = None if _is_platform_admin(payload) else subject
    existing_tenant = _current_user_tenant(payload)
    if existing_tenant is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "This user already has a tenant",
                "tenant": _tenant_summary(existing_tenant),
            },
        )

    try:
        cfg = TenantManager.create(
            tenant_id=body.tenant_id,
            display_name=body.display_name or body.tenant_id,
            tier=body.tier,
            scopes=body.scopes,
            algorithm_mode=body.algorithm_mode,
            rate_limit_rpm=body.rate_limit_rpm,
            notes=body.notes,
            owner_subject=owner_subject,
        )
        raw_key = cfg.api_key
        if owner_subject:
            UserManager.link_tenant(owner_subject, cfg.tenant_id)
            session = _user_token_response(UserManager.get(owner_subject), tenant_id=cfg.tenant_id)
        else:
            session = _admin_token_response(subject, tenant_id=cfg.tenant_id)
    except TenantAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except TenantValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except UserTenantLimitError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except (UserNotFoundError, UserValidationError) as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    return {
        "tenant_id": cfg.tenant_id,
        "display_name": cfg.display_name,
        "tier": cfg.tier,
        "scopes": cfg.scopes,
        "rate_limit_rpm": cfg.rate_limit_rpm,
        "active": cfg.active,
        "created_at": cfg.created_at,
        "owner_subject": cfg.owner_subject,
        "api_key": raw_key,
        "access_token": session["access_token"],
        "token_type": session["token_type"],
        "role": session["role"],
        "role_label": session["role_label"],
        "requires_tenant": False,
        "warning": "Save this api_key now — it will never be shown again.",
    }


@router.get("/tenants", tags=["Tenant Management"])
def list_tenants(
    active_only: bool = Query(False, description="If true, return only active tenants."),
    tier: Optional[str] = Query(None, description="Filter by tier."),
    payload: dict = Depends(_require_bearer_user),
):
    """
    List all tenants. API keys are never returned here for security.
    """
    owner_subject = None if _is_platform_admin(payload) else str(payload.get("sub") or "").strip()
    tenants = TenantManager.list_tenants(
        active_only=active_only,
        tier=tier,
        owner_subject=owner_subject or None,
    )
    return {
        "total": len(tenants),
        "tenants": [_tenant_summary(t) for t in tenants],
    }


@router.get("/tenants/{tenant_id}", tags=["Tenant Management"])
def get_tenant(
    tenant_id: str,
    payload: dict = Depends(_require_bearer_user),
):
    """
    Get configuration for a single tenant.
    """
    cfg = _get_authorized_tenant(tenant_id, payload)

    return {
        "tenant_id": cfg.tenant_id,
        "display_name": cfg.display_name,
        "tier": cfg.tier,
        "scopes": cfg.scopes,
        "rate_limit_rpm": cfg.rate_limit_rpm,
        "active": cfg.active,
        "algorithm_mode": cfg.algorithm_mode,
        "top_k": cfg.top_k,
        "max_recs": cfg.max_recs,
        "created_at": cfg.created_at,
        "updated_at": cfg.updated_at,
        "notes": cfg.notes,
        "webhook_url": cfg.webhook_url,
        "owner_subject": cfg.owner_subject,
        "api_key": cfg.api_key,
    }


@router.post("/tenants/{tenant_id}/activate", tags=["Tenant Management"])
def activate_tenant(
    tenant_id: str,
    payload: dict = Depends(_require_bearer_user),
):
    """
    Set a tenant as active for the bearer session and return a refreshed token.
    """
    cfg = _get_authorized_tenant(tenant_id, payload)
    if not cfg.active:
        raise HTTPException(status_code=409, detail=f"Tenant '{tenant_id}' is inactive.")

    subject = str(payload.get("sub") or "").strip()
    if _is_platform_admin(payload):
        session = _admin_token_response(subject, tenant_id=cfg.tenant_id)
    else:
        try:
            user = UserManager.get(subject)
            if user.tenant_id and user.tenant_id != cfg.tenant_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="This user is not allowed to activate that tenant",
                )
            if not user.tenant_id:
                user = UserManager.link_tenant(subject, cfg.tenant_id)
        except (UserNotFoundError, UserValidationError) as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        session = _user_token_response(user, tenant_id=cfg.tenant_id)

    return {
        "status": "activated",
        "tenant": _tenant_summary(cfg),
        **session,
    }


@router.patch("/tenants/{tenant_id}", tags=["Tenant Management"])
def update_tenant(
    tenant_id: str,
    body: _TenantUpdateRequest,
    payload: dict = Depends(_require_bearer_user),
):
    """
    Update one or more fields on an existing tenant.
    Only the fields you send are changed — omitted fields stay as-is.
    """
    _get_authorized_tenant(tenant_id, payload)
    try:
        cfg = TenantManager.update(
            tenant_id,
            display_name=body.display_name,
            tier=body.tier,
            scopes=body.scopes,
            algorithm_mode=body.algorithm_mode,
            rate_limit_rpm=body.rate_limit_rpm,
            active=body.active,
            notes=body.notes,
        )
    except (TenantNotFoundError, KeyError):
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found.")
    except TenantValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {
        "status": "updated",
        "tenant_id": cfg.tenant_id,
        "display_name": cfg.display_name,
        "tier": cfg.tier,
        "scopes": cfg.scopes,
        "rate_limit_rpm": cfg.rate_limit_rpm,
        "active": cfg.active,
        "updated_at": cfg.updated_at,
    }


@router.delete("/tenants/{tenant_id}", tags=["Tenant Management"])
def delete_tenant(
    tenant_id: str,
    delete_data: bool = Query(
        False,
        description="If true, permanently deletes the tenant's data and model folders. "
                    "If false (default), only removes the config — data is preserved on disk.",
    ),
    payload: dict = Depends(_require_bearer_user),
):
    """
    Delete a tenant.

    - `delete_data=false` (default) — removes only the config file. The tenant's
      data and trained models remain on disk and can be restored by recreating
      the tenant with the same `tenant_id`.
    - `delete_data=true` — **permanently** removes the tenant directory, all
      training data, and all trained models. This is irreversible.
    """
    cfg = _get_authorized_tenant(tenant_id, payload)
    try:
        TenantManager.delete(tenant_id, delete_data=delete_data)
    except (TenantNotFoundError, KeyError):
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found.")

    if not _is_platform_admin(payload) and cfg.owner_subject:
        UserManager.unlink_tenant(cfg.owner_subject, tenant_id)

    return {
        "status": "deleted",
        "tenant_id": tenant_id,
        "data_deleted": delete_data,
    }


@router.post("/tenants/{tenant_id}/rotate-key", tags=["Tenant Management"])
def rotate_tenant_api_key(
    tenant_id: str,
    payload: dict = Depends(_require_bearer_user),
):
    """
    Rotate the API key for a tenant.

    The old key is **immediately invalidated** — any clients using it will get
    401 errors. Distribute the new key to your client before rotating.

    **The new raw key is only returned once — save it immediately.**
    """
    _get_authorized_tenant(tenant_id, payload)
    try:
        new_key = TenantManager.rotate_api_key(tenant_id)
    except (TenantNotFoundError, KeyError):
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found.")

    return {
        "status": "rotated",
        "tenant_id": tenant_id,
        "api_key": new_key,
        "warning": "Save this api_key now — the old key is already invalid and this will not be shown again.",
    }
