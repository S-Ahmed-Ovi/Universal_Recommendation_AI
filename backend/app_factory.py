"""
app_factory.py – Application factory (v3)

Changes from v2:
  - TenantMiddleware injected when USE_V3_MULTITENANCY=true.
  - Modular v3 route files registered when USE_V3_ROUTES=true,
    with v2 monolithic router kept as fallback.
  - Admin router mounted under /admin (guarded separately).
  - Metering middleware added when USE_V3_METERING=true.
  - CORS tightened: origins driven by CORS_ORIGINS env var.
  - All feature-flag branches are explicit so each layer can be
    toggled independently during the v2 → v3 migration.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import (
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config.settings import (
    USE_V3_AUTH,
    USE_V3_METERING,
    USE_V3_MULTITENANCY,
    USE_V3_ROUTES,
)
from backend.utils.logger import get_logger

try:
    from swagger_ui_bundle import swagger_ui_path
except ImportError:
    swagger_ui_path = None

log = get_logger(__name__)

BASE_DIR       = Path(__file__).parent
FRONTEND_DIR   = BASE_DIR.parent / "frontend"
FRONTEND_DIST  = FRONTEND_DIR / "dist"
STATIC_DIR     = BASE_DIR / "static"
SWAGGER_ASSETS = ("swagger-ui-bundle.js", "swagger-ui.css")
INLINE_FAVICON = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+X2ioAAAAASUVORK5CYII="
)

load_dotenv(BASE_DIR.parent / ".env")
load_dotenv(BASE_DIR / ".env", override=True)
STATIC_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Swagger asset helpers (unchanged from v2)
# ---------------------------------------------------------------------------

def _missing_swagger_assets() -> list[str]:
    return [n for n in SWAGGER_ASSETS if not (STATIC_DIR / n).exists()]


def _ensure_swagger_assets() -> list[str]:
    missing = _missing_swagger_assets()
    if not missing or swagger_ui_path is None:
        return missing
    bundle_dir = Path(swagger_ui_path)
    for name in missing:
        src, dst = bundle_dir / name, STATIC_DIR / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    return _missing_swagger_assets()


# ---------------------------------------------------------------------------
# Middleware registration
# ---------------------------------------------------------------------------

def _register_middleware(app: FastAPI) -> None:
    """Attach middleware in outermost-first order."""

    # ── CORS ─────────────────────────────────────────────────────────────────
    import os
    raw_origins = os.getenv("CORS_ORIGINS", "*")
    origins = [o.strip() for o in raw_origins.split(",")] if raw_origins != "*" else ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    log.debug("CORS middleware registered (origins=%s)", origins)

    # ── Tenant extraction (v3) ────────────────────────────────────────────────
    if USE_V3_MULTITENANCY:
        from backend.api.gateway.middleware import TenantMiddleware
        app.add_middleware(TenantMiddleware)
        log.info("TenantMiddleware registered")

    # ── Metering / rate-limiting (v3) ─────────────────────────────────────────
    if USE_V3_METERING:
        from backend.api.gateway.metering import MeteringMiddleware
        app.add_middleware(MeteringMiddleware)
        log.info("MeteringMiddleware registered")


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

def _register_routes(app: FastAPI) -> None:
    """Mount all API routers."""

    if USE_V3_ROUTES:
        # ── Modular v3 routes ─────────────────────────────────────────────────
        from backend.api.routes.algorithms import router as algorithms_router
        from backend.api.routes.analytics  import router as analytics_router
        from backend.api.routes.benchmarks import router as benchmarks_router
        from backend.api.routes.batch      import router as batch_router
        from backend.api.routes.context    import router as context_router
        from backend.api.routes.deployment import router as deployment_router
        from backend.api.routes.explain    import router as explain_router
        from backend.api.routes.jobs       import router as jobs_router
        from backend.api.routes.models     import router as models_router
        from backend.api.routes.recommend  import router as recommend_router
        from backend.api.routes.scheduler  import router as scheduler_router
        from backend.api.routes.similar    import router as similar_router
        from backend.api.routes.training   import router as training_router

        app.include_router(algorithms_router)
        app.include_router(context_router)
        app.include_router(deployment_router)
        app.include_router(analytics_router)
        app.include_router(benchmarks_router)
        app.include_router(training_router)
        app.include_router(recommend_router)
        app.include_router(batch_router)
        app.include_router(similar_router)
        app.include_router(explain_router)
        app.include_router(models_router)
        app.include_router(jobs_router)
        app.include_router(scheduler_router)
        log.info("Registered v3 modular routes")

    else:
        # ── v2 monolithic fallback ─────────────────────────────────────────────
        from backend.api.routes import router as legacy_router
        app.include_router(legacy_router)
        log.info("Registered v2 legacy router (USE_V3_ROUTES=false)")

    # ── smart_db_csv_builder ──────────────────────────────────────────────────
    try:
        from fastapi import Depends
        from backend.core.tenants.context import require_scope
        from backend.smart_db_csv_builder.api.routers import build, connections, jobs, schema

        _training_dep = [Depends(require_scope("training:write"))]

        app.include_router(
            connections.router,
            prefix="/smart-db-csv/api/connections",
            tags=["SmartDB – Connections"],
            dependencies=_training_dep,
        )
        app.include_router(
            schema.router,
            prefix="/smart-db-csv/api/schema",
            tags=["SmartDB – Schema"],
            dependencies=_training_dep,
        )
        app.include_router(
            build.router,
            prefix="/smart-db-csv/api/build",
            tags=["SmartDB – Build"],
            dependencies=_training_dep,
        )
        app.include_router(
            jobs.router,
            prefix="/smart-db-csv/api/jobs",
            tags=["SmartDB – Jobs"],
            dependencies=_training_dep,
        )
        log.info("Registered smart_db_csv_builder routes (auth: training:write)")
    except ImportError:
        log.warning("smart_db_csv_builder not found — skipping its routes")

    # ── Admin dashboard (v3) ──────────────────────────────────────────────────
    try:
        from backend.admin.routes import router as admin_router
        app.include_router(admin_router, prefix="/admin")
        log.info("Registered admin routes")
    except ImportError:
        log.warning("admin routes not found — skipping")


# ---------------------------------------------------------------------------
# Static / frontend mounts
# ---------------------------------------------------------------------------

def _register_static(app: FastAPI) -> None:
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    assets_dir = FRONTEND_DIST / "assets"
    if FRONTEND_DIST.exists() and assets_dir.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(FRONTEND_DIST / "assets")),
            name="assets",
        )


# ---------------------------------------------------------------------------
# Built-in endpoints
# ---------------------------------------------------------------------------

def _register_builtin_endpoints(app: FastAPI) -> None:
    """Attach health, docs, and frontend convenience endpoints to the app."""

    def _frontend_available() -> bool:
        """Return whether a compiled frontend bundle is available on disk."""
        return (FRONTEND_DIST / "index.html").exists()

    def _frontend_response():
        """Serve the frontend index or return a build-missing diagnostic payload."""
        index = FRONTEND_DIST / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return {
            "message": (
                "Frontend build missing. "
                "Run: cd frontend && npm install && npm run build"
            )
        }

    @app.get("/health", include_in_schema=False)
    def health():
        """Return the application health payload used by probes and smoke tests."""
        return {"status": "ok", "version": app.version}

    @app.get("/", include_in_schema=False)
    def api_root():
        """Return a compact root document with docs and frontend links."""
        return {
            "name": app.title,
            "version": app.version,
            "docs": "/docs",
            "health": "/health",
            "frontend": "/app" if _frontend_available() else None,
        }

    @app.get("/app", include_in_schema=False)
    def serve_frontend():
        """Serve the primary frontend application entrypoint."""
        return _frontend_response()

    @app.get("/smart-db-csv", include_in_schema=False)
    def serve_frontend_alias():
        """Serve the frontend through the legacy smart-db-csv alias route."""
        return _frontend_response()

    @app.get("/docs", include_in_schema=False)
    def custom_swagger_ui():
        """Serve Swagger UI using the vendored static assets bundled with the repo."""
        missing = _ensure_swagger_assets()
        if missing:
            raise HTTPException(
                status_code=500,
                detail="Missing Swagger UI assets: " + ", ".join(missing),
            )
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} – Swagger UI",
            swagger_js_url="/static/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui.css",
            swagger_favicon_url=INLINE_FAVICON,
            swagger_ui_parameters=app.swagger_ui_parameters,
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            init_oauth=app.swagger_ui_init_oauth,
        )

    if app.swagger_ui_oauth2_redirect_url:
        @app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
        def swagger_redirect():
            """Handle the OAuth redirect target used by Swagger UI."""
            return get_swagger_ui_oauth2_redirect_html()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Assemble and return the FastAPI application."""
    app = FastAPI(
        title="Proactive AI",
        description=(
            "Multi-tenant recommendation platform. "
            "Train models, serve recommendations, and manage tenants via a single API.\n\n"
            "## Authentication\n\n"
            "Two auth methods are supported:\n\n"
            "- **X-API-Key** *(tenant operations)*: Use the `api_key` returned when you create "
            "a tenant via `POST /admin/tenants`. Pass it in the `X-API-Key` header. "
            "Required for all training, recommendation, jobs, and model endpoints.\n\n"
            "- **HTTPBearer** *(admin operations)*: Use the JWT token returned by "
            "`POST /admin/auth/login`. Required for tenant management endpoints under `/admin/tenants`."
        ),
        version="3.0.0",
        docs_url=None,   # served manually for custom Swagger UI
    )

    _register_static(app)
    _register_middleware(app)
    _register_routes(app)
    _register_builtin_endpoints(app)

    # ------------------------------------------------------------------
    # Custom OpenAPI schema: inject X-API-Key security scheme so that
    # the Swagger UI "Authorize" dialog shows both auth options.
    # ------------------------------------------------------------------
    from fastapi.openapi.utils import get_openapi

    def _custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )

        # Ensure the components section exists
        schema.setdefault("components", {})
        schema["components"].setdefault("securitySchemes", {})

        # Add X-API-Key scheme (tenant operations)
        schema["components"]["securitySchemes"]["ApiKeyAuth"] = {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": (
                "Tenant API key — returned once when you call POST /admin/tenants. "
                "Use this for all training, recommendation, jobs, and model endpoints."
            ),
        }

        # HTTPBearer is auto-added by FastAPI; ensure it's present and described
        schema["components"]["securitySchemes"].setdefault("HTTPBearer", {
            "type": "http",
            "scheme": "bearer",
            "description": (
                "Admin JWT token — returned by POST /admin/auth/login. "
                "Use this for tenant management endpoints under /admin/tenants."
            ),
        })

        # Apply both schemes globally so every endpoint shows the lock icon.
        # Individual endpoints that need only one can override via security=[].
        schema["security"] = [
            {"ApiKeyAuth": []},
            {"HTTPBearer": []},
        ]

        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = _custom_openapi  # type: ignore[method-assign]

    log.info(
        "App created (auth=%s, multitenancy=%s, metering=%s, v3_routes=%s)",
        USE_V3_AUTH,
        USE_V3_MULTITENANCY,
        USE_V3_METERING,
        USE_V3_ROUTES,
    )
    return app
