# Proactive AI v2 → Enterprise v3 Restructuring Guide

## Overview

Your v2 codebase is **single-tenant, monolithic, and CLI/basic-API focused**. The target enterprise structure is **multi-tenant, modular, with clear separation of concerns**. This guide shows exactly which v2 files map to the new structure and what refactoring is needed.

---

## Current v2 Structure

```
proactive-ai-v2/
├── algorithms/              # Algorithms (keep, relocate to core/)
├── api/routes.py            # Basic FastAPI routes (refactor → api/gateway + routes/)
├── benchmark/               # Benchmark engine (keep, relocate to core/)
├── config/                  # Settings (keep, enhance with env management)
├── data_processing/         # Data cleaning (keep, relocate to core/)
├── ingestion/               # CSV/DB loaders (relocate to smart_db_csv_builder/)
├── insights/                # Explainability (keep, relocate to core/)
├── models/                  # Model registry (keep, but tenant-scoped)
├── optimization/            # Optuna tuning (keep, relocate to core/)
├── pipeline/                # Training pipeline (keep, relocate to core/)
├── recommendation/          # Serving (keep, relocate to core/)
├── ui/dashboard.html        # Single dashboard (split: user UI + admin dashboard)
├── utils/                   # Utilities (keep)
└── main.py                  # CLI entrypoint (refactor → main.py + app_factory.py)
```

---

## Target Enterprise Structure

```
proactive-ai-v3/
├── tenants/                    ← NEW: Per-tenant data isolation
│   └── {tenant_id}/
│       ├── data/
│       ├── models/
│       └── config.json
│
├── core/                       ← Shared AI engine (relocated from v2)
│   ├── algorithms/
│   ├── benchmark/
│   ├── data_processing/
│   ├── optimization/
│   ├── pipeline/
│   ├── recommendation/
│   └── insights/
│
├── api/                        ← Gateway + routes (refactored from v2/api/)
│   ├── gateway/
│   │   ├── auth.py            ← NEW: JWT/API key validation
│   │   ├── middleware.py       ← NEW: Tenant extraction, routing
│   │   └── metering.py         ← NEW: Usage tracking, rate limits
│   └── routes/
│       ├── recommend.py        ← NEW: Split from v2/api/routes.py
│       ├── similar.py          ← NEW: Split from v2/api/routes.py
│       └── batch.py            ← NEW: Split from v2/api/routes.py
│
├── smart_db_csv_builder/       ← NEW: Refactored from v2/ingestion/
│   ├── connectors/
│   ├── services/
│   └── models/
│
├── admin/                      ← NEW: Admin-only dashboard & routes
│   ├── dashboard.html
│   └── routes.py
│
├── sdk/                        ← NEW: Client integration
│   ├── js_widget/
│   └── webhooks/
│
├── config/                     ← Enhanced (from v2/config/)
├── utils/                      ← Unchanged (from v2/utils/)
├── tests/                      ← NEW: Test suite
├── main.py                     ← Refactored entrypoint
├── app_factory.py              ← NEW: Tenant middleware + route registration
├── requirements.txt
└── docker-compose.yml
```

---

## File-by-File Migration Map

### Phase 1: Core AI Engine (No Breaking Changes)

These files move to `core/` with **minimal logic changes**:

| v2 Location | v3 Location | Changes | Priority |
|---|---|---|---|
| `algorithms/explicit_models/*.py` | `core/algorithms/explicit_models/` | Move as-is | High |
| `algorithms/implicit_models/*.py` | `core/algorithms/implicit_models/` | Move as-is | High |
| `algorithms/domain_models/*.py` | `core/algorithms/domain_models/` | Move as-is | High |
| `algorithms/__init__.py` | `core/algorithms/__init__.py` | Move as-is | High |
| `benchmark/benchmark_engine.py` | `core/benchmark/benchmark_engine.py` | Move as-is | High |
| `data_processing/*.py` | `core/data_processing/` | Move as-is | High |
| `optimization/optuna_tuner.py` | `core/optimization/optuna_tuner.py` | Move as-is | High |
| `pipeline/training_pipeline.py` | `core/pipeline/training_pipeline.py` | Add tenant_id param | High |
| `insights/explainability.py` | `core/insights/explainability.py` | Move as-is | Medium |
| `insights/model_explainer.py` | `core/insights/model_explainer.py` | NEW file (from review) | Medium |
| `recommendation/recommender_engine.py` | `core/recommendation/recommender_engine.py` | Move as-is | Medium |

---

### Phase 2: Data Ingestion (Refactor to `smart_db_csv_builder/`)

These files are refactored and reorganized:

| v2 Location | v3 Location | Changes | Priority |
|---|---|---|---|
| `ingestion/csv_loader.py` | `smart_db_csv_builder/connectors/csv.py` | Tenant-scope the loader | High |
| `ingestion/db_connector.py` | `smart_db_csv_builder/connectors/database.py` | Extract DB type to separate files | High |
| `ingestion/sql_query_executor.py` | `smart_db_csv_builder/services/executor.py` | Add tenant context | High |
| `ingestion/nosql_connector.py` | `smart_db_csv_builder/connectors/nosql.py` | Move as-is | Medium |
| *(new)* | `smart_db_csv_builder/services/builder.py` | Extract connection building logic | High |
| *(new)* | `smart_db_csv_builder/services/llm_planner.py` | For smart query generation (optional) | Low |

---

### Phase 3: API Gateway & Routes (Major Refactor)

**`v2/api/routes.py`** is split into multiple route files with auth middleware:

| v2 Code | v3 Location | Changes | Priority |
|---|---|---|---|
| `/health` → Stays in `main.py` | `main.py` | Move to root | High |
| `/algorithms` → Move to `api/routes/algorithms.py` (new) | `api/routes/algorithms.py` | Add tenant auth gate | High |
| `/train/file` | `api/routes/training.py` (new) | Add auth, tenant_id param, algorithm_mode | High |
| `/train/sql` | `api/routes/training.py` | Add auth, tenant_id param | High |
| `/recommend` | `api/routes/recommend.py` | Wire up (currently 503), add auth | High |
| `/recommend/batch` | `api/routes/batch.py` | Wire up, split from recommend.py, add auth | High |
| `/similar/{item_id}` | `api/routes/similar.py` | Create new, add auth | High |
| `/explain/{user}/{item}` | `api/routes/explain.py` (new) | Create new, add auth | Medium |
| `/models` | `api/routes/models.py` (new) | Create new, tenant-scoped, add auth | High |
| `/models/{id}/promote` | `api/routes/models.py` | Tenant-scoped promotion, add auth | High |
| `/jobs/{job_id}` | `api/routes/jobs.py` (new) | Async job tracking, tenant-scoped | Medium |

**New Gateway Files:**
- `api/gateway/auth.py` — JWT + API key validation, role scopes
- `api/gateway/middleware.py` — Extract tenant_id from request, validate, attach to context
- `api/gateway/metering.py` — Track usage, enforce rate limits per tenant

---

### Phase 4: Configuration & Settings

| v2 Location | v3 Location | Changes | Priority |
|---|---|---|---|
| `config/settings.py` | `config/settings.py` | Add feature flags, tenant config | High |
| `config/database.py` | `config/database.py` | Add multi-tenant DB pool logic | High |

---

### Phase 5: UI & Admin

| v2 Location | v3 Location | Changes | Priority |
|---|---|---|---|
| `ui/dashboard.html` | **Split into two** | | High |
| | `ui/user_dashboard.html` | Client-facing recommendation UI | High |
| | `admin/dashboard.html` | Admin model management, A/B controls, analytics | High |
| *(new)* | `admin/routes.py` | Admin endpoints (list tenants, view models, A/B test) | Medium |

---

### Phase 6: Entrypoints & App Factory

| v2 Location | v3 Location | Changes | Priority |
|---|---|---|---|
| `main.py` | `main.py` | Refactor: remove CLI, focus on FastAPI server | High |
| *(new)* | `app_factory.py` | Register tenant middleware, routes, error handlers | High |

---

### Phase 7: New Tenant Infrastructure

| v3 Location | Purpose | Priority |
|---|---|---|
| `tenants/{tenant_id}/config.json` | API key, tier, scopes, settings | High |
| `tenants/{tenant_id}/data/` | Tenant data files, interaction matrices | High |
| `tenants/{tenant_id}/models/` | Tenant model registry | High |
| *(new module)* `core/tenants/manager.py` | Tenant CRUD, isolation enforcement | High |
| *(new module)* `core/tenants/context.py` | Tenant context stack (request-scoped) | High |

---

## Step-by-Step Refactoring Plan

### **Step 1: Create Core Directory Structure**

```bash
mkdir -p core/{algorithms/{explicit_models,implicit_models,domain_models},benchmark,data_processing,optimization,pipeline,recommendation,insights}
mkdir -p api/{gateway,routes}
mkdir -p smart_db_csv_builder/{connectors,services,models}
mkdir -p admin
mkdir -p sdk/{js_widget,webhooks}
mkdir -p tests
mkdir -p tenants
```

### **Step 2: Move Core AI Files (No Logic Changes)**

```bash
# Algorithms
cp -r algorithms/* core/algorithms/

# Benchmark, data processing, optimization, pipeline, insights, recommendation
cp -r benchmark/* core/benchmark/
cp -r data_processing/* core/data_processing/
cp -r optimization/* core/optimization/
cp -r pipeline/* core/pipeline/
cp -r insights/* core/insights/
cp -r recommendation/* core/recommendation/
```

### **Step 3: Refactor Ingestion → `smart_db_csv_builder/`**

**File: `ingestion/csv_loader.py` → `smart_db_csv_builder/connectors/csv.py`**

```python
# OLD (v2)
class FileLoader:
    def load(self, path: str, file_format: str = "auto") -> pd.DataFrame:
        ...

# NEW (v3)
class CSVConnector:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.data_dir = f"tenants/{tenant_id}/data"
    
    def load(self, path: str, file_format: str = "auto") -> pd.DataFrame:
        # Validate path is within tenant's data dir
        safe_path = self._validate_tenant_path(path)
        ...
```

**File: `ingestion/db_connector.py` → `smart_db_csv_builder/connectors/database.py`**

```python
# Split by DB type:
# - smart_db_csv_builder/connectors/postgres.py
# - smart_db_csv_builder/connectors/mysql.py
# - smart_db_csv_builder/connectors/sqlite.py

class PostgreSConnector:
    def __init__(self, tenant_id: str, config: dict):
        self.tenant_id = tenant_id
        self.config = config
```

**File: `ingestion/sql_query_executor.py` → `smart_db_csv_builder/services/executor.py`**

```python
class SQLQueryExecutor:
    def __init__(self, tenant_id: str, sql_config: SQLConfig):
        self.tenant_id = tenant_id
        self.config = sql_config
    
    def execute_custom(self, sql: str) -> pd.DataFrame:
        # Log to tenant's audit trail
        self._log_query(sql)
        ...
```

---

### **Step 4: Refactor API Routes**

**Delete: `api/routes.py`** and create modular route files:

**`api/routes/training.py`**
```python
from api.gateway.auth import require_auth
from core.pipeline.training_pipeline import TrainingPipeline, TrainingConfig

@router.post("/train/file")
@require_auth(scope="training:write")
async def train_from_file(
    request: Request,
    file: UploadFile,
    algorithm_mode: str = Form("auto"),
    top_k: int = Form(10),
    n_trials: int = Form(-1),
):
    tenant_id = request.state.tenant_id  # From middleware
    user_id = request.state.user_id
    
    # Log for metering
    await metering.record_usage(tenant_id, "training", 1)
    
    # Validate top_k
    if top_k not in (5, 10):
        top_k = 10
    
    config = TrainingConfig(
        top_k=top_k,
        n_tuning_trials=n_trials,
        algorithm_mode=algorithm_mode,
    )
    pipeline = TrainingPipeline(config, tenant_id=tenant_id)
    ...
```

**`api/routes/recommend.py`** (was 503, now implemented)
```python
from api.gateway.auth import require_auth
from core.recommendation.recommender_engine import RecommenderEngine

@router.post("/recommend")
@require_auth(scope="recommendation:read")
async def get_recommendations(
    request: Request,
    user_id: str = Form(...),
    top_n: int = Form(10),
):
    tenant_id = request.state.tenant_id
    model_id = request.state.active_model_id  # From tenant config
    
    engine = RecommenderEngine(tenant_id=tenant_id, model_id=model_id)
    recs = engine.recommend(user_id, top_n)
    
    return {
        "user_id": user_id,
        "recommendations": recs,
        "model_id": model_id,
    }
```

**`api/routes/batch.py`** (Split from `/recommend/batch`)
```python
@router.post("/recommend/batch")
@require_auth(scope="recommendation:read")
async def batch_recommend(request: Request, req: BatchRecommendRequest):
    tenant_id = request.state.tenant_id
    # Parallelize recommendations
    results = await parallel_recommend(tenant_id, req.user_ids)
    return {"recommendations": results}
```

**`api/routes/models.py`** (New, tenant-scoped model management)
```python
@router.get("/models")
@require_auth(scope="models:read")
async def list_models(request: Request):
    tenant_id = request.state.tenant_id
    registry = ModelRegistry(tenant_id=tenant_id)
    return {"models": registry.list_models()}

@router.post("/models/{model_id}/promote")
@require_auth(scope="models:write")
async def promote_model(request: Request, model_id: str):
    tenant_id = request.state.tenant_id
    registry = ModelRegistry(tenant_id=tenant_id)
    registry.promote(model_id)
    return {"status": "promoted"}
```

---

### **Step 5: Create Gateway Middleware**

**`api/gateway/auth.py`**
```python
from functools import wraps
from fastapi import HTTPException, Header
import jwt

class AuthConfig:
    SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    ALGORITHM = "HS256"

def require_auth(scope: str = None):
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            auth_header = request.headers.get("Authorization", "")
            
            if not auth_header.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Missing auth token")
            
            token = auth_header.split(" ")[1]
            try:
                payload = jwt.decode(token, AuthConfig.SECRET_KEY, 
                                    algorithms=[AuthConfig.ALGORITHM])
                request.state.tenant_id = payload["tenant_id"]
                request.state.user_id = payload["user_id"]
                request.state.scopes = payload.get("scopes", [])
                
                if scope and scope not in request.state.scopes:
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
            except jwt.InvalidTokenError:
                raise HTTPException(status_code=401, detail="Invalid token")
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator
```

**`api/gateway/middleware.py`**
```python
class TenantMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Extract tenant_id from:
        # 1. Authorization header (JWT)
        # 2. X-Tenant-ID header (optional explicit)
        # 3. Subdomain (api-tenant123.example.com)
        
        tenant_id = extract_tenant_from_request(request)
        if not tenant_id:
            return JSONResponse({"error": "Missing tenant context"}, status_code=400)
        
        request.state.tenant_id = tenant_id
        response = await call_next(request)
        response.headers["X-Tenant-ID"] = tenant_id
        return response

def extract_tenant_from_request(request: Request) -> Optional[str]:
    # Priority: JWT > X-Tenant-ID header > subdomain
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            payload = jwt.decode(token, ..., options={"verify_signature": False})
            return payload.get("tenant_id")
        except:
            pass
    
    tenant_id = request.headers.get("X-Tenant-ID")
    if tenant_id:
        return tenant_id
    
    # Extract from subdomain
    host = request.headers.get("Host", "")
    if "." in host:
        subdomain = host.split(".")[0]
        if subdomain.startswith("api-"):
            return subdomain[4:]  # api-tenant123 → tenant123
    
    return None
```

**`api/gateway/metering.py`**
```python
class MeteringService:
    async def record_usage(self, tenant_id: str, operation: str, count: int = 1):
        """
        Track API usage per tenant for billing/rate limiting.
        operation: "training", "recommendation", "similar", etc.
        """
        key = f"usage:{tenant_id}:{operation}:{date.today()}"
        await redis_client.incr(key, count)
        
        # Check rate limits
        tier = await self.get_tenant_tier(tenant_id)
        limits = TIER_LIMITS[tier]
        current = await redis_client.get(key)
        
        if current > limits[operation]:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    async def get_tenant_tier(self, tenant_id: str) -> str:
        config = TenantConfigManager.load(tenant_id)
        return config.get("tier", "free")

TIER_LIMITS = {
    "free": {"training": 5, "recommendation": 100},
    "pro": {"training": 50, "recommendation": 10000},
    "enterprise": {"training": 500, "recommendation": 1000000},
}
```

---

### **Step 6: Tenant Infrastructure**

**`core/tenants/manager.py`**
```python
class TenantManager:
    @staticmethod
    def load_config(tenant_id: str) -> dict:
        path = f"tenants/{tenant_id}/config.json"
        if not os.path.exists(path):
            raise ValueError(f"Tenant {tenant_id} not found")
        with open(path) as f:
            return json.load(f)
    
    @staticmethod
    def create_tenant(tenant_id: str, config: dict):
        os.makedirs(f"tenants/{tenant_id}/data", exist_ok=True)
        os.makedirs(f"tenants/{tenant_id}/models", exist_ok=True)
        with open(f"tenants/{tenant_id}/config.json", "w") as f:
            json.dump(config, f)
    
    @staticmethod
    def validate_tenant_isolation(tenant_id: str, requested_path: str) -> bool:
        """Ensure requested path is within tenant's directory."""
        real_path = os.path.realpath(requested_path)
        tenant_base = os.path.realpath(f"tenants/{tenant_id}")
        return real_path.startswith(tenant_base)
```

**`tenants/{tenant_id}/config.json` Example**
```json
{
  "tenant_id": "acme-corp",
  "api_key": "sk_live_xxxxx",
  "tier": "pro",
  "scopes": [
    "training:write",
    "recommendation:read",
    "models:write",
    "analytics:read"
  ],
  "active_model_id": "model_v3_best",
  "settings": {
    "top_k_default": 10,
    "training_algorithm_mode": "auto",
    "enable_hybrid_mode": true
  }
}
```

---

### **Step 7: Update `main.py` & Create `app_factory.py`**

**`main.py`** (Simplified)
```python
from fastapi import FastAPI
from app_factory import create_app
import uvicorn

app = create_app()

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**`app_factory.py`** (New)
```python
from fastapi import FastAPI
from fastapi.middleware import Middleware
from api.gateway.middleware import TenantMiddleware
from api.routes import training, recommend, batch, similar, models, algorithms

def create_app() -> FastAPI:
    # Middleware stack (order matters)
    middleware = [
        Middleware(TenantMiddleware),
        Middleware(CORSMiddleware, allow_origins=["*"]),
    ]
    
    app = FastAPI(middleware=middleware)
    
    # Register routes
    app.include_router(training.router, prefix="/api", tags=["training"])
    app.include_router(recommend.router, prefix="/api", tags=["recommendations"])
    app.include_router(batch.router, prefix="/api", tags=["recommendations"])
    app.include_router(similar.router, prefix="/api", tags=["recommendations"])
    app.include_router(models.router, prefix="/api", tags=["models"])
    app.include_router(algorithms.router, prefix="/api", tags=["system"])
    
    # Admin routes
    admin_app = create_admin_app()
    app.mount("/admin", admin_app)
    
    return app

def create_admin_app() -> FastAPI:
    """Separate FastAPI app for admin dashboard."""
    admin = FastAPI()
    admin.include_router(admin_routes.router)
    return admin
```

---

## Import Adjustments Needed

### In `core/pipeline/training_pipeline.py`:
```python
# OLD
from algorithms import ALGORITHM_REGISTRY
from benchmark.benchmark_engine import BenchmarkEngine

# NEW
from core.algorithms import ALGORITHM_REGISTRY
from core.benchmark.benchmark_engine import BenchmarkEngine
from core.tenants.manager import TenantManager

# Add tenant_id to constructor
def __init__(self, config: TrainingConfig, tenant_id: str):
    self.tenant_id = tenant_id
    self.config = config
    # Model registry is now tenant-scoped
    self.registry = ModelRegistry(tenant_id=tenant_id)
```

### In `core/models/model_registry.py`:
```python
# OLD
MODEL_STORE_DIR = "model_store"

# NEW
def __init__(self, tenant_id: str):
    self.tenant_id = tenant_id
    self.store_dir = Path(f"tenants/{tenant_id}/models")
    self.store_dir.mkdir(parents=True, exist_ok=True)
```

### In `api/routes/training.py`:
```python
# OLD
from pipeline.training_pipeline import TrainingPipeline

# NEW
from core.pipeline.training_pipeline import TrainingPipeline
from core.data_processing.feedback_detector import FeedbackDetector
from core.insights.model_explainer import ModelExplainer
from api.gateway.metering import metering
```

---

## Testing Strategy

Add `tests/` with unit tests:

```python
# tests/test_tenant_isolation.py
def test_tenant_cannot_access_other_tenant_data():
    # Ensure tenant A cannot read/write tenant B's data
    
# tests/test_auth.py
def test_invalid_token_rejected()
def test_missing_auth_header_rejected()
def test_insufficient_scope_rejected()

# tests/test_training_pipeline.py (unchanged logic)

# tests/test_api_routes.py
def test_training_endpoint_records_usage()
def test_recommend_uses_active_model()
```

---

## Rollout Checklist

- [ ] Create new directory structure
- [ ] Move core AI files (no logic changes)
- [ ] Refactor ingestion to `smart_db_csv_builder/`
- [ ] Split `api/routes.py` into modular route files
- [ ] Implement gateway middleware (auth, tenants, metering)
- [ ] Add tenant infrastructure (manager, context, config)
- [ ] Update all imports throughout codebase
- [ ] Update `main.py` and create `app_factory.py`
- [ ] Implement admin dashboard
- [ ] Add comprehensive tests
- [ ] Database migrations (if using SQL for tenant config)
- [ ] Update Docker/deployment configs
- [ ] Update documentation
- [ ] Load test with multiple tenants
- [ ] Security audit (especially tenant isolation)
- [ ] Deploy to staging, then production

---

## Summary: What Changes vs. What Stays the Same

### ✅ **Stays Same (Logic Unchanged)**
- All 8+ recommendation algorithms
- Benchmark engine scoring logic
- Data cleaning and preprocessing
- Optuna tuning logic
- Model training and evaluation
- Interaction matrix builder
- Explainability insights

### 🔄 **Refactored (Logic + Structure)**
- API routes (split & add auth)
- Ingestion (tenant-scoped paths)
- Model registry (tenant-scoped storage)
- Training pipeline (add tenant_id context)
- Configuration (add tenant settings)

### ✨ **New (Enterprise Features)**
- Multi-tenant isolation
- JWT/API key authentication
- Role-based access control (scopes)
- Rate limiting & metering
- Admin dashboard
- Tenant config management
- Webhook/event-driven delivery (SDK)
- JavaScript widget for no-code clients

