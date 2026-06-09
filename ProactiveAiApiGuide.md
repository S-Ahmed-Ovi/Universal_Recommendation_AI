# Proactive AI API Guide

This guide matches the current codebase in `backend/` as of May 23, 2026.

## Run Guide

### 1. Install

```bash
python -m venv venv
venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 2. Minimal `.env`

Create `.env` in the repo root or `backend/.env`.

```env
API_HOST=0.0.0.0
API_PORT=8000

AUTH_USERNAME=admin
AUTH_PASSWORD=admin123
AUTH_SECRET_KEY=replace-me-in-production
AUTH_TOKEN_TTL_MINUTES=120

BASE_TENANTS_DIR=tenants
MODEL_STORE_DIR=model_store
LOG_LEVEL=INFO

USE_V3_AUTH=true
USE_V3_MULTITENANCY=true
USE_V3_METERING=true
USE_V3_ROUTES=true
```

### 3. Start the API

From `backend/`:

```bash
python main.py
```

That starts Uvicorn with reload enabled on `http://0.0.0.0:8000`.

Docs:

- `http://localhost:8000/docs`
- `http://localhost:8000/openapi.json`

### 4. CLI training mode

```bash
python main.py --file path/to/ratings.csv
```

Supported flags:

| Flag | Default | Notes |
|---|---:|---|
| `--file` | none | Required for CLI training mode |
| `--format` | `auto` | `csv`, `excel`, `json`, `parquet`, `auto` |
| `--topk` | `10` | Allowed: `5`, `10` |
| `--top-models` | `10` | Allowed: `5`, `10` |
| `--trials` | `-1` | Optuna adaptive when `-1` |
| `--mode` | `auto` | `explicit`, `implicit`, `hybrid`, `auto` |
| `--interactive` | off | Manual column mapping |
| `--force-all` | off | Run broader benchmark coverage |
| `--no-tune` | off | Skip Optuna |
| `--serve` | off | Force API mode |

## Authentication

There are two auth patterns in the current app.

### Admin bearer token

Used for admin and dashboard endpoints under `/admin/...`.

```bash
curl -X POST http://localhost:8000/admin/auth/login ^
  -H "Content-Type: application/json" ^
  -d "{\"username\":\"admin\",\"password\":\"admin123\"}"
```

Response:

```json
{
  "access_token": "eyJ...",
  "token_type": "bearer"
}
```

Then send:

```http
Authorization: Bearer eyJ...
```

### Tenant API key

Used for tenant-scoped training, recommendation, jobs, models, analytics, deployment, and SmartDB endpoints.

```http
X-API-Key: pai_...
```

### Verify tenant context

```bash
curl http://localhost:8000/tenant/context ^
  -H "X-API-Key: pai_..."
```

Response:

```json
{
  "tenant_id": "iqbal",
  "tier": "starter",
  "scopes": ["training:write", "recommend:read", "models:read"],
  "auth_type": "api_key"
}
```

## Health and System Endpoints

### `GET /health`

Root app health.

```json
{
  "status": "ok",
  "version": "3.0.0"
}
```

### `GET /admin/health`

Admin router health with timestamp.

```json
{
  "status": "ok",
  "timestamp": 1747980000.0
}
```

### `GET /admin/session/status`

Shows runtime state for a tenant.

Query:

- `tenant_id` optional, defaults to `admin`

### `GET /algorithms`
### `GET /admin/algorithms`

Both return algorithm metadata from `ALGORITHM_REGISTRY`.

## Tenant Management

The guide previously said there was no tenant registration API. That is no longer true.

### Create tenant

`POST /admin/tenants`

Auth: admin bearer token

```bash
curl -X POST http://localhost:8000/admin/tenants ^
  -H "Authorization: Bearer %TOKEN%" ^
  -H "Content-Type: application/json" ^
  -d "{
    \"tenant_id\":\"iqbal\",
    \"display_name\":\"Iqbal\",
    \"tier\":\"starter\",
    \"algorithm_mode\":\"auto\",
    \"rate_limit_rpm\":60,
    \"notes\":\"first tenant\"
  }"
```

Response includes the raw API key once:

```json
{
  "tenant_id": "iqbal",
  "display_name": "Iqbal",
  "tier": "starter",
  "scopes": ["training:write", "recommend:read", "models:read", "models:write", "explain:read", "batch:read", "similar:read", "jobs:read"],
  "rate_limit_rpm": 60,
  "active": true,
  "created_at": 1747980000.0,
  "api_key": "pai_...",
  "warning": "Save this api_key now - it will never be shown again."
}
```

### List tenants

`GET /admin/tenants`

Query:

- `active_only` optional bool
- `tier` optional string

### Get tenant

`GET /admin/tenants/{tenant_id}`

### Update tenant

`PATCH /admin/tenants/{tenant_id}`

### Delete tenant

`DELETE /admin/tenants/{tenant_id}?delete_data=false`

### Rotate tenant API key

`POST /admin/tenants/{tenant_id}/rotate-key`

### Webhook registration

Current mounted path is:

`PATCH /admin/admin/tenants/{tenant_id}/webhook`

This double `/admin` is real in the current code because the router is mounted with `/admin` and the route itself is declared as `/admin/tenants/...`.

Body:

```json
{
  "webhook_url": "https://example.com/hooks/proactive",
  "webhook_secret": "at-least-16-characters"
}
```

Webhook events:

- `job.completed`
- `job.failed`
- `model.promoted`

Signature header:

```http
X-ProactiveAI-Signature: sha256=<hex_digest>
```

## Training API

Auth: tenant API key with `training:write`

### `POST /train/file`

Multipart form upload. Returns `202 Accepted`.

Fields:

| Field | Type | Default |
|---|---|---|
| `file` | upload | required |
| `top_k` | int | `10` |
| `n_trials` | int | `-1` |
| `top_models` | int | `10` |
| `algorithm_mode` | string | `auto` |
| `format` | string | `auto` |

Allowed `top_k` and `top_models`: `5`, `10`

Allowed `algorithm_mode`: `explicit`, `implicit`, `hybrid`, `auto`

Example:

```bash
curl -X POST http://localhost:8000/train/file ^
  -H "X-API-Key: pai_..." ^
  -F "file=@ratings.csv" ^
  -F "top_k=10" ^
  -F "n_trials=-1" ^
  -F "top_models=10" ^
  -F "algorithm_mode=auto" ^
  -F "format=auto"
```

Response:

```json
{
  "job_id": "8ef2a7c0d3f1",
  "poll_url": "/jobs/8ef2a7c0d3f1"
}
```

### `POST /train/sql`

JSON body. Returns `202 Accepted`.

```json
{
  "dialect": "postgresql",
  "host": "db.example.com",
  "port": 5432,
  "database": "mydb",
  "username": "readonly",
  "password": "secret",
  "sql": "SELECT user_id, item_id, rating FROM ratings",
  "top_k": 10,
  "n_trials": -1,
  "top_models": 10,
  "algorithm_mode": "auto"
}
```

## Training Jobs

Auth:

- `GET /jobs` requires `jobs:read`
- `GET /jobs/{job_id}` requires `jobs:read`
- `DELETE /jobs/{job_id}` requires `training:write`

### `GET /jobs`

Query:

- `limit` default `25`, range `1..100`

### `GET /jobs/{job_id}`

Current v3 status values:

- `pending`
- `running`
- `completed`
- `failed`

This is different from older in-memory admin job shapes that used `done` and `error`.

### `DELETE /jobs/{job_id}`

Deletes a completed or failed job record. Running and pending jobs return `409`.

## Recommendation API

Auth: tenant API key with `recommend:read`

### `GET /recommend/options`

Optional query:

- `top_n_models` integer `>= 1`

Returns strategies, ranked model choices, promoted models, and available algorithms.

### `POST /recommend`

Body:

```json
{
  "user_id": "user_42",
  "top_n": 10,
  "strategy": null,
  "model_id": null,
  "algorithm": null,
  "models": [],
  "auto_normalize_weights": true
}
```

`top_n` must be `5` or `10`.

Weighted ensemble example:

```json
{
  "user_id": "user_42",
  "top_n": 10,
  "models": [
    {"algorithm": "SVD", "weight": 0.7},
    {"algorithm": "ALS", "weight": 0.3}
  ]
}
```

### `POST /recommend/batch`

Body is the same pattern, but with `user_ids`.

```json
{
  "user_ids": ["user_1", "user_2"],
  "top_n": 10
}
```

Notes:

- returns `400` if `user_ids` is empty
- returns partial `results` plus `errors`
- returns `422` only if every user fails

### `GET /similar/{item_id}`

Query:

- `top_n` default `10`
- `algorithm` optional

### `GET /explain/{user_id}/{item_id}`

Query:

- `score` default `0.0`
- `algorithm` optional

## Models and Deployment

### `GET /models`

Auth: `models:read`

Query:

- `algorithm` optional
- `promoted_only` default `false`

Response shape:

```json
{
  "tenant_id": "iqbal",
  "models": [],
  "total": 0
}
```

### `POST /models/{model_id}/promote`

Auth: `models:write`

Promotes the model, clears serving cache, records deployment metadata, and may emit a webhook.

### `GET /deployment/active`
### `GET /deployment/status`

Auth: `models:read`

Both expose the current active deployment view.

### `GET /deployment/history`

Auth: `models:read`

Query:

- `limit` default `12`, range `1..100`

### `POST /deployment/rollback`

Auth: `models:write`

Returns status `rolled_back` on success.

## Benchmarks and Analytics

### `GET /benchmarks`

Auth: `models:read`

Returns:

- `best_algorithm`
- `best_model_id`
- `leaderboard`
- `tuning`
- `resolved_mode`
- `top_model_recommendations`
- `model_selection_policy`
- `dataset_shape`
- `dataset_columns`
- `dataset_stats`
- `promoted_models`

### `GET /analytics`

Auth: `models:read`

Returns tenant usage counters from metering.

If no usage exists yet:

```json
{
  "tenant_id": "iqbal",
  "totals": {},
  "tier": "starter",
  "message": "No usage data recorded yet."
}
```

## SmartDB CSV Builder

Mounted under `/smart-db-csv/api/...`

These routes are included with dependency `training:write`.

### Connections

- `POST /smart-db-csv/api/connections`
- `POST /smart-db-csv/api/connections/nosql`
- `GET /smart-db-csv/api/connections`
- `DELETE /smart-db-csv/api/connections/{conn_id}`

### Schema

- `GET /smart-db-csv/api/schema/{conn_id}`
- `GET /smart-db-csv/api/schema/{conn_id}/{table}`
- `POST /smart-db-csv/api/schema/{conn_id}/plan`

The plan endpoint currently requires `ANTHROPIC_API_KEY` for the Claude-based planner.

### Build

- `POST /smart-db-csv/api/build`
- `GET /smart-db-csv/api/build/{conn_id}/preview`

`POST /smart-db-csv/api/build` returns `202` with a SmartDB job id.

### SmartDB jobs

- `GET /smart-db-csv/api/jobs`
- `GET /smart-db-csv/api/jobs/{job_id}`
- `DELETE /smart-db-csv/api/jobs/{job_id}`

SmartDB job status values:

- `pending`
- `running`
- `done`
- `failed`

## Scope Reference

Current scopes from `api/gateway/auth.py`:

| Scope | Meaning |
|---|---|
| `recommend:read` | recommendations, similar items, explanations |
| `training:write` | submit training jobs |
| `models:read` | list models, analytics, benchmarks, deployment reads |
| `models:write` | promote models, rollback deployment |
| `explain:read` | declared scope for explainability |
| `batch:read` | declared scope for batch reads |
| `similar:read` | declared scope for similar-item reads |
| `jobs:read` | read async job status |
| `admin:read` | admin read access |
| `admin:write` | all scopes |

Scope hierarchy:

- `admin:write` expands to all scopes
- `admin:read` expands to `recommend:read` and `models:read`

## Common Errors

| HTTP | Meaning in current app |
|---|---|
| `400` | invalid request shape such as empty `user_ids` |
| `401` | missing auth, invalid token, invalid API key |
| `403` | auth ok but missing scope or wrong tenant access |
| `404` | tenant, model, job, or connection not found |
| `409` | invalid state transition such as deleting a running job or impossible rollback |
| `413` | upload too large for `/train/file` |
| `422` | Pydantic validation or recommendation strategy error |
| `429` | tenant rate limit exceeded |
| `500` | server-side failure |
| `502` | downstream database/schema preview issue in SmartDB |
| `503` | recommendation or LLM planning unavailable |

## Practical Notes

- Root `/health` and `/admin/health` are not the same endpoint.
- Tenant creation should use `POST /admin/tenants`; the old Python-only bootstrap flow is no longer required.
- Admin tenant management uses bearer auth in the frontend.
- Tenant-facing API pages use `X-API-Key`, not the admin bearer token.
- Current Uvicorn startup in `main.py` uses `reload=True`, so code edits trigger server reloads during local development.
