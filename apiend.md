# Proactive AI — API Endpoints Reference

**Version:** 3.2  
**Status:** Production Ready  
**Audience:** Developers and Integrators

This document provides a comprehensive reference for the Proactive AI RESTful API. The API is logically divided into four main surfaces:

1.  **System Endpoints:** Health checks and built-in utilities.
2.  **Tenant API:** Core operations for training, recommendations, and model management.
3.  **SmartDB API:** Utilities for direct database ingestion and ETL automation.
4.  **Admin API:** Global platform management and dashboard support.

---

## 1. System Endpoints

These endpoints provide basic liveness and application routing. They are generally unauthenticated.

| Method | Path | Description | Access |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Basic liveness check. Returns `{"status": "ok", "version": "3.0.0"}`. | Public |
| `GET` | `/` | Serves the frontend React application (`index.html`) if built. | Public |
| `GET` | `/smart-db-csv` | Alias for `/`, used as a frontend entry point. | Public |
| `GET` | `/docs` | Interactive API documentation (Swagger UI). | Public |

---

## 2. Tenant API

These endpoints form the core of the recommendation platform. All endpoints require a valid Tenant API Key (`X-API-Key`) and the appropriate operational scope.

### 2.1 Training & Jobs
Manage asynchronous machine learning pipelines.

| Method | Path | Description | Required Scope |
| :--- | :--- | :--- | :--- |
| `POST` | `/train/file` | Submit a CSV/JSON/Parquet file to start an ML training job. | `training:write` |
| `POST` | `/train/sql` | Submit database credentials and a query to start a training job. | `training:write` |
| `GET` | `/jobs` | List recent jobs for the current tenant. | `jobs:read` |
| `GET` | `/jobs/{job_id}` | Poll the status and retrieve results of a specific job. | `jobs:read` |
| `DELETE` | `/jobs/{job_id}` | Remove a completed or failed job record. | `training:write` |

### 2.2 Recommendations & Serving
Retrieve personalized recommendations and item similarities.

| Method | Path | Description | Required Scope |
| :--- | :--- | :--- | :--- |
| `GET` | `/recommend/options` | Discover available strategies, algorithms, and promoted models. | `recommend:read` |
| `POST` | `/recommend` | Generate recommendations for a single user based on strategy. | `recommend:read` |
| `POST` | `/recommend/batch` | Generate recommendations for multiple users in a single request. | `recommend:read` |
| `GET` | `/similar/{item_id}` | Retrieve items structurally or behaviorally similar to a given item. | `recommend:read` |
| `GET` | `/explain/{user_id}/{item_id}`| Get human-readable reasons for why an item was recommended. | `recommend:read` |

### 2.3 Model Management
Manage trained artifacts and production deployments.

| Method | Path | Description | Required Scope |
| :--- | :--- | :--- | :--- |
| `GET` | `/models` | List all trained models available in the tenant's registry. | `models:read` |
| `POST` | `/models/{model_id}/promote` | Set a specific model as the active production deployment. | `models:write` |

---

## 3. SmartDB API

The SmartDB endpoints automate the extraction of training data from external SQL and NoSQL sources. They are mounted under `/smart-db-csv/api/` and require the `training:write` scope.

| Method | Path | Description | Required Scope |
| :--- | :--- | :--- | :--- |
| `POST` | `.../connections` | Register a new SQL database connection. | `training:write` |
| `POST` | `.../connections/nosql` | Register a new NoSQL (MongoDB, DynamoDB) connection. | `training:write` |
| `GET` | `.../connections` | List all registered connections for the tenant. | `training:write` |
| `DELETE`| `.../connections/{conn_id}` | Delete a registered database connection. | `training:write` |
| `GET` | `.../schema/{conn_id}` | Discover and list all tables in the connected database. | `training:write` |
| `GET` | `.../schema/{conn_id}/{table}` | Inspect table schema and retrieve a small data sample. | `training:write` |
| `POST` | `.../schema/{conn_id}/plan` | Use the LLM to auto-generate the optimal extraction query. | `training:write` |
| `POST` | `.../build` | Dispatch a job to extract data and build a training-ready CSV. | `training:write` |
| `GET` | `.../build/{conn_id}/preview`| Execute the extraction query and return a preview payload. | `training:write` |
| `GET` | `.../jobs` | List recent SmartDB extraction jobs. | `training:write` |
| `GET` | `.../jobs/{job_id}` | Poll the status of a specific SmartDB extraction job. | `training:write` |
| `DELETE`| `.../jobs/{job_id}` | Remove a SmartDB job record. | `training:write` |

---

## 4. Admin API

Admin APIs are used for global platform management and require a Bearer token obtained via `/admin/auth/login`.

### 4.1 Authentication & Status
| Method | Path | Description | Access |
| :--- | :--- | :--- | :--- |
| `POST` | `/admin/auth/login` | Authenticate with username/password to receive a Bearer token. | Public |
| `GET` | `/admin/auth/me` | Retrieve token claims and permissions for the current session. | Auth Required |
| `GET` | `/admin/health` | Diagnostic liveness check for the Admin router. | Public |
| `GET` | `/admin/session/status` | View global runtime state and algorithm availability. | Public |
| `GET` | `/admin/algorithms` | List globally registered algorithms and their configurations. | Public |

### 4.2 Tenant Management
| Method | Path | Description | Access |
| :--- | :--- | :--- | :--- |
| `POST` | `/admin/tenants` | Create a new tenant. Returns the API key (once only). | Auth Required |
| `GET` | `/admin/tenants` | List all tenants on the platform. | Auth Required |
| `GET` | `/admin/tenants/{tenant_id}` | Retrieve details for a specific tenant. | Auth Required |
| `PATCH` | `/admin/tenants/{tenant_id}` | Update tenant configuration (e.g., tier, rate limits). | Auth Required |
| `DELETE` | `/admin/tenants/{tenant_id}` | Delete a tenant (with option to preserve data). | Auth Required |
| `POST` | `/admin/tenants/{id}/rotate-key`| Invalidate the current API key and issue a new one. | Auth Required |
| `PATCH` | `/admin/admin/tenants/{id}/webhook` | Configure a webhook URL and secret for tenant events. | Auth Required |

*(Note: The admin interface also mirrors the Tenant API operations under the `/admin/` prefix to power the frontend dashboard.)*

---

## 5. Architectural Notes & Guidelines

*   **Authentication Modes:** Platform administrators use JWT Bearer tokens via the `/admin` routes. End-user applications use the raw `X-API-Key` via the root Tenant API routes.
*   **Job State Volatility:** By default (without Redis), asynchronous jobs (Training and SmartDB builds) are managed in memory and will not persist across server restarts.
*   **Rate Limiting:** Tenant API requests are subject to the `RATE_LIMIT_RPM` defined in the tenant's configuration. Exceeding this limit returns a `429 Too Many Requests` status.
*   **Webhook Path Quirk:** Due to current routing architecture, the webhook configuration endpoint is accessible at `/admin/admin/tenants/{tenant_id}/webhook`.