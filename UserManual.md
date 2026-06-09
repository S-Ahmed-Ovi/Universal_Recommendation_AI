# Proactive AI — User Manual

**Version:** 3.2  
**Status:** Production Ready  
**Audience:** Developers, ML Engineers, and Product Teams  
**Last Updated:** May 2026

---

## 1. Executive Summary

**Proactive AI** is an enterprise-grade, multi-tenant recommendation platform designed to bridge the gap between raw data and production-ready personalization. It automates the entire machine learning lifecycle—from data ingestion and cleaning to algorithm benchmarking, hyperparameter tuning, and high-concurrency serving.

### Core Capabilities
*   **Auto-ML Pipeline:** Automated feedback detection, algorithm selection, and Optuna-backed tuning.
*   **Multi-Tenancy:** Hard isolation of data, models, and configuration for multiple clients or projects.
*   **SmartDB Integration:** Native connectors for SQL (Postgres, MySQL, SQLite) and NoSQL (MongoDB, DynamoDB) to automate ETL.
*   **Flexible Serving:** Support for single-model pinning, weighted ensembles, and automated "best-promoted" strategy.
*   **Explainable AI:** Human-readable explanations for every recommendation, with optional LLM enrichment.
*   **Developer First:** RESTful API, interactive Swagger documentation, and a drop-in JS widget for rapid integration.

---

## 2. Core Concepts & Glossary

| Term | Definition |
| :--- | :--- |
| **Tenant** | An isolated workspace with its own configuration, API keys, datasets, and models. |
| **Job** | An asynchronous process (Training or SmartDB Build) identified by a unique `job_id`. |
| **Algorithm** | A specific mathematical model (e.g., ALS, BPR, SVD) used to generate recommendations. |
| **Model** | A trained and serialized artifact resulting from a training job. |
| **Promoted Model** | The canonical version of a model designated for production serving for a specific tenant. |
| **Strategy** | The logic used during inference to select or combine models (e.g., `ensemble_weighted`). |
| **Interaction** | A single data point representing a user's action on an item (e.g., a click or a rating). |

---

## 3. Quick Start Guide

Follow these steps to go from zero to serving your first recommendations.

### Step 1: Initialize the Environment
Ensure Python 3.11+ is installed. Clone the repository and install dependencies:

```bash
git clone https://github.com/your-org/proactive-ai.git
cd proactive-ai/backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Configure and Start
Create a `.env` file based on the provided guide and start the server:

```bash
python main.py
```
*The API is now live at `http://localhost:8000`. Access docs at `/docs`.*

### Step 3: Create a Tenant
Use the Admin API to create your first tenant (requires Admin credentials):

```bash
curl -X POST http://localhost:8000/admin/tenants \
  -H "Authorization: Bearer <ADMIN_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "my-app", "display_name": "My Application"}'
```
**Important:** Securely store the `api_key` returned in the response.

### Step 4: Train a Model
Upload a CSV with `user_id`, `item_id`, and `rating`/`interaction`:

```bash
curl -X POST http://localhost:8000/train/file \
  -H "X-API-Key: <TENANT_API_KEY>" \
  -F "file=@data.csv"
```

### Step 5: Serve Recommendations
Once the job is `completed` and the model is `promoted`, call the recommendation endpoint:

```bash
curl -X POST http://localhost:8000/recommend \
  -H "X-API-Key: <TENANT_API_KEY>" \
  -d '{"user_id": "user_123", "top_n": 10}'
```

---

## 4. Architecture Deep Dive

### 4.1 System Layers

Proactive AI operates with a clear separation of concerns:

1.  **API Gateway:** Handles authentication (API Keys/JWT), tenant context injection, and request metering.
2.  **Control Plane:** Manages tenant configurations, deployment state, and model promotion logic.
3.  **Data Ingestion (SmartDB):** Connects to external databases, executes queries, and prepares training-ready CSVs.
4.  **Training Engine:** Orchestrates data cleaning, benchmarking, and hyperparameter optimization (Optuna).
5.  **Inference Engine:** Loads models into memory and serves high-speed recommendation results.

### 4.2 The Training Lifecycle

When a training job is submitted, it moves through these stages:
1.  **Ingestion:** Data is loaded from a file or SQL query.
2.  **Profiling:** System detects if feedback is `explicit` (ratings) or `implicit` (binary).
3.  **Cleaning:** Filters outliers and users/items with insufficient history.
4.  **Benchmarking:** Multiple algorithms compete on metrics like `NDCG@K` and `HitRate@K`.
5.  **Tuning:** Top performers undergo hyperparameter search using Optuna.
6.  **Persistence:** Optimized models are saved to the Tenant Model Store.
7.  **Promotion:** The best model is automatically (or manually) promoted to the Production slot.

---

## 5. Implementation Guide

### 5.1 Authentication Models

Proactive AI supports two distinct authentication paths:

*   **Admin Access (Bearer Token):** Used for platform management (creating tenants, global monitoring). Obtain tokens via `POST /admin/auth/login`.
*   **Tenant Access (X-API-Key):** Used for all application-level operations (training, recommendations). The key is provided only once during tenant creation.

### 5.2 Recommendation Strategies

Customize how results are generated by passing a `strategy` in the `/recommend` request:

*   **`best_promoted_model` (Default):** Uses the currently promoted production model.
*   **`single_model`:** Forces the use of a specific `model_id` or `algorithm`.
*   **`ensemble_weighted`:** Blends multiple models. Example:
    ```json
    {
      "strategy": "ensemble_weighted",
      "models": [
        {"algorithm": "ALS", "weight": 0.7},
        {"algorithm": "BPR", "weight": 0.3}
      ]
    }
    ```

### 5.3 SmartDB Workflow

SmartDB eliminates manual ETL by pulling data directly from your database:
1.  **Register Connection:** `POST /smart-db-csv/api/connections`
2.  **Inspect Schema:** `GET /smart-db-csv/api/schema/{conn_id}`
3.  **Plan & Build:** Define mapping and submit a build job.
4.  **Train:** Use the generated CSV as input for `POST /train/file`.

---

## 6. Frontend Integration

### 6.1 Admin Dashboard
Access the built-in dashboard at `http://localhost:8000/app` to manage tenants, view training leaderboards, and monitor system health.

### 6.2 JS Widget
Integrate recommendations into any website with a single script tag:

```html
<div id="proactive-recs"></div>
<script>
  window.ProactiveAIConfig = {
    apiUrl: "https://api.yourdomain.com",
    apiKey: "YOUR_TENANT_KEY",
    userId: "current_user_id",
    mountId: "proactive-recs"
  };
</script>
<script src="/sdk/js_widget/index.js"></script>
```

---

## 7. Operational Reference

### 7.1 Configuration (.env)

| Variable | Description | Default |
| :--- | :--- | :--- |
| `AUTH_SECRET_KEY` | Key for JWT signing (Production: Use long random string) | `None` |
| `LOG_LEVEL` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` |
| `BASE_TENANTS_DIR` | Directory for tenant-scoped data and models | `tenants/` |
| `LLM_EXPLANATIONS_ENABLED` | Enable natural language explanations | `false` |
| `RATE_LIMIT_RPM` | Default requests per minute per tenant | `60` |

### 7.2 Webhook Events
Register a `webhook_url` in the tenant config to receive:
*   `job.completed`: Training or build success.
*   `job.failed`: Details on why a process failed.
*   `model.promoted`: Notification when the production model changes.

---

## 8. Troubleshooting & FAQ

### Common Issues

**Q: Why am I getting a 401 Unauthorized?**  
A: Ensure you are using the raw `X-API-Key` (starts with `pai_`) for tenant routes, and a valid Bearer token for admin routes.

**Q: A training job is stuck in `pending`.**  
A: Check if the background worker is running. In dev mode, this runs in a thread; in production, ensure Redis and the RQ worker are active.

**Q: Recommendation results are empty.**  
A: Ensure the user has at least `MIN_USER_INTERACTIONS` (default: 3) in the training data, or that a model has been promoted for the tenant.

**Q: How do I handle large datasets?**  
A: Proactive AI supports adaptive downsampling. For datasets >1M interactions, consider using the `Tier A` configuration settings in `.env`.

---

## 9. Best Practices

1.  **Data Freshness:** Retrain models weekly or daily depending on your catalog churn.
2.  **Hybrid Strategies:** Use `ensemble_weighted` to combine the strengths of Collaborative Filtering (ALS) with Ranking models (BPR).
3.  **Security:** Never commit `.env` files. Rotate Tenant API keys every 90 days using `POST /admin/tenants/{id}/rotate-key`.
4.  **Monitoring:** Use the `/health` and `/metrics` (Prometheus) endpoints to set up alerts for latency and error rates.

---
*For further assistance, visit our [GitHub Repository](https://github.com/your-org/proactive-ai) or contact support at support@proactive-ai.io.*
