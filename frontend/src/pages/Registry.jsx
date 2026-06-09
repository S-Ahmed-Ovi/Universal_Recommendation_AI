import { useCallback, useDeferredValue, useMemo, useState } from "react";
import { RefreshCw } from "lucide-react";
import { Link } from "react-router-dom";
import TenantRequiredState from "../components/TenantRequiredState";
import Layout from "../components/Layout";
import { formatTimestamp, tenantRequest } from "../api";
import { useAsyncData } from "../hooks/useAsyncData";
import { useTenantSession } from "../hooks/useAuthSession";

const EMPTY_MODELS = Object.freeze([]);

export default function Registry(){
  const [algorithmFilter, setAlgorithmFilter] = useState("");
  const deferredAlgorithmFilter = useDeferredValue(algorithmFilter);
  const {
    hasTenantApiKey: hasKey,
    tenantApiKey,
    tenantId,
  } = useTenantSession();
  const loadModels = useCallback(async ({ signal }) => {
    const data = await tenantRequest("/models", { signal });
    return data?.models || [];
  }, [tenantApiKey, tenantId]);
  const {
    data: models,
    error,
    loading,
    reload,
  } = useAsyncData(loadModels, {
    enabled: hasKey,
    initialData: EMPTY_MODELS,
  });

  const filteredModels = useMemo(() => {
    const value = deferredAlgorithmFilter.trim().toLowerCase();
    if (!value) {
      return models;
    }
    return models.filter((model) => String(model.algorithm || "").toLowerCase().includes(value));
  }, [deferredAlgorithmFilter, models]);

  return(
    <Layout
      summary="Model Registry is an artifact inventory. Promotion and rollback actions are handled only in Deployment & Production."
      title="Model Registry"
      heroStats={[
        { label: "Tenant", value: tenantId || "Not attached" },
        { label: "Artifacts", value: String(models.length) },
        { label: "Promoted", value: String(models.filter((model) => model.promoted).length) },
      ]}
    >
      {error ? <div className="form-error">{error}</div> : null}
      {!hasKey ? (
        <TenantRequiredState
          title="No tenant API key selected"
          description="Open Tenants, store a tenant API key, then come back to browse models."
        />
      ) : null}

      <div className="card">
        <div className="card-header">
          <h3>Artifact Inventory</h3>
          <div className="actions-row">
            <span className="subtle">{tenantId || "Current tenant key"}</span>
            <button
              aria-label={loading ? "Refreshing models" : "Refresh models"}
              className="btn btn-secondary btn-icon"
              disabled={!hasKey || loading}
              onClick={() => reload()}
              title={loading ? "Refreshing models" : "Refresh models"}
              type="button"
            >
              <RefreshCw aria-hidden="true" className={loading ? "icon-spin" : ""} size={18} />
            </button>
          </div>
        </div>

        <div className="field-grid">
          <div className="field">
            <label htmlFor="registry-algorithm-filter">Filter by algorithm</label>
            <input
              id="registry-algorithm-filter"
              onChange={(event) => setAlgorithmFilter(event.target.value)}
              placeholder="e.g. svd"
              value={algorithmFilter}
            />
          </div>
        </div>

        {loading ? (
          <p className="subtle">Loading models...</p>
        ) : filteredModels.length ? (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Model ID</th>
                  <th>Algorithm</th>
                  <th>Version</th>
                  <th>Status</th>
                  <th>Feedback</th>
                  <th>Created</th>
                  <th>Metrics</th>
                </tr>
              </thead>
              <tbody>
                {filteredModels.map((model) => (
                  <tr key={model.model_id}>
                    <td><strong>{model.model_id}</strong></td>
                    <td>{model.algorithm}</td>
                    <td>{model.version || "n/a"}</td>
                    <td>
                      <span className={`status-pill ${model.promoted ? "status-live" : "status-muted"}`}>
                        {model.promoted ? "Promoted" : "Saved"}
                      </span>
                    </td>
                    <td>{model.is_implicit ? "implicit" : "explicit"}</td>
                    <td>{formatTimestamp(model.created_at)}</td>
                    <td>{Object.keys(model.metrics || {}).join(", ") || "n/a"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="subtle">No models are registered for this tenant yet.</p>
        )}

        <div className="actions-row">
          <Link className="btn" to="/deployment">Go To Deployment & Production</Link>
        </div>
      </div>
    </Layout>
  )
}
