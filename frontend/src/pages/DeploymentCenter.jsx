import { useCallback, useMemo, useState } from "react";
import { Activity, RefreshCw } from "lucide-react";
import { Link } from "react-router-dom";
import Layout from "../components/Layout";
import TenantRequiredState from "../components/TenantRequiredState";
import { formatTimestamp, tenantRequest } from "../api";
import { useAsyncData } from "../hooks/useAsyncData";
import { useTenantSession } from "../hooks/useAuthSession";

function sortByCreatedDesc(models = []) {
  return [...models].sort((a, b) => Number(b.created_at || 0) - Number(a.created_at || 0));
}

function eventTone(status) {
  if (status === "active" || status === "healthy") {
    return "status-live";
  }
  if (status === "failed") {
    return "status-error";
  }
  return "status-muted";
}

const emptyDeploymentState = {
  deployment: null,
  history: [],
  models: [],
};

export default function DeploymentCenter() {
  const [actionError, setActionError] = useState("");
  const [workingAction, setWorkingAction] = useState("");
  const [message, setMessage] = useState("");
  const {
    hasTenantApiKey: hasKey,
    tenantApiKey,
    tenantId,
  } = useTenantSession();
  const loadDeploymentState = useCallback(async ({ signal }) => {
    const [deploymentResult, historyResult, modelResult] = await Promise.all([
      tenantRequest("/deployment/active", { signal }),
      tenantRequest("/deployment/history?limit=12", { signal }),
      tenantRequest("/models", { signal }),
    ]);

    return {
      deployment: deploymentResult || null,
      history: historyResult?.events || [],
      models: modelResult?.models || [],
    };
  }, [tenantApiKey, tenantId]);
  const {
    data,
    error,
    loading: isLoading,
    reload,
  } = useAsyncData(loadDeploymentState, {
    enabled: hasKey,
    initialData: emptyDeploymentState,
  });
  const deployment = data?.deployment || null;
  const models = data?.models || [];
  const history = data?.history || [];

  const orderedModels = useMemo(() => sortByCreatedDesc(models), [models]);
  const activeModel = deployment?.active_deployment || null;
  const rollbackCandidate = deployment?.rollback_candidate || null;
  const health = deployment?.health || null;
  const standbyModels = useMemo(
    () => orderedModels.filter((model) => model.model_id !== activeModel?.model_id).slice(0, 8),
    [orderedModels, activeModel],
  );

  async function promoteModel(modelId) {
    if (!modelId) return;
    setWorkingAction(`promote:${modelId}`);
    setActionError("");
    setMessage("");
    try {
      await tenantRequest(`/models/${encodeURIComponent(modelId)}/promote`, { method: "POST" });
      setMessage(`Model ${modelId} is now the active production deployment.`);
      await reload();
    } catch (actionError) {
      setActionError(actionError.message || "Promotion failed");
    } finally {
      setWorkingAction("");
    }
  }

  async function rollbackDeployment() {
    setWorkingAction("rollback");
    setActionError("");
    setMessage("");
    try {
      await tenantRequest("/deployment/rollback", { method: "POST" });
      setMessage("Rollback completed and the previous active deployment was restored.");
      await reload();
    } catch (actionError) {
      setActionError(actionError.message || "Rollback failed");
    } finally {
      setWorkingAction("");
    }
  }

  const healthState = health?.status_label || "No active deployment";
  const pageError = actionError || error;
  const pageLoading = isLoading;
  const summary = pageLoading
    ? "Loading canonical deployment state, health, and release history."
    : activeModel
      ? `Active deployment is ${activeModel.algorithm} (${activeModel.model_id}). This page reflects the backend deployment control plane, not raw registry state.`
      : "No active deployment yet. Promote one validated model to switch default production traffic.";

  return (
    <Layout
      title="Deployment & Production"
      summary={summary}
      heroEyebrow="Model Release Control"
      heroStats={[
        { label: "Tenant", value: tenantId || "Not attached" },
        { label: "Active Model", value: activeModel?.algorithm || "None" },
        { label: "Health", value: healthState },
      ]}
    >
      {!hasKey ? (
        <TenantRequiredState description="Attach a tenant key to manage deployment and production model versions." />
      ) : null}

      {message ? <div className="success-banner">{message}</div> : null}
      {pageError ? <div className="form-error">{pageError}</div> : null}

      <div className="grid">
        <div className="card">
          <div className="card-header">
            <h3>Active Deployment</h3>
            <span className={`status-pill ${eventTone(health?.status)}`}>{healthState}</span>
          </div>
          {activeModel ? (
            <div className="detail-list">
              <div><strong>Active algorithm:</strong> {activeModel.algorithm}</div>
              <div><strong>Model ID:</strong> {activeModel.model_id}</div>
              <div><strong>Version:</strong> {activeModel.version || "n/a"}</div>
              <div><strong>Promoted at:</strong> {formatTimestamp(activeModel.promoted_at)}</div>
              <div><strong>Model saved at:</strong> {formatTimestamp(activeModel.created_at)}</div>
              <div><strong>Loaded model:</strong> {health?.loaded_model_id || "Warming on next request"}</div>
              <div><strong>Metric keys:</strong> {Object.keys(activeModel.metrics || {}).join(", ") || "n/a"}</div>
              {health?.last_error ? <div><strong>Last error:</strong> {health.last_error}</div> : null}
            </div>
          ) : (
            <p className="subtle">No active deployment. Promote one from the standby candidates below.</p>
          )}

          <div className="actions-row">
            <button
              aria-label={pageLoading ? "Refreshing deployment status" : "Refresh deployment status"}
              className="btn btn-secondary btn-icon"
              disabled={pageLoading}
              onClick={() => reload()}
              title={pageLoading ? "Refreshing deployment status" : "Refresh deployment status"}
              type="button"
            >
              <RefreshCw aria-hidden="true" className={pageLoading ? "icon-spin" : ""} size={18} />
            </button>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3>Rollback Control</h3>
            <span className="subtle">Previous active release only</span>
          </div>
          {rollbackCandidate ? (
            <div className="detail-list">
              <div><strong>Rollback candidate:</strong> {rollbackCandidate.model_id}</div>
              <div><strong>Algorithm:</strong> {rollbackCandidate.algorithm}</div>
              <div><strong>Version:</strong> {rollbackCandidate.version || "n/a"}</div>
              <div><strong>Saved:</strong> {formatTimestamp(rollbackCandidate.created_at)}</div>
            </div>
          ) : (
            <p className="subtle">No previous active deployment is available for rollback.</p>
          )}
          <div className="actions-row">
            <button
              className="btn"
              disabled={!rollbackCandidate || Boolean(workingAction)}
              onClick={rollbackDeployment}
              type="button"
            >
              {workingAction === "rollback" ? "Rolling back..." : "Rollback"}
            </button>
          </div>
        </div>
      </div>

      <div className="card page-section">
        <div className="card-header">
          <h3>Standby Candidates</h3>
          <span className="subtle">Latest saved models, excluding the active deployment</span>
        </div>

        {pageLoading ? (
          <p className="subtle">Loading standby models...</p>
        ) : standbyModels.length ? (
          <>
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>Model ID</th>
                    <th>Algorithm</th>
                    <th>Version</th>
                    <th>Saved</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {standbyModels.map((model) => (
                    <tr key={model.model_id}>
                      <td><strong>{model.model_id}</strong></td>
                      <td>{model.algorithm}</td>
                      <td>{model.version || "n/a"}</td>
                      <td>{formatTimestamp(model.created_at)}</td>
                      <td>
                        <button
                          className="btn"
                          disabled={workingAction === `promote:${model.model_id}`}
                          onClick={() => promoteModel(model.model_id)}
                          type="button"
                        >
                          {workingAction === `promote:${model.model_id}` ? "Promoting..." : "Promote"}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="actions-row">
              <Link className="btn btn-secondary" to="/registry">Open Full Registry Inventory</Link>
            </div>
          </>
        ) : (
          <p className="subtle">No standby models are available yet.</p>
        )}
      </div>

      <div className="card page-section">
        <div className="card-header">
          <h3>Deployment History</h3>
          <span className="subtle">Authoritative deployment events from the backend</span>
        </div>
        {history.length ? (
          <div className="deployment-log-list">
            {history.map((event) => (
              <div className="deployment-log-row" key={event.event_id}>
                <span className={`status-pill ${eventTone(event.status)}`}>
                  <Activity size={12} />&nbsp;{event.status}
                </span>
                <div>
                  <strong>{event.message || event.event_type}</strong>
                  <p className="subtle">
                    {formatTimestamp(event.created_at)} · {event.algorithm || "n/a"} · {event.model_id || "n/a"}
                  </p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="subtle">No deployment events logged yet.</p>
        )}
      </div>
    </Layout>
  );
}
