import { useCallback, useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import TenantRequiredState from "../components/TenantRequiredState";
import Layout from "../components/Layout";
import { tenantRequest } from "../api";
import { useAsyncData } from "../hooks/useAsyncData";
import { useTenantSession } from "../hooks/useAuthSession";

const OPERATION_META = {
  default: {
    category: "Platform",
    description: "Authenticated control-plane and uncategorized platform traffic.",
    label: "Platform Requests",
  },
  explain: {
    category: "Serving",
    description: "Explanation requests that justify model output.",
    label: "Explainability",
  },
  model_promote: {
    category: "Lifecycle",
    description: "Promotion actions that push a model into serving.",
    label: "Model Promotion",
  },
  recommend: {
    category: "Serving",
    description: "Single-user recommendation traffic.",
    label: "Recommendations",
  },
  recommend_batch: {
    category: "Serving",
    description: "Bulk recommendation generation traffic.",
    label: "Batch Recommendations",
  },
  similar: {
    category: "Serving",
    description: "Similar-item lookup traffic.",
    label: "Similar Items",
  },
  training: {
    category: "Lifecycle",
    description: "Training and retraining jobs.",
    label: "Training Jobs",
  },
};

const CHART_COLORS = ["#ef6b4a", "#0fa38a", "#c8952d", "#38536a", "#7d8aa3", "#a16207"];

function formatCount(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toLocaleString() : "0";
}

function percent(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${numeric.toFixed(1)}%` : "0.0%";
}

function buildOperations(usage) {
  const totals = usage?.totals || {};
  const totalRequests = Object.values(totals).reduce((sum, value) => sum + Number(value || 0), 0);
  return Object.entries(totals)
    .map(([key, count]) => {
      const numericCount = Number(count || 0);
      const meta = OPERATION_META[key] || {
        category: "Platform",
        description: "Tracked tenant activity for a platform endpoint.",
        label: key.replace(/_/g, " "),
      };
      const share = totalRequests > 0 ? (numericCount / totalRequests) * 100 : 0;
      return {
        key,
        count: numericCount,
        share,
        ...meta,
      };
    })
    .sort((a, b) => b.count - a.count);
}

export default function Analytics() {
  const {
    hasTenantApiKey: hasKey,
    tenantApiKey,
    tenantId,
  } = useTenantSession();
  const loadAnalytics = useCallback(({ signal }) => (
    tenantRequest("/analytics", { signal })
  ), [tenantApiKey, tenantId]);
  const {
    data: usage,
    error,
    loading,
  } = useAsyncData(loadAnalytics, {
    enabled: hasKey,
    initialData: null,
  });

  const operations = useMemo(() => buildOperations(usage), [usage]);
  const totalRequests = operations.reduce((sum, item) => sum + item.count, 0);
  const topOperation = operations[0] || null;
  const activeOperationCount = operations.filter((item) => item.count > 0).length;

  const servingRequests = operations
    .filter((item) => item.category === "Serving")
    .reduce((sum, item) => sum + item.count, 0);
  const lifecycleRequests = operations
    .filter((item) => item.category === "Lifecycle")
    .reduce((sum, item) => sum + item.count, 0);
  const platformRequests = operations
    .filter((item) => item.category === "Platform")
    .reduce((sum, item) => sum + item.count, 0);

  const endpointChartData = operations.map((item) => ({ name: item.label, count: item.count }));
  const workflowMix = [
    { name: "Serving", value: servingRequests },
    { name: "Lifecycle", value: lifecycleRequests },
    { name: "Platform", value: platformRequests },
  ].filter((item) => item.value > 0);

  const summary = loading
    ? "Loading API tracking analytics from backend usage counters."
    : totalRequests
      ? "This page is API telemetry only: endpoint usage, traffic mix, and call distribution."
      : "No API tracking data available yet. Trigger tenant API calls to populate analytics.";

  return (
    <Layout
      title="Operational Analytics"
      summary={summary}
      heroEyebrow="API Tracking Analytics"
      heroStats={[
        { label: "Tenant", value: tenantId || usage?.tenant_id || "Not attached" },
        { label: "Total Calls", value: formatCount(totalRequests) },
        { label: "Top Endpoint", value: topOperation?.label || "No traffic yet" },
      ]}
    >
      {!hasKey ? (
        <TenantRequiredState description="Store a tenant key in Tenants before opening API telemetry analytics." />
      ) : null}

      {error ? <div className="form-error">{error}</div> : null}

      <div className="analytics-kpi-grid">
        <div className="card stat-card">
          <p className="eyebrow">Traffic</p>
          <div className="metric">{formatCount(totalRequests)}</div>
          <h3>Total API Calls</h3>
          <p className="subtle">All tracked endpoint calls for the active tenant.</p>
        </div>

        <div className="card stat-card">
          <p className="eyebrow">Coverage</p>
          <div className="metric">{activeOperationCount}</div>
          <h3>Active Endpoints</h3>
          <p className="subtle">Distinct endpoint types with non-zero request count.</p>
        </div>

        <div className="card stat-card">
          <p className="eyebrow">Serving Mix</p>
          <div className="metric">{percent(totalRequests ? (servingRequests / totalRequests) * 100 : 0)}</div>
          <h3>Serving Share</h3>
          <p className="subtle">Percentage of API calls related to recommendation serving.</p>
        </div>

        <div className="card stat-card">
          <p className="eyebrow">Lifecycle Mix</p>
          <div className="metric">{percent(totalRequests ? (lifecycleRequests / totalRequests) * 100 : 0)}</div>
          <h3>Lifecycle Share</h3>
          <p className="subtle">Percentage of API calls related to training and promotion flows.</p>
        </div>
      </div>

      <div className="analytics-layout page-section">
        <div className="card chart-card">
          <div className="card-header">
            <h3>Endpoint Usage</h3>
            <span className="subtle">Calls by endpoint category</span>
          </div>
          {endpointChartData.length ? (
            <div className="chart-shell">
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={endpointChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--line-soft)" vertical={false} />
                  <XAxis dataKey="name" tickLine={false} axisLine={false} />
                  <YAxis tickLine={false} axisLine={false} />
                  <Tooltip
                    contentStyle={{
                      background: "var(--surface-strong)",
                      border: "1px solid var(--line-soft)",
                      borderRadius: 16,
                    }}
                  />
                  <Bar dataKey="count" fill="#ef6b4a" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className="subtle">No endpoint usage data yet.</p>
          )}
        </div>

        <div className="card chart-card">
          <div className="card-header">
            <h3>Workflow Distribution</h3>
            <span className="subtle">Serving vs lifecycle vs platform traffic</span>
          </div>
          {workflowMix.length ? (
            <div className="chart-shell">
              <ResponsiveContainer width="100%" height={320}>
                <PieChart>
                  <Pie
                    data={workflowMix}
                    cx="50%"
                    cy="50%"
                    innerRadius={72}
                    outerRadius={108}
                    dataKey="value"
                    nameKey="name"
                    paddingAngle={3}
                  >
                    {workflowMix.map((entry, index) => (
                      <Cell key={entry.name} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      background: "var(--surface-strong)",
                      border: "1px solid var(--line-soft)",
                      borderRadius: 16,
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className="subtle">No workflow distribution data yet.</p>
          )}
        </div>
      </div>

      <div className="card page-section">
        <div className="card-header">
          <h3>API Tracking Breakdown</h3>
          <span className="subtle">Detailed call mix from backend metering</span>
        </div>
        {operations.length ? (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Endpoint</th>
                  <th>Category</th>
                  <th>Calls</th>
                  <th>Share</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                {operations.map((operation) => (
                  <tr key={operation.key}>
                    <td><strong>{operation.label}</strong></td>
                    <td>{operation.category}</td>
                    <td>{formatCount(operation.count)}</td>
                    <td>{percent(operation.share)}</td>
                    <td>{operation.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="subtle">No API tracking rows available yet.</p>
        )}
      </div>
    </Layout>
  );
}
