import { useEffect, useMemo, useState } from "react";
import { RefreshCw } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import Layout from "../components/Layout";
import { tenantContextRequest, tenantRequest } from "../api";
import { setTenantContext } from "../auth";
import { useTenantSession } from "../hooks/useAuthSession";
import ModelDecisionPanel from "../components/ModelDecisionPanel";
import { normalizeLeaderboardRows } from "../lib/benchmarkUtils";

function formatMetricValue(value, digits = 4) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : "n/a";
}

function resolvePrimaryMetric(row = {}) {
  const candidateKeys = ["NDCG@K", "RMSE", "MAP@K", "Precision@K", "Recall@K", "MAE"];
  return candidateKeys.find((key) => row[key] !== null && row[key] !== undefined) || "Composite Score";
}

function buildLeadNarrative(model, selectionPolicy) {
  if (!model) {
    return "The benchmark winner explanation will appear here after training and ranking data are available.";
  }
  return [
    model.performance_summary,
    model.fit_summary,
    model.reliability_summary,
    model.comparison_to_next || selectionPolicy?.serving_reason,
  ]
    .filter(Boolean)
    .join(" ");
}

function trainingStorageKey(tenantId) {
  return `neurorec_active_training_job_${tenantId || "default"}`;
}

function readActiveJobId(tenantId) {
  if (typeof window === "undefined") {
    return "";
  }
  return window.localStorage.getItem(trainingStorageKey(tenantId)) || "";
}

export default function Benchmarks() {
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [tenantWarning, setTenantWarning] = useState("");
  const [options, setOptions] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdatedAt, setLastUpdatedAt] = useState(null);
  const [activeJobId, setActiveJobId] = useState("");
  const [lastSyncedJobId, setLastSyncedJobId] = useState("");
  const [jobStatus, setJobStatus] = useState("");

  const {
    hasTenantApiKey: hasKey,
    tenantApiKey,
    tenantId,
  } = useTenantSession();

  async function fetchBenchmarks(sourceJobId = "") {
    if (!hasKey) {
      setBenchmarkData(null);
      setOptions(null);
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError("");
      setTenantWarning("");

      const [contextResult, benchmarksResult, recommendationOptionsResult] = await Promise.allSettled([
        tenantContextRequest(),
        tenantRequest(`/benchmarks?_ts=${Date.now()}`),
        tenantRequest("/recommend/options"),
      ]);

      if (benchmarksResult.status !== "fulfilled") {
        throw benchmarksResult.reason;
      }

      const context = contextResult.status === "fulfilled" ? contextResult.value : null;
      const benchmarks = benchmarksResult.value;
      const recommendationOptions = recommendationOptionsResult.status === "fulfilled"
        ? recommendationOptionsResult.value
        : null;

      if (context?.tenant_id && context.tenant_id !== tenantId) {
        if (tenantApiKey) {
          setTenantContext({ apiKey: tenantApiKey, tenantId: context.tenant_id });
        }
        setTenantWarning(
          `The active API key is authenticated as ${context.tenant_id}. The saved tenant label was ${tenantId || "not set"}, so the UI has been synchronized to the verified tenant.`
        );
      }

      setBenchmarkData(benchmarks || null);
      setOptions(recommendationOptions || null);
      setLastUpdatedAt(Date.now());
      if (sourceJobId) {
        setLastSyncedJobId(sourceJobId);
      }
    } catch (err) {
      setError(err.message || "Failed to fetch benchmarks");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchBenchmarks();
  }, [hasKey, tenantApiKey, tenantId]);

  useEffect(() => {
    if (!hasKey) {
      return undefined;
    }

    const currentJobId = readActiveJobId(tenantId);
    setActiveJobId(currentJobId);
    if (!currentJobId) {
      return undefined;
    }

    let cancelled = false;
    let timeoutId = null;

    async function watch() {
      try {
        const job = await tenantRequest(`/jobs/${encodeURIComponent(currentJobId)}`);
        if (cancelled) {
          return;
        }

        setJobStatus(job?.status || "");

        if (job?.status === "pending" || job?.status === "running") {
          timeoutId = window.setTimeout(watch, 2500);
          return;
        }

        await fetchBenchmarks(currentJobId);
      } catch {
        // keep benchmark view resilient
      }
    }

    watch();
    return () => {
      cancelled = true;
      if (timeoutId) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [hasKey, tenantId]);

  async function refreshNow() {
    setRefreshing(true);
    try {
      await fetchBenchmarks();
    } finally {
      setRefreshing(false);
    }
  }

  const leaderboard = benchmarkData?.leaderboard || [];
  const rankedModels = options?.ranked_models?.length
    ? options.ranked_models
    : benchmarkData?.top_model_recommendations || [];
  const selectionPolicy = options?.selection_policy || benchmarkData?.model_selection_policy || {};
  const primaryMetric = selectionPolicy?.primary_metric || resolvePrimaryMetric(leaderboard[0]);
  const normalizedLeaderboard = useMemo(
    () => normalizeLeaderboardRows(leaderboard, rankedModels),
    [leaderboard, rankedModels],
  );
  const benchmarkSummary = loading
    ? "Loading the latest benchmark leaderboard and selection policy."
    : hasKey
      ? leaderboard.length
        ? `${benchmarkData?.best_algorithm || leaderboard[0]?.Algorithm} currently leads ${leaderboard.length} benchmarked models for tenant ${tenantId || "current session"}.`
        : "No benchmark results are available yet. Run training to generate production ranking evidence."
      : "Attach a tenant API key to unlock benchmark and promotion workflows.";

  const heroStats = [
    { label: "Tenant", value: tenantId || "Not attached" },
    { label: "Candidates", value: String(leaderboard.length || 0) },
    { label: "Lead Model", value: benchmarkData?.best_algorithm || "n/a" },
  ];

  const chartData = useMemo(() => {
    return normalizedLeaderboard.map((row) => ({
      algorithm: row.algorithm,
      composite: Number(row.composite) || 0,
      rank: row.rank,
      selectionScore: Number(row.selectionScore) || 0,
      time: Number(row.trainingTime) || 0,
      primaryMetric: Number(row.raw?.[primaryMetric]) || 0,
    }));
  }, [normalizedLeaderboard, primaryMetric]);

  const radarData = useMemo(() => {
    if (!leaderboard.length) return [];
    const topModels = leaderboard.slice(0, 3);
    const metrics = ["Performance Score", "Speed Score", "Scalability Score", "Composite Score"];

    return metrics.map((metricName) => {
      const point = { metric: metricName };
      topModels.forEach((model) => {
        point[model.Algorithm] = Number(model[metricName]) || 0;
      });
      return point;
    });
  }, [leaderboard]);

  const leadRow = leaderboard[0];
  const leadModel = normalizedLeaderboard[0]?.rankedModel || rankedModels[0] || selectionPolicy?.selected_models?.[0] || null;
  const leadNarrative = buildLeadNarrative(leadModel, selectionPolicy);
  const datasetShape = benchmarkData?.dataset_shape || [];
  const datasetStats = benchmarkData?.dataset_stats || {};

  return (
    <Layout
      heroStats={heroStats}
      summary={benchmarkSummary}
      title="Benchmark & Model Comparison"
    >
      {!hasKey ? (
        <div className="card empty-state">
          <h3>Tenant API key required</h3>
          <p className="subtle">
            Store a tenant key in Tenants before comparing benchmark runs or promoting models.
          </p>
        </div>
      ) : null}

      {tenantWarning ? <div className="success-banner">{tenantWarning}</div> : null}
      {error ? <div className="form-error">{error}</div> : null}

        <div className="card">
          <div className="card-header">
            <h3>Benchmark Sync Status</h3>
          <button
            aria-label={refreshing ? "Refreshing benchmarks" : "Refresh benchmarks"}
            className="btn btn-secondary btn-icon"
            disabled={loading || refreshing}
            onClick={refreshNow}
            title={refreshing ? "Refreshing benchmarks" : "Refresh benchmarks"}
            type="button"
          >
            <RefreshCw aria-hidden="true" className={refreshing ? "icon-spin" : ""} size={18} />
          </button>
        </div>
        <div className="detail-list">
          <div><strong>Last updated:</strong> {lastUpdatedAt ? new Date(lastUpdatedAt).toLocaleString() : "n/a"}</div>
          <div><strong>Active training job:</strong> {activeJobId || "none"}</div>
          <div><strong>Last synced job:</strong> {lastSyncedJobId || "none"}</div>
          <div><strong>Active job status:</strong> {jobStatus || "n/a"}</div>
        </div>
      </div>

      {hasKey ? (
        <section className="mission-panel">
          <div>
            <p className="eyebrow">Model Lead Narrative</p>
            <h3>{benchmarkData?.best_algorithm || "No leader yet"}</h3>
            <div className="explanation-stack">
              {leadModel?.performance_summary ? (
                <div className="explanation-item">
                  <span>Performance</span>
                  <strong>{leadModel.performance_summary}</strong>
                </div>
              ) : null}
              {leadModel?.fit_summary ? (
                <div className="explanation-item">
                  <span>Dataset fit</span>
                  <strong>{leadModel.fit_summary}</strong>
                </div>
              ) : null}
              {leadModel?.reliability_summary ? (
                <div className="explanation-item">
                  <span>Reliability</span>
                  <strong>{leadModel.reliability_summary}</strong>
                </div>
              ) : null}
              {!leadModel ? <p className="subtle">{leadNarrative}</p> : null}
            </div>
          </div>
          <div className="mission-callout">
            <span className="status-dot"></span>
            <strong>
              {leadRow
                ? `${leadRow.Algorithm} is ranked #1 with ${primaryMetric} ${formatMetricValue(leadRow[primaryMetric])} and composite ${formatMetricValue(leadRow["Composite Score"])}.`
                : "Train a tenant model to populate leaderboard and production comparison signals."}
            </strong>
          </div>
        </section>
      ) : null}

      {hasKey && leaderboard.length ? (
        <div className="benchmark-grid">
          <div className="card chart-card">
            <div className="card-header">
              <h3>Model Capabilities Radar</h3>
              <span className="subtle">Comparative analysis of top 3 candidates</span>
            </div>
            <div className="chart-shell">
              <ResponsiveContainer width="100%" height={320}>
                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                  <PolarGrid stroke="var(--line-soft)" />
                  <PolarAngleAxis dataKey="metric" tick={{ fill: "var(--muted)", fontSize: 12 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 1]} tick={false} axisLine={false} />
                  {leaderboard.slice(0, 3).map((model, idx) => (
                    <Radar
                      key={model.Algorithm}
                      name={model.Algorithm}
                      dataKey={model.Algorithm}
                      stroke={idx === 0 ? "#ef6b4a" : idx === 1 ? "#0fa38a" : "#c8952d"}
                      fill={idx === 0 ? "#ef6b4a" : idx === 1 ? "#0fa38a" : "#c8952d"}
                      fillOpacity={0.4}
                    />
                  ))}
                  <Tooltip
                    contentStyle={{
                      background: "var(--surface-strong)",
                      border: "1px solid var(--line-soft)",
                      borderRadius: 16,
                    }}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="card chart-card">
            <div className="card-header">
              <h3>Selection Confidence</h3>
              <span className="subtle">Composite score and API serving confidence</span>
            </div>
            <div className="chart-shell">
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={chartData} barCategoryGap={18}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--line-soft)" vertical={false} />
                  <XAxis dataKey="algorithm" tickLine={false} axisLine={false} />
                  <YAxis tickLine={false} axisLine={false} />
                  <Tooltip
                    cursor={{ fill: "rgba(239,107,74,.08)" }}
                    contentStyle={{
                      background: "var(--surface-strong)",
                      border: "1px solid var(--line-soft)",
                      borderRadius: 16,
                    }}
                  />
                  <Bar dataKey="selectionScore" fill="#ef6b4a" radius={[10, 10, 0, 0]} name="Selection Score %" />
                  <Bar dataKey="composite" fill="#0fa38a" radius={[10, 10, 0, 0]} name="Composite Score" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="card chart-card">
            <div className="card-header">
              <h3>Quality vs Cost</h3>
              <span className="subtle">Primary metric vs training time (s)</span>
            </div>
            <div className="chart-shell">
              <ResponsiveContainer width="100%" height={320}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--line-soft)" vertical={false} />
                  <XAxis dataKey="algorithm" tickLine={false} axisLine={false} />
                  <YAxis yAxisId="left" tickLine={false} axisLine={false} />
                  <YAxis yAxisId="right" orientation="right" tickLine={false} axisLine={false} />
                  <Tooltip
                    contentStyle={{
                      background: "var(--surface-strong)",
                      border: "1px solid var(--line-soft)",
                      borderRadius: 16,
                    }}
                  />
                  <Line yAxisId="left" type="monotone" dataKey="primaryMetric" stroke="var(--ink)" strokeWidth={3} dot={{ r: 4 }} name={primaryMetric} />
                  <Line yAxisId="right" type="monotone" dataKey="time" stroke="#c8952d" strokeWidth={3} dot={{ r: 4 }} name="Training Time (s)" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="card chart-card">
            <div className="card-header">
              <h3>Expected Dataset Shape</h3>
              <span className="subtle">Live SDK/backend/frontend contract readiness</span>
            </div>
            {datasetShape.length ? (
              <>
                <div className="chart-shell">
                  <ResponsiveContainer width="100%" height={320}>
                    <BarChart data={datasetShape} layout="vertical" margin={{ left: 16, right: 12 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--line-soft)" horizontal={false} />
                      <XAxis type="number" domain={[0, 100]} tickLine={false} axisLine={false} />
                      <YAxis dataKey="field" type="category" tickLine={false} axisLine={false} width={90} />
                      <Tooltip
                        formatter={(value, key, payload) => {
                          if (key === "readiness") {
                            return [`${value}%`, payload?.payload?.requirement || "Contract"];
                          }
                          return [value, key];
                        }}
                        contentStyle={{
                          background: "var(--surface-strong)",
                          border: "1px solid var(--line-soft)",
                          borderRadius: 16,
                        }}
                      />
                      <Bar dataKey="readiness" fill="#38536a" radius={[0, 10, 10, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="detail-list">
                  <div><strong>Rows:</strong> {datasetStats?.rows ?? "n/a"}</div>
                  <div><strong>Columns:</strong> {datasetStats?.columns ?? "n/a"}</div>
                  <div><strong>Unique users:</strong> {datasetStats?.users ?? "n/a"}</div>
                  <div><strong>Unique items:</strong> {datasetStats?.items ?? "n/a"}</div>
                </div>
              </>
            ) : (
              <p className="subtle">Dataset contract is unavailable until a training run completes with source columns.</p>
            )}
          </div>
        </div>
      ) : null}

      {hasKey && (rankedModels.length || selectionPolicy?.selected_models?.length) ? (
        <div className="card page-section">
          <div className="card-header">
            <h3>Model Decision Intelligence</h3>
            <span className="subtle">Explanation returned by the recommendation API policy</span>
          </div>
          <ModelDecisionPanel
            bestAlgorithm={benchmarkData?.best_algorithm}
            bestModelId={benchmarkData?.best_model_id}
            options={options || { ranked_models: rankedModels, selection_policy: selectionPolicy }}
            policy={selectionPolicy}
            resolvedMode={benchmarkData?.resolved_mode}
          />
        </div>
      ) : null}

      <div className="card">
        <div className="card-header">
          <h3>Leaderboard</h3>
          <span className="subtle">Compare ranking quality before promotion</span>
        </div>

        {loading ? (
          <p className="subtle">Loading benchmarks...</p>
        ) : error ? (
          <p className="subtle">{error}</p>
        ) : !hasKey ? (
          <p className="subtle">Tenant benchmark actions are disabled until a tenant key is attached.</p>
        ) : leaderboard.length === 0 ? (
          <p className="subtle">No benchmark results yet. Train a model to see results here.</p>
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Algorithm</th>
                  <th>Time (s)</th>
                  <th>Composite Score</th>
                  <th>{primaryMetric}</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {leaderboard.map((row, idx) => (
                  <tr key={row.model_id || idx}>
                    <td>
                      <strong>{row.Algorithm}</strong>
                      <div className="subtle table-meta">{row.model_id}</div>
                    </td>
                    <td>{formatMetricValue(row["Time (s)"], 2)}</td>
                    <td>{formatMetricValue(row["Composite Score"])}</td>
                    <td>{formatMetricValue(row[primaryMetric])}</td>
                    <td>
                      <span className={`status-pill ${idx === 0 ? "status-live" : "status-muted"}`}>
                        {row.Promoted ? "Promoted" : idx === 0 ? "Lead candidate" : "Candidate"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </Layout>
  );
}
