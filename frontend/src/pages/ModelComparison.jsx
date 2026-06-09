import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
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
import TenantRequiredState from "../components/TenantRequiredState";
import Layout from "../components/Layout";
import { tenantRequest } from "../api";
import { useAsyncData } from "../hooks/useAsyncData";
import { useTenantSession } from "../hooks/useAuthSession";
import { formatNumber, normalizeLeaderboardRows } from "../lib/benchmarkUtils";

const COLORS = ["#ef6b4a", "#0fa38a", "#c8952d", "#38536a"];
const INITIAL_COMPARISON_DATA = Object.freeze({
  benchmarkData: null,
  options: null,
});

function productionReadiness(model) {
  const score = Number(model.selectionScore ?? model.composite ?? 0);
  if (score >= 75) return "High";
  if (score >= 50) return "Medium";
  return "Needs validation";
}

function interpretabilityForAlgorithm(algorithm) {
  const name = String(algorithm || "").toLowerCase();
  if (name.includes("svd") || name.includes("item") || name.includes("user")) return "High";
  if (name.includes("als") || name.includes("bpr") || name.includes("nmf")) return "Medium";
  return "Low-Medium";
}

function sparsityHandlingForAlgorithm(algorithm) {
  const name = String(algorithm || "").toLowerCase();
  if (name.includes("als") || name.includes("bpr") || name.includes("implicit")) return "Strong";
  if (name.includes("hybrid") || name.includes("item")) return "Strong";
  return "Moderate";
}

function scalabilityForAlgorithm(algorithm) {
  const name = String(algorithm || "").toLowerCase();
  if (name.includes("als") || name.includes("item") || name.includes("knn")) return "High";
  if (name.includes("svd") || name.includes("nmf")) return "Medium";
  return "Medium-Low";
}

function summarizeStrengths(model) {
  return [model.rankedModel?.performance_summary, model.rankedModel?.fit_summary]
    .filter(Boolean)
    .slice(0, 2)
    .join(" ") || "Strong candidate across ranking and benchmark metrics.";
}

function summarizeWeakness(model) {
  return model.rankedModel?.comparison_to_next || "Requires targeted A/B validation against adjacent ranked models.";
}

export default function ModelComparison() {
  const [selected, setSelected] = useState([]);
  const {
    hasTenantApiKey: hasKey,
    tenantApiKey,
    tenantId,
  } = useTenantSession();
  const loadComparisonData = useCallback(async ({ signal }) => {
    const [benchmarksResult, optionsResult] = await Promise.allSettled([
      tenantRequest("/benchmarks", { signal }),
      tenantRequest("/recommend/options", { signal }),
    ]);

    if (benchmarksResult.status !== "fulfilled") {
      throw benchmarksResult.reason;
    }

    return {
      benchmarkData: benchmarksResult.value || null,
      options: optionsResult.status === "fulfilled" ? optionsResult.value : null,
    };
  }, [tenantApiKey, tenantId]);
  const {
    data,
    error,
    loading,
  } = useAsyncData(loadComparisonData, {
    enabled: hasKey,
    initialData: INITIAL_COMPARISON_DATA,
  });
  const benchmarkData = data?.benchmarkData || null;
  const options = data?.options || null;

  const rows = useMemo(() => {
    const leaderboard = benchmarkData?.leaderboard || [];
    const rankedModels = options?.ranked_models?.length
      ? options.ranked_models
      : benchmarkData?.top_model_recommendations || [];
    return normalizeLeaderboardRows(leaderboard, rankedModels);
  }, [benchmarkData, options]);

  useEffect(() => {
    if (!rows.length) {
      setSelected([]);
      return;
    }

    setSelected((current) => {
      const next = current.filter((algorithm) => rows.some((row) => row.algorithm === algorithm));
      if (next.length) {
        return next.slice(0, 3);
      }
      return rows.slice(0, 3).map((row) => row.algorithm);
    });
  }, [rows]);

  const selectedRows = useMemo(() => rows.filter((row) => selected.includes(row.algorithm)), [rows, selected]);
  const lead = rows[0] || null;

  const radarData = useMemo(() => {
    const metrics = [
      { key: "composite", label: "Composite" },
      { key: "precision", label: "Precision" },
      { key: "recall", label: "Recall" },
      { key: "ndcg", label: "NDCG" },
    ];

    return metrics.map((metric) => {
      const point = { metric: metric.label };
      selectedRows.forEach((row) => {
        const value = Number(row[metric.key]);
        point[row.algorithm] = Number.isFinite(value) ? value : 0;
      });
      return point;
    });
  }, [selectedRows]);

  const gapData = useMemo(() => {
    if (!rows.length) return [];
    const leadScore = Number(rows[0]?.composite ?? 0);
    return rows.map((row) => ({
      algorithm: row.algorithm,
      gap: Number((leadScore - Number(row.composite ?? 0)).toFixed(4)),
    }));
  }, [rows]);

  const summary = loading
    ? "Loading side-by-side model comparison, score gaps, and production-fit signals."
    : rows.length
      ? "Compare model tradeoffs before promotion using benchmark metrics and recommendation policy context."
      : "No benchmark rows are available. Run training first.";

  return (
    <Layout
      title="Model Comparison"
      summary={summary}
      heroEyebrow="Candidate Evaluation"
      heroStats={[
        { label: "Tenant", value: tenantId || "Not attached" },
        { label: "Compared", value: String(selectedRows.length || 0) },
        { label: "Lead", value: lead?.algorithm || "n/a" },
      ]}
    >
      {!hasKey ? (
        <TenantRequiredState description="Attach a tenant key in Tenants to compare candidate models." />
      ) : null}

      {error ? <div className="form-error">{error}</div> : null}

      {hasKey ? (
        <div className="card">
          <div className="card-header">
            <h3>Comparison Scope</h3>
            <span className="subtle">Select up to three models</span>
          </div>
          <div className="actions-row">
            {rows.map((row) => {
              const active = selected.includes(row.algorithm);
              return (
                <button
                  className={active ? "btn" : "btn btn-secondary"}
                  disabled={!active && selected.length >= 3}
                  key={row.algorithm}
                  onClick={() => {
                    setSelected((current) => {
                      if (current.includes(row.algorithm)) {
                        return current.filter((item) => item !== row.algorithm);
                      }
                      return [...current, row.algorithm].slice(0, 3);
                    });
                  }}
                  type="button"
                >
                  {row.algorithm}
                </button>
              );
            })}
          </div>
        </div>
      ) : null}

      {selectedRows.length ? (
        <div className="comparison-card-grid">
          {selectedRows.map((row, idx) => (
            <article className="card comparison-model-card" key={row.algorithm}>
              <div className="card-header">
                <h3>{row.algorithm}</h3>
                <span className={`status-pill ${idx === 0 ? "status-live" : "status-muted"}`}>
                  Rank #{row.rank}
                </span>
              </div>
              <div className="result-stat-strip">
                <div className="result-stat-card">
                  <span>Composite</span>
                  <strong>{formatNumber(row.composite)}</strong>
                </div>
                <div className="result-stat-card">
                  <span>Precision</span>
                  <strong>{formatNumber(row.precision)}</strong>
                </div>
                <div className="result-stat-card">
                  <span>Recall</span>
                  <strong>{formatNumber(row.recall)}</strong>
                </div>
                <div className="result-stat-card">
                  <span>NDCG</span>
                  <strong>{formatNumber(row.ndcg)}</strong>
                </div>
              </div>

              <div className="detail-list">
                <div><strong>Scalability:</strong> {scalabilityForAlgorithm(row.algorithm)}</div>
                <div><strong>Sparsity handling:</strong> {sparsityHandlingForAlgorithm(row.algorithm)}</div>
                <div><strong>Production readiness:</strong> {productionReadiness(row)}</div>
                <div><strong>Interpretability:</strong> {interpretabilityForAlgorithm(row.algorithm)}</div>
                <div><strong>Strengths:</strong> {summarizeStrengths(row)}</div>
                <div><strong>Weaknesses:</strong> {summarizeWeakness(row)}</div>
                <div><strong>Recommendation reasons:</strong> {row.rankedModel?.reason || "No additional reason provided."}</div>
              </div>
            </article>
          ))}
        </div>
      ) : null}

      {selectedRows.length ? (
        <div className="benchmark-grid">
          <div className="card chart-card">
            <div className="card-header">
              <h3>Capability Radar</h3>
              <span className="subtle">Side-by-side model strengths</span>
            </div>
            <div className="chart-shell">
              <ResponsiveContainer width="100%" height={320}>
                <RadarChart cx="50%" cy="50%" outerRadius="78%" data={radarData}>
                  <PolarGrid stroke="var(--line-soft)" />
                  <PolarAngleAxis dataKey="metric" tick={{ fill: "var(--muted)", fontSize: 12 }} />
                  <PolarRadiusAxis tick={false} axisLine={false} />
                  {selectedRows.map((row, idx) => (
                    <Radar
                      key={row.algorithm}
                      name={row.algorithm}
                      dataKey={row.algorithm}
                      stroke={COLORS[idx % COLORS.length]}
                      fill={COLORS[idx % COLORS.length]}
                      fillOpacity={0.3}
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
              <h3>Score Gap Analysis</h3>
              <span className="subtle">Distance from current leader</span>
            </div>
            <div className="chart-shell">
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={gapData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--line-soft)" vertical={false} />
                  <XAxis dataKey="algorithm" tickLine={false} axisLine={false} />
                  <YAxis tickLine={false} axisLine={false} />
                  <Tooltip
                    contentStyle={{
                      background: "var(--surface-strong)",
                      border: "1px solid var(--line-soft)",
                      borderRadius: 16,
                    }}
                  />
                  <Bar dataKey="gap" radius={[10, 10, 0, 0]}>
                    {gapData.map((entry, index) => (
                      <Cell key={`${entry.algorithm}-cell`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      ) : null}

      {rows.length ? (
        <div className="card">
          <div className="card-header">
            <h3>Comparison Matrix</h3>
            <span className="subtle">Ranking heatmap and score deltas</span>
          </div>
          <div className="table-wrap">
            <table className="table comparison-matrix-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Rank</th>
                  <th>Composite</th>
                  <th>Selection %</th>
                  <th>Score diff vs lead</th>
                  <th>Heat</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => {
                  const leadComposite = Number(rows[0]?.composite ?? 0);
                  const value = Number(row.composite ?? 0);
                  const diff = leadComposite - value;
                  const heat = Math.max(0.05, Math.min(1, value || 0));
                  return (
                    <tr key={`${row.algorithm}-${row.modelId || row.rank}`}>
                      <td><strong>{row.algorithm}</strong></td>
                      <td>#{row.rank}</td>
                      <td>{formatNumber(row.composite)}</td>
                      <td>{row.selectionScore !== null ? `${row.selectionScore.toFixed(2)}%` : "n/a"}</td>
                      <td>{diff > 0 ? `-${diff.toFixed(4)}` : "Leader"}</td>
                      <td>
                        <div className="matrix-heat-cell" style={{ background: `rgba(15,163,138,${heat})` }}>
                          {Math.round(heat * 100)}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}
    </Layout>
  );
}
