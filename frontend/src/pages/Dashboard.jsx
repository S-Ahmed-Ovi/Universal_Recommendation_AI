import { useEffect, useState } from "react";
import Layout from "../components/Layout";
import { adminRequest, formatTimestamp, publicRequest, tenantContextRequest, tenantRequest } from "../api";
import { setTenantContext } from "../auth";
import { useTenantSession } from "../hooks/useAuthSession";

function buildDashboardNarrative(model) {
  if (!model) {
    return "This dashboard now adapts to tenant readiness, training posture, and the promoted-model strategy instead of relying on fixed operator copy.";
  }
  return [model.performance_summary, model.fit_summary, model.reliability_summary]
    .filter(Boolean)
    .join(" ");
}

function scoreLevel(value, mapping) {
  return mapping[value] ?? 0;
}

function normalizeAlgorithmName(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "");
}

function businessFitScore(algorithm, benchmarkRow) {
  const readiness = scoreLevel(algorithm.production_readiness, { low: 35, medium: 65, high: 90 });
  const scalability = scoreLevel(algorithm.scalability, { low: 30, medium: 65, high: 90 });
  const robustness = scoreLevel(algorithm.robustness, { low: 30, medium: 65, high: 90 });
  const speed = scoreLevel(algorithm.training_speed, { slow: 40, medium: 70, fast: 90 });
  const benchmarkBoost = benchmarkRow ? Math.min(10, Math.max(0, Number(benchmarkRow["Composite Score"] || 0) * 10)) : 0;
  const raw = 0.32 * readiness + 0.24 * scalability + 0.22 * robustness + 0.22 * speed + benchmarkBoost;
  return Math.round(Math.min(100, raw));
}

function businessUseCase(algorithm) {
  const domain = String(algorithm.domain || "").toLowerCase();
  const feedback = String(algorithm.feedback || "").toLowerCase();
  if (domain === "ecommerce") return "Conversion and cross-sell optimization";
  if (domain === "movie") return "Session depth and watch-time growth";
  if (feedback === "implicit") return "High-volume behavioral recommendation";
  if (feedback === "explicit") return "Rating-quality personalization";
  return "Balanced general-purpose recommendation";
}

function rolloutPriority(score) {
  if (score >= 85) return "Primary rollout";
  if (score >= 70) return "Controlled rollout";
  return "Experiment track";
}

function deliveryRisk(algorithm) {
  const readiness = String(algorithm.production_readiness || "");
  const speed = String(algorithm.training_speed || "");
  if (readiness === "high" && speed !== "slow") return "Low";
  if (readiness === "medium" || speed === "slow") return "Medium";
  return "High";
}

export default function Dashboard(){
  const {
    hasTenantApiKey: hasKey,
    tenantApiKey,
    tenantId,
  } = useTenantSession();
  const [state, setState] = useState({
    algorithms: [],
    benchmarkData: null,
    error: "",
    loading: true,
    modelData: null,
    profile: null,
    session: null,
    tenantWarning: "",
    tenants: [],
    usageData: null,
  });

  useEffect(() => {
    let ignore = false;

    async function load() {
      setState((current) => ({ ...current, error: "", loading: true }));

      try {
        let verifiedTenantId = tenantId;
        let tenantWarning = "";
        if (hasKey) {
          try {
            const context = await tenantContextRequest();
            if (context?.tenant_id && context.tenant_id !== tenantId) {
              verifiedTenantId = context.tenant_id;
              if (tenantApiKey) {
                setTenantContext({
                  apiKey: tenantApiKey,
                  tenantId: context.tenant_id,
                });
              }
              tenantWarning = `The active API key belongs to ${context.tenant_id}. The saved tenant label has been synchronized.`;
            } else if (context?.tenant_id) {
              verifiedTenantId = context.tenant_id;
            }
          } catch {
            verifiedTenantId = tenantId;
          }
        }

        const sessionPath = verifiedTenantId
          ? `/admin/session/status?tenant_id=${encodeURIComponent(verifiedTenantId)}`
          : "/admin/session/status";
        const [profile, tenants, algorithms, session] = await Promise.all([
          adminRequest("/admin/auth/me"),
          adminRequest("/admin/tenants"),
          publicRequest("/admin/algorithms"),
          publicRequest(sessionPath),
        ]);

        let benchmarkData = null;
        let modelData = null;
        let usageData = null;

        if (hasKey) {
          const [benchmarksResult, modelsResult, usageResult] = await Promise.allSettled([
            tenantRequest("/benchmarks"),
            tenantRequest("/models"),
            tenantRequest("/analytics"),
          ]);
          benchmarkData = benchmarksResult.status === "fulfilled" ? benchmarksResult.value : null;
          modelData = modelsResult.status === "fulfilled" ? modelsResult.value : null;
          usageData = usageResult.status === "fulfilled" ? usageResult.value : null;
        }

        if (!ignore) {
          setState({
            algorithms: algorithms?.algorithms || [],
            benchmarkData,
            error: "",
            loading: false,
            modelData,
            profile,
            session,
            tenantWarning,
            tenants: tenants?.tenants || [],
            usageData,
          });
        }
      } catch (error) {
        if (!ignore) {
          setState((current) => ({
            ...current,
            error: error.message || "Unable to load dashboard data",
            loading: false,
          }));
        }
      }
    }

    load();
    return () => {
      ignore = true;
    };
  }, [hasKey, tenantApiKey, tenantId]);

  const { algorithms, benchmarkData, error, loading, modelData, profile, session, tenantWarning, tenants, usageData } = state;
  const benchmarkRows = benchmarkData?.leaderboard || [];
  const models = modelData?.models || [];
  const promotedModels = models.filter((model) => model.promoted);
  const totalApiCalls = Object.values(usageData?.totals || {}).reduce((sum, value) => sum + Number(value || 0), 0);
  const leadModel =
    session?.recommendation_options?.ranked_models?.[0] ||
    session?.recommendation_options?.selected_models?.[0] ||
    session?.recommendation_options?.supported_models?.[0] ||
    null;
  const recommendedStrategy = session?.recommendation_options?.selection_policy?.recommended_strategy || "pending";
  const nextActions = tenantId
    ? session?.dataset_loaded
      ? [
          `Validate the promoted ${session?.last_best_algorithm || "model"} in Benchmarks before serving it broadly.`,
          `Use the API Playground with tenant ${tenantId} to verify recommendation payloads and explanations.`,
          `Review ranked candidate models to confirm whether ${recommendedStrategy} is still the right serving strategy.`,
        ]
      : [
          `Start a file or SQL training run for tenant ${tenantId} to populate live benchmark data.`,
          "Review the dataset schema before launching training so the user and item columns resolve cleanly.",
          "Come back here after training to inspect the winning model explanation and serving policy.",
        ]
    : [
        "Create or attach a tenant key before using training, registry, benchmark, or API validation flows.",
        "Use Tenants to bind a client context and unlock scoped API requests.",
        "After tenant setup, run an initial training job to hydrate the serving workspace.",
      ];
  const dashboardSummary = loading
    ? "Loading current workspace status, tenant context, and serving posture."
    : tenantId
      ? `Tenant ${tenantId} has ${benchmarkRows.length} benchmarked models, ${models.length} saved models, and ${totalApiCalls} tracked API calls.`
      : "Admin control plane is active. Attach a tenant to unlock model training and production benchmark workflows.";
  const heroStats = [
    { label: "Tenant", value: tenantId || "Not attached" },
    { label: "Benchmarks", value: String(benchmarkRows.length || 0) },
    { label: "API Calls", value: String(totalApiCalls || 0) },
  ];

  const metrics = [
    {
      label: "Operator Session",
      value: loading ? "..." : profile?.user || "admin",
      hint: "Admin bearer access",
    },
    {
      label: "Tenant Context",
      value: loading ? "..." : tenantId || "Not attached",
      hint: tenantId ? "Ready for live tenant APIs" : "Attach a key in Tenants",
    },
    {
      label: "Saved Models",
      value: loading ? "..." : String(models.length),
      hint: promotedModels.length ? `${promotedModels.length} promoted` : "No promoted model",
    },
    {
      label: "Benchmarked Models",
      value: loading ? "..." : String(benchmarkRows.length),
      hint: benchmarkData?.best_algorithm || "No benchmark leader yet",
    },
  ];

  const benchmarkByAlgorithm = new Map(
    benchmarkRows.map((row) => [normalizeAlgorithmName(row.Algorithm), row])
  );

  const businessCatalog = algorithms.map((algorithm) => {
    const benchmarkRow = benchmarkByAlgorithm.get(normalizeAlgorithmName(algorithm.name));
    const fitScore = businessFitScore(algorithm, benchmarkRow);
    return {
      ...algorithm,
      benchmarkRow,
      fitScore,
      useCase: businessUseCase(algorithm),
      priority: rolloutPriority(fitScore),
      risk: deliveryRisk(algorithm),
    };
  }).sort((a, b) => b.fitScore - a.fitScore);

  return(
    <Layout
      heroStats={heroStats}
      summary={dashboardSummary}
      title="Dashboard Overview"
    >
      {tenantWarning ? <div className="success-banner">{tenantWarning}</div> : null}
      {error ? <div className="form-error">{error}</div> : null}

      <section className="mission-panel">
        <div>
          <p className="eyebrow">Mission Snapshot</p>
          <h3>{session?.dataset_loaded ? "The workspace is carrying live model context." : "The workspace is ready for its next training cycle."}</h3>
          <p className="subtle">
            {buildDashboardNarrative(leadModel)}
          </p>
        </div>
        <div className="mission-callout">
          <span className="status-dot"></span>
          <strong className="mission-callout-copy">
            {tenantId
              ? `${tenantId} is attached${session?.last_best_algorithm ? ` and ${session.last_best_algorithm} is the latest leading algorithm.` : "."}`
              : "Attach a tenant key to enable training and serving flows."}
          </strong>
        </div>
      </section>

      <div className="grid">
        {metrics.map((metric)=>(
          <div className="card stat-card" key={metric.label}>
            <div className="eyebrow">{metric.label}</div>
            <div className="metric metric-truncate" title={metric.value}>{metric.value}</div>
            <p className="subtle metric-hint" title={metric.hint}>{metric.hint}</p>
          </div>
        ))}
      </div>

      <div className="grid page-section">
        <div className="card">
          <div className="card-header">
            <h3>Runtime Session</h3>
            <span className="status-pill status-muted">{session?.dataset_loaded ? "Hot" : "Waiting"}</span>
          </div>
          <div className="detail-list">
            <div><strong>Last best algorithm:</strong> {session?.last_best_algorithm || "n/a"}</div>
            <div><strong>Resolved mode:</strong> {session?.last_mode || "n/a"}</div>
            <div><strong>Cached serving pipelines:</strong> {(session?.cached_serving_algorithms || []).length}</div>
            <div><strong>Strategies:</strong> {(session?.recommendation_options?.strategies || []).join(", ") || "n/a"}</div>
            <div><strong>Supported models:</strong> {String(session?.recommendation_options?.supported_model_count || 0)}</div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3>Recommended Next Actions</h3>
            <span className="eyebrow">Dynamic workflow</span>
          </div>
          <div className="detail-list">
            {nextActions.map((action, index) => (
              <div key={action}><strong>{index + 1}.</strong> {action}</div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid page-section">
        <div className="card">
          <div className="card-header">
            <h3>Tenants In View</h3>
            <span className="subtle">{tenants.length} configured</span>
          </div>
          {tenants.length ? (
            <div className="tenant-stack">
              {tenants.slice(0, 5).map((tenant) => (
                <div className="tenant-row" key={tenant.tenant_id}>
                  <div className="tenant-row-main">
                    <strong title={tenant.display_name}>{tenant.display_name}</strong>
                    <p title={tenant.tenant_id}>{tenant.tenant_id}</p>
                  </div>
                  <div className="tenant-meta">
                    <span>{tenant.tier}</span>
                    <span>{tenant.algorithm_mode}</span>
                    <span>{tenant.active ? "active" : "inactive"}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="subtle">No tenants have been created yet.</p>
          )}
        </div>

        <div className="card">
          <div className="card-header">
            <h3>Session Metadata</h3>
            <span className="subtle">{algorithms.length} algos available</span>
          </div>
          <div className="detail-list">
            <div><strong>Token expires:</strong> {profile?.exp ? formatTimestamp(profile.exp) : "n/a"}</div>
            <div><strong>Selection policy:</strong> {session?.recommendation_options?.selection_policy?.recommended_strategy || "No policy active"}</div>
            {leadModel ? (
              <div className="explanation-stack">
                <div className="explanation-item">
                  <span>Performance</span>
                  <strong>{leadModel.performance_summary || "n/a"}</strong>
                </div>
                <div className="explanation-item">
                  <span>Dataset fit</span>
                  <strong>{leadModel.fit_summary || "n/a"}</strong>
                </div>
                <div className="explanation-item">
                  <span>Reliability</span>
                  <strong>{leadModel.reliability_summary || "n/a"}</strong>
                </div>
                {leadModel.comparison_to_next ? (
                  <div className="explanation-item">
                    <span>Why it stays ahead</span>
                    <strong>{leadModel.comparison_to_next}</strong>
                  </div>
                ) : null}
              </div>
            ) : (
              <div><strong>Recommended model explanation:</strong> No promoted model yet</div>
            )}
          </div>
        </div>
      </div>

      <div className="card page-section">
        <div className="card-header">
          <h3>Algorithm Business Catalog</h3>
          <span className="subtle">Business-fit ranking from live algorithm metadata and benchmark context</span>
        </div>
        <div className="catalog-grid">
          {businessCatalog.slice(0, 10).map((algorithm) => (
            <article className="catalog-card" key={algorithm.name}>
              <p className="eyebrow">{algorithm.domain}</p>
              <h4>{algorithm.name}</h4>
              <p>{algorithm.description}</p>
              <div className="detail-list">
                <div><strong>Business fit:</strong> {algorithm.fitScore}/100</div>
                <div><strong>Use case:</strong> {algorithm.useCase}</div>
                <div><strong>Rollout priority:</strong> {algorithm.priority}</div>
                <div><strong>Delivery risk:</strong> {algorithm.risk}</div>
                <div><strong>Benchmark context:</strong> {algorithm.benchmarkRow ? `Composite ${Number(algorithm.benchmarkRow["Composite Score"] || 0).toFixed(4)}` : "No benchmark row yet"}</div>
              </div>
              <div className="catalog-meta">
                <span>{algorithm.feedback}</span>
                <span>{algorithm.training_speed}</span>
                <span>{algorithm.production_readiness}</span>
              </div>
            </article>
          ))}
        </div>
      </div>
    </Layout>
  )
}
