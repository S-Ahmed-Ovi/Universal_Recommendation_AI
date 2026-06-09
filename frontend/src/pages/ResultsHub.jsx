import { Link } from "react-router-dom";
import { BarChart3, Code2, Rocket, TestTube2 } from "lucide-react";
import TenantRequiredState from "../components/TenantRequiredState";
import Layout from "../components/Layout";
import { tenantRequest } from "../api";
import { useCallback, useMemo } from "react";
import { useAsyncData } from "../hooks/useAsyncData";
import { useTenantSession } from "../hooks/useAuthSession";

const FLOW_STEPS = [
  {
    title: "Benchmark Results",
    description: "Validate leaderboard quality, ranking confidence, and benchmark charts.",
    to: "/benchmarks",
    cta: "Benchmark Results",
    icon: BarChart3,
  },
  {
    title: "Validation & Deploy",
    description: "Test recommendations, promote versions, and manage rollback readiness.",
    to: "/deployment",
    cta: "Deploy Model",
    icon: Rocket,
  },
  {
    title: "SDK & API Integration",
    description: "Copy production snippets, verify API auth, and complete integration checklist.",
    to: "/integrations",
    cta: "SDK Integration",
    icon: Code2,
  },
];

const QUICK_ACTIONS = [
  { label: "Test Recommendations", to: "/api" },
  { label: "API Usage", to: "/api" },
  { label: "Deploy Model", to: "/deployment" },
  { label: "SDK Integration", to: "/integrations" },
];
const INITIAL_HEALTH = Object.freeze({
  hasBenchmarks: false,
  hasModels: false,
  hasPromoted: false,
});

export default function ResultsHub() {
  const {
    hasTenantApiKey: hasKey,
    tenantApiKey,
    tenantId,
  } = useTenantSession();
  const loadHealth = useCallback(async ({ signal }) => {
    const [benchmarksResult, modelsResult] = await Promise.allSettled([
      tenantRequest("/benchmarks", { signal }),
      tenantRequest("/models", { signal }),
    ]);

    const leaderboard =
      benchmarksResult.status === "fulfilled" ? benchmarksResult.value?.leaderboard || [] : [];
    const models =
      modelsResult.status === "fulfilled" ? modelsResult.value?.models || [] : [];

    return {
      hasBenchmarks: leaderboard.length > 0,
      hasModels: models.length > 0,
      hasPromoted: models.some((model) => model.promoted),
    };
  }, [tenantApiKey, tenantId]);
  const {
    data: health,
    loading,
  } = useAsyncData(loadHealth, {
    enabled: hasKey,
    initialData: INITIAL_HEALTH,
  });

  const readinessChecklist = useMemo(() => ([
    {
      label: "Benchmark generated",
      done: health.hasBenchmarks,
      detail: health.hasBenchmarks ? "Leaderboard is available." : "Run or complete training first.",
    },
    {
      label: "Model artifacts available",
      done: health.hasModels,
      detail: health.hasModels ? "Registry contains saved versions." : "No model artifacts found yet.",
    },
    {
      label: "Production model promoted",
      done: health.hasPromoted,
      detail: health.hasPromoted ? "An active production model is set." : "Promote a candidate in Deployment.",
    },
    {
      label: "Integration path ready",
      done: hasKey,
      detail: hasKey ? "Tenant key is configured for SDK/API use." : "Add tenant key in Tenants page.",
    },
  ]), [health.hasBenchmarks, health.hasModels, health.hasPromoted, hasKey]);

  return (
    <Layout
      title="Post-Training Results Navigator"
      summary="Move through benchmark validation, deployment, and integration in a guided production workflow."
      heroEyebrow="Training Completed Workflow"
      heroStats={[
        { label: "Tenant", value: tenantId || "Not attached" },
        { label: "Flow Stages", value: String(FLOW_STEPS.length) },
        { label: "Readiness", value: loading ? "Checking..." : readinessChecklist.filter((item) => item.done).length + "/4 complete" },
      ]}
    >
      {!hasKey ? (
        <TenantRequiredState description="Add a tenant key in Tenants to continue through benchmark, deployment, and integration flows." />
      ) : null}

      

      <div className="results-flow-grid">
        {FLOW_STEPS.map((step, index) => (
          <article className="card result-step-card" key={step.title}>
            <div className="result-step-head">
              <span className="status-pill status-muted">Step {index + 1}</span>
              <step.icon size={18} />
            </div>
            <h3>{step.title}</h3>
            <p className="subtle">{step.description}</p>
            <Link className="btn" to={step.to}>{step.cta}</Link>
          </article>
        ))}
      </div>

      <div className="card">
        <div className="card-header">
          <h3>Production Readiness Checklist</h3>
          <span className="subtle">Execution-focused handoff checklist</span>
        </div>
        <div className="detail-list">
          {readinessChecklist.map((item) => (
            <div key={item.label}>
              <strong>{item.done ? "Done" : "Pending"}:</strong> {item.label} - {item.detail}
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h3>Quick Actions</h3>
          <span className="subtle">Direct jumps for operator workflows</span>
        </div>
        <div className="actions-row">
          {QUICK_ACTIONS.map((item) => (
            <Link className="btn btn-secondary" key={item.label} to={item.to}>{item.label}</Link>
          ))}
          <Link className="btn btn-secondary" to="/training">
            <TestTube2 size={16} />&nbsp;Back to Training Jobs
          </Link>
        </div>
      </div>
    </Layout>
  );
}
