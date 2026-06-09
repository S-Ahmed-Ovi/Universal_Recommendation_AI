import { useMemo } from "react";
import Sidebar from "./Sidebar";
import WorkspaceControls from "./WorkspaceControls";
import { useAdminProfile } from "../hooks/useAdminProfile";
import { useTenantSession } from "../hooks/useAuthSession";

const pageDescriptions = {
  "AI Training Pipeline": "Launch training jobs from file uploads or direct SQL pipelines.",
  "Analytics Dashboard": "Read adoption, latency, and model behavior from one command surface.",
  "Graph & Analytics": "Track sparsity, tuning progress, score distribution, and training telemetry in interactive charts.",
  "Operational Analytics": "Read adoption, latency, and model behavior from one command surface.",
  "API Playground": "Test tenant-authenticated endpoints against the live serving stack.",
  "Benchmark Results": "Rank candidate models with sortable metrics and production-focused leaderboard visualizations.",
  "Model Comparison": "Review side-by-side model tradeoffs before promotion decisions.",
  "Post-Training Results Navigator": "Navigate benchmark, comparison, analytics, and integration steps after training completes.",
  "Deployment & Production": "Manage active versions, promotions, rollback, and deployment health.",
  "SDK & Integration Center": "Validate SDK readiness, auth flows, snippets, and deployment integration guidance.",
  "Benchmark & Model Comparison": "Rank candidate models before promoting them to production.",
  "Dashboard Overview": "Monitor workspace health, tenant posture, and model readiness.",
  "Dataset Upload": "Prepare input data and route it into the training workspace.",
  "Model Registry": "Inspect saved models and decide what serves production traffic.",
  "Multi-Tenant Management": "Provision tenants, rotate keys, and govern client access.",
  "Notification Center": "Track deployment, billing, and platform events in one stream.",
  "Settings": "Tune workspace defaults, access boundaries, and operator preferences.",
};

function formatOperatorName(user) {
  const value = (user || "").trim();
  if (!value) {
    return "Administrator";
  }
  if (value.toLowerCase() === "admin") {
    return "Administrator";
  }
  return value
    .split(/[\s._-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function buildInitials(user) {
  const parts = (user || "Administrator")
    .split(/[\s._-]+/)
    .filter(Boolean);
  const joined = parts.slice(0, 2).map((part) => part.charAt(0).toUpperCase()).join("");
  return joined || "AD";
}

export default function Layout({ children, title, summary, heroEyebrow = "Enterprise Recommendation Suite", heroStats }){
  const { profile } = useAdminProfile();
  const { hasTenantApiKey: tenantReady, tenantId } = useTenantSession();

  const operatorName = formatOperatorName(profile.user);
  const initials = buildInitials(operatorName);
  const isPlatformAdmin = Boolean(profile.is_platform_admin);
  const accessLabel = tenantReady
    ? "Tenant Session"
    : isPlatformAdmin
      ? "Admin Access"
      : (profile.role_label || "User");
  const tenantSummary = tenantReady
    ? `Tenant scope: ${tenantId}`
    : isPlatformAdmin
      ? "Admin access: tenant management enabled"
      : "User session: create or attach a tenant";
  const operatorBadge = accessLabel;

  const defaultHeroStats = useMemo(() => ([
    { label: "Operator", value: operatorName },
    { label: "Access", value: tenantReady ? tenantId : accessLabel },
    { label: "Mode", value: tenantReady ? "Tenant ready" : (isPlatformAdmin ? "Admin only" : "User only") },
  ]), [accessLabel, isPlatformAdmin, operatorName, tenantId, tenantReady]);

  return(
    <div className="app-shell">
      <div className="ambient ambient-one"></div>
      <div className="ambient ambient-two"></div>
      <Sidebar />

      <main className="content">
        <div className="topbar">
          <div className="topbar-breadcrumb">Proactive AI / Workspace</div>
          <div className="topbar-profile">
            <div className="profile-chip">
              <div className="avatar">{initials}</div>
              <div className="profile-meta">
                <p className="profile-kicker">{accessLabel}</p>
                <strong>{operatorName}</strong>
                <span>{tenantSummary}</span>
              </div>
              <div className="profile-status">
                <span className={`status-pill ${tenantReady ? "status-live" : "status-muted"}`}>
                  {operatorBadge}
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="mobile-workspace-actions">
          <WorkspaceControls compact title="Quick Actions" />
        </div>

        <section className="page-hero">
          <div className="page-hero-copy">
            <p className="eyebrow">{heroEyebrow}</p>
            <h2 className="page-title">{title}</h2>
            <p className="page-summary">
              {summary || pageDescriptions[title] || "Operate your recommendation stack from a single premium workspace."}
            </p>
          </div>

          <div className="page-hero-band">
            {(heroStats || defaultHeroStats).map((stat) => (
              <div className="hero-stat" key={stat.label}>
                <span>{stat.label}</span>
                <strong>{stat.value}</strong>
              </div>
            ))}
          </div>
        </section>

        <div className="page-body section-stack">
          {children}
        </div>
      </main>
    </div>
  )
}
