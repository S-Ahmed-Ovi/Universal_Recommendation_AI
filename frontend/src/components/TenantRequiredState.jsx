import { Link } from "react-router-dom";

export default function TenantRequiredState({
  actionLabel = "Open Tenants",
  description,
  title = "Tenant API key required",
  to = "/tenants",
}) {
  return (
    <div className="card empty-state" role="status" aria-live="polite">
      <h3>{title}</h3>
      <p className="subtle">{description}</p>
      {actionLabel ? <Link className="btn" to={to}>{actionLabel}</Link> : null}
    </div>
  );
}
