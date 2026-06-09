
import Layout from "../components/Layout";
import WorkspaceControls from "../components/WorkspaceControls";

export default function Settings(){
  return(
    <Layout
      summary="Tune workspace appearance, operator session behavior, and tenant-access boundaries from one settings surface."
      title="Settings"
    >
      <div className="grid">
        <div className="card">
          <div className="card-header">
            <h3>Workspace</h3>
            <span className="subtle">Core console preferences</span>
          </div>
          <p className="subtle">
            Manage workspace-level preferences, appearance behavior, and your active operator session from one place.
          </p>
          <div className="page-section">
            <WorkspaceControls compact title="Settings" />
          </div>
        </div>
        <div className="card">
          <div className="card-header">
            <h3>API Keys</h3>
            <span className="subtle">Tenant-scoped access</span>
          </div>
          <p className="subtle">
            Use the Tenants page to attach, rotate, and validate tenant API keys for training, benchmarks, and live serving flows.
          </p>
        </div>
        <div className="card">
          <div className="card-header">
            <h3>Theme</h3>
            <span className="subtle">Display behavior</span>
          </div>
          <p className="subtle">
            Theme switching is available here and in the sidebar so operators can change modes without leaving the current workflow.
          </p>
        </div>
        <div className="card">
          <div className="card-header">
            <h3>Permissions</h3>
            <span className="subtle">Access boundaries</span>
          </div>
          <p className="subtle">
            Admin bearer access controls tenant management, while tenant API keys control training, recommendation, and benchmark operations.
          </p>
        </div>
      </div>
    </Layout>
  )
}
