import { useNavigate } from "react-router-dom";
import Layout from "../components/Layout";

export default function Datasets(){
  const navigate = useNavigate();

  return(
    <Layout
      summary="Prepare recommendation-ready interaction data before launching file or SQL training jobs."
      title="Dataset Upload"
    >
      <div className="upload">
        <h2>Prepare Training Data</h2>
        <p>
          File upload and database training now run from the training workspace, with tenant-scoped jobs, live polling, and benchmark output.
        </p>
        <button className="btn" onClick={() => navigate("/training")} type="button">
          Open Training Workspace
        </button>
      </div>

      <div className="grid page-section">
        <div className="card">
          <div className="card-header">
            <h3>Supported Inputs</h3>
            <span className="subtle">Production intake paths</span>
          </div>
          <div className="detail-list">
            <div><strong>File training:</strong> CSV, Excel, JSON, Parquet</div>
            <div><strong>Database training:</strong> PostgreSQL, MySQL, SQLite</div>
            <div><strong>Async jobs:</strong> Poll via `/jobs/{'{job_id}'}`</div>
            <div><strong>Best practice:</strong> Keep `user_id`, `item_id`, and interaction score or rating fields consistently typed.</div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3>Expected Dataset Shape</h3>
            <span className="subtle">Minimum interaction schema</span>
          </div>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>User</th>
                  <th>Item</th>
                  <th>Score</th>
                </tr>
              </thead>
              <tbody>
                <tr><td>1024</td><td>Gaming Laptop</td><td>0.92</td></tr>
                <tr><td>2055</td><td>Headphones</td><td>0.81</td></tr>
              </tbody>
            </table>
          </div>
          <p className="field-hint">The training service expects a dense enough interaction table to benchmark multiple candidate models reliably.</p>
        </div>
      </div>
    </Layout>
  )
}
