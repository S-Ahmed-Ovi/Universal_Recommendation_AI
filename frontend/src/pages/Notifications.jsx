import Layout from "../components/Layout";

const notices = [
  {
    title: "Deployment success",
    detail: "Transformer v4.2 cleared validation and is ready for controlled serving traffic.",
    status: "Ready",
    tone: "status-live",
  },
  {
    title: "Billing threshold warning",
    detail: "Nova Retail is approaching its current tenant usage ceiling and may require a higher plan tier.",
    status: "Attention",
    tone: "status-error",
  },
  {
    title: "Latency anomaly",
    detail: "Recommendation API response times briefly moved above the normal workspace baseline during the last observation window.",
    status: "Review",
    tone: "status-muted",
  },
];

export default function Notifications(){
  return(
    <Layout
      summary="Track rollout events, tenant warnings, and operational anomalies from one notification stream."
      title="Notification Center"
    >
      <div className="grid">
        {notices.map((notice) => (
          <article className="card card-nested" key={notice.title}>
            <div className="card-header">
              <div>
                <p className="eyebrow">Workspace event</p>
                <h3>{notice.title}</h3>
              </div>
              <span className={`status-pill ${notice.tone}`}>{notice.status}</span>
            </div>
            <p className="subtle">{notice.detail}</p>
          </article>
        ))}
      </div>
    </Layout>
  )
}
