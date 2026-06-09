import { useEffect, useState } from "react";
import { Copy, KeyRound, Webhook } from "lucide-react";
import { Link } from "react-router-dom";
import TenantRequiredState from "../components/TenantRequiredState";
import Layout from "../components/Layout";
import { useTenantSession } from "../hooks/useAuthSession";

const SNIPPETS = {
  javascript: `const response = await fetch("https://api.example.com/recommend", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": process.env.PROACTIVE_AI_KEY,
  },
  body: JSON.stringify({
    user_id: "u_1024",
    top_n: 10,
    strategy: "best_promoted_model",
  }),
});

const payload = await response.json();
console.log(payload.recommendations);`,
  python: `import requests

resp = requests.post(
  "https://api.example.com/recommend",
  headers={"X-API-Key": "YOUR_TENANT_KEY"},
  json={"user_id": "u_1024", "top_n": 10},
  timeout=10,
)
print(resp.json())`,
  react: `import { useEffect, useState } from "react";

export function useRecommendations(userId, apiKey) {
  const [data, setData] = useState([]);
  useEffect(() => {
    if (!userId || !apiKey) return;
    fetch("/recommend", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": apiKey,
      },
      body: JSON.stringify({ user_id: userId, top_n: 10 }),
    })
      .then((res) => res.json())
      .then((payload) => setData(payload.recommendations || []));
  }, [userId, apiKey]);
  return data;
}`,
  webhook: `from sdk.webhooks.push import verify_signature

# in your webhook handler
def handle_webhook(body: bytes, signature: str, secret: str):
    if not verify_signature(body, secret, signature):
        raise ValueError("invalid signature")
    # process event payload`,
  tracking: `// impression
window.dispatchEvent(new CustomEvent("pai.impression", {
  detail: { user_id: "u_1024", item_id: "i_43", model_id: "svd_2026_04" }
}));

// click
window.dispatchEvent(new CustomEvent("pai.click", {
  detail: { user_id: "u_1024", item_id: "i_43", position: 1 }
}));`,
  embed: `<div id="proactive-recommendations"></div>
<script>
  window.ProactiveAIConfig = {
    apiUrl: "https://api.example.com",
    apiKey: "YOUR_TENANT_KEY",
    userId: "u_1024",
    containerId: "proactive-recommendations",
    topN: 10,
    theme: "light"
  };
</script>
<script src="/sdk/js_widget/index.js"></script>`,
};

const SNIPPET_CARDS = [
  {
    key: "javascript",
    label: "JavaScript SDK",
    language: "JavaScript",
    description: "Server-side request example for production recommendation fetches.",
  },
  {
    key: "python",
    label: "Python API",
    language: "Python",
    description: "Backend integration example for service-to-service recommendation calls.",
  },
  {
    key: "react",
    label: "React Hook",
    language: "React",
    description: "Client-side consumption pattern for rendering recommendations in-app.",
  },
  {
    key: "webhook",
    label: "Webhook Verify",
    language: "Python",
    description: "Signature validation flow for inbound tenant webhook events.",
  },
  {
    key: "tracking",
    label: "Event Tracking",
    language: "JavaScript",
    description: "Impression and click instrumentation for analytics forwarding.",
  },
  {
    key: "embed",
    label: "Recommendation Carousel",
    language: "HTML",
    description: "Drop-in widget bootstrap for fast storefront or app embedding.",
  },
].map((snippet) => ({
  ...snippet,
  code: SNIPPETS[snippet.key],
}));

async function copyText(value) {
  if (typeof window === "undefined" || !window.navigator?.clipboard) {
    return "Clipboard not available";
  }

  try {
    await window.navigator.clipboard.writeText(value);
    return "Copied";
  } catch {
    return "Copy failed";
  }
}

function SnippetCard({ isCopied, onCopy, snippet }) {
  const titleId = `${snippet.key}-snippet-title`;
  const descriptionId = `${snippet.key}-snippet-description`;

  return (
    <article
      aria-describedby={descriptionId}
      aria-labelledby={titleId}
      className="card snippet-card"
    >
      <div className="snippet-card-header">
        <div className="snippet-card-copy">
          <p className="snippet-card-eyebrow">{snippet.language}</p>
          <h3 id={titleId}>{snippet.label}</h3>
          <p className="subtle snippet-card-description" id={descriptionId}>
            {snippet.description}
          </p>
        </div>
        <button
          aria-label={`Copy ${snippet.label} code sample`}
          className="btn btn-secondary snippet-copy-button"
          onClick={() => onCopy(snippet)}
          type="button"
        >
          <Copy aria-hidden="true" size={16} />
          <span>{isCopied ? "Copied" : "Copy Snippet"}</span>
        </button>
      </div>

      <div className="snippet-card-body">
        <pre
          aria-label={`${snippet.label} code sample`}
          className="code-block snippet-code-block"
          tabIndex={0}
        >
          <code>{snippet.code}</code>
        </pre>
      </div>
    </article>
  );
}

export default function Integrations() {
  const { hasTenantApiKey: hasKey, tenantId } = useTenantSession();
  const [copyFeedback, setCopyFeedback] = useState({ key: "", message: "" });

  useEffect(() => {
    if (!copyFeedback.message) {
      return undefined;
    }

    const timeoutId = window.setTimeout(() => {
      setCopyFeedback((current) => (
        current.message === copyFeedback.message ? { key: "", message: "" } : current
      ));
    }, 2400);

    return () => window.clearTimeout(timeoutId);
  }, [copyFeedback.message]);

  async function handleCopy(snippet) {
    const message = await copyText(snippet.code);

    setCopyFeedback({
      key: message === "Copied" ? snippet.key : "",
      message: `${snippet.label}: ${message}`,
    });
  }

  return (
    <Layout
      title="SDK & Integration Center"
      summary="Production integration workspace for SDK usage, API auth, webhooks, real-time recommendations, and deployment handoff."
      heroEyebrow="Integration Readiness"
      heroStats={[
        { label: "Tenant", value: tenantId || "Not attached" },
        { label: "Runtime", value: hasKey ? "Tenant linked" : "Tenant required" },
        { label: "API Key", value: hasKey ? "Configured" : "Missing" },
      ]}
    >
      {!hasKey ? (
        <TenantRequiredState
          actionLabel="Open Tenant Key Management"
          description="Save a tenant key before validating SDK and endpoint calls."
        />
      ) : null}

      <div
        aria-atomic="true"
        aria-live="polite"
        className={`integration-status${copyFeedback.message ? " integration-status-active" : ""}`}
        role="status"
      >
        {copyFeedback.message || "\u00A0"}
      </div>

      <div className="grid page-section">
        <div className="card">
          <div className="card-header">
            <h3><KeyRound size={18} /> API Key Management</h3>
          </div>
          <div className="detail-list">
            <div><strong>Current tenant:</strong> {tenantId || "none"}</div>
            <div><strong>Authentication:</strong> Tenant endpoints require `X-API-Key`.</div>
            <div><strong>Rotation:</strong> Rotate keys in Multi-Tenant Management and redeploy clients.</div>
            <div><strong>Security:</strong> Keep keys server-side for production SSR or backend-for-frontend proxies.</div>
          </div>
          <div className="actions-row">
            <Link className="btn btn-secondary" to="/tenants">Manage Keys</Link>
            <Link className="btn btn-secondary" to="/api">Open Endpoint Explorer</Link>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3><Webhook size={18} /> Deployment Guide</h3>
          </div>
          <div className="detail-list">
            <div><strong>1.</strong> Train and benchmark candidate models.</div>
            <div><strong>2.</strong> Promote selected model in Deployment Center.</div>
            <div><strong>3.</strong> Validate `/recommend` and `/similar` in API Playground.</div>
            <div><strong>4.</strong> Register tenant webhook for `job.completed` and `model.promoted` events.</div>
            <div><strong>5.</strong> Embed JS widget or use SDK hooks in your app.</div>
            <div><strong>6.</strong> Enable click/impression event forwarding in your analytics stack.</div>
          </div>
        </div>
      </div>

      <section aria-label="Integration code samples" className="snippet-grid page-section">
        {SNIPPET_CARDS.map((snippet) => (
          <SnippetCard
            isCopied={copyFeedback.key === snippet.key}
            key={snippet.key}
            onCopy={handleCopy}
            snippet={snippet}
          />
        ))}
      </section>

      <div className="card page-section">
        <div className="card-header">
          <h3>Endpoint Explorer</h3>
          <span className="subtle">Primary production endpoints</span>
        </div>
        <div className="detail-list">
          <div><strong>Recommendations:</strong> `POST /recommend`</div>
          <div><strong>Batch recommendations:</strong> `POST /recommend/batch`</div>
          <div><strong>Similar items:</strong> `GET /similar/{`{item_id}`}`</div>
          <div><strong>Explainability:</strong> `GET /explain/{`{user_id}`}/{`{item_id}`}`</div>
          <div><strong>Options:</strong> `GET /recommend/options`</div>
          <div><strong>Model promotion:</strong> `POST /models/{`{model_id}`}/promote`</div>
        </div>
        <div className="actions-row">
          <Link className="btn" to="/api">Test API Usage</Link>
        </div>
      </div>
    </Layout>
  );
}
