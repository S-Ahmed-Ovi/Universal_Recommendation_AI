import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { RefreshCw } from "lucide-react";
import Layout from "../components/Layout";
import { adminRequest, formatTimestamp, maskSecret, verifyTenantApiKey } from "../api";
import {
  clearTenantContext,
  setAuthToken,
  setTenantContext,
} from "../auth";
import { useAsyncData } from "../hooks/useAsyncData";
import { useAdminProfile } from "../hooks/useAdminProfile";
import { useAuthSession, useTenantSession } from "../hooks/useAuthSession";

const initialForm = {
  algorithm_mode: "auto",
  display_name: "",
  notes: "",
  rate_limit_rpm: 60,
  tenant_id: "",
  tier: "starter",
};
const INITIAL_TENANT_PAYLOAD = Object.freeze({ tenants: [] });

function validateTenantDraft(form) {
  const nextErrors = {};
  if (!form.tenant_id.trim()) {
    nextErrors.tenant_id = "Tenant ID is required.";
  }
  const rateLimit = Number(form.rate_limit_rpm);
  if (!Number.isFinite(rateLimit) || rateLimit <= 0) {
    nextErrors.rate_limit_rpm = "Rate limit RPM must be greater than zero.";
  }
  return nextErrors;
}

export default function Tenants(){
  const [form, setForm] = useState(initialForm);
  const [actionError, setActionError] = useState("");
  const [fieldErrors, setFieldErrors] = useState({});
  const [manualWorking, setManualWorking] = useState(false);
  const [secret, setSecret] = useState(null);
  const createRequestRef = useRef(false);
  const { isAuthenticated } = useAuthSession();
  const { tenantApiKey, tenantId } = useTenantSession();
  const { profile, loading: profileLoading } = useAdminProfile();
  const [manualTenantId, setManualTenantId] = useState(tenantId);
  const [manualApiKey, setManualApiKey] = useState(tenantApiKey || "");
  const [savedMessage, setSavedMessage] = useState("");
  const [workingTenant, setWorkingTenant] = useState("");
  const loadTenants = useCallback(({ signal }) => (
    adminRequest("/admin/tenants", { signal })
  ), []);
  const {
    data: tenantPayload,
    error,
    loading,
    reload,
  } = useAsyncData(loadTenants, {
    initialData: INITIAL_TENANT_PAYLOAD,
  });
  const tenants = tenantPayload?.tenants || [];
  const isPlatformAdmin = Boolean(profile?.is_platform_admin);
  const activeTenantId = profile?.tenant_id || tenantId || "";
  const existingTenant = tenants[0] || null;
  const hasExistingTenant = Boolean(!isPlatformAdmin && existingTenant);
  const createBusy = workingTenant === "create";
  const createDisabled = useMemo(
    () => !isAuthenticated || createBusy || loading || profileLoading || hasExistingTenant,
    [createBusy, hasExistingTenant, isAuthenticated, loading, profileLoading],
  );

  useEffect(() => {
    setManualTenantId(tenantId);
    setManualApiKey(tenantApiKey || "");
  }, [tenantApiKey, tenantId]);

  // Auto-load tenant context if the user has an assigned tenant and it's not yet in session
  useEffect(() => {
    if (loading || !tenants.length || isPlatformAdmin) {
      return;
    }

    const assignedTenantId = profile?.tenant_id;
    if (!assignedTenantId) {
      return;
    }

    const matched = tenants.find((t) => t.tenant_id === assignedTenantId);
    if (matched && matched.api_key && (!tenantApiKey || tenantId !== assignedTenantId)) {
      setTenantContext({
        apiKey: matched.api_key,
        tenantId: matched.tenant_id,
      });
      setSavedMessage(`Automatically loaded credentials for tenant ${matched.tenant_id}.`);
    }
  }, [loading, tenants, profile, tenantId, tenantApiKey, isPlatformAdmin]);

  useEffect(() => {
    if (secret?.api_key && (secret.tenant_id === activeTenantId || !activeTenantId)) {
      const nextTenantId = secret.tenant_id || "";
      setTenantContext({
        apiKey: secret.api_key,
        tenantId: nextTenantId,
      });
    }
  }, [secret, activeTenantId]);

  function isTenantActive(tenant) {
    return Boolean(
      tenant?.tenant_id
      && tenant.tenant_id === activeTenantId
      && tenantApiKey
    );
  }

  function updateForm(key, value) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  async function handleCreate(event) {
    event.preventDefault();
    setActionError("");
    setFieldErrors({});
    setSavedMessage("");

    if (createRequestRef.current) {
      return;
    }

    if (!isAuthenticated) {
      setActionError("Sign in before creating a tenant.");
      return;
    }

    if (hasExistingTenant) {
      setSavedMessage(`Tenant ${existingTenant.tenant_id} already exists. Showing the tenant overview.`);
      return;
    }

    const nextErrors = validateTenantDraft(form);
    if (Object.keys(nextErrors).length) {
      setFieldErrors(nextErrors);
      return;
    }

    createRequestRef.current = true;
    setWorkingTenant("create");

    try {
      const data = await adminRequest("/admin/tenants", {
        body: {
          ...form,
          rate_limit_rpm: Number(form.rate_limit_rpm),
        },
        method: "POST",
      });
      if (data.access_token) {
        setAuthToken(data.access_token);
      }
      setSecret(data);
      setManualTenantId(data.tenant_id || "");
      setManualApiKey(data.api_key || "");
      setTenantContext({
        apiKey: data.api_key || "",
        tenantId: data.tenant_id || "",
      });
      setSavedMessage(`Created tenant ${data.tenant_id} and stored its API key for this browser.`);
      setForm(initialForm);
      await reload({ silent: true });
    } catch (createError) {
      const existing = createError?.payload?.detail?.tenant;
      if (createError?.status === 409 && existing?.tenant_id) {
        setSecret(null);
        setForm(initialForm);
        setSavedMessage(`Tenant ${existing.tenant_id} already exists. Showing the tenant overview.`);
        await reload({ silent: true });
        return;
      }
      setActionError(createError.message || "Unable to create tenant");
    } finally {
      createRequestRef.current = false;
      setWorkingTenant("");
    }
  }

  async function handleRotateKey(tenantId) {
    if (workingTenant) {
      return;
    }
    setActionError("");
    setSavedMessage("");
    setWorkingTenant(tenantId);

    try {
      const data = await adminRequest(`/admin/tenants/${encodeURIComponent(tenantId)}/rotate-key`, {
        method: "POST",
      });
      setSecret(data);
    } catch (rotateError) {
      setActionError(rotateError.message || "Unable to rotate API key");
    } finally {
      setWorkingTenant("");
    }
  }

  async function persistManualKey() {
    setSavedMessage("");
    setActionError("");
    setFieldErrors({});

    if (!manualApiKey.trim()) {
      setFieldErrors({ manual_api_key: "Tenant API key is required." });
      return;
    }

    try {
      setManualWorking(true);
      const context = await verifyTenantApiKey(manualApiKey.trim());
      const resolvedTenantId = context?.tenant_id || manualTenantId.trim();
      setManualTenantId(resolvedTenantId);
      setTenantContext({
        apiKey: manualApiKey.trim(),
        tenantId: resolvedTenantId,
      });
      setSavedMessage(`Stored tenant API key for ${resolvedTenantId}.`);
    } catch (saveError) {
      setActionError(saveError.message || "Unable to verify tenant API key");
    } finally {
      setManualWorking(false);
    }
  }

  function useReturnedKey() {
    if (!secret?.api_key) {
      return;
    }

    const nextTenantId = secret.tenant_id || "";
    setManualTenantId(nextTenantId);
    setManualApiKey(secret.api_key);
    setTenantContext({
      apiKey: secret.api_key,
      tenantId: nextTenantId,
    });
    setSavedMessage(`Stored tenant API key for ${nextTenantId}.`);
  }

  const pageError = actionError || error;

  return(
    <Layout
      summary="Provision tenants, validate API keys, and control which client context is active across training and serving pages."
      title="Multi-Tenant Management"
    >
      {pageError ? <div className="form-error">{pageError}</div> : null}
      {savedMessage ? <div className="success-banner">{savedMessage}</div> : null}

      <div className="grid">
        <div className="card">
          <div className="card-header">
            <h3>Tenant API Key</h3>
          </div>
          <div className="field-grid">
            <div className="field">
              <label htmlFor="manual-tenant-id">Tenant ID</label>
              <input
                id="manual-tenant-id"
                onChange={(event) => setManualTenantId(event.target.value)}
                placeholder="my-client"
                value={manualTenantId}
              />
              <p className="field-hint">Optional. The verified tenant ID will override this label if the key belongs to a different tenant.</p>
            </div>
            <div className="field">
              <label htmlFor="manual-api-key">API Key</label>
              <input
                autoComplete="off"
                id="manual-api-key"
                onChange={(event) => setManualApiKey(event.target.value)}
                placeholder="pai_..."
                spellCheck={false}
                type="password"
                value={manualApiKey}
              />
              {fieldErrors.manual_api_key ? <p className="field-error">{fieldErrors.manual_api_key}</p> : null}
            </div>
          </div>
          <div className="actions-row">
            <button className="btn" disabled={manualWorking} onClick={persistManualKey} type="button">
              {manualWorking ? "Verifying..." : "Save Key"}
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => {
                setManualApiKey("");
                setManualTenantId("");
                setFieldErrors((current) => ({ ...current, manual_api_key: "" }));
                clearTenantContext();
                setSavedMessage("Cleared the stored tenant API key.");
              }}
              type="button"
            >
              Clear Key
            </button>
          </div>
          <p className="subtle">Recommendation endpoints use `X-API-Key`. The admin login token does not replace it.</p>
        </div>

        <div className="card">
          <div className="card-header">
            <h3>Create Tenant</h3>
          </div>
          {hasExistingTenant ? (
            <div className="detail-list">
              <div><strong>Tenant:</strong> {existingTenant.display_name}</div>
              <div><strong>ID:</strong> {existingTenant.tenant_id}</div>
              <div><strong>Status:</strong> {existingTenant.active ? "Active" : "Inactive"}</div>
              <div><strong>Created:</strong> {formatTimestamp(existingTenant.created_at)}</div>
              <p className="subtle">This account already owns a tenant. Additional tenant creation is blocked by the backend.</p>
            </div>
          ) : (
          <form aria-busy={createBusy} onSubmit={handleCreate}>
            <div className="field-grid">
              <div className="field">
                <label htmlFor="tenant-id">Tenant ID</label>
                <input disabled={createBusy} id="tenant-id" onChange={(event) => updateForm("tenant_id", event.target.value)} required value={form.tenant_id} />
                {fieldErrors.tenant_id ? <p className="field-error">{fieldErrors.tenant_id}</p> : null}
              </div>
              <div className="field">
                <label htmlFor="display-name">Display Name</label>
                <input disabled={createBusy} id="display-name" onChange={(event) => updateForm("display_name", event.target.value)} value={form.display_name} />
              </div>
              <div className="field">
                <label htmlFor="tenant-tier">Tier</label>
                <select disabled={createBusy} id="tenant-tier" onChange={(event) => updateForm("tier", event.target.value)} value={form.tier}>
                  <option value="free">free</option>
                  <option value="starter">starter</option>
                  <option value="pro">pro</option>
                  <option value="enterprise">enterprise</option>
                </select>
              </div>
              <div className="field">
                <label htmlFor="algorithm-mode">Algorithm Mode</label>
                <select disabled={createBusy} id="algorithm-mode" onChange={(event) => updateForm("algorithm_mode", event.target.value)} value={form.algorithm_mode}>
                  <option value="auto">auto</option>
                  <option value="explicit">explicit</option>
                  <option value="implicit">implicit</option>
                  <option value="hybrid">hybrid</option>
                </select>
              </div>
              <div className="field">
                <label htmlFor="rate-limit-rpm">Rate Limit RPM</label>
                <input disabled={createBusy} id="rate-limit-rpm" min="1" onChange={(event) => updateForm("rate_limit_rpm", event.target.value)} type="number" value={form.rate_limit_rpm} />
                {fieldErrors.rate_limit_rpm ? <p className="field-error">{fieldErrors.rate_limit_rpm}</p> : null}
              </div>
              <div className="field field-full">
                <label htmlFor="tenant-notes">Notes</label>
                <textarea disabled={createBusy} id="tenant-notes" onChange={(event) => updateForm("notes", event.target.value)} rows={3} value={form.notes} />
              </div>
            </div>
            <div className="actions-row">
              <button aria-busy={createBusy} className="btn" disabled={createDisabled} type="submit">
                {createBusy ? "Creating..." : "Create Tenant"}
              </button>
            </div>
          </form>
          )}
        </div>
      </div>

      {secret ? (
        <div className="card key-banner page-section">
          <div className="card-header">
            <h3>Returned API Key</h3>
            <button className="btn" onClick={useReturnedKey} type="button">Use In API Pages</button>
          </div>
          <div className="detail-list">
            <div><strong>Tenant:</strong> {secret.tenant_id}</div>
            <div><strong>API key:</strong> <code>{secret.api_key}</code></div>
            <div><strong>Warning:</strong> {secret.warning}</div>
            <div><strong>Handling:</strong> Copy or store this once. Future views only show masked values.</div>
          </div>
        </div>
      ) : null}

        <div className="card page-section">
          <div className="card-header">
            <h3>Tenants</h3>
          <button
            aria-label={loading ? "Refreshing tenants" : "Refresh tenants"}
            className="btn btn-secondary btn-icon"
            disabled={loading}
            onClick={() => reload()}
            title={loading ? "Refreshing tenants" : "Refresh tenants"}
            type="button"
          >
            <RefreshCw aria-hidden="true" className={loading ? "icon-spin" : ""} size={18} />
          </button>
        </div>
        {loading ? (
          <p className="subtle">Loading tenants...</p>
        ) : tenants.length ? (
          <div className="grid">
            {tenants.map((tenant) => (
              <div className="card card-nested" key={tenant.tenant_id}>
                <div className="card-header">
                  <h4>{tenant.display_name}</h4>
                  <span className={`status-pill ${tenant.active ? "status-live" : "status-muted"}`}>
                    {tenant.active ? "Active" : "Inactive"}
                  </span>
                </div>
                <div className="detail-list">
                  <div><strong>ID:</strong> {tenant.tenant_id}</div>
                  <div><strong>Tier:</strong> {tenant.tier}</div>
                  <div><strong>Mode:</strong> {tenant.algorithm_mode}</div>
                  <div><strong>Scopes:</strong> {tenant.scopes.length}</div>
                  <div><strong>RPM:</strong> {tenant.rate_limit_rpm}</div>
                  <div><strong>Created:</strong> {formatTimestamp(tenant.created_at)}</div>
                  <div><strong>Stored key:</strong> {isTenantActive(tenant) ? maskSecret(tenantApiKey) : (tenant.tenant_id === activeTenantId ? "Not configured" : "Not selected")}</div>
                </div>
                <div className="actions-row">
                  <button className="btn btn-secondary" disabled={Boolean(workingTenant)} onClick={() => handleRotateKey(tenant.tenant_id)} type="button">
                    {workingTenant === tenant.tenant_id ? "Rotating..." : "Rotate API Key"}
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="subtle">No tenants found yet.</p>
        )}
      </div>
    </Layout>
  )
}
