import { useCallback, useEffect, useRef, useState } from "react";
import { CalendarClock, ChevronDown, ChevronUp, Clock, Play, Plus, RefreshCw, Trash2, ToggleLeft, ToggleRight, History } from "lucide-react";
import Layout from "../components/Layout.jsx";
import { formatTimestamp, tenantRequest } from "../api.js";
import { useTenantSession } from "../hooks/useAuthSession.js";

// ─── constants ───────────────────────────────────────────────────────────────

const INTERVAL_UNITS  = ["hours", "days", "weeks"];
const ALGORITHM_MODES = ["auto", "explicit", "implicit", "hybrid"];
const DIALECTS        = ["postgresql", "mysql", "sqlite", "mssql"];
const POLL_INTERVAL   = 8_000; // ms

const defaultFileSource = { type: "file", path: "", format: "auto" };
const defaultSqlSource  = {
  type: "sql", dialect: "postgresql", host: "localhost", port: 5432,
  database: "", username: "root", password: "", sql:
    "SELECT user_id, item_id, rating FROM interactions LIMIT 100000",
};

const blankForm = {
  name: "", description: "", interval_value: 24, interval_unit: "hours",
  algorithm_mode: "auto", top_k: 10, n_trials: -1, top_models: 10,
  enabled: true, source_type: "file",
  file: { ...defaultFileSource },
  sql:  { ...defaultSqlSource },
};

// ─── helpers ─────────────────────────────────────────────────────────────────

function statusClass(s) {
  if (!s) return "status-muted";
  const v = s.toLowerCase();
  if (v === "completed") return "status-live";
  if (v === "failed")    return "status-error";
  return "status-muted";
}

function nextRunLabel(iso) {
  if (!iso) return "—";
  const diff = new Date(iso) - Date.now();
  if (diff <= 0) return "Imminent";
  const h = Math.floor(diff / 3_600_000);
  const m = Math.floor((diff % 3_600_000) / 60_000);
  if (h >= 24) return `in ${Math.floor(h / 24)}d ${h % 24}h`;
  if (h)       return `in ${h}h ${m}m`;
  return `in ${m}m`;
}

function validateForm(f) {
  const errs = {};
  if (!f.name.trim())              errs.name = "Name is required.";
  if (f.interval_value < 1)        errs.interval_value = "Must be ≥ 1.";
  if (f.interval_unit === "hours" && f.interval_value < 1)
    errs.interval_value = "Minimum interval is 1 hour.";
  if (f.source_type === "file" && !f.file.path.trim())
    errs.path = "Server-side file path is required.";
  if (f.source_type === "sql") {
    if (!f.sql.database.trim()) errs.database = "Database name is required.";
    if (!f.sql.username.trim()) errs.username  = "Username is required.";
    if (!f.sql.sql.trim())      errs.sql       = "SQL query is required.";
  }
  return errs;
}

function buildPayload(f) {
  const data_source = f.source_type === "file"
    ? { type: "file", path: f.file.path.trim(), format: f.file.format }
    : { type: "sql", ...f.sql, port: Number(f.sql.port) };
  return {
    name: f.name.trim(), description: f.description.trim() || null,
    interval_value: Number(f.interval_value),
    interval_unit: f.interval_unit,
    algorithm_mode: f.algorithm_mode,
    top_k: Number(f.top_k), n_trials: Number(f.n_trials),
    top_models: Number(f.top_models), enabled: f.enabled,
    data_source,
  };
}

// ─── sub-components ──────────────────────────────────────────────────────────

function Field({ label, error, children }) {
  return (
    <div className="form-field" style={{ display: "grid", gap: 6 }}>
      <label style={{ fontSize: 13, fontWeight: 600, color: "var(--ink)" }}>{label}</label>
      {children}
      {error && <span style={{ fontSize: 12, color: "#b4432c" }}>{error}</span>}
    </div>
  );
}

function Select({ value, onChange, options, disabled }) {
  return (
    <select value={value} onChange={e => onChange(e.target.value)} disabled={disabled}
      style={{ width: "100%", padding: "10px 14px", borderRadius: 12,
        border: "1px solid var(--line)", background: "var(--input-bg)",
        color: "var(--ink)", fontSize: 14, cursor: "pointer" }}>
      {options.map(o => (
        <option key={typeof o === "string" ? o : o.value}
          value={typeof o === "string" ? o : o.value}>
          {typeof o === "string" ? o : o.label}
        </option>
      ))}
    </select>
  );
}

function Input({ value, onChange, type = "text", placeholder, disabled, min }) {
  return (
    <input type={type} value={value} placeholder={placeholder} disabled={disabled}
      min={min} onChange={e => onChange(e.target.value)}
      style={{ width: "100%", padding: "10px 14px", borderRadius: 12,
        border: "1px solid var(--line)", background: "var(--input-bg)",
        color: "var(--ink)", fontSize: 14, boxSizing: "border-box" }} />
  );
}

function Textarea({ value, onChange, placeholder, rows = 3 }) {
  return (
    <textarea value={value} placeholder={placeholder} rows={rows}
      onChange={e => onChange(e.target.value)}
      style={{ width: "100%", padding: "10px 14px", borderRadius: 12,
        border: "1px solid var(--line)", background: "var(--input-bg)",
        color: "var(--ink)", fontSize: 13, resize: "vertical",
        fontFamily: "monospace", boxSizing: "border-box" }} />
  );
}

function HistoryDrawer({ scheduleId, onClose }) {
  const [history, setHistory] = useState(null);
  const [err, setErr]         = useState(null);

  useEffect(() => {
    let active = true;
    tenantRequest(`/scheduler/schedules/${scheduleId}/history`)
      .then(d => { if (active) setHistory(d.history || []); })
      .catch(e => { if (active) setErr(e.message); });
    return () => { active = false; };
  }, [scheduleId]);

  return (
    <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,.42)",
      zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }}
      onClick={onClose}>
      <div style={{ background: "var(--surface-strong)", borderRadius: 20, padding: 28,
        width: "min(560px, 94vw)", maxHeight: "76vh", display: "flex", flexDirection: "column",
        boxShadow: "var(--shadow)", overflow: "hidden" }}
        onClick={e => e.stopPropagation()}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 18 }}>
          <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700 }}>Run History</h3>
          <button className="btn btn-secondary" style={{ padding: "6px 14px", minHeight: "unset" }}
            onClick={onClose}>Close</button>
        </div>

        {err && <div className="form-error">{err}</div>}

        {!history && !err && (
          <p style={{ color: "var(--muted)", textAlign: "center", padding: 20 }}>Loading…</p>
        )}

        {history && history.length === 0 && (
          <div className="empty-state">
            <History size={28} style={{ opacity: .3 }} />
            <p style={{ color: "var(--muted)", fontSize: 14 }}>No runs yet.</p>
          </div>
        )}

        {history && history.length > 0 && (
          <div style={{ overflowY: "auto", display: "grid", gap: 10 }}>
            {history.map((entry, i) => (
              <div key={i} style={{ padding: "12px 16px", borderRadius: 14,
                border: "1px solid var(--line)", background: "var(--surface)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: 12, fontWeight: 600, color: "var(--muted)",
                    textTransform: "uppercase", letterSpacing: ".05em" }}>
                    {entry.triggered_by === "manual" ? "Manual" : "Scheduled"}
                  </span>
                  <span style={{ fontSize: 12, color: "var(--muted)" }}>
                    {formatTimestamp(entry.triggered_at)}
                  </span>
                </div>
                {entry.job_id && (
                  <p style={{ margin: "4px 0 0", fontSize: 13, color: "var(--ink)",
                    fontFamily: "monospace" }}>
                    job: {entry.job_id}
                  </p>
                )}
                {entry.error && (
                  <p style={{ margin: "4px 0 0", fontSize: 13, color: "#b4432c" }}>
                    Error: {entry.error}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ScheduleForm({ initial, onSave, onCancel, saving, globalError }) {
  const [form, setForm]   = useState(initial || { ...blankForm });
  const [errs, setErrs]   = useState({});

  function patch(key, val) { setForm(f => ({ ...f, [key]: val })); }
  function patchFile(key, val) { setForm(f => ({ ...f, file: { ...f.file, [key]: val } })); }
  function patchSql(key, val)  { setForm(f => ({ ...f, sql:  { ...f.sql,  [key]: val } })); }

  function handleSubmit() {
    const e = validateForm(form);
    setErrs(e);
    if (Object.keys(e).length) return;
    onSave(buildPayload(form));
  }

  const isSql = form.source_type === "sql";

  return (
    <div style={{ display: "grid", gap: 20 }}>
      {globalError && <div className="form-error">{globalError}</div>}

      {/* ── Basic info ── */}
      <div className="card-nested" style={{ display: "grid", gap: 16 }}>
        <p style={{ fontSize: 12, fontWeight: 700, textTransform: "uppercase",
          letterSpacing: ".07em", color: "var(--muted)", margin: 0 }}>Schedule Info</p>

        <Field label="Schedule Name *" error={errs.name}>
          <Input value={form.name} onChange={v => patch("name", v)} placeholder="e.g. Nightly Retraining" />
        </Field>

        <Field label="Description">
          <Input value={form.description} onChange={v => patch("description", v)}
            placeholder="Optional notes about this schedule" />
        </Field>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <Field label="Interval Value *" error={errs.interval_value}>
            <Input type="number" value={form.interval_value} min="1"
              onChange={v => patch("interval_value", v)} />
          </Field>
          <Field label="Unit">
            <Select value={form.interval_unit} onChange={v => patch("interval_unit", v)}
              options={INTERVAL_UNITS} />
          </Field>
        </div>
      </div>

      {/* ── Training params ── */}
      <div className="card-nested" style={{ display: "grid", gap: 16 }}>
        <p style={{ fontSize: 12, fontWeight: 700, textTransform: "uppercase",
          letterSpacing: ".07em", color: "var(--muted)", margin: 0 }}>Training Parameters</p>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
          <Field label="Algorithm Mode">
            <Select value={form.algorithm_mode} onChange={v => patch("algorithm_mode", v)}
              options={ALGORITHM_MODES} />
          </Field>
          <Field label="Top K">
            <Select value={String(form.top_k)} onChange={v => patch("top_k", v)}
              options={[{ value: "5", label: "5" }, { value: "10", label: "10" }]} />
          </Field>
          <Field label="Top Models">
            <Select value={String(form.top_models)} onChange={v => patch("top_models", v)}
              options={[{ value: "5", label: "5" }, { value: "10", label: "10" }]} />
          </Field>
        </div>

        <Field label="Optuna Trials (-1 = auto)">
          <Input type="number" value={form.n_trials} onChange={v => patch("n_trials", v)} />
        </Field>
      </div>

      {/* ── Data source ── */}
      <div className="card-nested" style={{ display: "grid", gap: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <p style={{ fontSize: 12, fontWeight: 700, textTransform: "uppercase",
            letterSpacing: ".07em", color: "var(--muted)", margin: 0 }}>Data Source</p>
          <div style={{ display: "flex", gap: 8 }}>
            {["file", "sql"].map(t => (
              <button key={t} onClick={() => patch("source_type", t)}
                style={{ padding: "5px 14px", borderRadius: 99, fontSize: 13, fontWeight: 600,
                  border: "1px solid",
                  borderColor: form.source_type === t ? "var(--teal)" : "var(--line)",
                  background: form.source_type === t ? "var(--accent-soft)" : "transparent",
                  color: form.source_type === t ? "var(--teal)" : "var(--muted)",
                  cursor: "pointer" }}>
                {t.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        {!isSql && (
          <>
            <Field label="Server-side File Path *" error={errs.path}>
              <Input value={form.file.path} onChange={v => patchFile("path", v)}
                placeholder="/data/interactions.csv" />
            </Field>
            <Field label="Format">
              <Select value={form.file.format} onChange={v => patchFile("format", v)}
                options={["auto", "csv", "json", "parquet", "xlsx"]} />
            </Field>
          </>
        )}

        {isSql && (
          <>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <Field label="Dialect">
                <Select value={form.sql.dialect} onChange={v => patchSql("dialect", v)}
                  options={DIALECTS} />
              </Field>
              <Field label="Port">
                <Input type="number" value={form.sql.port} onChange={v => patchSql("port", v)} />
              </Field>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <Field label="Host">
                <Input value={form.sql.host} onChange={v => patchSql("host", v)} placeholder="localhost" />
              </Field>
              <Field label="Database *" error={errs.database}>
                <Input value={form.sql.database} onChange={v => patchSql("database", v)} placeholder="mydb" />
              </Field>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <Field label="Username *" error={errs.username}>
                <Input value={form.sql.username} onChange={v => patchSql("username", v)} />
              </Field>
              <Field label="Password">
                <Input type="password" value={form.sql.password} onChange={v => patchSql("password", v)} />
              </Field>
            </div>
            <Field label="SQL Query *" error={errs.sql}>
              <Textarea value={form.sql.sql} onChange={v => patchSql("sql", v)} rows={4}
                placeholder="SELECT user_id, item_id, rating FROM interactions" />
            </Field>
          </>
        )}
      </div>

      {/* ── Actions ── */}
      <div className="actions-row">
        <button className="btn" onClick={handleSubmit} disabled={saving}>
          {saving ? "Saving…" : "Save Schedule"}
        </button>
        <button className="btn btn-secondary" onClick={onCancel} disabled={saving}>
          Cancel
        </button>
      </div>
    </div>
  );
}

function ScheduleCard({ schedule, onToggle, onDelete, onTrigger, onHistory, busy }) {
  const [expanded, setExpanded] = useState(false);
  const ds = schedule.data_source || {};

  return (
    <div className="card" style={{ padding: "22px 24px", transition: "box-shadow .2s" }}>
      {/* ── Header row ── */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 16 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
            <CalendarClock size={18} style={{ color: "var(--teal)", flexShrink: 0 }} />
            <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: "var(--ink)" }}>
              {schedule.name}
            </h3>
            <span className={`status-pill ${schedule.enabled ? "status-live" : "status-muted"}`}
              style={{ fontSize: 11 }}>
              {schedule.enabled ? "Active" : "Paused"}
            </span>
            <span className="status-pill status-muted" style={{ fontSize: 11 }}>
              {ds.type === "sql" ? "SQL" : "File"}
            </span>
          </div>

          {schedule.description && (
            <p style={{ margin: "6px 0 0 28px", fontSize: 13, color: "var(--muted)" }}>
              {schedule.description}
            </p>
          )}

          <div style={{ margin: "10px 0 0 28px", display: "flex", gap: 20, flexWrap: "wrap" }}>
            <span style={{ fontSize: 12, color: "var(--muted)" }}>
              <Clock size={12} style={{ marginRight: 4, verticalAlign: "middle" }} />
              Every {schedule.interval_value} {schedule.interval_unit}
            </span>
            <span style={{ fontSize: 12, color: "var(--muted)" }}>
              Next: <strong style={{ color: "var(--ink)" }}>
                {schedule.enabled ? nextRunLabel(schedule.next_run_iso) : "—"}
              </strong>
            </span>
            <span style={{ fontSize: 12, color: "var(--muted)" }}>
              Runs: <strong style={{ color: "var(--ink)" }}>{schedule.run_count ?? 0}</strong>
            </span>
            <span style={{ fontSize: 12, color: "var(--muted)" }}>
              Mode: <strong style={{ color: "var(--ink)" }}>{schedule.algorithm_mode}</strong>
            </span>
          </div>
        </div>

        {/* ── Action buttons ── */}
        <div style={{ display: "flex", gap: 8, flexShrink: 0, alignItems: "center" }}>
          <button title={schedule.enabled ? "Pause" : "Resume"}
            onClick={() => onToggle(schedule)} disabled={busy}
            style={{ background: "none", border: "none", cursor: "pointer",
              color: schedule.enabled ? "var(--teal)" : "var(--muted)",
              padding: "6px", borderRadius: 8, display: "flex", alignItems: "center" }}>
            {schedule.enabled
              ? <ToggleRight size={22} />
              : <ToggleLeft size={22} />}
          </button>

          <button title="Trigger now" onClick={() => onTrigger(schedule)} disabled={busy}
            style={{ background: "none", border: "none", cursor: "pointer",
              color: "var(--ink)", padding: "6px", borderRadius: 8,
              display: "flex", alignItems: "center" }}>
            <Play size={16} />
          </button>

          <button title="Run history" onClick={() => onHistory(schedule.id)} disabled={busy}
            style={{ background: "none", border: "none", cursor: "pointer",
              color: "var(--muted)", padding: "6px", borderRadius: 8,
              display: "flex", alignItems: "center" }}>
            <History size={16} />
          </button>

          <button title="Delete" onClick={() => onDelete(schedule)} disabled={busy}
            style={{ background: "none", border: "none", cursor: "pointer",
              color: "#b4432c", padding: "6px", borderRadius: 8,
              display: "flex", alignItems: "center" }}>
            <Trash2 size={16} />
          </button>

          <button onClick={() => setExpanded(x => !x)}
            style={{ background: "none", border: "none", cursor: "pointer",
              color: "var(--muted)", padding: "6px", borderRadius: 8,
              display: "flex", alignItems: "center" }}>
            {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
        </div>
      </div>

      {/* ── Expanded details ── */}
      {expanded && (
        <div style={{ marginTop: 18, paddingTop: 18, borderTop: "1px solid var(--line)",
          display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12 }}>
          {[
            ["Top K",        schedule.top_k],
            ["Top Models",   schedule.top_models],
            ["Optuna Trials", schedule.n_trials === -1 ? "Auto" : schedule.n_trials],
            ["Created",      formatTimestamp(schedule.created_at)],
            ["Last Run",     schedule.last_run_at ? formatTimestamp(schedule.last_run_at) : "Never"],
            ["Source",       ds.type === "file"
              ? (ds.path?.split("/").pop() || "file")
              : `${ds.dialect} / ${ds.database}`],
          ].map(([k, v]) => (
            <div key={k} style={{ padding: "10px 14px", background: "var(--surface)",
              borderRadius: 12, border: "1px solid var(--line-soft)" }}>
              <p style={{ margin: 0, fontSize: 11, color: "var(--muted)",
                textTransform: "uppercase", letterSpacing: ".06em" }}>{k}</p>
              <p style={{ margin: "4px 0 0", fontSize: 14, fontWeight: 600,
                color: "var(--ink)" }}>{String(v)}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── main page ───────────────────────────────────────────────────────────────

export default function Scheduler() {
  const [schedules,   setSchedules]   = useState([]);
  const [loading,     setLoading]     = useState(true);
  const [error,       setError]       = useState(null);
  const [showForm,    setShowForm]    = useState(false);
  const [saving,      setSaving]      = useState(false);
  const [formError,   setFormError]   = useState(null);
  const [toast,       setToast]       = useState(null);
  const [busyIds,     setBusyIds]     = useState(new Set());
  const [historyId,   setHistoryId]   = useState(null);
  const [refreshing,  setRefreshing]  = useState(false);
  const pollRef = useRef(null);
  const { hasTenantApiKey: tenantReady } = useTenantSession();

  const showToast = useCallback((msg, type = "success") => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 4000);
  }, []);

  const load = useCallback(async (quiet = false) => {
    if (!quiet) setLoading(true);
    setError(null);
    try {
      const data = await tenantRequest("/scheduler/schedules");
      setSchedules(data.schedules || []);
    } catch (e) {
      setError(e.message);
    } finally {
      if (!quiet) setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  // Polling while any schedule has a last run very recent (could be running)
  useEffect(() => {
    clearInterval(pollRef.current);
    if (!tenantReady) return;
    pollRef.current = setInterval(() => load(true), POLL_INTERVAL);
    return () => clearInterval(pollRef.current);
  }, [load, tenantReady]);

  async function handleRefresh() {
    setRefreshing(true);
    await load(true);
    setRefreshing(false);
  }

  async function handleCreate(payload) {
    setSaving(true);
    setFormError(null);
    try {
      await tenantRequest("/scheduler/schedules", { method: "POST", body: payload });
      showToast("Schedule created.");
      setShowForm(false);
      await load(true);
    } catch (e) {
      setFormError(e.message);
    } finally {
      setSaving(false);
    }
  }

  async function handleToggle(schedule) {
    setBusyIds(s => new Set([...s, schedule.id]));
    try {
      await tenantRequest(`/scheduler/schedules/${schedule.id}`, {
        method: "PATCH",
        body: { enabled: !schedule.enabled },
      });
      showToast(schedule.enabled ? "Schedule paused." : "Schedule resumed.");
      await load(true);
    } catch (e) {
      showToast(e.message, "error");
    } finally {
      setBusyIds(s => { const n = new Set(s); n.delete(schedule.id); return n; });
    }
  }

  async function handleDelete(schedule) {
    if (!window.confirm(`Delete schedule "${schedule.name}"? This cannot be undone.`)) return;
    setBusyIds(s => new Set([...s, schedule.id]));
    try {
      await tenantRequest(`/scheduler/schedules/${schedule.id}`, { method: "DELETE" });
      showToast("Schedule deleted.");
      await load(true);
    } catch (e) {
      showToast(e.message, "error");
    } finally {
      setBusyIds(s => { const n = new Set(s); n.delete(schedule.id); return n; });
    }
  }

  async function handleTrigger(schedule) {
    if (!window.confirm(`Trigger an immediate retrain for "${schedule.name}"?`)) return;
    setBusyIds(s => new Set([...s, schedule.id]));
    try {
      const res = await tenantRequest(`/scheduler/schedules/${schedule.id}/trigger`, { method: "POST" });
      showToast(`Retrain triggered — job ${res.job_id?.slice(0, 8)}…`);
    } catch (e) {
      showToast(e.message, "error");
    } finally {
      setBusyIds(s => { const n = new Set(s); n.delete(schedule.id); return n; });
    }
  }

  const active  = schedules.filter(s => s.enabled).length;
  const paused  = schedules.filter(s => !s.enabled).length;
  const totalRuns = schedules.reduce((acc, s) => acc + (s.run_count || 0), 0);

  const heroStats = [
    { label: "Total Schedules", value: schedules.length },
    { label: "Active",          value: active },
    { label: "Paused",          value: paused },
    { label: "Total Runs",      value: totalRuns },
  ];

  return (
    <Layout title="Scheduler" heroEyebrow="Automated Retraining" heroStats={heroStats}
      summary="Configure recurring retraining schedules so your models stay fresh without manual intervention.">

      {/* ── Tenant warning ── */}
      {!tenantReady && (
        <div className="form-error">
          A tenant API key is required to manage schedules. Configure one in Settings.
        </div>
      )}

      {/* ── Toast ── */}
      {toast && (
        <div style={{ position: "fixed", bottom: 28, right: 28, zIndex: 200,
          padding: "14px 20px", borderRadius: 16,
          background: toast.type === "error" ? "rgba(239,107,74,.12)" : "rgba(15,163,138,.12)",
          border: `1px solid ${toast.type === "error" ? "rgba(239,107,74,.24)" : "rgba(15,163,138,.22)"}`,
          color: toast.type === "error" ? "#a03d27" : "#0f7c6b",
          fontSize: 14, fontWeight: 600, boxShadow: "var(--shadow)",
          animation: "fadeIn .2s ease" }}>
          {toast.msg}
        </div>
      )}

      {/* ── History drawer ── */}
      {historyId && <HistoryDrawer scheduleId={historyId} onClose={() => setHistoryId(null)} />}

      {/* ── Create form ── */}
      {showForm && (
        <div className="card">
          <div className="card-header" style={{ marginBottom: 20 }}>
            <h3 style={{ margin: 0 }}>New Retraining Schedule</h3>
          </div>
          <ScheduleForm onSave={handleCreate} onCancel={() => { setShowForm(false); setFormError(null); }}
            saving={saving} globalError={formError} />
        </div>
      )}

      {/* ── Toolbar ── */}
      {!showForm && (
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12 }}>
          <p style={{ margin: 0, color: "var(--muted)", fontSize: 14 }}>
            {loading ? "Loading…" : `${schedules.length} schedule${schedules.length !== 1 ? "s" : ""} configured`}
          </p>
          <div style={{ display: "flex", gap: 10 }}>
            <button
              aria-label={refreshing ? "Refreshing schedules" : "Refresh schedules"}
              className="btn btn-secondary btn-icon"
              disabled={refreshing || loading}
              onClick={handleRefresh}
              title={refreshing ? "Refreshing schedules" : "Refresh schedules"}
              type="button"
            >
              <RefreshCw aria-hidden="true" className={refreshing ? "icon-spin" : ""} size={18} />
            </button>
            <button className="btn" onClick={() => { setShowForm(true); setFormError(null); }}
              disabled={!tenantReady}
              style={{ display: "flex", alignItems: "center", gap: 7 }}>
              <Plus size={16} /> New Schedule
            </button>
          </div>
        </div>
      )}

      {/* ── Error banner ── */}
      {error && <div className="form-error">{error}</div>}

      {/* ── Schedule list ── */}
      {!loading && !showForm && schedules.length === 0 && (
        <div className="card">
          <div className="empty-state">
            <CalendarClock size={36} style={{ opacity: .25 }} />
            <p style={{ fontWeight: 600, margin: 0 }}>No schedules yet</p>
            <p style={{ color: "var(--muted)", fontSize: 14, margin: 0 }}>
              Create one to automatically retrain your models on a fixed cadence.
            </p>
            {tenantReady && (
              <button className="btn" onClick={() => setShowForm(true)}
                style={{ marginTop: 8, display: "flex", alignItems: "center", gap: 7 }}>
                <Plus size={15} /> Create First Schedule
              </button>
            )}
          </div>
        </div>
      )}

      {schedules.length > 0 && !showForm && (
        <div style={{ display: "grid", gap: 14 }}>
          {schedules.map(s => (
            <ScheduleCard key={s.id} schedule={s} busy={busyIds.has(s.id)}
              onToggle={handleToggle} onDelete={handleDelete}
              onTrigger={handleTrigger} onHistory={setHistoryId} />
          ))}
        </div>
      )}

      {/* ── Spin animation ── */}
      <style>{`@keyframes spin { to { transform: rotate(360deg); } } @keyframes fadeIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }`}</style>
    </Layout>
  );
}
