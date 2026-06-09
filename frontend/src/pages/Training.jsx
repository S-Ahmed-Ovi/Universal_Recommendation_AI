import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { FileUp, RefreshCw, X } from "lucide-react";
import Layout from "../components/Layout";
import { formatTimestamp, tenantRequest } from "../api";
import { useTenantSession } from "../hooks/useAuthSession";

const steps = ["Data Ingest", "Benchmarking", "Hyperparameter Tuning", "Validation", "Deployment"];
const DATASET_FILE_ACCEPT = ".csv,.xlsx,.json,.parquet";
const DATASET_FILE_EXTENSIONS = new Set(["csv", "xlsx", "json", "parquet"]);
const ACTIVE_STATUSES = new Set(["pending", "running"]);
const RESULT_ACTIONS = [
  { label: "View Benchmark Results", to: "/benchmarks" },
  { label: "Test Recommendations", to: "/api" },
  { label: "Deploy Model", to: "/deployment" },
  { label: "SDK Integration", to: "/integrations" },
  { label: "API Usage", to: "/api" },
];
const initialSqlForm = {
  algorithm_mode: "auto",
  database: "",
  dialect: "postgresql",
  host: "localhost",
  n_trials: -1,
  password: "",
  port: 5432,
  sql: "SELECT user_id, item_id, rating FROM interactions LIMIT 100000",
  top_k: 10,
  top_models: 10,
  username: "root",
};

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function normalizeJobStatus(value) {
  const status = String(value || "pending").toLowerCase();
  if (status === "done") {
    return "completed";
  }
  if (status === "error") {
    return "failed";
  }
  return status;
}

function normalizeJobRecord(job) {
  if (!job || typeof job !== "object") {
    return null;
  }
  return {
    ...job,
    status: normalizeJobStatus(job.status),
  };
}

function sortJobRecords(jobs) {
  return jobs
    .map(normalizeJobRecord)
    .filter(Boolean)
    .sort((left, right) => Number(right?.created_at || 0) - Number(left?.created_at || 0));
}

function mergeJobRecord(current, nextJob) {
  const job = normalizeJobRecord(nextJob);
  if (!job?.job_id) {
    return current;
  }
  return sortJobRecords([job, ...current.filter((entry) => entry.job_id !== job.job_id)]);
}

function isActiveJobStatus(status) {
  return ACTIVE_STATUSES.has(normalizeJobStatus(status));
}

function trainingStatusTone(status) {
  const normalized = normalizeJobStatus(status);
  if (normalized === "completed") {
    return "status-live";
  }
  if (normalized === "failed") {
    return "status-error";
  }
  return "status-muted";
}

function runSourceLabel(run) {
  if (run?.source_label) {
    return run.source_label;
  }
  if (run?.source_type === "sql") {
    return "SQL training run";
  }
  if (run?.source_type === "file") {
    return "File training run";
  }
  return "Training run";
}

function validateTrialBudget(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) && (numeric === -1 || numeric > 0);
}

function isSupportedDatasetFile(file) {
  if (!file?.name) {
    return false;
  }
  const extension = file.name.split(".").pop()?.toLowerCase() || "";
  return DATASET_FILE_EXTENSIONS.has(extension);
}

function validateFileTraining(selectedFile, trainConfig) {
  const nextErrors = {};
  if (!selectedFile) {
    nextErrors.file_dataset = "Choose a dataset file before starting training.";
  } else if (!isSupportedDatasetFile(selectedFile)) {
    nextErrors.file_dataset = "Unsupported file type. Upload a csv, xlsx, json, or parquet file.";
  }
  if (Number(trainConfig.top_k) <= 0) {
    nextErrors.file_top_k = "Top K must be greater than zero.";
  }
  if (Number(trainConfig.top_models) <= 0) {
    nextErrors.file_top_models = "Top Models must be greater than zero.";
  }
  if (!validateTrialBudget(trainConfig.n_trials)) {
    nextErrors.file_n_trials = "Trials must be -1 for auto or a positive number.";
  }
  return nextErrors;
}

function validateSqlTraining(sqlForm) {
  const nextErrors = {};
  const isSqlite = sqlForm.dialect === "sqlite";

  if (!sqlForm.database.trim()) {
    nextErrors.sql_database = isSqlite
      ? "Database path is required for SQLite."
      : "Database name is required.";
  }
  if (!isSqlite && !sqlForm.host.trim()) {
    nextErrors.sql_host = "Host is required.";
  }
  if (!isSqlite && !sqlForm.username.trim()) {
    nextErrors.sql_username = "Username is required.";
  }
  if (!isSqlite && Number(sqlForm.port) <= 0) {
    nextErrors.sql_port = "Port must be greater than zero.";
  }
  if (!sqlForm.sql.trim()) {
    nextErrors.sql_query = "SQL query is required.";
  }
  if (Number(sqlForm.top_k) <= 0) {
    nextErrors.sql_top_k = "Top K must be greater than zero.";
  }
  if (Number(sqlForm.top_models) <= 0) {
    nextErrors.sql_top_models = "Top Models must be greater than zero.";
  }
  if (!validateTrialBudget(sqlForm.n_trials)) {
    nextErrors.sql_n_trials = "Trials must be -1 for auto or a positive number.";
  }

  return nextErrors;
}

function resolveInputFile(inputRef) {
  return inputRef.current?.files?.[0] || null;
}

function notifyMissingTenantContext(message) {
  if (typeof window !== "undefined") {
    window.alert(message);
  }
}

function trainingStorageKey(tenantId) {
  return `neurorec_active_training_job_${tenantId || "default"}`;
}

function loadStoredJob(tenantId) {
  if (typeof window === "undefined") {
    return "";
  }
  return window.localStorage.getItem(trainingStorageKey(tenantId)) || "";
}

function saveStoredJob(tenantId, jobId) {
  if (typeof window === "undefined") {
    return;
  }
  if (jobId) {
    window.localStorage.setItem(trainingStorageKey(tenantId), jobId);
    return;
  }
  window.localStorage.removeItem(trainingStorageKey(tenantId));
}

export default function Training() {
  const [activeTab, setActiveTab] = useState("file");
  const [error, setError] = useState("");
  const [fieldErrors, setFieldErrors] = useState({});
  const [fileFormat, setFileFormat] = useState("auto");
  const [deletingJobId, setDeletingJobId] = useState("");
  const [jobData, setJobData] = useState(null);
  const [jobHistory, setJobHistory] = useState([]);
  const [jobId, setJobId] = useState("");
  const [loadingJob, setLoadingJob] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [sqlForm, setSqlForm] = useState(initialSqlForm);
  const [submitting, setSubmitting] = useState("");
  const [trainConfig, setTrainConfig] = useState({
    algorithm_mode: "auto",
    n_trials: -1,
    top_k: 10,
    top_models: 10,
  });
  const fileInputRef = useRef(null);
  const { hasTenantApiKey: hasKey, tenantId } = useTenantSession();

  async function loadJobHistory(options = {}) {
    if (!hasKey) {
      setJobHistory([]);
      return [];
    }

    try {
      const data = await tenantRequest("/jobs?limit=25");
      const jobs = sortJobRecords(data?.jobs || []);
      setJobHistory(jobs);
      return jobs;
    } catch (loadError) {
      if (!options.silent) {
        setError(loadError.message || "Unable to load training history");
      }
      return [];
    }
  }

  useEffect(() => {
    if (!hasKey) {
      setJobId("");
      setJobData(null);
      setJobHistory([]);
      return;
    }

    setJobData(null);
    setJobHistory([]);
    setJobId(loadStoredJob(tenantId));
  }, [hasKey, tenantId]);

  useEffect(() => {
    let cancelled = false;

    async function hydrateHistory() {
      const jobs = await loadJobHistory({ silent: true });
      if (cancelled || jobId) {
        return;
      }

      const nextActiveJob = jobs.find((job) => isActiveJobStatus(job.status));
      if (nextActiveJob) {
        setJobId(nextActiveJob.job_id);
        setJobData(nextActiveJob);
        saveStoredJob(tenantId, nextActiveJob.job_id);
      }
    }

    if (hasKey) {
      hydrateHistory();
    }

    return () => {
      cancelled = true;
    };
  }, [hasKey, tenantId]);

  useEffect(() => {
    if (!jobId || !hasKey) {
      return;
    }

    let cancelled = false;
    let timeoutId = null;

    async function poll() {
      try {
        const data = normalizeJobRecord(await tenantRequest(`/jobs/${encodeURIComponent(jobId)}`));
        if (cancelled || !data) {
          return;
        }

        setJobData(data);
        setJobHistory((current) => mergeJobRecord(current, data));

        if (isActiveJobStatus(data.status)) {
          saveStoredJob(tenantId, data.job_id || jobId);
          timeoutId = window.setTimeout(poll, 2500);
          return;
        }

        saveStoredJob(tenantId, "");
      } catch (pollError) {
        if (cancelled) {
          return;
        }

        if ((pollError.message || "").toLowerCase().includes("not found")) {
          saveStoredJob(tenantId, "");
          setJobId("");
          setJobData(null);
          void loadJobHistory({ silent: true });
        }
        setError(pollError.message || "Unable to fetch training job status");
      }
    }

    poll();
    return () => {
      cancelled = true;
      if (timeoutId) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [jobId, hasKey, tenantId]);

  function updateTrainConfig(key, value) {
    setTrainConfig((current) => ({ ...current, [key]: value }));
    setFieldErrors((current) => ({ ...current, [`file_${key}`]: "" }));
  }

  function updateSqlForm(key, value) {
    if (key === "dialect") {
      setSqlForm((current) => {
        const nextDialect = value;
        const next = { ...current, dialect: nextDialect };
        const currentPort = Number(current.port);

        if (nextDialect === "mysql") {
          next.host = "localhost";
          next.username = "root";
          if (!currentPort || currentPort === 5432 || currentPort === 0) {
            next.port = 3306;
          }
        } else if (nextDialect === "sqlite") {
          next.host = "";
          next.port = 0;
        } else {
          if (!current.host || current.host === "localhost") {
            next.host = "localhost";
          }
          if (!currentPort || currentPort === 3306 || currentPort === 0) {
            next.port = 5432;
          }
        }

        return next;
      });
      setFieldErrors((current) => ({
        ...current,
        sql_database: "",
        sql_host: "",
        sql_password: "",
        sql_port: "",
        sql_username: "",
      }));
      return;
    }

    setSqlForm((current) => ({ ...current, [key]: value }));
    const errorKey = key === "sql" ? "sql_query" : `sql_${key}`;
    setFieldErrors((current) => ({ ...current, [errorKey]: "" }));
  }

  function handleFilePick(event) {
    const nextFile = event.target.files?.[0] || null;
    if (!nextFile) {
      setSelectedFile(null);
      setFieldErrors((current) => ({ ...current, file_dataset: "" }));
      return;
    }

    if (!isSupportedDatasetFile(nextFile)) {
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
      setSelectedFile(null);
      setFieldErrors((current) => ({
        ...current,
        file_dataset: "Unsupported file type. Upload a csv, xlsx, json, or parquet file.",
      }));
      return;
    }

    setSelectedFile(nextFile);
    setFieldErrors((current) => ({ ...current, file_dataset: "" }));
  }

  function clearSelectedFile() {
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    setSelectedFile(null);
    setFieldErrors((current) => ({ ...current, file_dataset: "" }));
  }

  function openFilePicker() {
    if (submitting === "file") {
      return;
    }
    fileInputRef.current?.click();
  }

  function ensureTenantContext() {
    if (!hasKey) {
      const message = "Tenant API key missing. Please add a tenant key first.";
      setError(message);
      notifyMissingTenantContext(message);
      return false;
    }

    if (!tenantId) {
      const message = "Tenant ID missing. Please select or save a tenant first.";
      setError(message);
      notifyMissingTenantContext(message);
      return false;
    }

    return true;
  }

  async function refreshJob() {
    if (!hasKey) {
      return;
    }

    setLoadingJob(true);
    setError("");
    try {
      if (jobId) {
        const data = normalizeJobRecord(await tenantRequest(`/jobs/${encodeURIComponent(jobId)}`));
        setJobData(data);
        if (data) {
          setJobHistory((current) => mergeJobRecord(current, data));
          if (isActiveJobStatus(data.status)) {
            saveStoredJob(tenantId, data.job_id || jobId);
          } else {
            saveStoredJob(tenantId, "");
          }
        }
      }
      await loadJobHistory({ silent: true });
    } catch (loadError) {
      setError(loadError.message || "Unable to load training job");
    } finally {
      setLoadingJob(false);
    }
  }

  async function deleteHistoryRun(run) {
    if (!run?.job_id || isActiveJobStatus(run.status)) {
      return;
    }

    const confirmed = typeof window === "undefined"
      ? true
      : window.confirm(`Delete training run ${run.job_id} from history?`);
    if (!confirmed) {
      return;
    }

    setDeletingJobId(run.job_id);
    setError("");
    try {
      await tenantRequest(`/jobs/${encodeURIComponent(run.job_id)}`, {
        method: "DELETE",
      });

      setJobHistory((current) => current.filter((entry) => entry.job_id !== run.job_id));
      setJobData((current) => (current?.job_id === run.job_id ? null : current));
      if (jobId === run.job_id) {
        setJobId("");
        saveStoredJob(tenantId, "");
      }
    } catch (deleteError) {
      setError(deleteError.message || "Unable to delete training run");
    } finally {
      setDeletingJobId("");
    }
  }

  async function submitFileTraining(event) {
    event.preventDefault();
    setFieldErrors({});

    const currentFile = selectedFile || resolveInputFile(fileInputRef);

    if (!ensureTenantContext()) {
      return;
    }

    const nextErrors = validateFileTraining(currentFile, trainConfig);
    if (Object.keys(nextErrors).length) {
      setFieldErrors(nextErrors);
      return;
    }

    setSubmitting("file");
    setError("");

    try {
      const body = new FormData();
      body.append("file", currentFile);
      body.append("top_k", String(trainConfig.top_k));
      body.append("top_models", String(trainConfig.top_models));
      body.append("n_trials", String(trainConfig.n_trials));
      body.append("algorithm_mode", String(trainConfig.algorithm_mode));
      body.append("format", fileFormat);

      const data = await tenantRequest("/train/file", {
        body,
        method: "POST",
      });

      const nextJobId = data?.job_id || "";
      const optimisticJob = normalizeJobRecord({
        algorithm_mode: trainConfig.algorithm_mode,
        created_at: Date.now() / 1000,
        job_id: nextJobId,
        poll_url: data?.poll_url,
        source_label: currentFile?.name || "uploaded dataset",
        source_type: "file",
        status: "pending",
        top_k: Number(trainConfig.top_k),
        top_models: Number(trainConfig.top_models),
      });

      setJobId(nextJobId);
      saveStoredJob(tenantId, nextJobId);
      setJobData(optimisticJob);
      setJobHistory((current) => mergeJobRecord(current, optimisticJob));
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
      setSelectedFile(null);
    } catch (submitError) {
      setError(submitError.message || "Unable to start file training");
    } finally {
      setSubmitting("");
    }
  }

  async function submitSqlTraining(event) {
    event.preventDefault();
    setFieldErrors({});

    if (!ensureTenantContext()) {
      return;
    }

    const nextErrors = validateSqlTraining(sqlForm);
    if (Object.keys(nextErrors).length) {
      setFieldErrors(nextErrors);
      return;
    }

    setSubmitting("sql");
    setError("");

    try {
      const data = await tenantRequest("/train/sql", {
        body: {
          ...sqlForm,
          n_trials: Number(sqlForm.n_trials),
          port: Number(sqlForm.port),
          top_k: Number(sqlForm.top_k),
          top_models: Number(sqlForm.top_models),
        },
        method: "POST",
      });

      const nextJobId = data?.job_id || "";
      const optimisticJob = normalizeJobRecord({
        algorithm_mode: sqlForm.algorithm_mode,
        created_at: Date.now() / 1000,
        job_id: nextJobId,
        poll_url: data?.poll_url,
        source_label: `${sqlForm.dialect} / ${sqlForm.database}`,
        source_type: "sql",
        status: "pending",
        top_k: Number(sqlForm.top_k),
        top_models: Number(sqlForm.top_models),
      });

      setJobId(nextJobId);
      saveStoredJob(tenantId, nextJobId);
      setJobData(optimisticJob);
      setJobHistory((current) => mergeJobRecord(current, optimisticJob));
    } catch (submitError) {
      setError(submitError.message || "Unable to start SQL training");
    } finally {
      setSubmitting("");
    }
  }

  const featuredJob = jobId ? (normalizeJobRecord(jobData) || jobHistory.find((job) => job.job_id === jobId) || null) : null;
  const status = featuredJob?.status || "idle";
  const statusClass = trainingStatusTone(status);
  const result = featuredJob?.result || null;
  const canRefresh = Boolean(hasKey && (jobId || jobHistory.length));
  const completedRuns = jobHistory.filter((run) => run.status === "completed").length;
  const trainingSummary = hasKey
    ? featuredJob
      ? `Training history is retained per tenant. ${jobHistory.length} runs are visible and ${completedRuns} completed successfully.`
      : "Launch a tenant-scoped training run from file upload or SQL. Completed runs remain visible here as history."
    : "Attach a tenant API key before starting file or SQL model training.";
  const canSubmitFileTraining = submitting !== "file" && Boolean(selectedFile);
  const heroStats = [
    { label: "Tenant", value: tenantId || "Not attached" },
    { label: "Job", value: featuredJob?.job_id || jobId || "No runs yet" },
    { label: "Status", value: status },
  ];

  return (
    <Layout
      heroStats={heroStats}
      summary={trainingSummary}
      title="AI Training Pipeline"
    >
      {!hasKey ? (
        <div className="card empty-state">
          <h3>Tenant API key required</h3>
          <p className="subtle">
            Training endpoints use `X-API-Key`. Save a tenant key in Tenants before uploading a file or starting SQL training.
          </p>
        </div>
      ) : null}

      {error ? <div className="form-error">{error}</div> : null}

      <div className="stage-grid">
        {steps.map((step, index) => (
          <div className="card progress-card" key={step}>
            <h3>{step}</h3>
            <div className="progress-track" aria-hidden="true">
              <div className="progress-fill" style={{ width: `${(index + 1) * 18}%` }}></div>
            </div>
          </div>
        ))}
      </div>

      <div className="card page-section">
        <div className="card-header">
          <h3>Training Inputs</h3>
          <span className="subtle">{tenantId ? `Tenant: ${tenantId}` : "Tenant key loaded"}</span>
        </div>

        <div className="segmented">
          <button className={activeTab === "file" ? "segment-active" : ""} onClick={() => setActiveTab("file")} type="button">
            Train From File
          </button>
          <button className={activeTab === "sql" ? "segment-active" : ""} onClick={() => setActiveTab("sql")} type="button">
            Train From SQL
          </button>
        </div>

        {activeTab === "file" ? (
          <form className="form-stack page-section" onSubmit={submitFileTraining}>
            <fieldset aria-busy={submitting === "file"} disabled={submitting === "file"}>
              <div className="field-grid">
                <div className="field field-full">
                  <label htmlFor="training-dataset-file">Dataset File</label>
                  <input
                    accept={DATASET_FILE_ACCEPT}
                    aria-describedby="training-dataset-file-help training-dataset-file-formats training-dataset-file-error"
                    aria-invalid={fieldErrors.file_dataset ? "true" : "false"}
                    className="upload-input"
                    id="training-dataset-file"
                    onChange={handleFilePick}
                    ref={fileInputRef}
                    type="file"
                  />
                  <button
                    aria-describedby="training-dataset-file-help training-dataset-file-formats"
                    className={`upload-dropzone${selectedFile ? " upload-dropzone-selected" : ""}${fieldErrors.file_dataset ? " upload-dropzone-error" : ""}`}
                    onClick={openFilePicker}
                    type="button"
                  >
                    <span className="upload-dropzone-icon" aria-hidden="true">
                      <FileUp size={20} />
                    </span>
                    <span className="upload-dropzone-copy">
                      <span className="upload-dropzone-title">
                        {selectedFile ? selectedFile.name : "Click to upload dataset file"}
                      </span>
                      <span className="upload-dropzone-description" id="training-dataset-file-help">
                        Upload a recommendation dataset for file-based training.
                      </span>
                      <span className="upload-dropzone-formats" id="training-dataset-file-formats">
                        Accepted formats: csv, xlsx, json, parquet
                      </span>
                    </span>
                    <span className="upload-dropzone-action">
                      {selectedFile ? "Replace file" : "Browse files"}
                    </span>
                  </button>
                  {selectedFile ? (
                    <div className="upload-dropzone-footer">
                      <p className="field-hint">Selected file ready for upload.</p>
                      <button
                        className="upload-clear"
                        onClick={(event) => {
                          event.preventDefault();
                          clearSelectedFile();
                        }}
                        type="button"
                      >
                        <X size={14} />
                        Remove
                      </button>
                    </div>
                  ) : null}
                  {fieldErrors.file_dataset ? <p className="field-error" id="training-dataset-file-error">{fieldErrors.file_dataset}</p> : null}
                </div>
                <div className="field">
                  <label htmlFor="training-file-format">Format</label>
                  <select id="training-file-format" onChange={(event) => setFileFormat(event.target.value)} value={fileFormat}>
                    <option value="auto">auto</option>
                    <option value="csv">csv</option>
                    <option value="excel">excel</option>
                    <option value="json">json</option>
                    <option value="parquet">parquet</option>
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="training-file-top-k">Top K</label>
                  <select id="training-file-top-k" onChange={(event) => updateTrainConfig("top_k", event.target.value)} value={trainConfig.top_k}>
                    <option value={5}>5</option>
                    <option value={10}>10</option>
                  </select>
                  {fieldErrors.file_top_k ? <p className="field-error">{fieldErrors.file_top_k}</p> : null}
                </div>
                <div className="field">
                  <label htmlFor="training-file-top-models">Top Models</label>
                  <select id="training-file-top-models" onChange={(event) => updateTrainConfig("top_models", event.target.value)} value={trainConfig.top_models}>
                    <option value={5}>5</option>
                    <option value={10}>10</option>
                  </select>
                  {fieldErrors.file_top_models ? <p className="field-error">{fieldErrors.file_top_models}</p> : null}
                </div>
                <div className="field">
                  <label htmlFor="training-file-trials">Trials</label>
                  <input id="training-file-trials" onChange={(event) => updateTrainConfig("n_trials", event.target.value)} type="number" value={trainConfig.n_trials} />
                  <p className="field-hint">Use `-1` to let the backend choose an automatic trial budget.</p>
                  {fieldErrors.file_n_trials ? <p className="field-error">{fieldErrors.file_n_trials}</p> : null}
                </div>
                <div className="field">
                  <label htmlFor="training-file-algorithm-mode">Algorithm Mode</label>
                  <select id="training-file-algorithm-mode" onChange={(event) => updateTrainConfig("algorithm_mode", event.target.value)} value={trainConfig.algorithm_mode}>
                    <option value="auto">auto</option>
                    <option value="explicit">explicit</option>
                    <option value="implicit">implicit</option>
                    <option value="hybrid">hybrid</option>
                  </select>
                </div>
              </div>
            </fieldset>

            <div className="actions-row">
              <button className="btn" disabled={!canSubmitFileTraining} type="submit">
                {submitting === "file" ? "Uploading..." : "Upload And Train"}
              </button>
            </div>
          </form>
        ) : (
          <form className="form-stack page-section" onSubmit={submitSqlTraining}>
            <fieldset aria-busy={submitting === "sql"} disabled={submitting === "sql"}>
              <div className="field-grid">
                <div className="field">
                  <label htmlFor="training-sql-dialect">Dialect</label>
                  <select id="training-sql-dialect" onChange={(event) => updateSqlForm("dialect", event.target.value)} value={sqlForm.dialect}>
                    <option value="postgresql">postgresql</option>
                    <option value="mysql">mysql</option>
                    <option value="sqlite">sqlite</option>
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="training-sql-host">Host</label>
                  <input id="training-sql-host" onChange={(event) => updateSqlForm("host", event.target.value)} placeholder={sqlForm.dialect === "mysql" ? "localhost" : "db.example.com"} value={sqlForm.host} />
                  {fieldErrors.sql_host ? <p className="field-error">{fieldErrors.sql_host}</p> : null}
                </div>
                <div className="field">
                  <label htmlFor="training-sql-port">Port</label>
                  <input id="training-sql-port" onChange={(event) => updateSqlForm("port", event.target.value)} type="number" value={sqlForm.port} />
                  {fieldErrors.sql_port ? <p className="field-error">{fieldErrors.sql_port}</p> : null}
                </div>
                <div className="field">
                  <label htmlFor="training-sql-database">Database</label>
                  <input id="training-sql-database" onChange={(event) => updateSqlForm("database", event.target.value)} placeholder={sqlForm.dialect === "sqlite" ? "data/recommendations.db" : "analytics"} value={sqlForm.database} />
                  {fieldErrors.sql_database ? <p className="field-error">{fieldErrors.sql_database}</p> : null}
                </div>
                <div className="field">
                  <label htmlFor="training-sql-username">Username</label>
                  <input id="training-sql-username" onChange={(event) => updateSqlForm("username", event.target.value)} placeholder={sqlForm.dialect === "mysql" ? "root" : ""} value={sqlForm.username} />
                  {fieldErrors.sql_username ? <p className="field-error">{fieldErrors.sql_username}</p> : null}
                </div>
                <div className="field">
                  <label htmlFor="training-sql-password">Password</label>
                  <input autoComplete="current-password" id="training-sql-password" onChange={(event) => updateSqlForm("password", event.target.value)} type="password" value={sqlForm.password} />
                  {fieldErrors.sql_password ? <p className="field-error">{fieldErrors.sql_password}</p> : null}
                </div>
                <div className="field">
                  <label htmlFor="training-sql-top-k">Top K</label>
                  <select id="training-sql-top-k" onChange={(event) => updateSqlForm("top_k", event.target.value)} value={sqlForm.top_k}>
                    <option value={5}>5</option>
                    <option value={10}>10</option>
                  </select>
                  {fieldErrors.sql_top_k ? <p className="field-error">{fieldErrors.sql_top_k}</p> : null}
                </div>
                <div className="field">
                  <label htmlFor="training-sql-top-models">Top Models</label>
                  <select id="training-sql-top-models" onChange={(event) => updateSqlForm("top_models", event.target.value)} value={sqlForm.top_models}>
                    <option value={5}>5</option>
                    <option value={10}>10</option>
                  </select>
                  {fieldErrors.sql_top_models ? <p className="field-error">{fieldErrors.sql_top_models}</p> : null}
                </div>
                <div className="field">
                  <label htmlFor="training-sql-trials">Trials</label>
                  <input id="training-sql-trials" onChange={(event) => updateSqlForm("n_trials", event.target.value)} type="number" value={sqlForm.n_trials} />
                  <p className="field-hint">Use `-1` to let the backend choose an automatic trial budget.</p>
                  {fieldErrors.sql_n_trials ? <p className="field-error">{fieldErrors.sql_n_trials}</p> : null}
                </div>
                <div className="field">
                  <label htmlFor="training-sql-algorithm-mode">Algorithm Mode</label>
                  <select id="training-sql-algorithm-mode" onChange={(event) => updateSqlForm("algorithm_mode", event.target.value)} value={sqlForm.algorithm_mode}>
                    <option value="auto">auto</option>
                    <option value="explicit">explicit</option>
                    <option value="implicit">implicit</option>
                    <option value="hybrid">hybrid</option>
                  </select>
                </div>
                <div className="field field-full field-mono">
                  <label htmlFor="training-sql-query">SQL Query</label>
                  <textarea id="training-sql-query" onChange={(event) => updateSqlForm("sql", event.target.value)} rows={7} value={sqlForm.sql} />
                  {fieldErrors.sql_query ? <p className="field-error">{fieldErrors.sql_query}</p> : null}
                </div>
              </div>
            </fieldset>

            <div className="actions-row">
              <button className="btn" disabled={!hasKey || submitting === "sql"} type="submit">
                {submitting === "sql" ? "Starting..." : "Connect And Train"}
              </button>
            </div>
          </form>
        )}
      </div>

      <div className="card page-section">
          <div className="card-header">
            <h3>Current Training Run</h3>
            <div className="actions-row">
              <span className={`status-pill ${statusClass}`}>{status}</span>
              <button
                aria-label={loadingJob ? "Refreshing training status" : "Refresh training status"}
                className="btn btn-secondary btn-icon"
                disabled={!canRefresh || loadingJob}
                onClick={refreshJob}
                title={loadingJob ? "Refreshing training status" : "Refresh training status"}
                type="button"
              >
                <RefreshCw aria-hidden="true" className={loadingJob ? "icon-spin" : ""} size={18} />
              </button>
            </div>
          </div>

        {featuredJob ? (
          <div className="detail-list">
            <div><strong>Job ID:</strong> {featuredJob.job_id || jobId}</div>
            <div><strong>Source:</strong> {runSourceLabel(featuredJob)}</div>
            <div><strong>Created:</strong> {featuredJob.created_at ? formatTimestamp(featuredJob.created_at) : "n/a"}</div>
            <div><strong>Started:</strong> {featuredJob.started_at ? formatTimestamp(featuredJob.started_at) : "n/a"}</div>
            <div><strong>Finished:</strong> {featuredJob.finished_at ? formatTimestamp(featuredJob.finished_at) : "n/a"}</div>
            {featuredJob.error ? <div><strong>Error:</strong> {featuredJob.error}</div> : null}
            {result ? (
              <div className="training-result-grid">
                <div className="result-stat-strip">
                  <div className="result-stat-card">
                    <span>Best algorithm</span>
                    <strong>{result.best_algorithm || "n/a"}</strong>
                  </div>
                  <div className="result-stat-card">
                    <span>Resolved mode</span>
                    <strong>{result.resolved_mode || "n/a"}</strong>
                  </div>
                  <div className="result-stat-card">
                    <span>Elapsed</span>
                    <strong>{result.elapsed_s === 0 || result.elapsed_s ? `${result.elapsed_s}s` : "n/a"}</strong>
                  </div>
                  <div className="result-stat-card">
                    <span>Recommended strategy</span>
                    <strong>{result.model_selection_policy?.recommended_strategy || "n/a"}</strong>
                  </div>
                </div>

                {status === "completed" ? (
                  <div className="success-banner training-complete-banner">
                    <strong>Training Completed.</strong> Use this production path: Benchmark to Deploy to Integrate.
                  </div>
                ) : null}

                {status === "completed" ? (
                  <div className="card page-section">
                    <div className="card-header">
                      <h3>Post-Training Workflow</h3>
                      <span className="subtle">Use the production handoff path in order.</span>
                    </div>
                    <div className="training-workflow-path">
                      <span>Benchmark</span>
                      <span>Deploy</span>
                      <span>Integrate</span>
                    </div>
                    <div className="actions-row">
                      {RESULT_ACTIONS.map((action) => (
                        <Link className="btn btn-secondary" key={action.label} to={action.to}>
                          {action.label}
                        </Link>
                      ))}
                      <Link className="btn" to="/results">
                        Open Results Navigator
                      </Link>
                    </div>
                  </div>
                ) : null}

                <details className="result-debug">
                  <summary>Raw training payload</summary>
                  <pre className="code-block">{pretty(result)}</pre>
                </details>
              </div>
            ) : null}
          </div>
        ) : (
          <p className="subtle">No training job started yet. When one starts, it stays visible here and in the tenant run history.</p>
        )}
      </div>

      <div className="card page-section">
        <div className="card-header">
          <h3>Training Run History</h3>
          <span className="subtle">{jobHistory.length ? `${completedRuns} completed / ${jobHistory.length} total` : "No stored runs yet"}</span>
        </div>

        {jobHistory.length ? (
          <div className="training-history-grid">
            {jobHistory.map((run) => {
              const historyResult = run.result || null;
              return (
                <article className="card card-nested training-history-card" key={run.job_id}>
                  <div className="card-header training-history-header">
                    <div className="training-run-heading">
                      <h4 title={runSourceLabel(run)}>{runSourceLabel(run)}</h4>
                      <p className="subtle training-run-id" title={run.job_id}>{run.job_id}</p>
                    </div>
                    <span className={`status-pill ${trainingStatusTone(run.status)}`}>{run.status}</span>
                  </div>

                  <div className="detail-list training-history-meta">
                    <div><strong>Created:</strong> {run.created_at ? formatTimestamp(run.created_at) : "n/a"}</div>
                    <div><strong>Started:</strong> {run.started_at ? formatTimestamp(run.started_at) : "n/a"}</div>
                    <div><strong>Finished:</strong> {run.finished_at ? formatTimestamp(run.finished_at) : "n/a"}</div>
                    {run.error ? <div><strong>Error:</strong> {run.error}</div> : null}
                  </div>

                  {historyResult ? (
                    <>
                      <div className="result-stat-strip">
                        <div className="result-stat-card">
                          <span>Best algorithm</span>
                          <strong>{historyResult.best_algorithm || "n/a"}</strong>
                        </div>
                        <div className="result-stat-card">
                          <span>Resolved mode</span>
                          <strong>{historyResult.resolved_mode || "n/a"}</strong>
                        </div>
                        <div className="result-stat-card">
                          <span>Elapsed</span>
                          <strong>{historyResult.elapsed_s === 0 || historyResult.elapsed_s ? `${historyResult.elapsed_s}s` : "n/a"}</strong>
                        </div>
                      </div>
                      <details className="result-debug">
                        <summary>Raw training payload</summary>
                        <pre className="code-block">{pretty(historyResult)}</pre>
                      </details>
                    </>
                  ) : null}

                  <div className="training-history-actions">
                    <button
                      className="btn btn-secondary"
                      disabled={deletingJobId === run.job_id || isActiveJobStatus(run.status)}
                      onClick={() => deleteHistoryRun(run)}
                      type="button"
                    >
                      {deletingJobId === run.job_id ? "Deleting..." : "Delete Log"}
                    </button>
                  </div>
                </article>
              );
            })}
          </div>
        ) : (
          <p className="subtle">Completed and failed training runs will remain visible here as tenant history.</p>
        )}
      </div>
    </Layout>
  );
}
