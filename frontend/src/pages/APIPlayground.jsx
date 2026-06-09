import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { RefreshCw } from "lucide-react";
import EnsembleModelBuilder from "../components/EnsembleModelBuilder";
import Layout from "../components/Layout";
import TenantRequiredState from "../components/TenantRequiredState";
import { tenantRequest } from "../api";
import { useAsyncData } from "../hooks/useAsyncData";
import { useTenantSession } from "../hooks/useAuthSession";
import { buildDynamicModelChoices } from "../lib/benchmarkUtils";

const DEFAULT_ENSEMBLE_ROWS = [
  { algorithm: "svd", model_id: "", weight: "0.60" },
  { algorithm: "als", model_id: "", weight: "0.40" },
];
const emptyMetadata = {
  models: [],
  options: null,
};

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function createRow(id, patch = {}) {
  return {
    algorithm: "",
    id,
    model_id: "",
    weight: "",
    ...patch,
  };
}

function initialEnsembleRows() {
  return DEFAULT_ENSEMBLE_ROWS.map((row, index) => createRow(`ensemble-${index + 1}`, row));
}

function createDefaultRows(modelOptions, nextRowId) {
  const seeds = modelOptions.slice(0, 2).map((option, index) =>
    createRow(nextRowId(), {
      algorithm: option.algorithm || "",
      model_id: option.model_id || "",
      weight: index === 0 ? "0.60" : "0.40",
    })
  );

  if (seeds.length >= 2) {
    return seeds;
  }

  const fallback = [...seeds];
  while (fallback.length < 2) {
    fallback.push(createRow(nextRowId()));
  }
  return fallback;
}

function buildSelectionKey(model = {}) {
  return model.model_id || model.algorithm || "";
}

function validatePlaygroundForm(form, action, ensembleValidation) {
  const nextErrors = {};
  const parsedTopN = Number(form.topN);

  if ((action === "recommend" || action === "explain") && !form.userId.trim()) {
    nextErrors.userId = "User ID is required for this request.";
  }

  if ((action === "similar" || action === "explain") && !form.itemId.trim()) {
    nextErrors.itemId = "Item ID is required for this request.";
  }

  if ((action === "recommend" || action === "similar") && (!Number.isFinite(parsedTopN) || parsedTopN <= 0)) {
    nextErrors.topN = "Top N must be greater than zero.";
  }

  if (action === "explain" && !Number.isFinite(Number(form.score))) {
    nextErrors.score = "Explain score must be numeric.";
  }

  if (action === "recommend" && form.strategy === "ensemble_weighted") {
    if (ensembleValidation.payload.length < 2) {
      nextErrors.ensembleModels = "Add at least two benchmark-ranked models for a weighted ensemble.";
    } else if (ensembleValidation.hasBlockingErrors) {
      nextErrors.ensembleModels = "Resolve the highlighted ensemble model selections before calling /recommend.";
    }
  }

  return nextErrors;
}

export default function APIPlayground() {
  const [ensembleRows, setEnsembleRows] = useState(() => initialEnsembleRows());
  const [actionError, setActionError] = useState("");
  const [fieldErrors, setFieldErrors] = useState({});
  const [loadingAction, setLoadingAction] = useState("");
  const [responseText, setResponseText] = useState(pretty({
    hint: "Select a tenant API key in Tenants, then call a live endpoint.",
  }));
  const [form, setForm] = useState({
    algorithm: "",
    // itemId: "item-1001",
    modelId: "",
    score: "0",
    strategy: "best_promoted_model",
    topN: 10,
    userId: "1024",
  });

  const rowIdRef = useRef(ensembleRows.length);
  const {
    hasTenantApiKey: hasKey,
    tenantApiKey,
    tenantId,
  } = useTenantSession();
  const loadMetadata = useCallback(async ({ signal }) => {
    const [optionsPayload, modelPayload] = await Promise.all([
      tenantRequest("/recommend/options", { signal }),
      tenantRequest("/models", { signal }),
    ]);

    return {
      models: modelPayload?.models || [],
      options: optionsPayload || null,
    };
  }, [tenantApiKey, tenantId]);
  const {
    data: metadata,
    error,
    loading: metadataLoading,
    reload: reloadMetadata,
    setData: setMetadata,
  } = useAsyncData(loadMetadata, {
    enabled: hasKey,
    initialData: emptyMetadata,
  });
  const options = metadata?.options || null;
  const models = metadata?.models || [];

  const nextRowId = () => {
    rowIdRef.current += 1;
    return `ensemble-${rowIdRef.current}`;
  };

  const dynamicModelChoices = useMemo(
    () => buildDynamicModelChoices(options || {}, models),
    [models, options],
  );
  const benchmarkModelOptions = dynamicModelChoices.benchmarkModels;
  const supportedModels = dynamicModelChoices.supportedModels;
  const algorithmChoices = dynamicModelChoices.algorithmChoices;

  const modelOptions = useMemo(
    () => (benchmarkModelOptions.length ? benchmarkModelOptions : supportedModels),
    [benchmarkModelOptions, supportedModels],
  );

  const modelOptionMap = useMemo(() => {
    const map = new Map();
    for (const option of modelOptions) {
      const key = buildSelectionKey(option);
      if (key && !map.has(key)) {
        map.set(key, option);
      }
    }
    return map;
  }, [modelOptions]);

  const algorithmSet = useMemo(
    () => new Set(algorithmChoices.map((option) => option.algorithm).filter(Boolean)),
    [algorithmChoices],
  );

  useEffect(() => {
    if (!modelOptions.length) {
      return;
    }

    setForm((current) => {
      const next = { ...current };
      if (current.modelId && !modelOptionMap.has(current.modelId)) {
        const fallback = current.algorithm
          ? modelOptions.find((option) => option.algorithm === current.algorithm)
          : null;
        next.modelId = fallback ? buildSelectionKey(fallback) : "";
      }
      if (current.algorithm && !algorithmSet.has(current.algorithm)) {
        next.algorithm = current.modelId && modelOptionMap.get(current.modelId)?.algorithm
          ? modelOptionMap.get(current.modelId).algorithm
          : "";
      }
      return next;
    });

    setEnsembleRows((current) => {
      if (!current.length) {
        return createDefaultRows(modelOptions, nextRowId);
      }

      return current.map((row) => {
        const currentKey = buildSelectionKey(row);
        const exact = currentKey ? modelOptionMap.get(currentKey) : null;
        const fallback = !exact && row.algorithm
          ? modelOptions.find((option) => option.algorithm === row.algorithm)
          : null;
        const resolved = exact || fallback;
        if (!resolved) {
          return row;
        }
        return {
          ...row,
          algorithm: resolved.algorithm || row.algorithm,
          model_id: resolved.model_id || row.model_id,
        };
      });
    });
  }, [algorithmSet, modelOptionMap, modelOptions]);

  const ensembleValidation = useMemo(() => {
    const rowErrors = ensembleRows.map(() => ({}));
    const seen = new Set();
    let hasBlockingErrors = false;
    let totalWeight = 0;

    const payload = ensembleRows.reduce((items, row, index) => {
      const selectionKey = buildSelectionKey(row);
      const resolvedOption = selectionKey ? modelOptionMap.get(selectionKey) : null;
      const weightValue = Number(row.weight);

      if (!selectionKey) {
        rowErrors[index].model = "Choose a benchmark-ranked model.";
        hasBlockingErrors = true;
      } else if (seen.has(selectionKey)) {
        rowErrors[index].model = "Duplicate model selections are not allowed.";
        hasBlockingErrors = true;
      } else {
        seen.add(selectionKey);
      }

      if (!row.weight.trim()) {
        rowErrors[index].weight = "Enter a positive weight.";
        hasBlockingErrors = true;
      } else if (!Number.isFinite(weightValue) || weightValue <= 0) {
        rowErrors[index].weight = "Weight must be a positive number.";
        hasBlockingErrors = true;
      } else {
        totalWeight += weightValue;
      }

      if (!rowErrors[index].model && !rowErrors[index].weight) {
        items.push({
          algorithm: resolvedOption?.algorithm || row.algorithm,
          model_id: resolvedOption?.model_id || row.model_id || undefined,
          weight: weightValue,
        });
      }

      return items;
    }, []);

    const summaryWarning =
      payload.length >= 2 && totalWeight > 0 && Math.abs(totalWeight - 1) > 0.001
        ? `Weights currently sum to ${totalWeight.toFixed(2)}. The backend can auto-normalize them, but a total of 1.00 is easier to reason about.`
        : "";

    return {
      hasBlockingErrors,
      jsonPreview: pretty(payload),
      payload,
      rowErrors,
      summaryWarning,
    };
  }, [ensembleRows, modelOptionMap]);

  function updateField(key, value) {
    setForm((current) => {
      const next = { ...current, [key]: value };
      if (key === "modelId" && value) {
        const selectedModel = modelOptionMap.get(value);
        if (selectedModel?.algorithm) {
          next.algorithm = selectedModel.algorithm;
        }
      }
      if (key === "strategy" && value === "ensemble_weighted" && !ensembleRows.length) {
        setEnsembleRows(createDefaultRows(modelOptions, nextRowId));
      }
      return next;
    });
    setFieldErrors((current) => ({ ...current, [key]: "", ensembleModels: "" }));
  }

  function handleEnsembleRowChange(rowId, selectionKey, nextWeight) {
    setEnsembleRows((current) =>
      current.map((row) => {
        if (row.id !== rowId) {
          return row;
        }

        const resolvedOption = selectionKey ? modelOptionMap.get(selectionKey) : null;
        return {
          ...row,
          algorithm: resolvedOption?.algorithm || "",
          model_id: resolvedOption?.model_id || "",
          weight: nextWeight ?? row.weight,
        };
      })
    );
    setFieldErrors((current) => ({ ...current, ensembleModels: "" }));
  }

  function addEnsembleRow() {
    setEnsembleRows((current) => [...current, createRow(nextRowId())]);
    setFieldErrors((current) => ({ ...current, ensembleModels: "" }));
  }

  function removeEnsembleRow(rowId) {
    setEnsembleRows((current) => current.filter((row) => row.id !== rowId));
    setFieldErrors((current) => ({ ...current, ensembleModels: "" }));
  }

  async function runRecommend() {
    const nextErrors = validatePlaygroundForm(form, "recommend", ensembleValidation);
    if (Object.keys(nextErrors).some((key) => nextErrors[key])) {
      setFieldErrors(nextErrors);
      return;
    }

    setLoadingAction("recommend");
    setActionError("");
    setFieldErrors({});

    try {
      const body = {
        top_n: Number(form.topN),
        user_id: form.userId.trim(),
        strategy: form.strategy,
      };

      if (form.strategy === "single_model") {
        const selectedModel = form.modelId.trim() ? modelOptionMap.get(form.modelId.trim()) : null;
        if (selectedModel?.model_id) {
          body.model_id = selectedModel.model_id;
        }
        if (selectedModel?.algorithm) {
          body.algorithm = selectedModel.algorithm;
        } else if (form.algorithm.trim()) {
          body.algorithm = form.algorithm.trim();
        }
      } else if (form.strategy === "ensemble_weighted") {
        body.models = ensembleValidation.payload;
      }

      const data = await tenantRequest("/recommend", {
        body,
        method: "POST",
      });
      setResponseText(pretty(data));
    } catch (requestError) {
      setActionError(requestError.message || "Recommendation call failed");
    } finally {
      setLoadingAction("");
    }
  }

  /*
  async function runSimilar() {
    const nextErrors = validatePlaygroundForm(form, "similar", ensembleValidation);
    if (Object.keys(nextErrors).some((key) => nextErrors[key])) {
      setFieldErrors(nextErrors);
      return;
    }

    setLoadingAction("similar");
    setActionError("");
    setFieldErrors({});

    try {
      const query = new URLSearchParams({ top_n: String(form.topN) });
      if (form.algorithm.trim()) {
        query.set("algorithm", form.algorithm.trim());
      }

      const data = await tenantRequest(`/similar/${encodeURIComponent(form.itemId.trim())}?${query.toString()}`);
      setResponseText(pretty(data));
    } catch (requestError) {
      setActionError(requestError.message || "Similar-items call failed");
    } finally {
      setLoadingAction("");
    }
  }
    */

  async function runExplain() {
    const nextErrors = validatePlaygroundForm(form, "explain", ensembleValidation);
    if (Object.keys(nextErrors).some((key) => nextErrors[key])) {
      setFieldErrors(nextErrors);
      return;
    }

    setLoadingAction("explain");
    setActionError("");
    setFieldErrors({});

    try {
      const query = new URLSearchParams({ score: String(form.score || "0") });
      if (form.algorithm.trim()) {
        query.set("algorithm", form.algorithm.trim());
      }

      const path = `/explain/${encodeURIComponent(form.userId.trim())}/${encodeURIComponent(form.itemId.trim())}?${query.toString()}`;
      const data = await tenantRequest(path);
      setResponseText(pretty(data));
    } catch (requestError) {
      setActionError(requestError.message || "Explain call failed");
    } finally {
      setLoadingAction("");
    }
  }

  async function runListModels() {
    setLoadingAction("models");
    setActionError("");

    try {
      const data = await tenantRequest("/models");
      setResponseText(pretty(data));
      setMetadata((current) => ({
        ...current,
        models: data?.models || [],
      }));
    } catch (requestError) {
      setActionError(requestError.message || "Model listing failed");
    } finally {
      setLoadingAction("");
    }
  }

  const pageError = actionError || error;
  const busy = Boolean(loadingAction) || metadataLoading;

  return (
    <Layout
      summary="Exercise live tenant-authenticated recommendation endpoints with benchmark-aware request inputs and readable responses."
      title="API Playground"
    >
      {!hasKey ? (
        <TenantRequiredState description="The live recommendation endpoints use `X-API-Key`, not the admin login token. Create or rotate a tenant key in Tenants, then come back here." />
      ) : null}

      {pageError ? <div className="form-error">{pageError}</div> : null}

      <div className="grid">
        <div className="card">
          <div className="card-header">
            <h3>Request Builder</h3>
            <span className="subtle">{tenantId ? `Tenant: ${tenantId}` : "Tenant key loaded"}</span>
          </div>

          <div className="field-grid">
            <div className="field">
              <label htmlFor="playground-user-id">User ID</label>
              <input id="playground-user-id" onChange={(event) => updateField("userId", event.target.value)} value={form.userId} />
              {fieldErrors.userId ? <p className="field-error">{fieldErrors.userId}</p> : null}
            </div>
            <div className="field">
              <label htmlFor="playground-item-id">Item ID</label>
              <input id="playground-item-id" onChange={(event) => updateField("itemId", event.target.value)} value={form.itemId} />
              {fieldErrors.itemId ? <p className="field-error">{fieldErrors.itemId}</p> : null}
            </div>
            <div className="field">
              <label htmlFor="playground-top-n">Top N</label>
              <select id="playground-top-n" onChange={(event) => updateField("topN", event.target.value)} value={form.topN}>
                <option value={5}>5</option>
                <option value={10}>10</option>
              </select>
              {fieldErrors.topN ? <p className="field-error">{fieldErrors.topN}</p> : null}
            </div>
            <div className="field">
              <label htmlFor="playground-strategy">Strategy</label>
              <select id="playground-strategy" onChange={(event) => updateField("strategy", event.target.value)} value={form.strategy}>
                <option value="best_promoted_model">best_promoted_model</option>
                <option value="single_model">single_model</option>
                <option value="ensemble_weighted">ensemble_weighted</option>
              </select>
            </div>

            {form.strategy === "ensemble_weighted" ? (
              <div className="field field-full">
                <label>Weighted Ensemble Builder</label>
                <EnsembleModelBuilder
                  jsonPreview={ensembleValidation.jsonPreview}
                  modelOptions={modelOptions}
                  onAddRow={addEnsembleRow}
                  onChangeRow={handleEnsembleRowChange}
                  onRemoveRow={removeEnsembleRow}
                  rowErrors={ensembleValidation.rowErrors}
                  rows={ensembleRows}
                  summaryWarning={ensembleValidation.summaryWarning}
                />
                {fieldErrors.ensembleModels ? <p className="field-error">{fieldErrors.ensembleModels}</p> : null}
              </div>
            ) : (
              <div className="field">
                <label htmlFor="playground-model-id">Benchmark Model</label>
                <select id="playground-model-id" onChange={(event) => updateField("modelId", event.target.value)} value={form.modelId}>
                  <option value="">Automatic</option>
                  {modelOptions.map((model) => (
                    <option key={buildSelectionKey(model)} value={buildSelectionKey(model)}>
                      {model.optionLabel}
                    </option>
                  ))}
                </select>
                <p className="field-hint">This dropdown is populated from the latest benchmark-ranked models for the active tenant.</p>
              </div>
            )}

            <div className="field">
              <label htmlFor="playground-algorithm">Benchmark Algorithm</label>
              <select id="playground-algorithm" onChange={(event) => updateField("algorithm", event.target.value)} value={form.algorithm}>
                <option value="">Automatic</option>
                {algorithmChoices.map((algorithm) => (
                  <option key={algorithm.algorithm} value={algorithm.algorithm}>{algorithm.label}</option>
                ))}
              </select>
              <p className="field-hint">
                Similar and explain routes use this as an algorithm filter. Single-model recommend uses it as a fallback when no specific model is picked.
              </p>
            </div>

            <div className="field">
              <label htmlFor="playground-score">Explain Score</label>
              <input id="playground-score" onChange={(event) => updateField("score", event.target.value)} value={form.score} />
              {fieldErrors.score ? <p className="field-error">{fieldErrors.score}</p> : null}
            </div>
          </div>

          {!benchmarkModelOptions.length && hasKey ? (
            <p className="field-hint">
              No benchmark-ranked models are available yet. Run training first, then refresh metadata to populate the model selectors.
            </p>
          ) : null}

          <div className="actions-row">
            <button className="btn" disabled={!hasKey || busy} onClick={runRecommend} type="button">
              {loadingAction === "recommend" ? "Calling /recommend..." : "POST /recommend"}
            </button>

                              {/* 
                  <button
                    className="btn btn-secondary"
                    disabled={!hasKey || busy}
                    onClick={runSimilar}
                    type="button"
                  >
                    {loadingAction === "similar"
                      ? "Calling /similar..."
                      : "GET /similar"}
                  </button>
                  */}
            <button className="btn btn-secondary" disabled={!hasKey || busy} onClick={runExplain} type="button">
              {loadingAction === "explain" ? "Calling /explain..." : "GET /explain"}
            </button>
            <button className="btn btn-secondary" disabled={!hasKey || busy} onClick={runListModels} type="button">
              {loadingAction === "models" ? "Loading models..." : "GET /models"}
            </button>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3>Live Response</h3>
            <button
              aria-label={metadataLoading ? "Refreshing metadata" : "Refresh metadata"}
              className="btn btn-secondary btn-icon"
              disabled={!hasKey || busy || metadataLoading}
              onClick={() => reloadMetadata()}
              title={metadataLoading ? "Refreshing metadata" : "Refresh metadata"}
              type="button"
            >
              <RefreshCw aria-hidden="true" className={metadataLoading ? "icon-spin" : ""} size={18} />
            </button>
          </div>
          <pre className="code-block">{responseText}</pre>
        </div>
      </div>

      <div className="grid page-section">
        <div className="card">
          <div className="card-header">
            <h3>Benchmark-Ranked Models</h3>
            <span className="subtle">{benchmarkModelOptions.length ? `${benchmarkModelOptions.length} dynamic options` : "Awaiting benchmark results"}</span>
          </div>
          {modelOptions.length ? (
            <div className="detail-list">
              {modelOptions.map((model) => (
                <div key={buildSelectionKey(model)}>
                  <strong>{model.algorithm}</strong>
                  <span className="subtle"> {model.model_id || "no model id"} - {model.metric_name || "metric"} {model.metric_value ?? "n/a"}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="subtle">No promoted or benchmark-ranked tenant models are available yet.</p>
          )}
        </div>

        <div className="card">
          <div className="card-header">
            <h3>Endpoint Notes</h3>
          </div>
          <div className="detail-list">
            <div><strong>Admin login endpoint:</strong> /admin/auth/login</div>
            <div><strong>Recommendation options:</strong> /recommend/options</div>
            <div><strong>Single-user recommend:</strong> /recommend</div>
            <div><strong>Similar items:</strong> /similar/{`{item_id}`}</div>
            <div><strong>Explainability:</strong> /explain/{`{user_id}`}/{`{item_id}`}</div>
          </div>
        </div>
      </div>
    </Layout>
  );
}
