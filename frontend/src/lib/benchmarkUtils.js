export const METRIC_ALIASES = {
  precision: ["Precision@K", "Precision", "precision", "precision_at_k"],
  recall: ["Recall@K", "Recall", "recall", "recall_at_k"],
  ndcg: ["NDCG@K", "NDCG", "ndcg", "ndcg_at_k"],
  rmse: ["RMSE", "rmse"],
  trainingTime: ["Time (s)", "training_time", "elapsed_s", "time"],
  composite: ["Composite Score", "composite_score", "score"],
};

export function toNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

export function formatNumber(value, digits = 4) {
  const numeric = toNumber(value);
  return numeric === null ? "n/a" : numeric.toFixed(digits);
}

export function formatDurationSeconds(value) {
  const numeric = toNumber(value);
  if (numeric === null) {
    return "n/a";
  }
  if (numeric < 60) {
    return `${numeric.toFixed(2)}s`;
  }
  const mins = Math.floor(numeric / 60);
  const secs = numeric % 60;
  return `${mins}m ${secs.toFixed(1)}s`;
}

export function metricValue(row, aliasKey) {
  const keys = METRIC_ALIASES[aliasKey] || [];
  for (const key of keys) {
    const numeric = toNumber(row?.[key]);
    if (numeric !== null) {
      return numeric;
    }
  }
  return null;
}

function modelIdentity(model = {}) {
  if (model.model_id) {
    return `model:${model.model_id}`;
  }
  if (model.algorithm) {
    return `algorithm:${String(model.algorithm).toLowerCase()}`;
  }
  return "";
}

function firstMetricEntry(metrics = {}) {
  for (const [name, value] of Object.entries(metrics || {})) {
    const numeric = toNumber(value);
    if (numeric !== null) {
      return { metric_name: name, metric_value: numeric };
    }
  }
  return { metric_name: null, metric_value: null };
}

export function dedupeModelRows(rows = []) {
  const seen = new Set();
  const out = [];
  for (const row of rows || []) {
    if (!row || typeof row !== "object") {
      continue;
    }
    const key = modelIdentity(row);
    if (!key || seen.has(key)) {
      continue;
    }
    seen.add(key);
    out.push(row);
  }
  return out;
}

export function buildRankedModelLookup(rankedModels = []) {
  const byAlgorithm = new Map();
  const byModelId = new Map();

  for (const model of dedupeModelRows(rankedModels)) {
    if (model?.model_id && !byModelId.has(model.model_id)) {
      byModelId.set(model.model_id, model);
    }
    if (model?.algorithm && !byAlgorithm.has(model.algorithm)) {
      byAlgorithm.set(model.algorithm, model);
    }
  }

  return { byAlgorithm, byModelId };
}

export function matchRankedModel(row, rankedLookup) {
  if (!row || !rankedLookup) {
    return null;
  }
  if (row.model_id && rankedLookup.byModelId.has(row.model_id)) {
    return rankedLookup.byModelId.get(row.model_id);
  }
  if (row.Algorithm && rankedLookup.byAlgorithm.has(row.Algorithm)) {
    return rankedLookup.byAlgorithm.get(row.Algorithm);
  }
  return null;
}

function buildOptionLabel(model = {}) {
  const rank = model.rank ? `#${model.rank}` : null;
  const algorithm = model.algorithm || "Unknown";
  const metricName = model.metric_name || null;
  const metricValue = toNumber(model.metric_value);
  const metricSegment =
    metricName && metricValue !== null ? `${metricName} ${metricValue.toFixed(4)}` : null;
  return [rank, algorithm, metricSegment].filter(Boolean).join(" · ");
}

function registryModelToChoice(model = {}) {
  const metric = firstMetricEntry(model.metrics);
  return {
    algorithm: model.algorithm || "",
    label: buildOptionLabel({
      algorithm: model.algorithm,
      metric_name: metric.metric_name,
      metric_value: metric.metric_value,
    }),
    metric_name: metric.metric_name,
    metric_value: metric.metric_value,
    model_id: model.model_id || "",
    promoted: Boolean(model.promoted),
    rank: null,
  };
}

export function buildDynamicModelChoices(options = {}, registryModels = []) {
  const benchmarkModels = dedupeModelRows(
    options?.ranked_models?.length
      ? options.ranked_models
      : options?.selected_models?.length
        ? options.selected_models
        : options?.selection_policy?.selected_models?.length
          ? options.selection_policy.selected_models
          : options?.supported_models || []
  );

  const supportedModels = dedupeModelRows([
    ...benchmarkModels,
    ...(options?.supported_models || []),
    ...(registryModels || []).map(registryModelToChoice),
  ]).map((model) => ({
    ...model,
    optionLabel: model.label || buildOptionLabel(model),
  }));

  const primaryModels = (benchmarkModels.length ? benchmarkModels : supportedModels).map((model) => ({
    ...model,
    optionLabel: model.label || buildOptionLabel(model),
  }));

  const algorithmChoices = [];
  const seenAlgorithms = new Set();
  for (const model of primaryModels) {
    const algorithm = model?.algorithm;
    if (!algorithm || seenAlgorithms.has(algorithm)) {
      continue;
    }
    seenAlgorithms.add(algorithm);
    algorithmChoices.push({
      algorithm,
      label: algorithm,
      model_id: model?.model_id || "",
      optionLabel: algorithm,
      rank: model?.rank ?? null,
    });
  }

  return {
    algorithmChoices,
    benchmarkModels: primaryModels,
    supportedModels,
  };
}

export function normalizeLeaderboardRows(leaderboard = [], rankedModels = []) {
  const rankedLookup = buildRankedModelLookup(rankedModels);
  return leaderboard.map((row, index) => {
    const rankedModel = matchRankedModel(row, rankedLookup);

    return {
      rank: index + 1,
      algorithm: row.Algorithm || rankedModel?.algorithm || "Unknown",
      modelId: row.model_id || rankedModel?.model_id || "",
      precision: metricValue(row, "precision"),
      recall: metricValue(row, "recall"),
      ndcg: metricValue(row, "ndcg"),
      rmse: metricValue(row, "rmse"),
      trainingTime: metricValue(row, "trainingTime"),
      composite: metricValue(row, "composite"),
      selectionScore: toNumber(rankedModel?.selection_score_pct),
      primaryMetricName: rankedModel?.metric_name || null,
      primaryMetricValue: toNumber(rankedModel?.metric_value),
      isPromoted: Boolean(row.Promoted),
      raw: row,
      rankedModel,
    };
  });
}

export function sortRows(rows, sortKey, sortDirection) {
  const direction = sortDirection === "asc" ? 1 : -1;
  const compareValue = (row) => {
    if (sortKey === "algorithm") {
      return row.algorithm.toLowerCase();
    }
    if (sortKey === "rank") {
      return row.rank;
    }
    if (sortKey === "status") {
      return row.isPromoted ? 0 : 1;
    }
    return row[sortKey];
  };

  const withIndex = rows.map((row, index) => ({ row, index }));
  withIndex.sort((a, b) => {
    const left = compareValue(a.row);
    const right = compareValue(b.row);

    if (left === null || left === undefined) {
      return 1;
    }
    if (right === null || right === undefined) {
      return -1;
    }

    if (typeof left === "string" || typeof right === "string") {
      const cmp = String(left).localeCompare(String(right));
      if (cmp !== 0) {
        return cmp * direction;
      }
      return a.index - b.index;
    }

    if (left === right) {
      return a.index - b.index;
    }

    return left > right ? direction : -direction;
  });

  return withIndex.map((item) => item.row);
}
