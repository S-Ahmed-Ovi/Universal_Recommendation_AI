function formatMetric(value, digits = 4) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : "n/a";
}

function formatScore(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${numeric.toFixed(2)}%` : "n/a";
}

function buildWinnerHeadline(model) {
  if (!model) {
    return "No model explanation is available yet.";
  }
  if (model.selection_score_pct !== null && model.selection_score_pct !== undefined) {
    return `${model.algorithm || "This model"} leads with ${formatScore(model.selection_score_pct)} selection confidence.`;
  }
  return `${model.algorithm || "This model"} is currently the highest-ranked model.`;
}

export default function ModelDecisionPanel({
  bestAlgorithm,
  bestModelId,
  options,
  policy,
  resolvedMode,
}) {
  const selectionPolicy = policy || options?.selection_policy || {};
  const rankedModels = options?.ranked_models?.length
    ? options.ranked_models
    : options?.selected_models?.length
      ? options.selected_models
      : options?.supported_models?.length
        ? options.supported_models
        : selectionPolicy?.selected_models || [];
  const visibleModels =
    (selectionPolicy?.selected_models && selectionPolicy.selected_models.length
      ? selectionPolicy.selected_models
      : rankedModels
    );
  const leadModel = visibleModels[0] || rankedModels[0] || null;
  const explanation =
    leadModel?.performance_summary ||
    options?.best_model_explanation ||
    selectionPolicy?.best_model_explanation ||
    selectionPolicy?.reason ||
    "No model explanation is available yet.";
  const summaryBlocks = [
    {
      label: "Performance",
      text: leadModel?.performance_summary,
    },
    {
      label: "Dataset fit",
      text: leadModel?.fit_summary,
    },
    {
      label: "Reliability",
      text: leadModel?.reliability_summary,
    },
    {
      label: "Competitive edge",
      text: leadModel?.comparison_to_next,
    },
  ].filter((item) => item.text);

  return (
    <div className="decision-layout">
      <div className="decision-summary">
        <div className="decision-summary-card">
          <p className="eyebrow">Why It Won</p>
          <h3>{bestAlgorithm || options?.best_algorithm || "No ranked model yet"}</h3>
          <p className="decision-summary-lead">{buildWinnerHeadline(leadModel)}</p>
          <p className="subtle">{explanation}</p>

          {summaryBlocks.length ? (
            <div className="decision-summary-points">
              {summaryBlocks.map((block) => (
                <div className="decision-summary-point" key={block.label}>
                  <span>{block.label}</span>
                  <strong>{block.text}</strong>
                </div>
              ))}
            </div>
          ) : null}
        </div>

        <div className="decision-kpis">
          <div className="decision-kpi">
            <span>Resolved mode</span>
            <strong>{resolvedMode || options?.resolved_mode || "n/a"}</strong>
          </div>
          <div className="decision-kpi">
            <span>Serving strategy</span>
            <strong>{selectionPolicy?.recommended_strategy || "n/a"}</strong>
          </div>
          <div className="decision-kpi">
            <span>Ranked models</span>
            <strong>{options?.supported_model_count || rankedModels.length || 0}</strong>
          </div>
          <div className="decision-kpi">
            <span>Best model ID</span>
            <strong>{bestModelId || options?.best_model_id || "n/a"}</strong>
          </div>
        </div>
      </div>

      <div className="decision-model-grid">
        {visibleModels.length ? (
          visibleModels.map((model, index) => (
            <article className="decision-model-card" key={model.model_id || `${model.algorithm}-${index}`}>
              <div className="decision-model-head">
                <div>
                  <p className="eyebrow">Rank {model.rank || index + 1}</p>
                  <h4>{model.algorithm || "Unknown model"}</h4>
                </div>
                <span className={`status-pill ${index === 0 ? "status-live" : "status-muted"}`}>
                  {index === 0 ? "Lead" : "Candidate"}
                </span>
              </div>

              <div className="decision-metric-row">
                <div>
                  <span>Selection score</span>
                  <strong>{formatScore(model.selection_score_pct)}</strong>
                </div>
                <div>
                  <span>{model.metric_name || "Primary metric"}</span>
                  <strong>{formatMetric(model.metric_value)}</strong>
                </div>
                <div>
                  <span>Composite score</span>
                  <strong>{formatMetric(model.composite_score)}</strong>
                </div>
              </div>

              <div className="decision-story">
                {model.performance_summary ? (
                  <div className="decision-story-block">
                    <span>Performance</span>
                    <p>{model.performance_summary}</p>
                  </div>
                ) : null}
                {model.fit_summary ? (
                  <div className="decision-story-block">
                    <span>Dataset fit</span>
                    <p>{model.fit_summary}</p>
                  </div>
                ) : null}
                {model.reliability_summary ? (
                  <div className="decision-story-block">
                    <span>Reliability</span>
                    <p>{model.reliability_summary}</p>
                  </div>
                ) : null}
              </div>

              {model.comparison_to_next ? (
                <div className="decision-note">{model.comparison_to_next}</div>
              ) : null}

              {model.reasons?.length ? (
                <div className="decision-reason-list">
                  {model.reasons.slice(0, 4).map((reason) => (
                    <div className="decision-reason-item" key={reason}>
                      {reason}
                    </div>
                  ))}
                </div>
              ) : null}
            </article>
          ))
        ) : (
          <div className="decision-empty">
            <p className="subtle">No ranked model candidates are available yet.</p>
          </div>
        )}
      </div>

      {selectionPolicy?.decision_rules?.length ? (
        <div className="decision-rules">
          <div className="card-header">
            <h4>Decision Rules</h4>
            <span className="subtle">{selectionPolicy?.display_title || "Selection policy"}</span>
          </div>
          <div className="decision-rule-list">
            {selectionPolicy.decision_rules.map((rule) => (
              <div className="decision-rule" key={rule}>
                {rule}
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}
