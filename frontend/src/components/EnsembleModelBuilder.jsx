export default function EnsembleModelBuilder({
  jsonPreview,
  modelOptions,
  onAddRow,
  onChangeRow,
  onRemoveRow,
  rowErrors,
  rows,
  summaryWarning,
}) {
  return (
    <div className="ensemble-builder">
      <div className="ensemble-list">
        {rows.map((row, index) => {
          const currentKey = row.model_id || row.algorithm;
          const errors = rowErrors[index] || {};
          return (
            <div className="ensemble-row" key={row.id}>
              <div className="field">
                <label htmlFor={`ensemble-model-${row.id}`}>Benchmark Model</label>
                <select
                  id={`ensemble-model-${row.id}`}
                  onChange={(event) => onChangeRow(row.id, event.target.value)}
                  value={currentKey}
                >
                  <option value="">Select a benchmark-ranked model</option>
                  {modelOptions.map((option) => {
                    const optionKey = option.model_id || option.algorithm;
                    const usedElsewhere = rows.some(
                      (candidate) =>
                        candidate.id !== row.id &&
                        (candidate.model_id || candidate.algorithm) === optionKey
                    );
                    return (
                      <option disabled={usedElsewhere} key={optionKey} value={optionKey}>
                        {option.optionLabel}
                      </option>
                    );
                  })}
                </select>
                {errors.model ? <p className="field-error">{errors.model}</p> : null}
              </div>

              <div className="field">
                <label htmlFor={`ensemble-weight-${row.id}`}>Weight</label>
                <input
                  id={`ensemble-weight-${row.id}`}
                  inputMode="decimal"
                  min="0"
                  onChange={(event) => onChangeRow(row.id, currentKey, event.target.value)}
                  placeholder="0.50"
                  step="0.05"
                  type="number"
                  value={row.weight}
                />
                {errors.weight ? <p className="field-error">{errors.weight}</p> : null}
              </div>

              <div className="ensemble-row-actions">
                <button
                  className="btn btn-secondary"
                  disabled={rows.length <= 1}
                  onClick={() => onRemoveRow(row.id)}
                  type="button"
                >
                  Remove
                </button>
              </div>
            </div>
          );
        })}
      </div>

      <div className="actions-row ensemble-actions">
        <button className="btn btn-secondary" onClick={onAddRow} type="button">
          + Add Model
        </button>
      </div>

      <p className="field-hint">
        Pick benchmark-ranked models and weights here. The backend still receives the same `models` JSON array and
        can auto-normalize weights if they do not sum to 1.0.
      </p>

      {summaryWarning ? <div className="form-warning">{summaryWarning}</div> : null}

      <div className="ensemble-preview">
        <div className="card-header">
          <h4>Generated JSON Preview</h4>
          <span className="subtle">Read-only compatibility payload</span>
        </div>
        <pre className="code-block ensemble-code-block">{jsonPreview}</pre>
      </div>
    </div>
  );
}
