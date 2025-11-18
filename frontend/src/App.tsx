// src/App.tsx
import { useState } from "react";
import type { ChangeEvent } from "react";
import api from "./api/client";
import type {
  UploadResponse,
  EDAResponse,
  ModelingResponse,
  InsightsResponse,
  ColumnTypeLabel,
} from "./types";
import "./App.css";

type OverridesMap = Record<string, ColumnTypeLabel>;

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [uploadMeta, setUploadMeta] = useState<UploadResponse | null>(null);
  const [eda, setEda] = useState<EDAResponse | null>(null);
  const [modeling, setModeling] = useState<ModelingResponse | null>(null);
  const [insights, setInsights] = useState<InsightsResponse | null>(null);

  const [target, setTarget] = useState<string>("");
  const [overrides, setOverrides] = useState<OverridesMap>({});

  const [isUploading, setIsUploading] = useState(false);
  const [isRunningEda, setIsRunningEda] = useState(false);
  const [isRunningModel, setIsRunningModel] = useState(false);
  const [isLoadingInsights, setIsLoadingInsights] = useState(false);
  const [isDownloadingPdf, setIsDownloadingPdf] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const backendUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setUploadMeta(null);
      setEda(null);
      setModeling(null);
      setInsights(null);
      setTarget("");
      setOverrides({});
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a CSV file first.");
      return;
    }
    setError(null);
    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await api.post<UploadResponse>("/api/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setUploadMeta(res.data);
      setEda(null);
      setModeling(null);
      setInsights(null);
      setOverrides({});
    } catch (err: any) {
      console.error(err);
      setError(err?.response?.data?.detail || "Upload failed.");
    } finally {
      setIsUploading(false);
    }
  };
  const handleResetAll = async () => {
    const confirmed = window.confirm(
      "This will delete all uploaded datasets, cached profiles and plots on the server, and clear the UI. Continue?"
    );
    if (!confirmed) return;

    setError(null);
    try {
      await api.post("/api/reset");
      setFile(null);
      setUploadMeta(null);
      setEda(null);
      setModeling(null);
      setInsights(null);
      setTarget("");
      setOverrides({});
    } catch (err: any) {
      console.error(err);
      setError(err?.response?.data?.detail || "Failed to reset server state.");
    }
  };
  const handleRunEda = async () => {
    if (!uploadMeta) {
      setError("Upload a dataset first.");
      return;
    }
    setError(null);
    setIsRunningEda(true);
    try {
      const res = await api.post<EDAResponse>(`/api/eda/${uploadMeta.dataset_id}`, {
        overrides,
      });
      setEda(res.data);
    } catch (err: any) {
      console.error(err);
      setError(err?.response?.data?.detail || "EDA failed.");
    } finally {
      setIsRunningEda(false);
    }
  };

  const handleRunModel = async () => {
    if (!uploadMeta) {
      setError("Upload a dataset first.");
      return;
    }
    if (!target) {
      setError("Please choose a target column.");
      return;
    }
    setError(null);
    setIsRunningModel(true);
    try {
      const res = await api.post<ModelingResponse>(
        `/api/model/${uploadMeta.dataset_id}`,
        {
          target,
          test_size: 0.2,
          random_state: 42,
          overrides,
        }
      );
      setModeling(res.data);
    } catch (err: any) {
      console.error(err);
      setError(err?.response?.data?.detail || "Modeling failed.");
    } finally {
      setIsRunningModel(false);
    }
  };

  const handleLoadInsights = async () => {
    if (!uploadMeta) {
      setError("Upload a dataset first.");
      return;
    }
    setError(null);
    setIsLoadingInsights(true);
    try {
      const params = target ? { target } : {};
      const res = await api.get<InsightsResponse>(
        `/api/insights/${uploadMeta.dataset_id}`,
        { params }
      );
      setInsights(res.data);
    } catch (err: any) {
      console.error(err);
      setError(err?.response?.data?.detail || "Failed to load insights.");
    } finally {
      setIsLoadingInsights(false);
    }
  };

  const handleOverrideChange = (colName: string, value: ColumnTypeLabel) => {
    setOverrides((prev) => ({ ...prev, [colName]: value }));
  };

  const handleOpenReport = () => {
    if (!uploadMeta) return;
    const url = `${backendUrl}/api/report/${uploadMeta.dataset_id}?format=html${
      target ? `&target=${encodeURIComponent(target)}` : ""
    }`;
    window.open(url, "_blank");
  };

  const handleDownloadPdf = async () => {
    if (!uploadMeta) return;
    setError(null);
    setIsDownloadingPdf(true);

    try {
      const params = target ? { target, format: "pdf" } : { format: "pdf" };
      const res = await api.get(`/api/report/${uploadMeta.dataset_id}`, {
        params,
        responseType: "blob",
      });

      const url = window.URL.createObjectURL(new Blob([res.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", `report_${uploadMeta.dataset_id}.pdf`);
      document.body.appendChild(link);
      link.click();

      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      console.error(err);
      setError("Failed to download PDF report.");
    } finally {
      setIsDownloadingPdf(false);
    }
  };

  return (
    <div className="app-root">
      <header className="app-header">
        <div className="title-block">
          <h1>Auto-Analyst</h1>
          <p>Upload a CSV, explore your data, and train baseline models with automated insights.</p>
        </div>
        <div className="header-right">
          <div className="env-chip">Backend: {backendUrl}</div>
          <button className="danger-btn" onClick={handleResetAll}>
            Clear all data
          </button>
        </div>
      </header>

      <main className="app-main">
        <section className="card">
          <div className="card-header">
            <span className="step-badge">1</span>
            <h2>Upload Dataset</h2>
          </div>
          <p className="card-subtitle">
            Choose a CSV file. We&rsquo;ll infer schema, run profiling, and build models from it.
          </p>
          <div className="row">
            <input type="file" accept=".csv" onChange={handleFileChange} />
            <button onClick={handleUpload} disabled={!file || isUploading}>
              {isUploading ? "Uploading..." : "Upload"}
            </button>
          </div>
          {uploadMeta && (
            <div className="meta">
              <p>
                <strong>Dataset ID:</strong> {uploadMeta.dataset_id}
              </p>
              <p>
                <strong>Shape:</strong> {uploadMeta.n_rows} rows × {uploadMeta.n_cols} columns
              </p>
              <p>
                <strong>Columns:</strong> {uploadMeta.columns.join(", ")}
              </p>
            </div>
          )}
        </section>

        <section className="card">
          <div className="card-header">
            <span className="step-badge">2</span>
            <h2>Exploratory Data Analysis</h2>
          </div>
          <p className="card-subtitle">
            Review inferred column types and adjust them before running EDA and plots.
          </p>

          {eda && (
            <p className="small-info">
              Rows: {eda.profile.n_rows}, Columns: {eda.profile.n_cols}. Rows with missing values:{" "}
              {eda.profile.missing.rows_with_missing}.
            </p>
          )}

          {eda && (
            <>
              <h3 className="section-title">Columns &amp; Types</h3>
              <div className="columns-grid">
                {Object.values(eda.profile.columns).map((col) => {
                  const currentOverride =
                    overrides[col.name] ?? (col.overridden_type || (col.effective_type as ColumnTypeLabel));
                  const labelType = col.overridden_type
                    ? `${col.overridden_type} (override)`
                    : col.effective_type;

                  return (
                    <div key={col.name} className="column-card">
                      <div className="column-header">
                        <h4>{col.name}</h4>
                        <span className="chip">{labelType}</span>
                      </div>
                      <p className="muted">
                        Missing: {col.missing_count} ({col.missing_pct.toFixed(1)}%) • Distinct:{" "}
                        {col.distinct_count}
                      </p>

                      <label className="type-label">
                        Column role:
                        <select
                          value={currentOverride}
                          onChange={(e) =>
                            handleOverrideChange(col.name, e.target.value as ColumnTypeLabel)
                          }
                        >
                          <option value="numeric">numeric</option>
                          <option value="categorical">categorical</option>
                          <option value="datetime">datetime</option>
                          <option value="boolean">boolean</option>
                          <option value="id">id (ignore in modeling)</option>
                          <option value="text">text (ignore in modeling)</option>
                        </select>
                      </label>

                      {col.numeric_summary && (
                        <p className="muted">
                          Mean: {col.numeric_summary.mean.toFixed(2)}, Std:{" "}
                          {col.numeric_summary.std.toFixed(2)}
                        </p>
                      )}
                      {col.categorical_summary && (
                        <p className="muted">
                          Unique values: {col.categorical_summary.unique}
                        </p>
                      )}
                    </div>
                  );
                })}
              </div>
            </>
          )}

          {!eda && (
            <p className="muted">
              Run EDA to see inferred types, missingness and plots. You can edit types after the
              first run.
            </p>
          )}

          <div className="row row-end">
            <button onClick={handleRunEda} disabled={!uploadMeta || isRunningEda}>
              {isRunningEda ? "Running EDA..." : eda ? "Re-run EDA with overrides" : "Run EDA"}
            </button>
          </div>

          {eda && (
            <>
              <h3 className="section-title">Plots</h3>

              <h4 className="plot-group-title">Numeric</h4>
              <div className="plots-grid">
                {eda.plots.numeric.map((p) => (
                  <div key={p.column} className="plot-card">
                    <h5>{p.column}</h5>
                    {p.histogram && (
                      <img
                        src={`${backendUrl}${p.histogram}`}
                        alt={`Histogram of ${p.column}`}
                      />
                    )}
                    {p.boxplot && (
                      <img
                        src={`${backendUrl}${p.boxplot}`}
                        alt={`Boxplot of ${p.column}`}
                      />
                    )}
                  </div>
                ))}
              </div>

              <h4 className="plot-group-title">Categorical</h4>
              <div className="plots-grid">
                {eda.plots.categorical.map((p) => (
                  <div key={p.column} className="plot-card">
                    <h5>{p.column}</h5>
                    {p.barplot && (
                      <img
                        src={`${backendUrl}${p.barplot}`}
                        alt={`Barplot of ${p.column}`}
                      />
                    )}
                  </div>
                ))}
              </div>

              <h4 className="plot-group-title">Correlation Heatmap</h4>
              {eda.plots.correlation_heatmap ? (
                <img
                  className="heatmap-image"
                  src={`${backendUrl}${eda.plots.correlation_heatmap.path}`}
                  alt="Correlation Heatmap"
                />
              ) : (
                <p className="muted">Not enough numeric columns for a correlation heatmap.</p>
              )}
            </>
          )}
        </section>

        {/* ... Modeling and Insights sections ... */}
        <section className="card">
          <div className="card-header">
            <span className="step-badge">3</span>
            <h2>Baseline Modeling</h2>
          </div>
          {/* ... Keep existing modeling section code exactly as is ... */}
             <p className="card-subtitle">
            Choose a target variable. The system will detect the task type and train baseline
            models using your column-type overrides.
          </p>
          <div className="row">
            <label className="inline-label">
              Target column:
              <select
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                disabled={!uploadMeta}
              >
                <option value="">Select target...</option>
                {uploadMeta?.columns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
            </label>
            <button
              onClick={handleRunModel}
              disabled={!uploadMeta || !target || isRunningModel}
            >
              {isRunningModel ? "Running model..." : "Run baseline model"}
            </button>
          </div>

          {modeling && (
            <div className="model-results">
              <h3>
                Task: {modeling.task_type} on target "{modeling.target}"
              </h3>

              <p className="muted">
                Best candidate:{" "}
                <strong>
                  {modeling.best_model} ({modeling.best_model_class})
                </strong>{" "}
                &mdash; test size: {(modeling.test_size * 100).toFixed(0)}%, random_state:{" "}
                {modeling.random_state}
              </p>

              <h4>Best Model Metrics</h4>
              <ul>
                {Object.entries(modeling.metrics).map(([name, value]) => (
                  <li key={name}>
                    {name}: {value.toFixed(3)}
                  </li>
                ))}
              </ul>

              <h4>Key Hyperparameters</h4>
              <ul>
                {Object.entries(modeling.best_model_params)
                  .slice(0, 8)
                  .map(([name, value]) => (
                    <li key={name}>
                      {name}: <code>{String(value)}</code>
                    </li>
                  ))}
              </ul>
              <p className="muted">Showing a subset of the best model's hyperparameters.</p>

              <h4>All Candidate Models</h4>
              <ul>
                {Object.entries(modeling.models).map(([name, info]) => (
                  <li key={name}>
                    <strong>{name}</strong>
                    {info.class_name ? ` (${info.class_name})` : ""}{" "}
                    {info.metrics &&
                      (() => {
                        const entries = Object.entries(info.metrics);
                        if (!entries.length) return null;
                        const [metricName, metricVal] = entries[0];
                        return `– ${metricName}: ${metricVal.toFixed(3)}`;
                      })()}
                  </li>
                ))}
              </ul>

              {modeling.feature_importances && (
                <>
                  <h4>Top Features (best model)</h4>
                  <ul>
                    {modeling.feature_importances.slice(0, 10).map((fi) => (
                      <li key={fi.feature}>
                        {fi.feature}: {fi.importance.toFixed(3)}
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </div>
          )}
        </section>

        <section className="card">
          <div className="card-header">
            <span className="step-badge">4</span>
            <h2>Insights &amp; Report</h2>
          </div>
          <p className="card-subtitle">
            Generate a narrative summary of your dataset and modeling results, or open a full HTML
            report.
          </p>
          <div className="row">
            <button
              onClick={handleLoadInsights}
              disabled={!uploadMeta || isLoadingInsights}
            >
              {isLoadingInsights ? "Loading insights..." : "Generate insights"}
            </button>
            <button onClick={handleOpenReport} disabled={!uploadMeta}>
              Open HTML report
            </button>
            <button
              onClick={handleDownloadPdf}
              disabled={!uploadMeta || isDownloadingPdf}
            >
              {isDownloadingPdf ? "Downloading..." : "Download PDF"}
            </button>
          </div>

          {insights && (
            <div className="insights">
              <h3>Overview</h3>
              <p>{insights.insights.overview}</p>

              <h3>Missingness</h3>
              <p>{insights.insights.missingness}</p>

              <h3>Target Balance</h3>
              <p>{insights.insights.target_balance}</p>

              <h3>Modeling</h3>
              <p>{insights.insights.modeling}</p>

              <h3>Feature Importance</h3>
              <p>{insights.insights.feature_importance}</p>
            </div>
          )}
        </section>

        {error && (
          <section className="card error-card">
            <strong>Error:</strong> {error}
          </section>
        )}
      </main>
    </div>
  );
}

export default App;