// src/App.tsx
import { useState } from "react";
import type { ChangeEvent } from "react";
import api from "./api/client";
import type { UploadResponse, EDAResponse, ModelingResponse, InsightsResponse, ColumnTypeLabel } from "./types";
import "./App.css";

type OverridesMap = Record<string, ColumnTypeLabel>;

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [uploadMeta, setUploadMeta] = useState<UploadResponse | null>(null);
  const [eda, setEda] = useState<EDAResponse | null>(null);
  const [modelHistory, setModelHistory] = useState<ModelingResponse[]>([]);
  const [modeling, setModeling] = useState<ModelingResponse | null>(null);
  const [insights, setInsights] = useState<InsightsResponse | null>(null);

  const [target, setTarget] = useState<string>("");
  const [overrides, setOverrides] = useState<OverridesMap>({});
  const [colsToDrop, setColsToDrop] = useState<Set<string>>(new Set());
  
  const [fillNaStrategy, setFillNaStrategy] = useState<string>("");
  const [fillNaValue, setFillNaValue] = useState<string>("");
  const [filterCol, setFilterCol] = useState<string>("");
  const [filterOp, setFilterOp] = useState<string>("==");
  const [filterVal, setFilterVal] = useState<string>("");

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [testSize, setTestSize] = useState(0.2);
  const [valSize, setValSize] = useState(0.0);
  const [tuneHyperparams, setTuneHyperparams] = useState(false);
  const [tuningTrials, setTuningTrials] = useState(20);
  const [optimizeMetric, setOptimizeMetric] = useState("");

  const [predictionInput, setPredictionInput] = useState<Record<string, string>>({});
  const [predictionResult, setPredictionResult] = useState<any>(null);

  const [isUploading, setIsUploading] = useState(false);
  const [isRunningEda, setIsRunningEda] = useState(false);
  const [isDropping, setIsDropping] = useState(false);
  const [isRunningModel, setIsRunningModel] = useState(false);
  const [isLoadingInsights, setIsLoadingInsights] = useState(false);
  const [isDownloadingPdf, setIsDownloadingPdf] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const backendUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) {
      setFile(e.target.files[0]);
      setUploadMeta(null); setEda(null); setModeling(null); setModelHistory([]); setInsights(null);
      setTarget(""); setOverrides({}); setColsToDrop(new Set()); setError(null); setPredictionResult(null);
      setFillNaStrategy(""); setFillNaValue(""); setFilterCol(""); setFilterVal("");
    }
  };

  const handleUpload = async () => {
    if (!file) return setError("Select file first.");
    setError(null); setIsUploading(true);
    try {
      const fd = new FormData(); fd.append("file", file);
      const res = await api.post<UploadResponse>("/api/upload", fd);
      setUploadMeta(res.data);

      setError(null);
      setIsRunningEda(true);
      try {
        const edaRes = await api.post<EDAResponse>(`/api/eda/${res.data.dataset_id}`, { overrides });
        setEda(edaRes.data);
      } catch (err: any) {
        setError(err?.response?.data?.detail || "EDA failed.");
      } finally {
        setIsRunningEda(false);
      }

    } catch (err: any) { 
      setError(err?.response?.data?.detail || "Upload failed."); 
    } 
    finally { 
      setIsUploading(false); 
    }
  };

  const handleResetAll = async () => {
    if (!window.confirm("Reset all data?")) return;
    try { await api.post("/api/reset"); window.location.reload(); } catch { setError("Reset failed."); }
  };

  const handleToggleDrop = (col: string) => {
    const next = new Set(colsToDrop);
    if (next.has(col)) next.delete(col); else next.add(col);
    setColsToDrop(next);
  };

  const handleTransform = async () => {
    if (!uploadMeta) return;
    setIsDropping(true);
    const filterConditions = [];
    if (filterCol && filterVal) filterConditions.push({ column: filterCol, operator: filterOp, value: filterVal });

    try {
      const res = await api.post<UploadResponse>(`/api/transform/${uploadMeta.dataset_id}`, { 
        drop_columns: Array.from(colsToDrop),
        fill_na_strategy: fillNaStrategy || null,
        fill_na_value: fillNaValue || null,
        filter_conditions: filterConditions
      });
      setUploadMeta(res.data); 
      setColsToDrop(new Set()); setEda(null); 
      setFilterCol(""); setFilterVal(""); setFillNaStrategy("");
    } catch (err: any) { setError(err?.response?.data?.detail || "Transform failed."); }
    finally { setIsDropping(false); }
  };

  const handleRunEda = async () => {
    if (!uploadMeta) return;
    setError(null); setIsRunningEda(true);
    try {
      const res = await api.post<EDAResponse>(`/api/eda/${uploadMeta.dataset_id}`, { overrides });
      setEda(res.data);
    } catch (err: any) { setError(err?.response?.data?.detail || "EDA failed."); }
    finally { setIsRunningEda(false); }
  };

  const handleRunModel = async () => {
    if (!uploadMeta || !target) return;
    setError(null); setIsRunningModel(true);
    try {
      const res = await api.post<ModelingResponse>(`/api/model/${uploadMeta.dataset_id}`, {
        target, test_size: testSize, val_size: valSize, random_state: 42, overrides,
        tune_hyperparameters: tuneHyperparams, tuning_trials: tuningTrials, optimize_metric: optimizeMetric || null
      });
      setModeling(res.data); setModelHistory(prev => [...prev, res.data]);
    } catch (err: any) { setError(err?.response?.data?.detail || "Modeling failed."); }
    finally { setIsRunningModel(false); }
  };

  const handleLoadInsights = async () => {
    if (!uploadMeta) return;
    setIsLoadingInsights(true);
    try {
      const res = await api.get<InsightsResponse>(`/api/insights/${uploadMeta.dataset_id}`, { params: target ? { target } : {} });
      setInsights(res.data);
    } catch { setError("Insights failed."); }
    finally { setIsLoadingInsights(false); }
  };

  const handleOverrideChange = (colName: string, value: ColumnTypeLabel) => {
    setOverrides((prev) => ({ ...prev, [colName]: value }));
  };

  const handleOpenReport = () => {
    if (!uploadMeta) return;
    window.open(`${backendUrl}/api/report/${uploadMeta.dataset_id}?format=html${target ? `&target=${encodeURIComponent(target)}` : ""}`, "_blank");
  };

  const handleDownloadPdf = async () => {
    if (!uploadMeta) return;
    setIsDownloadingPdf(true);
    try {
      const res = await api.get(`/api/report/${uploadMeta.dataset_id}`, { params: { target, format: "pdf" }, responseType: "blob" });
      const url = window.URL.createObjectURL(new Blob([res.data]));
      const link = document.createElement("a"); link.href = url; link.setAttribute("download", "report.pdf");
      document.body.appendChild(link); link.click(); link.remove();
    } catch { setError("PDF failed."); }
    finally { setIsDownloadingPdf(false); }
  };

  const handleDownloadModel = () => {
    if (!uploadMeta) return;
    window.open(`${backendUrl}/api/model/${uploadMeta.dataset_id}/download`, "_blank");
  };

  const handlePredict = async () => {
    if (!uploadMeta) return;
    try {
      const res = await api.post(`/api/model/${uploadMeta.dataset_id}/predict`, predictionInput);
      setPredictionResult(res.data);
    } catch (err: any) { setError(err?.response?.data?.detail || "Prediction failed."); }
  };

  const handleInputChange = (feature: string, value: string) => {
    setPredictionInput(prev => ({ ...prev, [feature]: value }));
  };

  const isValidSplit = testSize + valSize < 1.0;
  const canTrain = uploadMeta && target && isValidSplit && !isRunningModel;

  const hasDrops = colsToDrop.size > 0;
  const hasFill = !!fillNaStrategy;
  const hasFilter = !!(filterCol && filterVal);
  const canApplyChanges = (hasDrops || hasFill || hasFilter) && !isDropping;

  return (
    <div className="app-root">
      <header className="app-header">
        <div className="title-block"><h1>Auto-Analyst</h1><p>Advanced ML Platform</p></div>
        <div className="header-right"><button className="danger-btn" onClick={handleResetAll}>Reset All</button></div>
      </header>

      <main className="app-main">
        {!uploadMeta && (
          <div className="hero-section">
            <h2>🚀 AI-Powered Data Analysis</h2>
            <p>Upload a CSV file to unlock automated insights, data cleaning, and machine learning models in seconds.</p>
            <div className="hero-features">
               <span className="hero-chip">1. Data Upload</span>
               <span className="hero-chip">2. EDA</span>
               <span className="hero-chip">3. Data Cleaning</span>
               <span className="hero-chip">4. Modeling</span>
               <span className="hero-chip">5. Report</span>
            </div>
          </div>
        )}

        <section className="card">
          <div className="card-header"><span className="step-badge">1</span><h2>Data Upload</h2></div>
          <div className="row">
            <input type="file" accept=".csv" onChange={handleFileChange} />
            <button onClick={handleUpload} disabled={!file || isUploading}>{isUploading ? "Uploading..." : "Upload CSV"}</button>
          </div>
          {uploadMeta && <div className="meta">ID: {uploadMeta.dataset_id} • {uploadMeta.n_rows}x{uploadMeta.n_cols} • {uploadMeta.columns.join(", ")}</div>}
        </section>

        {uploadMeta && (
          <>
            <section className="card">
                <div className="card-header"><span className="step-badge">2</span><h2>EDA</h2></div>
                <div className="row row-end"><button onClick={handleRunEda} disabled={!uploadMeta || isRunningEda}>{isRunningEda ? "Running..." : eda ? "Re-run" : "Run EDA"}</button></div>
                {eda && (
                  <>
                    <div className="columns-grid">
                      {Object.values(eda.profile.columns).map((col) => (
                        <div key={col.name} className="column-card">
                          <div className="column-header"><h4>{col.name}</h4><span className="chip">{col.overridden_type || col.effective_type}</span></div>
                          <p className="muted">Missing: {col.missing_count} ({col.missing_pct.toFixed(1)}%) • Distinct: {col.distinct_count}</p>
                          <select value={overrides[col.name] || ""} onChange={(e) => handleOverrideChange(col.name, e.target.value as any)}>
                              <option value="">(Auto)</option><option value="numeric">Numeric</option><option value="categorical">Categorical</option>
                              <option value="text">Text (Ignore)</option><option value="id">ID (Ignore)</option>
                          </select>
                          {col.numeric_summary && <p className="muted">Mean: {col.numeric_summary.mean.toFixed(2)}, Std: {col.numeric_summary.std.toFixed(2)}</p>}
                          {col.categorical_summary && <p className="muted">Unique: {col.categorical_summary.unique}</p>}
                        </div>
                      ))}
                    </div>
                    <div className="plots-grid">
                        {eda.plots.numeric.map(p => (
                            <div key={p.column} className="plot-card">
                              <h5>{p.column}</h5>
                              {p.histogram && <img src={`${backendUrl}${p.histogram}`} />}
                              {p.boxplot && <img src={`${backendUrl}${p.boxplot}`} />}
                            </div>
                        ))}
                        {eda.plots.categorical.map(p => (
                            <div key={p.column} className="plot-card">
                              <h5>{p.column}</h5>
                              {p.barplot && <img src={`${backendUrl}${p.barplot}`} />}
                            </div>
                        ))}
                    </div>
                    {eda.plots.correlation_heatmap && <img className="heatmap-image" src={`${backendUrl}${eda.plots.correlation_heatmap.path}`} />}
                  </>
                )}
            </section>

            <section className="card">
                <div className="card-header"><span className="step-badge">3</span><h2>Data Cleaning</h2></div>
                <div style={{marginBottom: '1rem'}}>
                    <strong>Drop Columns:</strong>
                    <div className="row" style={{flexWrap: 'wrap', gap: '8px', marginTop: '0.5rem'}}>
                        {uploadMeta.columns.map(col => (
                            <label key={col} style={{background: '#f1f5f9', padding: '4px 8px', borderRadius: '4px', display: 'flex', alignItems: 'center'}}>
                                <input type="checkbox" checked={colsToDrop.has(col)} onChange={() => handleToggleDrop(col)} /> {col}
                            </label>
                        ))}
                    </div>
                </div>
                <div className="row" style={{marginBottom: '1rem', alignItems: 'center'}}>
                    <strong>Fill Missing:</strong>
                    <select value={fillNaStrategy} onChange={e => setFillNaStrategy(e.target.value)}>
                        <option value="">(None)</option><option value="mean">Mean</option><option value="median">Median</option><option value="mode">Mode</option><option value="drop">Drop Rows</option><option value="constant">Constant</option>
                    </select>
                    {fillNaStrategy === "constant" && <input placeholder="Value" value={fillNaValue} onChange={e => setFillNaValue(e.target.value)} style={{width: '100px'}} />}
                </div>
                <div className="row" style={{marginBottom: '1rem', alignItems: 'center'}}>
                    <strong>Filter Rows:</strong>
                    <select value={filterCol} onChange={e => setFilterCol(e.target.value)}><option value="">(Col)</option>{uploadMeta.columns.map(c => <option key={c} value={c}>{c}</option>)}</select>
                    <select value={filterOp} onChange={e => setFilterOp(e.target.value)}><option value="==">==</option><option value="!=">!=</option><option value=">">&gt;</option><option value="<">&lt;</option><option value=">=">&gt;=</option><option value="<=">&lt;=</option></select>
                    <input placeholder="Value" value={filterVal} onChange={e => setFilterVal(e.target.value)} style={{width: '100px'}} />
                </div>
                <div className="row row-end">
                  <button 
                    onClick={handleTransform} 
                    disabled={!canApplyChanges} 
                    className="primary-btn"
                  >
                    {isDropping ? "Processing..." : "Apply Changes"}
                  </button>
                </div>
            </section>

            <section className="card">
              <div className="card-header"><span className="step-badge">4</span><h2>Modeling</h2></div>
              <div className="config-panel">
                <div className="row">
                    <label>Target: <select value={target} onChange={e => setTarget(e.target.value)}><option value="">--</option>{uploadMeta?.columns.map(c => <option key={c} value={c}>{c}</option>)}</select></label>
                    <button className="secondary-btn" onClick={() => setShowAdvanced(!showAdvanced)} disabled={!uploadMeta}>{showAdvanced ? "Hide" : "Advanced"}</button>
                </div>
                {showAdvanced && (
                    <div className="advanced-options">
                      <label>Test: {(testSize*100).toFixed(0)}% <input type="range" min="0.1" max="0.5" step="0.05" value={testSize} onChange={e => setTestSize(parseFloat(e.target.value))} /></label>
                      <label>Val: {(valSize*100).toFixed(0)}% <input type="range" min="0.0" max="0.3" step="0.05" value={valSize} onChange={e => setValSize(parseFloat(e.target.value))} /></label>
                      <label>Metric: <select value={optimizeMetric} onChange={e => setOptimizeMetric(e.target.value)}><option value="">Auto</option><option value="accuracy">Acc</option><option value="f1">F1</option><option value="rmse">RMSE</option></select></label>
                      <label><input type="checkbox" checked={tuneHyperparams} onChange={e => setTuneHyperparams(e.target.checked)} /> Tune (Optuna)</label>
                      {tuneHyperparams && <label>Trials: <input type="number" value={tuningTrials} onChange={e => setTuningTrials(parseInt(e.target.value))} /></label>}
                    </div>
                )}
                <div className="row row-end"><button onClick={handleRunModel} disabled={!canTrain}>{isRunningModel ? "Training..." : "Train"}</button></div>
                {!isValidSplit && <p style={{color: 'red'}}>Split Error</p>}
              </div>
              
                {modelHistory.length > 0 && (
                <div className="history-board">
                  <h4>History</h4>
                  <table style={{width: '100%', fontSize: '0.9rem'}}>
                    <thead><tr><th>Model</th><th>Metric</th><th>Score</th><th>Tuned?</th></tr></thead>
                    <tbody>
                    {modelHistory.map((m, i) => {
                      let primaryMetric: string;

                      if (optimizeMetric === "f1" && "f1_macro" in m.metrics) {
                      primaryMetric = "f1_macro";
                      } else if (optimizeMetric === "accuracy" && "accuracy" in m.metrics) {
                      primaryMetric = "accuracy";
                      } else if ("f1_macro" in m.metrics) {
                      primaryMetric = "f1_macro";
                      } else if ("accuracy" in m.metrics) {
                      primaryMetric = "accuracy";
                      } else {
                      primaryMetric = Object.keys(m.metrics)[0];
                      }

                      const score = m.metrics[primaryMetric];

                      return (
                      <tr key={i} style={{ background: m === modeling ? '#f0f9ff' : '' }}>
                        <td>{m.best_model_class}</td>
                        <td>{primaryMetric}</td>
                        <td>{score.toFixed(4)}</td>
                        <td>{m.tuned ? 'Yes' : 'No'}</td>
                      </tr>
                      );
                    })}
                    </tbody>
                  </table>
                </div>
                )}

              {modeling && (
                <div className="model-results">
                    <h3>Best: {modeling.best_model_class} <button onClick={handleDownloadModel} className="small-btn">Download</button></h3>
                    <div className="metrics-grid">{Object.entries(modeling.metrics).map(([k, v]) => <div key={k} className="metric-box"><span>{k}</span><span>{v.toFixed(4)}</span></div>)}</div>
                    {modeling.tuned && <pre>{JSON.stringify(modeling.best_model_params, null, 2)}</pre>}
                    
                    <div className="playground">
                      <h4>Playground</h4>
                      <div className="playground-inputs">
                          {[...modeling.preprocessing.numeric_features, ...modeling.preprocessing.categorical_features].map(f => (
                            <input key={f} placeholder={f} onChange={e => handleInputChange(f, e.target.value)} />
                          ))}
                      </div>
                      <button onClick={handlePredict}>Predict</button>
                      {predictionResult && <div>Result: <strong>{predictionResult.prediction}</strong></div>}
                    </div>
                </div>
              )}
            </section>

            <section className="card">
                <div className="card-header"><span className="step-badge">5</span><h2>Report</h2></div>
                <div className="row">
                    <button onClick={handleLoadInsights} disabled={isLoadingInsights}>{isLoadingInsights ? "..." : "Insights"}</button>
                    <button onClick={handleOpenReport} disabled={!uploadMeta}>HTML</button>
                    <button onClick={handleDownloadPdf} disabled={isDownloadingPdf}>PDF</button>
                </div>
                {/* UPDATED: Render all insights */}
                {insights && (
                  <div className="insights">
                    <h3>Overview</h3>
                    <p>{insights.insights.overview}</p>
                    
                    <h3>Missing Data</h3>
                    <p>{insights.insights.missingness}</p>
                    
                    {insights.insights.target_balance && (
                       <>
                         <h3>Target Analysis</h3>
                         <p>{insights.insights.target_balance}</p>
                       </>
                    )}

                    {insights.insights.modeling && (
                       <>
                         <h3>Modeling Results</h3>
                         <p>{insights.insights.modeling}</p>
                         {insights.insights.feature_importance && <p>{insights.insights.feature_importance}</p>}
                       </>
                    )}
                  </div>
                )}
            </section>
            </>
        )}
        {error && <div className="error-banner">{error}</div>}
      </main>
    </div>
  );
}

export default App;
