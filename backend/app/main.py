from uuid import uuid4
from typing import Dict, Any
import io, joblib, pandas as pd, numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from app.utils.storage import save_dataset, get_dataset, save_metadata, get_metadata, get_plot, r
from app.eda.profiler import profile_dataset
from app.eda.visualizer import generate_plots
from app.eda.insights import generate_insights
from app.reporting.builder import build_html_report
from app.reporting.pdf_export import html_to_pdf
from app.ml.modeling import run_baseline_models
from app.schemas import EDARequest, ModelingRequest, TransformRequest

app = FastAPI(title="Auto-Analyst API", version="0.5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def read_root(): return {"message": "Backend running"}

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"): raise HTTPException(400, "CSV only")
    dataset_id = str(uuid4())
    try:
        df = pd.read_csv(file.file)
        save_dataset(dataset_id, df)
    except Exception as e: raise HTTPException(400, f"Error: {e}")
    
    meta = {"filename": file.filename, "n_rows": len(df), "n_cols": df.shape[1], "columns": df.columns.tolist()}
    save_metadata(dataset_id, meta)
    return {"dataset_id": dataset_id, **meta}

@app.post("/api/transform/{dataset_id}")
def transform(dataset_id: str, req: TransformRequest):
    try: df = get_dataset(dataset_id)
    except: raise HTTPException(404, "Dataset not found")

    # 1. Drop Columns
    if req.drop_columns: 
        df.drop(columns=[c for c in req.drop_columns if c in df.columns], inplace=True)
    
    # 2. Filter Rows
    if req.filter_conditions:
        for c in req.filter_conditions:
            if c.column in df.columns:
                col, val = df[c.column], c.value
                if pd.api.types.is_numeric_dtype(col):
                    try: val = float(val)
                    except: continue
                if c.operator == "==": df = df[col == val]
                elif c.operator == "!=": df = df[col != val]
                elif c.operator == ">": df = df[col > val]
                elif c.operator == "<": df = df[col < val]

    # 3. Fill NA
    if req.fill_na_strategy:
        if req.fill_na_strategy == "drop": df.dropna(inplace=True)
        elif req.fill_na_strategy == "constant": 
            try: val = float(req.fill_na_value)
            except: val = req.fill_na_value
            df.fillna(val, inplace=True)
        else:
            # Mean/Median only for numeric
            for col in df.select_dtypes(include=np.number).columns:
                if req.fill_na_strategy == "mean": df[col].fillna(df[col].mean(), inplace=True)
                elif req.fill_na_strategy == "median": df[col].fillna(df[col].median(), inplace=True)
    
    save_dataset(dataset_id, df)
    meta = get_metadata(dataset_id)
    meta.update({"n_rows": len(df), "n_cols": df.shape[1], "columns": df.columns.tolist()})
    if "last_profile" in meta: del meta["last_profile"]
    save_metadata(dataset_id, meta)
    return {"dataset_id": dataset_id, **meta}

@app.post("/api/eda/{dataset_id}")
def run_eda(dataset_id: str, req: EDARequest):
    try: df = get_dataset(dataset_id)
    except: raise HTTPException(404, "Dataset not found")
    
    profile = profile_dataset(df, overrides=req.overrides)
    plots = generate_plots(df, dataset_id, profile)
    meta = get_metadata(dataset_id)
    meta.update({"overrides": req.overrides or {}, "last_profile": profile})
    save_metadata(dataset_id, meta)
    return {"dataset_id": dataset_id, "profile": profile, "plots": plots}

@app.post("/api/model/{dataset_id}")
def run_model(dataset_id: str, req: ModelingRequest):
    try: df = get_dataset(dataset_id)
    except: raise HTTPException(404, "Dataset not found")
    
    try:
        res, model = run_baseline_models(df, req.target, req.test_size, req.val_size, req.random_state, req.overrides, req.tune_hyperparameters, req.tuning_trials, req.optimize_metric)
        buf = io.BytesIO()
        joblib.dump(model, buf)
        r.set(f"dataset:{dataset_id}:model", buf.getvalue())
        meta = get_metadata(dataset_id)
        meta["modeling"] = res
        save_metadata(dataset_id, meta)
        return res
    except Exception as e: raise HTTPException(500, f"Model error: {e}")

@app.get("/api/model/{dataset_id}/download")
def download(dataset_id: str):
    data = r.get(f"dataset:{dataset_id}:model")
    if not data: raise HTTPException(404, "Model not found")
    return Response(content=data, media_type="application/octet-stream", headers={"Content-Disposition": f"attachment; filename=model_{dataset_id}.pkl"})

@app.post("/api/model/{dataset_id}/predict")
def predict(dataset_id: str, payload: Dict[str, Any]):
    data = r.get(f"dataset:{dataset_id}:model")
    if not data: raise HTTPException(404, "Model not found")
    try:
        model = joblib.load(io.BytesIO(data))
        df = pd.DataFrame([payload])
        for c in df.columns: 
            try: df[c] = pd.to_numeric(df[c])
            except: pass
        return {"prediction": model.predict(df).tolist()[0]}
    except Exception as e: raise HTTPException(400, f"Predict error: {e}")

@app.get("/api/insights/{dataset_id}")
def insights(dataset_id: str, target: str = None):
    try: df = get_dataset(dataset_id)
    except: raise HTTPException(404, "Dataset not found")
    meta = get_metadata(dataset_id)
    return {"dataset_id": dataset_id, "insights": generate_insights(meta.get("last_profile") or profile_dataset(df), meta.get("modeling"), target)}

@app.get("/api/images/{dataset_id}/{filename}")
def image(dataset_id: str, filename: str):
    from app.utils.storage import get_plot
    data = get_plot(dataset_id, filename)
    if not data: raise HTTPException(404, "Image not found")
    return Response(content=data, media_type="image/png")

@app.get("/api/report/{dataset_id}")
def report(dataset_id: str, format: str = "html", target: str = None):
    try: df = get_dataset(dataset_id)
    except: raise HTTPException(404, "Dataset not found")
    meta = get_metadata(dataset_id)
    profile = meta.get("last_profile") or profile_dataset(df)
    plots = generate_plots(df, dataset_id, profile)
    insights = generate_insights(profile, meta.get("modeling"), target)
    
    if format == "html": return Response(content=build_html_report(dataset_id, profile, plots, insights, meta.get("modeling"), target), media_type="text/html")
    if format == "pdf":
        pdf = html_to_pdf(build_html_report(dataset_id, profile, plots, insights, meta.get("modeling"), target, embed_images=True))
        return Response(content=pdf, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=report_{dataset_id}.pdf"})
    raise HTTPException(400, "Invalid format")

@app.post("/api/reset")
def reset():
    r.flushall()
    return {"status": "ok"}