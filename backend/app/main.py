from uuid import uuid4
from typing import Dict, Any, Optional
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from app.utils.storage import (
    save_dataset, get_dataset, save_metadata, get_metadata, get_plot
)
from app.eda.profiler import profile_dataset
from app.eda.visualizer import generate_plots
from app.eda.insights import generate_insights
from app.reporting.builder import build_html_report
from app.reporting.pdf_export import html_to_pdf
from app.ml.modeling import run_baseline_models
from app.schemas import EDARequest, ModelingRequest

app = FastAPI(title="Auto-Analyst API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Auto-Analyst backend (Redis) is running"}

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    dataset_id = str(uuid4())
    
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    # Save to Redis
    try:
        save_dataset(dataset_id, df)
    except ImportError as e:
        # Catch pyarrow missing error
        raise HTTPException(status_code=500, detail=f"Server configuration error: {e}")
        
    # Save initial metadata
    meta = {
        "filename": filename,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
    }
    save_metadata(dataset_id, meta)

    return {
        "dataset_id": dataset_id,
        "n_rows": meta["n_rows"],
        "n_cols": meta["n_cols"],
        "columns": meta["columns"],
    }

def _compute_eda(dataset_id: str, overrides: dict | None = None) -> Dict:
    try:
        df = get_dataset(dataset_id)
        meta = get_metadata(dataset_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")

    profile = profile_dataset(df, overrides=overrides)
    plots = generate_plots(df, dataset_id, profile)

    meta["overrides"] = overrides or {}
    meta["last_profile"] = profile
    save_metadata(dataset_id, meta)

    return {"dataset_id": dataset_id, "profile": profile, "plots": plots}

@app.post("/api/eda/{dataset_id}")
def run_eda_post(dataset_id: str, req: EDARequest):
    return _compute_eda(dataset_id, overrides=req.overrides or {})

@app.post("/api/model/{dataset_id}")
def run_modeling(dataset_id: str, req: ModelingRequest):
    try:
        df = get_dataset(dataset_id)
        meta = get_metadata(dataset_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if req.target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{req.target}' not found.")

    try:
        result = run_baseline_models(
            df,
            target_col=req.target,
            test_size=req.test_size,
            random_state=req.random_state,
            overrides=req.overrides,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Modeling failed: {e}")

    meta["modeling"] = result
    save_metadata(dataset_id, meta)
    return result

@app.get("/api/insights/{dataset_id}")
def get_insights_endpoint(dataset_id: str, target: str | None = None):
    try:
        df = get_dataset(dataset_id)
        meta = get_metadata(dataset_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")

    profile = meta.get("last_profile")
    if not profile:
        profile = profile_dataset(df, overrides=meta.get("overrides"))

    modeling = meta.get("modeling")
    insights = generate_insights(profile, modeling, target_col=target)
    return {"dataset_id": dataset_id, "target": target, "insights": insights}

@app.get("/api/images/{dataset_id}/{filename}")
def get_image(dataset_id: str, filename: str):
    image_bytes = get_plot(dataset_id, filename)
    if not image_bytes:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(content=image_bytes, media_type="image/png")

@app.get("/api/report/{dataset_id}")
def get_report(dataset_id: str, format: str = "html", target: str | None = None):
    try:
        df = get_dataset(dataset_id)
        meta = get_metadata(dataset_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")

    profile = meta.get("last_profile") or profile_dataset(df, overrides=meta.get("overrides"))
    plots = generate_plots(df, dataset_id, profile)
    modeling = meta.get("modeling")
    insights = generate_insights(profile, modeling, target_col=target)

    if format == "html":
        # For HTML, use URLs so browser fetches from server
        html = build_html_report(dataset_id, profile, plots, insights, modeling, target, embed_images=False)
        return Response(content=html, media_type="text/html")
    
    elif format == "pdf":
        # For PDF, embed images as base64 to avoid path resolution issues
        html = build_html_report(dataset_id, profile, plots, insights, modeling, target, embed_images=True)
        pdf_bytes = html_to_pdf(html)
        
        if not pdf_bytes:
            raise HTTPException(status_code=500, detail="Failed to generate PDF.")
        
        return Response(
            content=pdf_bytes, 
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=report_{dataset_id}.pdf"}
        )

    raise HTTPException(status_code=400, detail=f"Format '{format}' not supported.")

@app.post("/api/reset")
def reset_server():
    import redis
    from app.config import REDIS_URL
    r = redis.from_url(REDIS_URL)
    r.flushall()
    return {"status": "ok", "message": "Redis cache cleared."}