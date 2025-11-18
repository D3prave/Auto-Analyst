from uuid import uuid4
from typing import Dict, Any, Optional

import pandas as pd
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.utils.file_storage import save_temp_csv, get_dataset_path
from app.eda.profiler import profile_dataset
from app.eda.visualizer import generate_plots
from app.eda.insights import generate_insights
from app.reporting.builder import build_html_report
from app.reporting.pdf_export import html_to_pdf
from app.ml.modeling import run_baseline_models
from app.schemas import EDARequest, ModelingRequest
from app.config import OUTPUT_DIR, DATA_DIR


app = FastAPI(title="Auto-Analyst API", version="0.1.0")

# In dev, allow all origins. In prod, restrict.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory dataset registry.
DATASETS: Dict[str, Dict] = {}

def _compute_eda(dataset_id: str, overrides: dict | None = None) -> Dict:
    meta = DATASETS.get(dataset_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Dataset not found")

    path = get_dataset_path(dataset_id)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")

    profile = profile_dataset(df, overrides=overrides)
    plots = generate_plots(df, dataset_id, profile)

    # cache overrides + profile for later (report/insights)
    meta["overrides"] = overrides or {}
    meta["last_profile"] = profile

    return {"dataset_id": dataset_id, "profile": profile, "plots": plots}


# Serve plots and other outputs as static files
app.mount(
    "/outputs",
    StaticFiles(directory=OUTPUT_DIR),
    name="outputs",
)

@app.get("/")
def read_root():
    return {"message": "Auto-Analyst backend is running"}


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV file and register it as a dataset."""
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    dataset_id = str(uuid4())
    path = save_temp_csv(dataset_id, file)

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    DATASETS[dataset_id] = {
        "path": str(path),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
    }

    return {
        "dataset_id": dataset_id,
        "n_rows": DATASETS[dataset_id]["n_rows"],
        "n_cols": DATASETS[dataset_id]["n_cols"],
        "columns": DATASETS[dataset_id]["columns"],
    }


@app.post("/api/eda/{dataset_id}")
def run_eda_post(dataset_id: str, req: EDARequest):
    return _compute_eda(dataset_id, overrides=req.overrides or {})

@app.post("/api/model/{dataset_id}")
def run_modeling(dataset_id: str, req: ModelingRequest):
    meta = DATASETS.get(dataset_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Dataset not found")

    path = get_dataset_path(dataset_id)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")

    if req.target not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{req.target}' not found in dataset.",
        )

    try:
        result = run_baseline_models(
            df,
            target_col=req.target,
            test_size=req.test_size,
            random_state=req.random_state,
            overrides=req.overrides,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Modeling failed: {e}",
        )

    meta["modeling"] = result
    return result


@app.get("/api/insights/{dataset_id}")
def get_insights(dataset_id: str, target: str | None = None):
    meta = DATASETS.get(dataset_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Dataset not found")

    path = get_dataset_path(dataset_id)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")

    profile = meta.get("last_profile")
    if profile is None:
        profile = profile_dataset(df, overrides=meta.get("overrides"))

    modeling = meta.get("modeling")
    insights = generate_insights(profile, modeling, target_col=target)
    return {"dataset_id": dataset_id, "target": target, "insights": insights}


@app.get("/api/report/{dataset_id}")
def get_report(dataset_id: str, format: str = "html", target: str | None = None):
    meta = DATASETS.get(dataset_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Dataset not found")

    path = get_dataset_path(dataset_id)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")

    profile = profile_dataset(df, overrides=meta.get("overrides"))
    plots = generate_plots(df, dataset_id, profile)
    modeling = meta.get("modeling")
    insights = generate_insights(profile, modeling, target_col=target)

    # 1. Build HTML
    if format == "html":
        # For browser, base_path is "/", so paths become /outputs/...
        html = build_html_report(
            dataset_id, profile, plots, insights, modeling, target, base_path="/"
        )
        return Response(content=html, media_type="text/html")
    
    elif format == "pdf":
        # PDF MODE: Use relative paths (e.g., "outputs/...") 
        # WeasyPrint will resolve these against base_url="file:///app"
        html = build_html_report(
            dataset_id,
            profile,
            plots,
            insights,
            modeling,
            target,
            base_path="",  # Empty string makes paths relative
        )
        
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
    DATASETS.clear()
    def _clean_dir(path):
        if not path.exists():
            return
        for child in path.iterdir():
            if child.is_file():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                shutil.rmtree(child, ignore_errors=True)

    _clean_dir(DATA_DIR)
    _clean_dir(OUTPUT_DIR)
    return {"status": "ok", "message": "All datasets, cache, and plots have been cleared."}