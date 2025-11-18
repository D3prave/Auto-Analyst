import base64, copy
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape
from app.config import BASE_DIR
from app.utils.storage import get_plot

_env = Environment(loader=FileSystemLoader(BASE_DIR / "app" / "reporting" / "templates"), autoescape=select_autoescape(["html"]))

def _embed(plot_obj, keys, dataset_id):
    """Replace Redis image URL with Base64 data URI for PDF embedding."""
    for key in keys:
        url = plot_obj.get(key)
        if url and url.startswith("/api/images/"):
            data = get_plot(dataset_id, url.split("/")[-1])
            if data: plot_obj[key] = f"data:image/png;base64,{base64.b64encode(data).decode('utf-8')}"

def build_html_report(dataset_id, profile, plots, insights, modeling=None, target=None, embed_images=False):
    final_plots = copy.deepcopy(plots) if embed_images else plots
    if embed_images:
        for p in final_plots.get("numeric", []): _embed(p, ["histogram", "boxplot"], dataset_id)
        for p in final_plots.get("categorical", []): _embed(p, ["barplot"], dataset_id)
        if final_plots.get("correlation_heatmap"): _embed(final_plots["correlation_heatmap"], ["path"], dataset_id)

    return _env.get_template("report.html").render(
        dataset_id=dataset_id, profile=profile, plots=final_plots,
        insights=insights, modeling=modeling, target=target
    )