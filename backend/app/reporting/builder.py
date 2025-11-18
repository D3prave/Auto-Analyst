import base64
import copy
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape
from app.config import BASE_DIR
from app.utils.storage import get_plot

_templates_path = BASE_DIR / "app" / "reporting" / "templates"
_env = Environment(
    loader=FileSystemLoader(_templates_path),
    autoescape=select_autoescape(["html", "xml"]),
)

def _embed_plot_in_struct(plot_obj: Dict[str, Any], keys: list, dataset_id: str):
    for key in keys:
        url = plot_obj.get(key)
        if url and url.startswith("/api/images/"):
            filename = url.split("/")[-1]
            img_bytes = get_plot(dataset_id, filename)
            if img_bytes:
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                plot_obj[key] = f"data:image/png;base64,{b64}"

def build_html_report(
    dataset_id: str,
    profile: Dict[str, Any],
    plots: Dict[str, Any],
    insights: Dict[str, str],
    modeling: Optional[Dict[str, Any]] = None,
    target: Optional[str] = None,
    embed_images: bool = False,
) -> str:
    
    final_plots = plots
    
    # If PDF mode, create a copy and embed images directly
    if embed_images:
        final_plots = copy.deepcopy(plots)
        
        # Numeric
        for p in final_plots.get("numeric", []):
            _embed_plot_in_struct(p, ["histogram", "boxplot"], dataset_id)
            
        # Categorical
        for p in final_plots.get("categorical", []):
            _embed_plot_in_struct(p, ["barplot"], dataset_id)
            
        # Heatmap
        if final_plots.get("correlation_heatmap"):
             _embed_plot_in_struct(final_plots["correlation_heatmap"], ["path"], dataset_id)

    tmpl = _env.get_template("report.html")
    html = tmpl.render(
        dataset_id=dataset_id,
        profile=profile,
        plots=final_plots,
        insights=insights,
        modeling=modeling,
        target=target,
    )
    return html