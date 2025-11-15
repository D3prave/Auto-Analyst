from pathlib import Path
from typing import Dict, Any, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.config import BASE_DIR

_templates_path = BASE_DIR / "app" / "reporting" / "templates"
_env = Environment(
    loader=FileSystemLoader(_templates_path),
    autoescape=select_autoescape(["html", "xml"]),
)


def build_html_report(
    dataset_id: str,
    profile: Dict[str, Any],
    plots: Dict[str, Any],
    insights: Dict[str, str],
    modeling: Optional[Dict[str, Any]] = None,
    target: Optional[str] = None,
) -> str:
    tmpl = _env.get_template("report.html")
    html = tmpl.render(
        dataset_id=dataset_id,
        profile=profile,
        plots=plots,
        insights=insights,
        modeling=modeling,
        target=target,
    )
    return html