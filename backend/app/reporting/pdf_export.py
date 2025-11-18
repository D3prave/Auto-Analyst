from typing import Optional
from weasyprint import HTML
import logging
import sys
import os

# Configure WeasyPrint's logger to print to stdout so you can see it in docker logs
logger = logging.getLogger("weasyprint")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

def html_to_pdf(html: str) -> Optional[bytes]:
    """
    Convert HTML string to PDF bytes using WeasyPrint.
    """
    
    # --- DEBUG: Check file existence ---
    plot_dir = "/app/outputs/plots"
    print(f"[DEBUG] Checking plot directory: {plot_dir}")
    if os.path.exists(plot_dir):
        # Only list the first few files to avoid spamming logs if there are many
        files = os.listdir(plot_dir)
        print(f"[DEBUG] Files found (first 10): {files[:10]}")
    else:
        print(f"[DEBUG] DIRECTORY NOT FOUND: {plot_dir}")
    # -----------------------------------

    try:
        # base_url="file:///app" ensures WeasyPrint treats this as a filesystem path
        # Relative paths in HTML (like "outputs/plots/...") will be appended to this.
        # So "outputs/..." becomes "file:///app/outputs/..."
        pdf_bytes = HTML(string=html, base_url="file:///app").write_pdf()
        return pdf_bytes
    except Exception as e:
        print(f"PDF generation error: {e}")
        return None