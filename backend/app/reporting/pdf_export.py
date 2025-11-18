from typing import Optional
from weasyprint import HTML
import logging
import sys

# Configure logger
logger = logging.getLogger("weasyprint")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

def html_to_pdf(html: str) -> Optional[bytes]:
    try:
        pdf_bytes = HTML(string=html).write_pdf()
        return pdf_bytes
    except Exception as e:
        print(f"PDF generation error: {e}")
        return None