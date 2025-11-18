from typing import Optional
from weasyprint import HTML
import logging, sys

# Configure WeasyPrint logger to visible stdout
logger = logging.getLogger("weasyprint")
logger.setLevel(logging.ERROR)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

def html_to_pdf(html: str) -> Optional[bytes]:
    try:
        return HTML(string=html).write_pdf()
    except Exception as e:
        print(f"PDF Error: {e}")
        return None