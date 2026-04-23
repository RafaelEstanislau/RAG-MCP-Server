import io

import fitz  # PyMuPDF
from docx import Document


def extract_text_pdf(data: bytes) -> list[dict]:
    """Extract per-page text from PDF bytes. Returns list of {page_number, text} dicts."""
    pages = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                pages.append({"page_number": page_num, "text": text})
    return pages


def extract_text_docx(data: bytes) -> list[dict]:
    """Extract text from DOCX bytes. Returns a single-element list (no page tracking in DOCX)."""
    doc = Document(io.BytesIO(data))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs)
    if not text:
        return []
    return [{"page_number": 0, "text": text}]
