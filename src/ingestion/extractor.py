from pathlib import Path

import fitz  # PyMuPDF
from docx import Document


def extract_text_pdf(path: Path) -> list[dict]:
    """Extract per-page text from a PDF. Returns list of {page_number, text} dicts."""
    pages = []
    with fitz.open(str(path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                pages.append({"page_number": page_num, "text": text})
    return pages


def extract_text_docx(path: Path) -> list[dict]:
    """Extract text from a DOCX file. Returns a single-element list (no page tracking in DOCX)."""
    doc = Document(str(path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs)
    if not text:
        return []
    return [{"page_number": 0, "text": text}]
