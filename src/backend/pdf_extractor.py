"""
PDF and plain-text proposal extractor.
Supports .pdf (via pypdf) and .txt files.
"""

import io
from typing import Tuple


def extract_text_from_upload(filename: str, content: bytes) -> Tuple[str, str]:
    """
    Extracts raw text from an uploaded file.
    Returns (extracted_text, detected_format).
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else "txt"

    if ext == "pdf":
        if content.startswith(b"%PDF"):
            pdf_text = _extract_pdf(content)
            if not pdf_text.startswith("[Failed"):
                return pdf_text, "pdf"
        # If not a binary PDF or if PDF parsing failed, try UTF-8 / latin-1 decoding
        try:
            txt = content.decode("utf-8").strip()
            if len(txt) > 20:
                return txt, "txt"
        except UnicodeDecodeError:
            try:
                txt = content.decode("latin-1").strip()
                if len(txt) > 20:
                    return txt, "txt"
            except Exception:
                pass
        return _extract_pdf(content), "pdf"
    elif ext in ("doc", "docx"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(content))
            paras = [p.text for p in doc.paragraphs if p.text]
            if paras:
                return "\n\n".join(paras).strip(), "docx"
        except Exception:
            pass
        try:
            return content.decode("utf-8", errors="ignore").strip(), "docx"
        except Exception:
            return "", "docx"
    else:
        # Plain text / JSONL fallback
        try:
            return content.decode("utf-8").strip(), "txt"
        except UnicodeDecodeError:
            return content.decode("latin-1").strip(), "txt"


def _extract_pdf(content: bytes) -> str:
    """Extract text from PDF bytes using pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        result = "\n\n".join(pages)
        if not result.strip():
            return "[PDF appears to contain only images or scanned content. Please paste the proposal as text.]"
        return result
    except Exception as e:
        return f"[Failed to extract PDF text: {e}. Please paste the proposal as text instead.]"
