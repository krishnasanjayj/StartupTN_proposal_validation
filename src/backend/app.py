"""
StartupTN AI Proposal Evaluation & Chat — FastAPI Backend
Serves the web UI and exposes REST APIs for proposal upload and conversational chat.
"""

import sys
import os
import json
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Project root on sys.path ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.llm.evaluator import ProposalEvaluator
from src.backend.pdf_extractor import extract_text_from_upload

# ── Global state ─────────────────────────────────────────────────────────────
evaluator: Optional[ProposalEvaluator] = None
model_ready: bool = False
model_error: Optional[str] = None
MODEL_NAME = os.getenv("STARTUPTN_MODEL", "Qwen/Qwen3-0.6B")


# ── Lifespan: load model at startup ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global evaluator, model_ready, model_error
    print(f"[Startup] Loading model '{MODEL_NAME}'…")
    loop = asyncio.get_event_loop()
    try:
        evaluator = await loop.run_in_executor(
            None, lambda: ProposalEvaluator(model_name_or_path=MODEL_NAME)
        )
        model_ready = True
        print("[Startup] Model loaded and ready.")
    except Exception as e:
        model_error = str(e)
        print(f"[Startup] ERROR loading model: {e}")
    yield
    print("[Shutdown] Releasing model.")
    evaluator = None


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="StartupTN AI Proposal Evaluator",
    version="1.0.0",
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Pydantic schemas ─────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []
    proposal_text: Optional[str] = None
    evaluation: Optional[Dict[str, Any]] = None


class EvaluationRequest(BaseModel):
    proposal_text: str
    industry: Optional[str] = None
    category: Optional[str] = None
    stage: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
async def health():
    return {
        "status": "ready" if model_ready else ("error" if model_error else "loading"),
        "model": MODEL_NAME,
        "error": model_error,
    }


@app.post("/api/upload")
async def upload_proposal(
    file: UploadFile = File(...),
    industry: str = Form(""),
    category: str = Form(""),
    stage: str = Form(""),
):
    """
    Accept a PDF or TXT proposal file, extract text, run AI evaluation,
    and return the structured assessment.
    """
    if not model_ready:
        raise HTTPException(503, detail="Model is still loading. Please wait and retry.")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(400, detail="Uploaded file is empty.")

    proposal_text, fmt = extract_text_from_upload(file.filename or "upload.txt", content)

    if len(proposal_text.strip()) < 50:
        raise HTTPException(400, detail="Could not extract meaningful text from the uploaded file.")

    metadata = {
        "industry": industry or "Not specified",
        "category": category or "Not specified",
        "stage": stage or "Not specified",
    }

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: evaluator.evaluate(proposal_text, metadata=metadata),
    )

    return JSONResponse({
        "proposal_text": proposal_text,
        "file_format": fmt,
        "filename": file.filename,
        "metadata": metadata,
        "evaluation": result["evaluation"],
        "thinking_process": result.get("thinking_process", ""),
    })


@app.post("/api/evaluate")
async def evaluate_text(req: EvaluationRequest):
    """
    Evaluate a proposal submitted as plain text (no file upload).
    """
    if not model_ready:
        raise HTTPException(503, detail="Model is still loading. Please wait and retry.")

    metadata = {
        "industry": req.industry or "Not specified",
        "category": req.category or "Not specified",
        "stage": req.stage or "Not specified",
    }

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: evaluator.evaluate(req.proposal_text, metadata=metadata),
    )

    return JSONResponse({
        "proposal_text": req.proposal_text,
        "metadata": metadata,
        "evaluation": result["evaluation"],
        "thinking_process": result.get("thinking_process", ""),
    })


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    Multi-turn conversational endpoint grounded in the uploaded proposal + evaluation.
    """
    if not model_ready:
        raise HTTPException(503, detail="Model is still loading. Please wait and retry.")

    if not req.message.strip():
        raise HTTPException(400, detail="Message cannot be empty.")

    loop = asyncio.get_event_loop()
    reply = await loop.run_in_executor(
        None,
        lambda: evaluator.chat(
            message=req.message,
            history=req.history,
            proposal_context=req.proposal_text,
            evaluation_context=req.evaluation,
        ),
    )

    return JSONResponse({"reply": reply})
