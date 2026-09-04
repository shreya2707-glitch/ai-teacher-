"""
AI Teacher - FastAPI backend
Wraps lesson_engine.py + rag_pipeline.py into HTTP endpoints for the frontend.

Requires:
    pip install fastapi uvicorn python-multipart --break-system-packages
    (plus everything lesson_engine.py and rag_pipeline.py need)

Run:
    uvicorn api:app --reload --port 8000

Session model:
    In-memory dict keyed by session_id (fine for a hackathon demo, single-process).
    Swap SESSIONS for Redis if you need multi-worker / persistence across restarts.
"""

import os
import shutil
import uuid
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from lesson_engine import (
    LessonState, plan_lesson, explain_concept, generate_check_question,
    evaluate_answer, adapt_or_advance, generate_final_report,
)
from rag_pipeline import ingest_document, retrieve_context

app = FastAPI(title="AI Teacher API")

# Loosen for hackathon dev; tighten allow_origins before you demo on a public URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# session_id -> {"state": LessonState, "doc_id": str | None}
SESSIONS: dict[str, dict] = {}


def _get_session(session_id: str) -> dict:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Call /session/start first.")
    return session


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------

class StartSessionRequest(BaseModel):
    topic: str
    level: str                  # beginner | intermediate | advanced
    language: str                # e.g. "Hindi", "English", "Hinglish"
    time_budget_min: int
    doc_id: Optional[str] = None  # pass this if a file was already uploaded via /upload


class AnswerRequest(BaseModel):
    session_id: str
    student_answer: str


# ---------------------------------------------------------------------------
# 1. Upload material (optional - topic-only mode skips this)
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload_material(file: UploadFile = File(...)):
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ("pdf", "docx", "pptx", "txt"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: .{ext}")

    save_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{file.filename}")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        doc_id = ingest_document(save_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"doc_id": doc_id, "filename": file.filename}


# ---------------------------------------------------------------------------
# 2. Start session -> runs PLAN
# ---------------------------------------------------------------------------

@app.post("/session/start")
def start_session(req: StartSessionRequest):
    session_id = uuid.uuid4().hex[:12]

    planning_context = ""
    if req.doc_id:
        planning_context = retrieve_context(query=req.topic, doc_id=req.doc_id, top_k=8)

    state = plan_lesson(
        topic=req.topic, level=req.level, language=req.language,
        time_budget_min=req.time_budget_min, retrieved_context=planning_context,
    )

    SESSIONS[session_id] = {"state": state, "doc_id": req.doc_id}

    return {
        "session_id": session_id,
        "concepts": [c.name for c in state.concepts],
        "lesson_complete": False,
    }


# ---------------------------------------------------------------------------
# 3. Get next explanation (EXPLAIN step for whatever concept is current)
# ---------------------------------------------------------------------------

@app.get("/session/{session_id}/explain")
def get_explanation(session_id: str):
    session = _get_session(session_id)
    state: LessonState = session["state"]

    if state.current_idx >= len(state.concepts):
        return {"lesson_complete": True}

    concept = state.current_concept

    # re-retrieve, scoped to this specific concept, for tighter grounding
    if session["doc_id"]:
        state.retrieved_context = retrieve_context(
            query=concept.name, doc_id=session["doc_id"], top_k=4
        )

    retry_hint = concept.last_misconception if concept.retry_count > 0 else None
    explanation = explain_concept(state, retry_hint=retry_hint)

    return {
        "lesson_complete": False,
        "concept": concept.name,
        "is_retry": retry_hint is not None,
        **explanation,   # spoken_script, on_screen_text, worked_example, visual_spec
    }


# ---------------------------------------------------------------------------
# 4. Get check question (CHECK step)
# ---------------------------------------------------------------------------

@app.get("/session/{session_id}/question")
def get_question(session_id: str):
    session = _get_session(session_id)
    state: LessonState = session["state"]

    if state.current_idx >= len(state.concepts):
        raise HTTPException(status_code=400, detail="Lesson already complete")

    question = generate_check_question(state)
    # stash the question on the session so /answer can reference it without the
    # frontend needing to round-trip the whole object back
    session["pending_question"] = question
    return question


# ---------------------------------------------------------------------------
# 5. Submit answer -> EVALUATE + ADAPT/ADVANCE
# ---------------------------------------------------------------------------

@app.post("/session/answer")
def submit_answer(req: AnswerRequest):
    session = _get_session(req.session_id)
    state: LessonState = session["state"]
    question = session.get("pending_question")

    if not question:
        raise HTTPException(status_code=400, detail="No pending question. Call /question first.")

    evaluation = evaluate_answer(state, question, req.student_answer)
    decision = adapt_or_advance(state, evaluation)
    session["pending_question"] = None

    return {
        "classification": evaluation["classification"],
        "misconception_type": evaluation.get("misconception_type"),
        "feedback": evaluation["feedback"],
        "decision": decision,            # "advance" | "retry" | "give_up_and_advance"
        "lesson_complete": state.current_idx >= len(state.concepts),
    }


# ---------------------------------------------------------------------------
# 6. Final report (ASSESS + REPORT)
# ---------------------------------------------------------------------------

@app.get("/session/{session_id}/report")
def get_report(session_id: str):
    session = _get_session(session_id)
    state: LessonState = session["state"]
    return generate_final_report(state)


# ---------------------------------------------------------------------------
# 7. Debug/demo helper - dump full transcript for showing "adaptation in action"
#    in your demo video, per the state-machine-on-screen idea.
# ---------------------------------------------------------------------------

@app.get("/session/{session_id}/transcript")
def get_transcript(session_id: str):
    session = _get_session(session_id)
    state: LessonState = session["state"]
    return {"transcript": state.transcript}


@app.get("/health")
def health():
    return {"status": "ok"}