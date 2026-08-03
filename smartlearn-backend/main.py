import os

from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

load_dotenv()

from .services import rag

app = FastAPI(title="SmartLearn Lite API")

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# In-memory storage: chat_id -> Day 3 RAG-ready document record
documents: dict[str, dict] = {}

# Artifact / upload locations for the backend routes
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_ROOT = os.getenv("UPLOAD_ROOT", os.path.join(BACKEND_DIR, "uploads"))
ARTIFACT_ROOT = os.getenv(
    "ARTIFACT_ROOT", os.path.join(BACKEND_DIR, "artifacts", "rag")
)

CHUNK_MODE = os.getenv("CHUNK_MODE", "character_overlap")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
OVERLAP = int(os.getenv("OVERLAP", "120"))
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
CANDIDATE_POOL = 60
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "poolside/laguna-s-2.1:free")

# Optional PostgreSQL history store (Lab C Appendix A)
_db_url = os.getenv("DAY3_DB_URL", "").strip()
_session_factory = None
if _db_url:
    try:
        _engine = rag.build_history_engine(_db_url)
        _session_factory = rag.build_history_session_factory(_engine)
        rag.ensure_history_tables(_engine)
    except Exception as exc:
        print(f"PostgreSQL history store disabled: {exc}")
        _session_factory = None


@app.get("/")
def root():
    return {"message": "SmartLearn Lite API is running"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/upload")
async def upload(
    file: UploadFile,
    chat_id: str = Query(..., description="Chat session identifier"),
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()

    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        document = rag.prepare_rag_chat_record(
            chat_id=chat_id,
            filename=file.filename,
            pdf_bytes=pdf_bytes,
            upload_root=UPLOAD_ROOT,
            chunk_mode=CHUNK_MODE,
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP,
            model_name=EMBED_MODEL_NAME,
            batch_size=32,
            artifact_root=ARTIFACT_ROOT,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    documents.pop(chat_id, None)
    documents[chat_id] = document

    return rag.build_upload_response(document)


@app.get("/documents/{chat_id}/file")
def get_document_file(chat_id: str):
    document = documents.get(chat_id)
    if document is None:
        raise HTTPException(
            status_code=404,
            detail=f"No document found for chat_id '{chat_id}'. Please upload a PDF first.",
        )

    file_path = document.get("saved_pdf_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"Saved PDF for chat_id '{chat_id}' is missing.",
        )

    return FileResponse(file_path, media_type="application/pdf")


class ChatRequest(BaseModel):
    chat_id: str = "day2-demo"
    message: str = Field(..., min_length=2, max_length=2000)


@app.post("/chat")
def chat(req: ChatRequest):
    document = documents.get(req.chat_id)
    if document is None:
        raise HTTPException(
            status_code=404,
            detail=f"No document found for chat_id '{req.chat_id}'. Please upload a PDF first.",
        )

    if _session_factory is not None:
        result = rag.answer_chat_turn_with_history_store(
            document,
            req.chat_id,
            req.message,
            session_factory=_session_factory,
            top_k=TOP_K,
            candidate_pool=CANDIDATE_POOL,
            answer_model=ANSWER_MODEL,
        )
    else:
        result = rag.answer_chat_turn(
            document,
            req.message,
            top_k=TOP_K,
            candidate_pool=CANDIDATE_POOL,
            answer_model=ANSWER_MODEL,
        )

    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "sources": result["sources"],
    }
