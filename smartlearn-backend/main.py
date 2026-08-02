import os
import re

from fastapi import FastAPI, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .services.pdf import extract_pages
from .services.llm import answer_from_pages

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

# 临时内存存储
documents: dict[str, list[dict]] = {}


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
        pages = extract_pages(pdf_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    total_chars = sum(len(p["text"]) for p in pages)
    if total_chars == 0:
        raise HTTPException(
            status_code=422,
            detail="PDF contains no readable text. Scanned/image-only PDFs are not supported (OCR is not available).",
        )

    documents[chat_id] = pages

    return {
        "status": "ok",
        "filename": file.filename,
        "pages": len(pages),
        "characters": total_chars,
    }


class ChatRequest(BaseModel):
    chat_id: str = "day2-demo"
    message: str = Field(..., min_length=2, max_length=2000)


@app.post("/chat")
def chat(req: ChatRequest):
    pages = documents.get(req.chat_id)
    if pages is None:
        raise HTTPException(
            status_code=404,
            detail=f"No document found for chat_id '{req.chat_id}'. Please upload a PDF first.",
        )

    try:
        raw_answer = answer_from_pages(pages, req.message)
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="The AI service is currently unavailable. Please try again later.",
        )

    found_pages = set(int(n) for n in re.findall(r"\[Page\s+(\d+)\]", raw_answer))
    valid_page_numbers = {p["page"] for p in pages}
    citations = sorted(found_pages & valid_page_numbers)

    return {
        "answer": raw_answer,
        "citations": citations,
    }