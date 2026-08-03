"""
RAG helpers: text cleaning, page loading, chunking, embeddings, retrieval,
and artifact management.

This module is the single implementation target for Day 3 logic.
All functions are reusable from both the notebooks and the backend routes.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from io import BytesIO
from pathlib import Path
from typing import Union

import numpy as np
from pypdf import PdfReader


# ---------------------------------------------------------------------------
# 1. Text cleaning
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Normalize one extracted page of PDF text.

    Removes null bytes, soft hyphens, repeated whitespace, and noisy
    line breaks while keeping paragraph-level structure.
    """
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = text.replace("\u00ad", "")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 2. Page loading
# ---------------------------------------------------------------------------


def extract_pages_for_rag(
    pdf_path: Union[str, Path],
    page_limit: int | None = None,
) -> list[dict]:
    """Read a PDF page by page and return only readable ``{page, text}`` records.

    Accepts a file path instead of raw bytes, keeps original page numbers,
    applies ``clean_text``, and drops empty pages. ``page_limit`` optionally
    caps how many pages are read (default: no limit).
    """
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))
    pages: list[dict] = []
    for page_number, page in enumerate(reader.pages, start=1):
        if page_limit is not None and page_number > page_limit:
            break
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
        if cleaned:
            pages.append({"page": page_number, "text": cleaned})
    return pages


def extract_pages_from_bytes_for_rag(pdf_bytes: bytes) -> list[dict]:
    """Extract cleaned page records from raw uploaded PDF bytes."""
    reader = PdfReader(BytesIO(pdf_bytes))
    pages: list[dict] = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
        if cleaned:
            pages.append({"page": page_number, "text": cleaned})
    return pages


# ---------------------------------------------------------------------------
# 3. JSON artifact helpers
# ---------------------------------------------------------------------------


def save_json(obj: object, path: Union[str, Path]) -> Path:
    """Save one Python object to a UTF-8 JSON file, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_json(path: Union[str, Path]) -> object:
    """Read one saved JSON artifact back into Python."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative_path_str(path: Union[str, Path], base: Union[str, Path]) -> str:
    """Return a shorter display path for ``path`` relative to ``base``."""
    try:
        return os.path.relpath(str(path), str(base))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# 4. Preview helper
# ---------------------------------------------------------------------------


def preview_records(records: list[dict], columns: list[str], rows: int = 5):
    """Show a small notebook table for chosen columns."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for preview_records") from exc

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    usable_columns = [c for c in columns if c in frame.columns]
    return frame[usable_columns].head(rows)


# ---------------------------------------------------------------------------
# 5. Chunking helpers
# ---------------------------------------------------------------------------


def slice_long_text(text: str, chunk_size: int) -> list[str]:
    """Split a single oversized text block into smaller pieces.

    Prefers natural boundaries (space) and avoids splits in the middle of
    words whenever possible.
    """
    if len(text) <= chunk_size:
        return [text]

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            pieces.append(text[start:])
            break
        space_idx = text.rfind(" ", start, end)
        if space_idx > start:
            pieces.append(text[start:space_idx])
            start = space_idx + 1
        else:
            pieces.append(text[start:end])
            start = end
    return [p for p in pieces if p.strip()]


def chunk_by_paragraph(pages: list[dict], chunk_size: int) -> list[dict]:
    """Convert paragraph-level records into chunks.

    Preserves page numbers and paragraph order. When a single paragraph
    exceeds ``chunk_size``, it is sliced via ``slice_long_text``.
    """
    chunks: list[dict] = []
    chunk_id = 0
    for record in pages:
        page = record["page"]
        paragraphs = [p.strip() for p in record["text"].split("\n\n") if p.strip()]
        for para in paragraphs:
            if len(para) <= chunk_size:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page,
                        "text": para,
                        "chunk_mode": "paragraph",
                    }
                )
                chunk_id += 1
            else:
                for piece in slice_long_text(para, chunk_size):
                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "page": page,
                            "text": piece,
                            "chunk_mode": "paragraph",
                        }
                    )
                    chunk_id += 1
    return chunks


def chunk_by_characters(
    pages: list[dict],
    chunk_size: int,
    overlap: int = 0,
) -> list[dict]:
    """Create fixed-size sliding-window chunks across all pages.

    When ``overlap > 0``, consecutive chunks share ``overlap`` characters.
    Each chunk records the page number of the page that contributed its
    starting character.
    """
    full_text = ""
    page_boundaries: list[tuple[int, int, int]] = []  # (start, end, page)
    for record in pages:
        start = len(full_text)
        full_text += record["text"] + " "
        end = len(full_text)
        page_boundaries.append((start, end, record["page"]))

    if not full_text.strip():
        return []

    step = max(chunk_size - overlap, 1)
    mode = "character_overlap" if overlap > 0 else "character"

    chunks: list[dict] = []
    chunk_id = 0
    pos = 0
    while pos < len(full_text):
        end = pos + chunk_size
        text = full_text[pos:end].strip()
        if not text:
            pos += step
            continue
        page = page_boundaries[-1][2]
        for pb_start, pb_end, pb_page in page_boundaries:
            if pb_start <= pos < pb_end:
                page = pb_page
                break
        chunks.append(
            {
                "chunk_id": chunk_id,
                "page": page,
                "text": text,
                "chunk_mode": mode,
            }
        )
        chunk_id += 1
        pos += step
    return chunks


def build_chunks(
    pages: list[dict],
    chunk_mode: str = "character_overlap",
    chunk_size: int = 700,
    overlap: int = 120,
) -> list[dict]:
    """Select the requested chunking strategy and return a uniform chunk list."""
    if chunk_mode == "paragraph":
        return chunk_by_paragraph(pages, chunk_size)
    elif chunk_mode == "character":
        return chunk_by_characters(pages, chunk_size, overlap=0)
    elif chunk_mode == "character_overlap":
        return chunk_by_characters(pages, chunk_size, overlap=overlap)
    elif chunk_mode == "langchain_recursive":
        return chunk_with_langchain_recursive(pages, chunk_size, chunk_overlap=overlap)
    else:
        raise ValueError(f"Unknown chunk_mode: {chunk_mode!r}")


# ---------------------------------------------------------------------------
# 6. LangChain recursive splitter (Lab A Appendix A, optional)
# ---------------------------------------------------------------------------


def chunk_with_langchain_recursive(
    pages: list[dict],
    chunk_size: int = 700,
    chunk_overlap: int = 120,
    separators: list[str] | None = None,
) -> list[dict]:
    """Split pages using LangChain ``RecursiveCharacterTextSplitter``."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise ImportError(
            "langchain-text-splitters is required for langchain_recursive mode. "
            "Install it with: pip install langchain-text-splitters"
        ) from exc

    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
    )

    chunks: list[dict] = []
    chunk_id = 0
    for record in pages:
        page = record["page"]
        pieces = splitter.split_text(record["text"])
        for piece in pieces:
            text = piece.strip()
            if not text:
                continue
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page": page,
                    "text": text,
                    "chunk_mode": "langchain_recursive",
                }
            )
            chunk_id += 1
    return chunks


# ---------------------------------------------------------------------------
# 7. Embedding helpers
# ---------------------------------------------------------------------------

_model_cache: dict[str, object] = {}


def model_tag(model_name: str) -> str:
    """Turn a model name into a safe filename suffix."""
    short = model_name.rsplit("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9]+", "_", short)


def resolve_model_source(
    model_name: str,
    artifact_root: Union[str, Path, None] = None,
) -> str:
    """Prefer a local cached model folder when it already exists.

    Checks two candidate locations:
    1. ``<artifact_root>/hf_models/<local_name>``
    2. ``<backend_dir>/artifacts/rag/hf_models/<local_name>``

    Falls back to the original ``model_name`` for online download.
    """
    local_name = model_name.rsplit("/", 1)[-1]
    required_file = "modules.json"
    candidates: list[Path] = []
    if artifact_root is not None:
        candidates.append(Path(artifact_root) / "hf_models" / local_name)
    backend_dir = Path(__file__).resolve().parent.parent
    candidates.append(backend_dir / "artifacts" / "rag" / "hf_models" / local_name)

    for path in candidates:
        if (path / required_file).exists():
            return str(path)
    return model_name


def get_device() -> str:
    """Choose CPU or CUDA for the current machine."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def load_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    artifact_root: Union[str, Path, None] = None,
    model_cache_dir: Union[str, Path, None] = None,
):
    """Create or reuse one SentenceTransformer model instance."""
    from sentence_transformers import SentenceTransformer

    cache_key = model_name
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    source = model_cache_dir or resolve_model_source(model_name, artifact_root)
    device = get_device()
    model = SentenceTransformer(
        source,
        device=device,
        model_kwargs={"use_safetensors": False},
    )
    _model_cache[cache_key] = model
    return model


def embed_texts(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    model_cache_dir: Union[str, Path, None] = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Encode a list of texts into normalized ``float32`` vectors.

    ``model_cache_dir`` lets callers point directly at a local cached model
    folder (for example the one recorded in a prepared document record).
    """
    if isinstance(texts, str):
        texts = [texts]
    model = load_model(model_name, model_cache_dir=model_cache_dir)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# 8. Artifact path management
# ---------------------------------------------------------------------------


def artifact_paths_for(
    document_id: str,
    chunk_mode: str,
    model_name: str,
    artifact_root: Union[str, Path] = Path("artifacts"),
) -> dict[str, Path]:
    """Decide where pages, chunks, embeddings, manifests, and indexes are saved."""
    root = Path(artifact_root)
    tag = model_tag(model_name)
    return {
        "raw_pages": root / "raw_pages" / f"{document_id}_pages.json",
        "chunks": root / "chunks" / f"{document_id}_{chunk_mode}.json",
        "embeddings": root / "embeddings" / f"{document_id}_{chunk_mode}_{tag}.npy",
        "manifest": root
        / "embeddings"
        / f"{document_id}_{chunk_mode}_{tag}.manifest.json",
        "faiss_index": root / "indexes" / f"{document_id}_{chunk_mode}_{tag}.faiss",
    }


def ensure_artifact_dirs(
    artifact_root: Union[str, Path, None] = None,
) -> dict[str, Path]:
    """Return all artifact folders, creating them if needed."""
    root = Path(artifact_root) if artifact_root is not None else Path("artifacts")
    dirs = {
        "raw_pages": root / "raw_pages",
        "chunks": root / "chunks",
        "embeddings": root / "embeddings",
        "indexes": root / "indexes",
        "reports": root / "reports",
        "chroma": root / "chroma",
        "hf_models": root / "hf_models",
    }
    for folder in dirs.values():
        folder.mkdir(parents=True, exist_ok=True)
    return dirs


# ---------------------------------------------------------------------------
# 9. Full artifact pipeline (Lab A)
# ---------------------------------------------------------------------------


def ensure_artifacts(
    document_id: str,
    pdf_name: str,
    pages: list[dict],
    chunk_mode: str = "character_overlap",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 700,
    overlap: int = 120,
    batch_size: int = 32,
    artifact_root: Union[str, Path] = Path("artifacts"),
) -> dict:
    """Build or reuse the full pages -> chunks -> embeddings -> manifest bundle.

    Returns a dict with keys: ``pages``, ``chunks``, ``embeddings``, ``manifest``.
    """
    paths = artifact_paths_for(document_id, chunk_mode, model_name, artifact_root)

    manifest_path = paths["manifest"]
    if manifest_path.exists():
        existing = load_json(manifest_path)
        sig_match = (
            existing.get("document_id") == document_id
            and existing.get("chunk_mode") == chunk_mode
            and existing.get("chunk_size") == chunk_size
            and existing.get("overlap") == overlap
            and existing.get("model_name") == model_name
            and existing.get("num_pages") == len(pages)
        )
        if sig_match and paths["embeddings"].exists() and paths["chunks"].exists():
            print(f"Reusing cached artifacts for {document_id} ({chunk_mode})")
            return {
                "pages": pages,
                "chunks": load_json(paths["chunks"]),
                "embeddings": np.load(str(paths["embeddings"])),
                "manifest": existing,
            }

    save_json(pages, paths["raw_pages"])
    print(f"Saved raw pages: {paths['raw_pages']}")

    chunks = build_chunks(
        pages, chunk_mode=chunk_mode, chunk_size=chunk_size, overlap=overlap
    )
    save_json(chunks, paths["chunks"])
    print(f"Saved {len(chunks)} chunks: {paths['chunks']}")

    device = get_device()
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(
        texts,
        model_name=model_name,
        model_cache_dir=resolve_model_source(model_name, artifact_root),
        batch_size=batch_size,
    )

    paths["embeddings"].parent.mkdir(parents=True, exist_ok=True)
    np.save(str(paths["embeddings"]), embeddings)
    print(f"Saved embeddings {embeddings.shape}: {paths['embeddings']}")

    manifest = {
        "document_id": document_id,
        "pdf_name": pdf_name,
        "num_pages": len(pages),
        "chunk_mode": chunk_mode,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "model_name": model_name,
        "num_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "device": device,
        "chunk_path": str(paths["chunks"]),
        "embedding_path": str(paths["embeddings"]),
        "raw_pages_path": str(paths["raw_pages"]),
    }
    save_json(manifest, paths["manifest"])
    print(f"Saved manifest: {paths['manifest']}")

    return {
        "pages": pages,
        "chunks": chunks,
        "embeddings": embeddings,
        "manifest": manifest,
    }


# ---------------------------------------------------------------------------
# 10. FAISS index helpers (Lab B)
# ---------------------------------------------------------------------------


def build_faiss_index(embeddings: np.ndarray):
    """Build a searchable FAISS inner-product index from normalized vectors."""
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.asarray(embeddings, dtype=np.float32))
    return index


def save_faiss_index(index, index_path: Union[str, Path]) -> None:
    """Write one FAISS index as a binary ``.faiss`` file."""
    import faiss

    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))


def load_faiss_index(index_path: Union[str, Path]):
    """Load one saved FAISS index back into memory."""
    import faiss

    return faiss.read_index(str(index_path))


def _index_dir_for(
    document_id: str,
    chunk_mode: str,
    chunk_size: int,
    overlap: int,
    model_name: str,
    artifact_root: Union[str, Path],
) -> Path:
    tag = model_tag(model_name)
    return (
        Path(artifact_root)
        / document_id
        / f"{chunk_mode}_c{chunk_size}_o{overlap}_{tag}"
    )


def ensure_index(
    document_id: str,
    pdf_name: str,
    pages: list[dict] | None = None,
    pdf_path: Union[str, Path, None] = None,
    chunk_mode: str = "character_overlap",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 700,
    overlap: int = 120,
    batch_size: int = 32,
    artifact_root: Union[str, Path, None] = None,
) -> dict:
    """Build or reuse chunks, embeddings, manifest, and a FAISS index.

    Returns a bundle dict with ``pages``, ``chunks``, ``embeddings``,
    ``manifest``, ``index``, ``index_path``, and model metadata.
    """
    root = Path(artifact_root) if artifact_root is not None else Path("artifacts")

    if pages is None:
        if pdf_path is None:
            raise ValueError("pages or pdf_path must be provided")
        pages = extract_pages_for_rag(pdf_path)

    artifacts = ensure_artifacts(
        document_id,
        pdf_name,
        pages,
        chunk_mode=chunk_mode,
        model_name=model_name,
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=batch_size,
        artifact_root=root,
    )

    index_dir = _index_dir_for(
        document_id, chunk_mode, chunk_size, overlap, model_name, root
    )
    index_path = index_dir / "index.faiss"
    meta_path = index_dir / "index.meta.json"

    signature = {
        "document_id": document_id,
        "pdf_name": pdf_name,
        "chunk_mode": chunk_mode,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "model_name": model_name,
        "num_chunks": len(artifacts["chunks"]),
        "embedding_dim": int(artifacts["embeddings"].shape[1]),
    }

    index = None
    if index_path.exists() and meta_path.exists():
        try:
            meta = load_json(meta_path)
            if meta.get("signature") == signature:
                index = load_faiss_index(index_path)
        except Exception:
            index = None

    if index is None:
        index = build_faiss_index(artifacts["embeddings"])
        save_faiss_index(index, index_path)
        save_json({"signature": signature}, meta_path)
        print(f"Saved FAISS index: {index_path}")

    paths = artifact_paths_for(document_id, chunk_mode, model_name, root)
    return {
        "pages": pages,
        "chunks": artifacts["chunks"],
        "embeddings": artifacts["embeddings"],
        "manifest": artifacts["manifest"],
        "index": index,
        "index_path": index_path,
        "chunk_path": paths["chunks"],
        "embedding_path": paths["embeddings"],
        "manifest_path": paths["manifest"],
        "model_name": model_name,
        "model_source": resolve_model_source(model_name, root),
    }


# ---------------------------------------------------------------------------
# 11. Retrieval helpers (Lab B)
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "to",
    "for",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "be",
    "by",
    "with",
    "as",
    "at",
    "from",
    "which",
    "what",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "used",
    "does",
    "do",
    "name",
    "model",
    "metric",
    "cover",
    "covers",
    "have",
    "has",
}


def keyword_set(text: str) -> set[str]:
    """Return lightweight lexical tokens from one question or chunk text."""
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {t for t in tokens if t not in _STOP_WORDS and len(t) > 1}


def _query_hits(
    question: str,
    index,
    chunks: list[dict],
    model_name: str,
    model_source: Union[str, None],
    top_k: int,
    candidate_pool: int,
    batch_size: int = 1,
) -> list[dict]:
    """Embed the question, search the index, and return ranked hits."""
    query_vec = embed_texts(
        [question],
        model_name=model_name,
        model_cache_dir=model_source,
        batch_size=batch_size,
    )
    scores, positions = index.search(
        np.asarray(query_vec, dtype=np.float32), candidate_pool
    )

    hits: list[dict] = []
    for score, pos in zip(scores[0], positions[0]):
        if pos < 0 or pos >= len(chunks):
            continue
        hits.append(
            {
                "chunk_id": chunks[pos]["chunk_id"],
                "page": chunks[pos]["page"],
                "text": chunks[pos]["text"],
                "score": float(score),
            }
        )

    q_tokens = keyword_set(question)
    for h in hits:
        h_tokens = keyword_set(h["text"])
        lexical = (
            (len(q_tokens & h_tokens) / max(len(q_tokens), 1)) if q_tokens else 0.0
        )
        h["combined"] = h["score"] + lexical
    hits.sort(key=lambda h: h["combined"], reverse=True)
    for h in hits:
        h.pop("combined", None)
    return hits[:top_k]


def search_bundle(
    question: str,
    bundle: dict,
    top_k: int = 3,
    candidate_pool: int = 60,
    batch_size: int = 1,
    history: list[dict] | None = None,
) -> list[dict]:
    """Search an in-memory index bundle and return top-k hits."""
    return _query_hits(
        question,
        bundle["index"],
        bundle["chunks"],
        bundle.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        bundle.get("model_source"),
        top_k,
        candidate_pool,
        batch_size=batch_size,
    )


def search_document(
    question: str,
    document: dict,
    top_k: int = 3,
    candidate_pool: int = 60,
    history: list[dict] | None = None,
) -> list[dict]:
    """Load the saved FAISS index for a prepared document and search it."""
    index_path = document["artifacts"]["index"]
    index = load_faiss_index(index_path)
    return _query_hits(
        question,
        index,
        document["chunks"],
        document.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        document.get("model_source"),
        top_k,
        candidate_pool,
    )


def split_sentences(text: str) -> list[str]:
    """Split one text block into candidate answer sentences."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def best_sentence_answer(question: str, hits: list[dict]) -> str:
    """Return one short local answer sentence with a page tag when possible."""
    q_tokens = keyword_set(question)
    best: tuple | None = None
    best_score = -1.0
    for hit in hits:
        for sent in split_sentences(hit["text"]):
            tokens = keyword_set(sent)
            if not tokens:
                continue
            score = (len(q_tokens & tokens) / max(len(q_tokens), 1)) + (
                len(q_tokens & tokens) / max(len(tokens), 1)
            )
            if score > best_score:
                best_score = score
                best = (hit, sent)
    if best:
        hit, sent = best
        return f"{sent} [Page {hit['page']}]"
    if hits:
        return f"{hits[0]['text'][:200]} [Page {hits[0]['page']}]"
    return "I could not find enough evidence in the document."


# ---------------------------------------------------------------------------
# 12. Project-facing wrapper (Lab B / Lab C)
# ---------------------------------------------------------------------------


def prepare_rag_document(
    document_id: str,
    filename: str,
    pages: list[dict],
    chunk_mode: str = "character_overlap",
    chunk_size: int = 700,
    overlap: int = 120,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    artifact_root: Union[str, Path, None] = None,
) -> dict:
    """Prepare one server-style document record with retrieval assets."""
    bundle = ensure_index(
        document_id,
        filename,
        pages=pages,
        chunk_mode=chunk_mode,
        model_name=model_name,
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=batch_size,
        artifact_root=artifact_root,
    )
    return {
        "document_id": document_id,
        "filename": filename,
        "pages": pages,
        "chunks": bundle["chunks"],
        "chunk_size": len(bundle["chunks"]),
        "embedding_dim": int(bundle["embeddings"].shape[1]),
        "model_name": model_name,
        "model_source": bundle["model_source"],
        "history": [],
        "artifacts": {
            "index": str(bundle["index_path"]),
            "chunks": str(bundle["chunk_path"]),
            "embeddings": str(bundle["embedding_path"]),
            "manifest": str(bundle["manifest_path"]),
        },
        "rag": {
            "document_id": document_id,
            "index_path": str(bundle["index_path"]),
            "model_name": model_name,
        },
    }


def extract_citations(answer: str, hits: list[dict] | None = None) -> list[int]:
    """Return numeric PDF page citations, preferring retrieved pages."""
    pages: set[int] = set()
    if hits:
        pages.update(int(h["page"]) for h in hits)
    if answer:
        pages.update(int(n) for n in re.findall(r"\[Page\s+(\d+)\]", answer))
    return sorted(pages)


def build_sources(hits: list[dict]) -> list[dict]:
    """Return frontend-friendly source objects with page, chunk id, score, preview."""
    return [
        {
            "page": h["page"],
            "chunk_id": h["chunk_id"],
            "score": h["score"],
            "preview": h["text"][:200],
        }
        for h in hits
    ]


def _llm_answer_from_hits(
    question: str,
    hits: list[dict],
    history: list[dict] | None = None,
    answer_model: str = "poolside/laguna-s-2.1:free",
) -> str:
    """Ask an OpenRouter LLM using only the retrieved chunks as evidence."""
    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    prompt = build_grounded_user_prompt(question, hits, history=history)
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    response = client.chat.completions.create(
        model=answer_model,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer only from the supplied evidence. "
                    "Cite factual claims with [Page X]. "
                    "If the evidence does not contain the answer, say so."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


def build_grounded_user_prompt(
    question: str,
    hits: list[dict],
    history: list[dict] | None = None,
) -> str:
    """Build one grounded prompt string from history and retrieved hits."""
    parts: list[str] = []
    if history:
        parts.append("### Conversation history")
        for turn in history[-4:]:
            parts.append(
                f"User: {turn.get('question', '')}\nAssistant: {turn.get('answer', '')}"
            )
    parts.append("### Retrieved evidence")
    for hit in hits:
        parts.append(f"[Page {hit['page']}]\n{hit['text']}")
    parts.append(f"### Question\n{question}")
    return "\n\n".join(parts)


def answer_document(
    document: dict,
    question: str,
    top_k: int = 3,
    candidate_pool: int = 60,
    answer_model: str = "poolside/laguna-s-2.1:free",
) -> dict:
    """Answer one question from retrieval, returning answer/citations/sources."""
    hits = search_document(
        question, document, top_k=top_k, candidate_pool=candidate_pool
    )

    answer = None
    if os.getenv("OPENROUTER_API_KEY"):
        try:
            answer = _llm_answer_from_hits(
                question,
                hits,
                history=document.get("history"),
                answer_model=answer_model,
            )
        except Exception:
            answer = None
    if not answer:
        answer = best_sentence_answer(question, hits)

    return {
        "answer": answer,
        "citations": extract_citations(answer, hits),
        "sources": build_sources(hits),
    }


def append_history(document: dict, question: str, result: dict) -> list[dict]:
    """Append one completed turn to the in-memory history and return it."""
    history = document.setdefault("history", [])
    history.append(
        {
            "question": question,
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
        }
    )
    return history


# ---------------------------------------------------------------------------
# 13. Lab C: upload record and multi-turn chat helpers
# ---------------------------------------------------------------------------


def prepare_rag_chat_record(
    chat_id: str,
    filename: str,
    pdf_bytes: bytes | None = None,
    pages: list[dict] | None = None,
    upload_root: Union[str, Path, None] = None,
    chunk_mode: str = "character_overlap",
    chunk_size: int = 700,
    overlap: int = 120,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    artifact_root: Union[str, Path, None] = None,
) -> dict:
    """Build an upload-time ``documents[chat_id]`` record for the backend route."""
    if pages is None:
        if pdf_bytes is None:
            raise ValueError("pages or pdf_bytes must be provided")
        pages = extract_pages_from_bytes_for_rag(pdf_bytes)

    upload_dir = (
        Path(upload_root)
        if upload_root is not None
        else Path(__file__).resolve().parent.parent / "uploads"
    )
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_pdf_path = upload_dir / f"{chat_id}.pdf"
    if pdf_bytes is not None:
        saved_pdf_path.write_bytes(pdf_bytes)

    document = prepare_rag_document(
        chat_id,
        filename,
        pages,
        chunk_mode=chunk_mode,
        chunk_size=chunk_size,
        overlap=overlap,
        model_name=model_name,
        batch_size=batch_size,
        artifact_root=artifact_root,
    )
    document["chat_id"] = chat_id
    document["file_path"] = str(saved_pdf_path)
    document["saved_pdf_path"] = str(saved_pdf_path)
    return document


def build_upload_response(document: dict) -> dict:
    """Build the visible Day 2-style upload success JSON for the frontend."""
    pages = document["pages"]
    return {
        "status": "ok",
        "filename": document.get("filename", ""),
        "pages": len(pages),
        "characters": sum(len(p["text"]) for p in pages),
    }


def answer_document_turn(
    document: dict,
    question: str,
    top_k: int = 3,
    candidate_pool: int = 60,
    answer_model: str = "poolside/laguna-s-2.1:free",
) -> dict:
    """Answer one question, append the turn to history, and return the result."""
    result = answer_document(
        document,
        question,
        top_k=top_k,
        candidate_pool=candidate_pool,
        answer_model=answer_model,
    )
    result["history"] = append_history(document, question, result)
    return result


def answer_chat_turn(
    document: dict,
    message: str,
    top_k: int = 3,
    candidate_pool: int = 60,
    answer_model: str = "poolside/laguna-s-2.1:free",
) -> dict:
    """Route-facing helper: fresh retrieval + answer + in-memory history update."""
    return answer_document_turn(
        document,
        message,
        top_k=top_k,
        candidate_pool=candidate_pool,
        answer_model=answer_model,
    )


# ---------------------------------------------------------------------------
# 14. Simple evaluation helpers (Lab B)
# ---------------------------------------------------------------------------


def normalize_for_match(text: str) -> str:
    """Normalize extracted text or a gold answer for simple string scoring."""
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_any_answer(text: str, answers: list[str]) -> bool:
    """Return whether any acceptable answer appears after normalization."""
    normalized = normalize_for_match(text)
    return any(normalize_for_match(a) in normalized for a in answers)


def evaluate_questions(
    eval_set: list[dict],
    documents_by_name: dict[str, dict],
    top_k: int = 3,
    candidate_pool: int = 60,
):
    """Score one short-answer question set and return one DataFrame row each."""
    import pandas as pd

    rows: list[dict] = []
    for item in eval_set:
        pdf_name = item["pdf_name"]
        document = documents_by_name[pdf_name]
        question = item["question"]
        answers = item["answers"]

        hits = search_document(
            question, document, top_k=top_k, candidate_pool=candidate_pool
        )
        local_answer = best_sentence_answer(question, hits)
        retrieval_hit = any(contains_any_answer(h["text"], answers) for h in hits)
        answer_hit = contains_any_answer(local_answer, answers)

        rows.append(
            {
                "pdf_name": pdf_name,
                "question": question,
                "local_answer": local_answer,
                "pages": sorted({h["page"] for h in hits}),
                "retrieval_hit": bool(retrieval_hit),
                "answer_hit": bool(answer_hit),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 15. Lab B Appendix A: optional Chroma branch
# ---------------------------------------------------------------------------


def _require_chromadb():
    """Return the chromadb module or raise a clear ImportError."""
    try:
        import chromadb

        return chromadb
    except ImportError as exc:
        raise ImportError(
            "chromadb is required for the Chroma branch. "
            "Install it with: pip install chromadb"
        ) from exc


def build_chroma_collection(
    document_id: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    persist_dir: Union[str, Path],
) -> dict:
    """Create or replace one persistent Chroma collection for a document."""
    chromadb = _require_chromadb()
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        name=document_id, metadata={"hnsw:space": "cosine"}
    )

    ids = [str(c["chunk_id"]) for c in chunks]
    metadatas = [{"page": c["page"]} for c in chunks]
    documents = [c["text"] for c in chunks]

    if collection.count() > 0:
        collection.delete(ids=collection.get()["ids"])

    collection.add(
        ids=ids,
        embeddings=np.asarray(embeddings, dtype=np.float32).tolist(),
        documents=documents,
        metadatas=metadatas,
    )
    return {"collection_name": document_id, "item_count": collection.count()}


def query_chroma_collection(
    document_id: str,
    query_embedding: np.ndarray,
    persist_dir: Union[str, Path],
    top_k: int,
) -> list[dict]:
    """Query one Chroma collection and return top-k hits."""
    chromadb = _require_chromadb()
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name=document_id)

    q = np.asarray(query_embedding)
    if q.ndim == 2:
        q = q[0]

    result = collection.query(
        query_embeddings=q.tolist(),
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )

    hits: list[dict] = []
    ids = result["ids"][0]
    distances = result["distances"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    for i, cid in enumerate(ids):
        hits.append(
            {
                "chunk_id": cid,
                "page": metadatas[i].get("page"),
                "text": documents[i],
                "score": float(1.0 - distances[i]),
            }
        )
    return hits


def search_document_with_chroma(
    question: str,
    document: dict,
    persist_dir: Union[str, Path],
    top_k: int = 3,
    batch_size: int = 1,
) -> list[dict]:
    """Search a document via the optional Chroma branch."""
    query_vec = embed_texts(
        [question],
        model_name=document.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        model_cache_dir=document.get("model_source"),
        batch_size=batch_size,
    )
    return query_chroma_collection(
        document["document_id"], query_vec, persist_dir, top_k
    )


def answer_document_with_chroma(
    document: dict,
    question: str,
    persist_dir: Union[str, Path],
    top_k: int = 3,
    answer_model: str = "poolside/laguna-s-2.1:free",
) -> dict:
    """Answer one question through the optional Chroma branch."""
    hits = search_document_with_chroma(question, document, persist_dir, top_k=top_k)

    answer = None
    if os.getenv("OPENROUTER_API_KEY"):
        try:
            answer = _llm_answer_from_hits(
                question,
                hits,
                history=document.get("history"),
                answer_model=answer_model,
            )
        except Exception:
            answer = None
    if not answer:
        answer = best_sentence_answer(question, hits)

    return {
        "answer": answer,
        "citations": extract_citations(answer, hits),
        "sources": build_sources(hits),
    }


# ---------------------------------------------------------------------------
# 16. Lab C Appendix A: optional PostgreSQL history store
# ---------------------------------------------------------------------------


def _require_sqlalchemy() -> dict[str, object]:
    """Return SQLAlchemy pieces or raise a clear ImportError."""
    try:
        from sqlalchemy import (
            Column,
            DateTime,
            ForeignKey,
            Integer,
            String,
            Text,
            create_engine,
        )
        from sqlalchemy.orm import declarative_base, relationship, sessionmaker

        return {
            "create_engine": create_engine,
            "sessionmaker": sessionmaker,
            "declarative_base": declarative_base,
            "Column": Column,
            "Integer": Integer,
            "String": String,
            "Text": Text,
            "DateTime": DateTime,
            "ForeignKey": ForeignKey,
            "relationship": relationship,
        }
    except ImportError as exc:
        raise ImportError(
            "sqlalchemy and psycopg[binary] are required for the PostgreSQL "
            'appendix. Install them with: pip install sqlalchemy "psycopg[binary]"'
        ) from exc


def _get_db_models() -> dict[str, object]:
    """Define (once) and return the Conversation / Message ORM models."""
    tools = _require_sqlalchemy()

    Base = tools["declarative_base"]()

    class Conversation(Base):
        __tablename__ = "conversations"
        id = tools["Column"](tools["Integer"], primary_key=True, autoincrement=True)
        chat_id = tools["Column"](
            tools["String"](255), unique=True, nullable=False, index=True
        )
        filename = tools["Column"](tools["String"](255), nullable=True)
        messages = tools["relationship"](
            "Message", back_populates="conversation", cascade="all, delete-orphan"
        )

    class Message(Base):
        __tablename__ = "messages"
        id = tools["Column"](tools["Integer"], primary_key=True, autoincrement=True)
        conversation_id = tools["Column"](
            tools["Integer"], tools["ForeignKey"]("conversations.id")
        )
        role = tools["Column"](tools["String"](32))
        content = tools["Column"](tools["Text"])
        citations_json = tools["Column"](tools["Text"], nullable=True)
        conversation = tools["relationship"]("Conversation", back_populates="messages")

    return {
        "Base": Base,
        "Conversation": Conversation,
        "Message": Message,
        "tools": tools,
    }


def build_history_engine(db_url: str):
    """Build one SQLAlchemy engine from a ``DAY3_DB_URL``."""
    tools = _require_sqlalchemy()
    return tools["create_engine"](db_url, future=True)


def build_history_session_factory(engine):
    """Return a factory that creates short database sessions."""
    tools = _require_sqlalchemy()
    return tools["sessionmaker"](bind=engine, future=True, expire_on_commit=False)


def ensure_history_tables(engine) -> None:
    """Create the conversation and message tables if they do not exist."""
    models = _get_db_models()
    models["Base"].metadata.create_all(bind=engine)


def get_or_create_conversation(session, chat_id: str, filename: str | None = None):
    """Return the matching conversation row, creating it when missing."""
    models = _get_db_models()
    conversation = (
        session.query(models["Conversation"]).filter_by(chat_id=chat_id).first()
    )
    if conversation is None:
        conversation = models["Conversation"](chat_id=chat_id, filename=filename)
        session.add(conversation)
        session.commit()
    return conversation


def load_history_from_db(session, chat_id: str) -> list[dict]:
    """Load previous turns in the same history-list shape used in memory."""
    models = _get_db_models()
    conversation = (
        session.query(models["Conversation"]).filter_by(chat_id=chat_id).first()
    )
    if conversation is None:
        return []

    messages = (
        session.query(models["Message"])
        .filter_by(conversation_id=conversation.id)
        .order_by(models["Message"].id)
        .all()
    )

    history: list[dict] = []
    pending_question: str | None = None
    for msg in messages:
        if msg.role == "user":
            pending_question = msg.content
        elif msg.role == "assistant":
            try:
                citations = json.loads(msg.citations_json or "[]")
            except (json.JSONDecodeError, TypeError):
                citations = []
            history.append(
                {
                    "question": pending_question or "",
                    "answer": msg.content,
                    "citations": citations,
                }
            )
            pending_question = None
    return history


def store_history_turn(
    session,
    chat_id: str,
    question: str,
    result: dict,
    filename: str | None = None,
) -> list[dict]:
    """Save one user + assistant turn and return the updated history list."""
    models = _get_db_models()
    conversation = get_or_create_conversation(session, chat_id, filename=filename)

    session.add(
        models["Message"](
            conversation_id=conversation.id,
            role="user",
            content=question,
        )
    )
    session.add(
        models["Message"](
            conversation_id=conversation.id,
            role="assistant",
            content=result.get("answer", ""),
            citations_json=json.dumps(result.get("citations", [])),
        )
    )
    session.commit()
    return load_history_from_db(session, chat_id)


def answer_chat_turn_with_history_store(
    document: dict,
    chat_id: str,
    message: str,
    session_factory,
    top_k: int = 3,
    candidate_pool: int = 60,
    answer_model: str = "poolside/laguna-s-2.1:free",
) -> dict:
    """Run one RAG turn with history loaded from and saved to PostgreSQL."""
    with session_factory() as session:
        history = load_history_from_db(session, chat_id)

    hits = search_document(
        message, document, top_k=top_k, candidate_pool=candidate_pool
    )

    answer = None
    if os.getenv("OPENROUTER_API_KEY"):
        try:
            answer = _llm_answer_from_hits(
                message, hits, history=history, answer_model=answer_model
            )
        except Exception:
            answer = None
    if not answer:
        answer = best_sentence_answer(message, hits)

    result = {
        "answer": answer,
        "citations": extract_citations(answer, hits),
        "sources": build_sources(hits),
    }

    with session_factory() as session:
        history = store_history_turn(
            session, chat_id, message, result, filename=document.get("filename")
        )

    result["history"] = history
    return result
