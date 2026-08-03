"""
RAG helpers: text cleaning, page loading, chunking, embeddings, and artifact management.

This module is the single implementation target for Day 3 logic.
All functions are reusable from both the notebook and the backend routes.
"""

from __future__ import annotations

import json
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
    # Remove null bytes
    text = text.replace("\x00", "")
    # Remove soft hyphens
    text = text.replace("\u00ad", "")
    # Normalize unicode (NFC)
    text = unicodedata.normalize("NFC", text)
    # Replace carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse three or more newlines into two (keep paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace single newlines that are not paragraph breaks with a space
    # (keep double newlines as paragraph separators)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Collapse repeated spaces / tabs into one space
    text = re.sub(r"[ \t]+", " ", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


# ---------------------------------------------------------------------------
# 2. Page loading
# ---------------------------------------------------------------------------


def extract_pages_for_rag(pdf_path: Union[str, Path]) -> list[dict]:
    """Read a PDF page by page and return only readable ``{page, text}`` records.

    Unlike ``pdf.extract_pages``, this helper:
    - accepts a file path instead of raw bytes,
    - does **not** hard-code a 30-page limit,
    - applies ``clean_text`` to every page,
    - drops pages whose cleaned text is empty.
    """
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))
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
        # Try to find the last space within the window to avoid word splits
        space_idx = text.rfind(" ", start, end)
        if space_idx > start:
            pieces.append(text[start:space_idx])
            start = space_idx + 1  # skip the space
        else:
            # No space found — hard cut
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
    # Build a flat text with page boundary tracking
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
        # Determine which page this chunk starts in
        page = page_boundaries[-1][2]  # fallback
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
    """Select the requested chunking strategy and return a uniform chunk list.

    Supported ``chunk_mode`` values:
    - ``"paragraph"``
    - ``"character"``
    - ``"character_overlap"``
    - ``"langchain_recursive"`` (requires langchain-text-splitters)
    """
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
# 6. LangChain recursive splitter (Appendix A, optional)
# ---------------------------------------------------------------------------


def chunk_with_langchain_recursive(
    pages: list[dict],
    chunk_size: int = 700,
    chunk_overlap: int = 120,
    separators: list[str] | None = None,
) -> list[dict]:
    """Split pages using LangChain ``RecursiveCharacterTextSplitter``.

    The splitter tries larger separators first (``\\n\\n``, ``\\n``, ``" "``,
    ``""``) so that chunk boundaries tend to land at paragraph or sentence
    edges when possible.

    Raises ``ImportError`` with a clear message if the package is missing.
    """
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
    """Turn a model name into a safe filename suffix.

    Example: ``"sentence-transformers/all-MiniLM-L6-v2"`` -> ``"all_MiniLM_L6_v2"``
    """
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
    # Also check relative to this file (backend dir)
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
):
    """Create or reuse one SentenceTransformer model instance."""
    from sentence_transformers import SentenceTransformer

    cache_key = model_name
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    source = resolve_model_source(model_name, artifact_root)
    device = get_device()
    model = SentenceTransformer(
        source,
        device=device,
        model_kwargs={"use_safetensors": False},
    )
    _model_cache[cache_key] = model
    return model


def embed_texts(
    model,
    texts: list[str],
    batch_size: int = 32,
) -> np.ndarray:
    """Encode a list of texts into normalized ``float32`` vectors."""
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return vectors.astype(np.float32)


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


# ---------------------------------------------------------------------------
# 9. Full artifact pipeline
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

    # --- Check if existing manifest matches the current signature ---
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

    # --- Save raw pages ---
    save_json(pages, paths["raw_pages"])
    print(f"Saved raw pages: {paths['raw_pages']}")

    # --- Build chunks ---
    chunks = build_chunks(
        pages, chunk_mode=chunk_mode, chunk_size=chunk_size, overlap=overlap
    )
    save_json(chunks, paths["chunks"])
    print(f"Saved {len(chunks)} chunks: {paths['chunks']}")

    # --- Generate embeddings ---
    model = load_model(model_name, artifact_root=artifact_root)
    device = get_device()
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(model, texts, batch_size=batch_size)

    paths["embeddings"].parent.mkdir(parents=True, exist_ok=True)
    np.save(str(paths["embeddings"]), embeddings)
    print(f"Saved embeddings {embeddings.shape}: {paths['embeddings']}")

    # --- Save manifest ---
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
