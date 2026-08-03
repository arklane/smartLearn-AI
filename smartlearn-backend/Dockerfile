FROM python:3.11-slim

WORKDIR /app

# System libraries needed by faiss-cpu / torch / sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the MiniLM model into the folder rag.py resolves first.
# This avoids downloading from HuggingFace at request time on Railway.
RUN python - <<'EOF'
from pathlib import Path

from sentence_transformers import SentenceTransformer

model_dir = Path("/app/artifacts/rag/hf_models/all-MiniLM-L6-v2")
model_dir.mkdir(parents=True, exist_ok=True)
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
    model_kwargs={"use_safetensors": False},
)
model.save(str(model_dir))
print("MiniLM model saved to:", model_dir)
EOF

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
