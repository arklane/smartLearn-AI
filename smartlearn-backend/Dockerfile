FROM python:3.11-slim

WORKDIR /app

# System libraries needed by faiss-cpu / torch / sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the MiniLM model files into the folder rag.py resolves first.
# snapshot_download pulls the full repo (pytorch_model.bin + model.safetensors),
# so load_model works regardless of which weight format it prefers.
RUN python - <<'EOF'
from huggingface_hub import snapshot_download

model_dir = "/app/artifacts/rag/hf_models/all-MiniLM-L6-v2"
snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    local_dir=model_dir,
    local_dir_use_symlinks=False,
)
print("MiniLM model downloaded to:", model_dir)
EOF

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
