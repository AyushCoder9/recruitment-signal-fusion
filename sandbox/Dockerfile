# Backup sandbox image (HF Spaces is primary). CPU-only.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Pre-cache the embedding model so the container needs no network at run time.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-small-en-v1.5')"

EXPOSE 8501
CMD ["streamlit", "run", "sandbox/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
