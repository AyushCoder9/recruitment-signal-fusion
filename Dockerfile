# Hugging Face Space (Docker SDK) — Streamlit candidate-ranking sandbox. CPU-only.
# HF Spaces expect the app on port 7860.
FROM python:3.12-slim

ENV HF_HOME=/app/.cache \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Pre-cache the embedding model into the image so the running container needs no network.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-small-en-v1.5')"

EXPOSE 7860
# enableXsrfProtection=false: Streamlit's file_uploader otherwise returns HTTP 403 when the
# app is served through the Hugging Face Spaces iframe/proxy.
CMD ["streamlit", "run", "sandbox/app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false"]
