# ─────────────────────────────────────────────────────────────────────────────
# RapRank — NLP Scoring Service
# Runs locally (called by Spring Boot backend on port 8000)
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
# ffmpeg: required by librosa for mp3 decoding (beat-sync feature)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (own layer for build cache efficiency) ────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY . .

# ── Port 8000 — matches app.python.service.url in Spring Boot ────────────────
EXPOSE 8000

# ── Environment ───────────────────────────────────────────────────────────────
# Override WHISPER_API_URL to point at a local whisper instance during dev:
#   docker run -e WHISPER_API_URL=http://localhost:7860 raprank-nlp
ENV WHISPER_API_URL=https://mechreaper007x-raprank-whisper.hf.space

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
