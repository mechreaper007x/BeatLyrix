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
COPY raprank-nlp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user for Hugging Face Spaces / Render compatibility
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user raprank-nlp/ $HOME/app

EXPOSE 8000
EXPOSE 7860

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
