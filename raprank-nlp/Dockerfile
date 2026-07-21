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

# Create non-root user for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

EXPOSE 7860
EXPOSE 8000

ENV WHISPER_API_URL=https://mechreaper007x-raprank-whisper.hf.space

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
