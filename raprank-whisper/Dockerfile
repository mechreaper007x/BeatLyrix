# ─────────────────────────────────────────────────────────────
# RapRank · Whisper Transcription Microservice
# Hugging Face Docker Space — port 7860 is mandatory
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────
# ffmpeg is required by faster-whisper to decode mp3/wav/ogg/etc.
# Skipping this will cause silent decode failures at runtime.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (own layer for cache efficiency) ─────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ───────────────────────────────────────
COPY . .

# ── Hugging Face Docker Spaces mandate port 7860 ────────────
EXPOSE 7860

# ── Start FastAPI via uvicorn ────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
