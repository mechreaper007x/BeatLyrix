# ─────────────────────────────────────────────────────────────────────────────
# RapRank · Hindi NER Microservice
# Hugging Face Docker Space — port 7860 mandatory
#
# Model: ai4bharat/IndicNER
#   - bert-base-multilingual-uncased fine-tuned on 11 Indian languages
#   - Supports Devanagari (Hindi) + Romanized Hinglish via subword tokenizer
#   - MIT License
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Dependencies first (separate layer for cache efficiency)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download and cache the model at build time so cold starts are instant
# The model is ~591MB; baking it into the image avoids per-request downloads.
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForTokenClassification; \
AutoTokenizer.from_pretrained('ai4bharat/IndicNER'); \
AutoModelForTokenClassification.from_pretrained('ai4bharat/IndicNER')\
"

COPY . .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
