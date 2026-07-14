# ─────────────────────────────────────────────────────────────────────────────
# RapRank · Hindi Semantic Scoring Microservice
# Hugging Face Docker Space — port 7860 mandatory
#
# Models:
#   - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (embeddings;
#     cosine-meaningful, multilingual incl. Hindi/Hinglish, ~470MB)
#   - google/muril-base-cased (masked-LM surprisal only; Devanagari +
#     Romanized Indian text, ~900MB)
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Dependencies first (separate layer for cache efficiency)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download and cache both models at build time so cold starts are instant
# (baking them into the image avoids per-request downloads).
RUN python -c "\
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM; \
E='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'; \
M='google/muril-base-cased'; \
AutoTokenizer.from_pretrained(E); AutoModel.from_pretrained(E); \
AutoTokenizer.from_pretrained(M); AutoModelForMaskedLM.from_pretrained(M)\
"

COPY . .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
