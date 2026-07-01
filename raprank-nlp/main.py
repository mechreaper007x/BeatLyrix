from fastapi import FastAPI
from models.schemas import AnalyzeRequest, ScoreBreakdown
from services import syllable_service, alliteration_service

app = FastAPI(title="RapRank NLP Service", version="1.0.0")

@app.post("/analyze", response_model=ScoreBreakdown)
async def analyze_lyrics(request: AnalyzeRequest):
    lyrics = request.lyrics
    
    # Calculate scores
    syllable_score, avg_syllables = syllable_service.calculate(lyrics)
    alliteration_score, pairs = alliteration_service.calculate(lyrics)
    
    # Flow score = weighted combination (from prompt specification)
    flow_score = (syllable_score * 0.6) + (alliteration_score * 0.4)
    
    # Total weighted score
    total_score = (
        syllable_score * 0.40 +
        alliteration_score * 0.35 +
        flow_score * 0.25
    )
    
    word_count = len(lyrics.split())
    line_count = len(lyrics.strip().split('\n'))
    
    return ScoreBreakdown(
        syllable_score=round(syllable_score, 2),
        alliteration_score=round(alliteration_score, 2),
        flow_score=round(flow_score, 2),
        total_score=round(total_score, 2),
        word_count=word_count,
        line_count=line_count,
        avg_syllables_per_word=round(avg_syllables, 2),
        alliteration_pairs=pairs
    )

@app.get("/health")
async def health():
    return {"status": "ok", "service": "raprank-nlp"}
