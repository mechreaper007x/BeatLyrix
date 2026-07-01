from pydantic import BaseModel
from typing import List

class AnalyzeRequest(BaseModel):
    lyrics: str

class ScoreBreakdown(BaseModel):
    syllable_score: float
    alliteration_score: float
    flow_score: float
    total_score: float
    word_count: int
    line_count: int
    avg_syllables_per_word: float
    alliteration_pairs: List[str]
